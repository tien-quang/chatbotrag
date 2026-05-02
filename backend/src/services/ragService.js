'use strict';
/**
 * backend/src/services/ragService.js
 *
 * LangGraph Multi-Agent RAG Pipeline
 * ───────────────────────────────────
 * Graph 1 — CHAT:
 *   START → memory → router → [doc_agent | product_agent | both_agent]
 *         → rerank → guardrail → generate → END
 *
 * Graph 2 — INDEX:
 *   START → extract → chunk → embed → save_chroma → update_mongo → END
 *
 * Graph 3 — PRODUCT:
 *   START → build_text → embed → upsert → END
 *
 * Fix trong version này:
 *  ✓ Context-aware: nhớ lịch sử chat để trả lời câu hỏi ngắn tiếp theo
 *  ✓ Query expansion: mở rộng câu hỏi ngắn dựa trên context
 *  ✓ Retrieval tốt hơn: nới ngưỡng distance, fallback khi rỗng
 *  ✓ Citation sạch: "Nguồn: tên_file" thay vì [1][2][3][4]
 *  ✓ Prompt ngắn gọn, tự nhiên
 *  ✓ Không có message "Bạn có thể: ..." dài dòng
 */

const fs   = require('fs');
const path = require('path');
const { ChromaClient }  = require('chromadb');
const OpenAI            = require('openai');
const { StateGraph, END, START, Annotation } = require('@langchain/langgraph');
const KnowledgeDocument = require('../models/KnowledgeDocument');
const Department        = require('../models/Department');

// ── Config ────────────────────────────────────────────────────────────
const CHUNK_SIZE    = 800;
const CHUNK_OVERLAP = 150;
const TOP_K         = 6;
const DOC_DIST      = 1.9;   // nới rộng (cũ 1.5 → bỏ sót nhiều)
const PROD_DIST     = 2.0;

// ── Singletons ────────────────────────────────────────────────────────
let _chroma = null;
let _openai = null;
const getChroma = () => {
  if (!_chroma) _chroma = new ChromaClient({ path: process.env.CHROMA_URL || 'http://localhost:8000' });
  return _chroma;
};
const getOpenAI = () => {
  if (!_openai) _openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  return _openai;
};
const toColName = (code) => `tttn_${code.toLowerCase()}_docs`;

// ══════════════════════════════════════════════════════════════════════
// BASE HELPERS
// ══════════════════════════════════════════════════════════════════════

async function extractText(filePath, fileType) {
  const ext = (fileType || path.extname(filePath).slice(1)).toLowerCase();
  if (ext === 'pdf') {
    return (await require('pdf-parse')(fs.readFileSync(filePath))).text;
  }
  if (ext === 'docx') {
    return (await require('mammoth').extractRawText({ path: filePath })).value;
  }
  if (['xlsx','xls'].includes(ext)) {
    const XLSX = require('xlsx');
    const wb   = XLSX.readFile(filePath);
    return wb.SheetNames.map(s => `Sheet: ${s}\n${XLSX.utils.sheet_to_csv(wb.Sheets[s])}`).join('\n\n');
  }
  return fs.readFileSync(filePath, 'utf-8');
}
function chunkText(text, size = CHUNK_SIZE, overlap = CHUNK_OVERLAP) {
  const chunks = [];
  let start = 0;
  while (start < text.length) {
    let end = start + size;
    if (end < text.length) {
      const bp = Math.max(text.lastIndexOf('.', end), text.lastIndexOf('\n', end));
      if (bp > start + size * 0.5) end = bp + 1;
    }
    const c = text.slice(start, end).trim();
    if (c.length > 50) chunks.push(c);
    start = end - overlap;
  }
  return chunks;
}

async function embedTexts(texts) {
  const all = [];
  for (let i = 0; i < texts.length; i += 50) {
    const res = await getOpenAI().embeddings.create({
      model: 'text-embedding-3-small',
      input: texts.slice(i, i + 50),
    });
    all.push(...res.data.map(d => d.embedding));
  }
  return all;
}

async function llm(messages, opts = {}) {
  const res = await getOpenAI().chat.completions.create({
    model:       'gpt-3.5-turbo',
    messages,
    temperature: opts.temperature ?? 0.1,
    max_tokens:  opts.maxTokens  ?? 1200,
    ...(opts.json ? { response_format: { type: 'json_object' } } : {}),
  });
  return {
    content: res.choices[0].message.content,
    tokens:  res.usage?.total_tokens || 0,
  };
}

// ══════════════════════════════════════════════════════════════════════
// GRAPH 1 — CHAT GRAPH (Multi-Agent)
// ══════════════════════════════════════════════════════════════════════

const ChatState = Annotation.Root({
  // Input
  question:       Annotation({ reducer: (_, b) => b ?? _, default: () => '' }),
  deptCode:       Annotation({ reducer: (_, b) => b ?? _, default: () => '' }),
  deptName:       Annotation({ reducer: (_, b) => b ?? _, default: () => '' }),
  systemPrompt:   Annotation({ reducer: (_, b) => b ?? _, default: () => '' }),
  history:        Annotation({ reducer: (_, b) => b ?? _, default: () => [] }),
  isAdmin:        Annotation({ reducer: (_, b) => b ?? _, default: () => false }),
  // Intermediate
  expandedQ:      Annotation({ reducer: (_, b) => b ?? _, default: () => '' }),
  queryType:      Annotation({ reducer: (_, b) => b ?? _, default: () => 'doc' }),
  qVec:           Annotation({ reducer: (_, b) => b ?? _, default: () => [] }),
  chunks:         Annotation({ reducer: (_, b) => b ?? _, default: () => [] }),
  hasCtx:         Annotation({ reducer: (_, b) => b ?? _, default: () => true }),
  // Output
  answer:         Annotation({ reducer: (_, b) => b ?? _, default: () => '' }),
  sources:        Annotation({ reducer: (_, b) => b ?? _, default: () => [] }),
  tokens:         Annotation({ reducer: (_, b) => b ?? _, default: () => 0 }),
});

// ── Node 1: Memory + Query Expansion ────────────────────────────────
// Đây là key fix: câu hỏi ngắn ("học trường gì?") được mở rộng
// dựa trên context lịch sử ("tiến thực tập ở đâu?")
async function nodeMemory(state) {
  const history = state.history || [];
  const q       = state.question.trim();

  // Nếu câu hỏi đã đủ ngữ cảnh hoặc không có lịch sử → giữ nguyên
  if (q.length > 30 || history.length === 0) {
    return { expandedQ: q };
  }

  // Lấy 4 turns gần nhất để làm context mở rộng câu hỏi
  const recentCtx = history.slice(-4)
    .map(m => `${m.role === 'user' ? 'Người dùng' : 'AI'}: ${m.content.slice(0, 120)}`)
    .join('\n');

  try {
    const { content } = await llm([{
      role: 'user',
      content: `Dựa vào cuộc hội thoại:\n${recentCtx}\n\nCâu hỏi ngắn mới: "${q}"\n\nHãy viết lại câu hỏi đầy đủ, rõ nghĩa (1 câu, tiếng Việt). Chỉ trả về câu hỏi, không giải thích.`,
    }], { temperature: 0, maxTokens: 80 });

    const expanded = content.trim().replace(/^["']|["']$/g, '');
    console.log(`[Memory] "${q}" → "${expanded}"`);
    return { expandedQ: expanded };
  } catch {
    return { expandedQ: q };
  }
}

// ── Node 2: Router ────────────────────────────────────────────────────
async function nodeRouter(state) {
  const q = state.expandedQ || state.question;
  try {
    const { content } = await llm([{
      role: 'user',
      content: `Phân loại câu hỏi: "${q}"\n→ "product" nếu hỏi về sản phẩm/giá/kho\n→ "both" nếu cần cả tài liệu lẫn sản phẩm\n→ "doc" cho mọi trường hợp còn lại\nJSON: {"type":"doc"|"product"|"both"}`,
    }], { temperature: 0, maxTokens: 40, json: true });
    const t = JSON.parse(content).type;
    const queryType = ['doc','product','both'].includes(t) ? t : 'doc';
    console.log(`[Router] type=${queryType} q="${q.slice(0, 50)}"`);
    return { queryType, expandedQ: q };
  } catch {
    return { queryType: 'doc', expandedQ: q };
  }
}

// ── Helper: query 1 ChromaDB collection ──────────────────────────────
async function queryCollection(name, vec, dist, topK) {
  try {
    const chroma = getChroma();
    const col    = await chroma.getCollection({ name });
    const count  = await col.count();
    if (count === 0) return [];

    const res = await col.query({
      queryEmbeddings: [vec],
      nResults:        Math.min(topK, count),
      include:         ['documents', 'metadatas', 'distances'],
    });

    return (res.documents?.[0] || []).map((doc, i) => ({
      text:     doc,
      docName:  res.metadatas[0][i]?.document_name   || path.basename(name),
      deptCode: res.metadatas[0][i]?.department_code || '',
      dist:     res.distances[0][i],
      isProd:   false,
    })).filter(c => c.dist < dist);
  } catch { return []; }
}
// ── Node 3: Doc Agent ─────────────────────────────────────────────────
async function nodeDocAgent(state) {
  console.log(`[DocAgent] dept=${state.deptCode} admin=${state.isAdmin}`);
  const [vec] = await embedTexts([state.expandedQ]);

  let deptCodes = [];
  if (state.isAdmin) {
    try {
      const depts = await Department.find({ isActive: true }).select('code').lean();
      deptCodes   = depts.map(d => d.code);
    } catch { deptCodes = ['HR','IT','SALES','ACCOUNTING']; }
  } else {
    deptCodes = [state.deptCode];
  }

  const results = await Promise.all(
    deptCodes.map(c => queryCollection(toColName(c), vec, DOC_DIST, TOP_K))
  );
  const chunks = results.flat().sort((a, b) => a.dist - b.dist).slice(0, TOP_K);
  console.log(`[DocAgent] ${chunks.length} chunks`);
  return { qVec: vec, chunks, hasCtx: chunks.length > 0 };
}

// ── Node 4: Product Agent ─────────────────────────────────────────────
async function nodeProductAgent(state) {
  console.log('[ProductAgent] searching...');
  const vec = state.qVec?.length ? state.qVec : (await embedTexts([state.expandedQ]))[0];
  const chunks = await queryCollection('tttn_products', vec, PROD_DIST, TOP_K);
  const prodChunks = chunks.map(c => ({ ...c, isProd: true, docName: c.docName || 'Catalog sản phẩm' }));
  console.log(`[ProductAgent] ${prodChunks.length} products`);
  return { qVec: vec, chunks: prodChunks, hasCtx: prodChunks.length > 0 };
}

// ── Node 5: Both Agent (docs + products) ─────────────────────────────
async function nodeBothAgent(state) {
  console.log('[BothAgent] searching docs + products...');
  const [vec] = await embedTexts([state.expandedQ]);

  let deptCodes = state.isAdmin
    ? (await Department.find({ isActive: true }).select('code').lean().catch(() => [])).map(d => d.code)
    : [state.deptCode];

  const [docResults, prodResults] = await Promise.all([
    Promise.all(deptCodes.map(c => queryCollection(toColName(c), vec, DOC_DIST, TOP_K))).then(r => r.flat()),
    queryCollection('tttn_products', vec, PROD_DIST, TOP_K).then(r => r.map(c => ({ ...c, isProd: true }))),
  ]);

  const chunks = [...docResults, ...prodResults].sort((a, b) => a.dist - b.dist).slice(0, TOP_K + 2);
  return { qVec: vec, chunks, hasCtx: chunks.length > 0 };
}

// Conditional edge
function routeAgent(state) {
  return { doc: 'doc_agent', product: 'product_agent', both: 'both_agent' }[state.queryType] || 'doc_agent';
}

// ── Node 6: Rerank ────────────────────────────────────────────────────
async function nodeRerank(state) {
  const { chunks, expandedQ } = state;
  if (!chunks?.length) return { hasCtx: false };
  if (chunks.length <= 3) return { hasCtx: true };

  try {
const list = chunks.map((c, i) => `[${i}] ${c.text.slice(0, 200)}`).join('\n\n');
    const { content } = await llm([{
      role: 'user',
      content: `Câu hỏi: "${expandedQ}"\n\nChọn tối đa 4 đoạn liên quan nhất:\n${list}\n\nJSON: {"ids":[<các số 0-based>]}`,
    }], { temperature: 0, maxTokens: 80, json: true });

    const { ids } = JSON.parse(content);
    if (Array.isArray(ids) && ids.length > 0) {
      const ranked = ids.filter(i => i >= 0 && i < chunks.length).map(i => chunks[i]);
      return { chunks: ranked, hasCtx: ranked.length > 0 };
    }
  } catch {}
  return { chunks: chunks.slice(0, 4), hasCtx: true };
}

// ── Node 7: Guardrail ─────────────────────────────────────────────────
async function nodeGuardrail(state) {
  if (!state.hasCtx || !state.chunks?.length) return { hasCtx: false };
  // Nếu ít nhất 1 chunk → tiếp tục (guardrail đơn giản để tránh over-reject)
  const topDist = state.chunks[0]?.dist || 2;
  return { hasCtx: topDist < 2.2 };
}

// ── Node 8: Generate ──────────────────────────────────────────────────
async function nodeGenerate(state) {
  // Không có context → trả lời ngắn gọn
  if (!state.hasCtx || !state.chunks?.length) {
    console.log('[Generate] No context found');
    return {
      answer:  `Tôi chưa có thông tin về "${state.expandedQ || state.question}" trong tài liệu nội bộ.`,
      sources: [],
      tokens:  0,
    };
  }

  const chunks = state.chunks;

  // Build context — KHÔNG dùng số [1][2][3]
  // Mỗi chunk ghi rõ tên file nguồn
  const contextText = chunks.map(c => {
    const src = c.isProd ? `[Sản phẩm]` : `[${c.docName}]`;
    return `${src}\n${c.text}`;
  }).join('\n\n---\n\n');

  // Lấy context lịch sử ngắn (3 turns gần nhất)
  const histCtx = (state.history || []).slice(-6)
    .map(m => `${m.role === 'user' ? 'Người dùng' : 'AI'}: ${m.content.slice(0, 200)}`)
    .join('\n');

  const basePrompt = state.systemPrompt ||
    'Bạn là trợ lý AI nội bộ của công ty TTTN, trả lời bằng tiếng Việt.';

  // System prompt ngắn gọn, tự nhiên
  const systemMsg =
`${basePrompt}

Dựa trên TÀI LIỆU NỘI BỘ dưới đây để trả lời. Sau câu trả lời, ghi "(Nguồn: tên_file)" cho từng thông tin quan trọng. Không bịa thêm thông tin ngoài tài liệu.${histCtx ? `\n\nNgữ cảnh cuộc trò chuyện:\n${histCtx}` : ''}

Tài liệu:
${contextText}`;

  const messages = [{ role: 'system', content: systemMsg }];
  // Thêm 4 turns history gần nhất để model hiểu mạch
  (state.history || []).slice(-4).forEach(m => {
    if (['user','assistant'].includes(m.role)) {
      messages.push({ role: m.role, content: m.content });
    }
  });
  messages.push({ role: 'user', content: state.expandedQ || state.question });
const { content, tokens } = await llm(messages, { temperature: 0.15, maxTokens: 1000 });

  // Build sources list (dedup)
  const seen = new Set();
  const sources = [];
  chunks.forEach(c => {
    const key = c.docName;
    if (!seen.has(key)) {
      seen.add(key);
      sources.push({
        documentName:   c.isProd ? `[Sản phẩm] ${c.docName}` : c.docName,
        departmentCode: c.deptCode,
        isProduct:      c.isProd || false,
      });
    }
  });

  console.log(`[Generate] ✓ ${tokens} tokens | ${sources.length} sources`);
  return { answer: content, sources, tokens };
}

// ── Compile Chat Graph ────────────────────────────────────────────────
let _chatGraph = null;
function getChatGraph() {
  if (_chatGraph) return _chatGraph;

  _chatGraph = new StateGraph(ChatState)
    .addNode('memory',       nodeMemory)
    .addNode('router',       nodeRouter)
    .addNode('doc_agent',    nodeDocAgent)
    .addNode('product_agent',nodeProductAgent)
    .addNode('both_agent',   nodeBothAgent)
    .addNode('rerank',       nodeRerank)
    .addNode('guardrail',    nodeGuardrail)
    .addNode('generate',     nodeGenerate)
    // Edges
    .addEdge(START,          'memory')
    .addEdge('memory',       'router')
    .addConditionalEdges('router', routeAgent, {
      doc_agent:     'doc_agent',
      product_agent: 'product_agent',
      both_agent:    'both_agent',
    })
    .addEdge('doc_agent',     'rerank')
    .addEdge('product_agent', 'rerank')
    .addEdge('both_agent',    'rerank')
    .addEdge('rerank',        'guardrail')
    .addEdge('guardrail',     'generate')
    .addEdge('generate',       END)
    .compile();

  console.log('[LangGraph] ✓ Chat: memory→router→[doc|product|both]→rerank→guardrail→generate');
  return _chatGraph;
}

// ══════════════════════════════════════════════════════════════════════
// GRAPH 2 — INDEX GRAPH
// ══════════════════════════════════════════════════════════════════════

const IndexState = Annotation.Root({
  filePath:   Annotation({ reducer: (_, b) => b ?? _, default: () => '' }),
  fileType:   Annotation({ reducer: (_, b) => b ?? _, default: () => '' }),
  docId:      Annotation({ reducer: (_, b) => b ?? _, default: () => '' }),
  docName:    Annotation({ reducer: (_, b) => b ?? _, default: () => '' }),
  deptCode:   Annotation({ reducer: (_, b) => b ?? _, default: () => '' }),
  deptId:     Annotation({ reducer: (_, b) => b ?? _, default: () => '' }),
  rawText:    Annotation({ reducer: (_, b) => b ?? _, default: () => '' }),
  chunks:     Annotation({ reducer: (_, b) => b ?? _, default: () => [] }),
  vectors:    Annotation({ reducer: (_, b) => b ?? _, default: () => [] }),
chunkCount: Annotation({ reducer: (_, b) => b ?? _, default: () => 0 }),
  colName:    Annotation({ reducer: (_, b) => b ?? _, default: () => '' }),
  error:      Annotation({ reducer: (_, b) => b ?? _, default: () => '' }),
});

async function idxExtract(state) {
  console.log(`[Index·extract] ${state.fileType} → ${state.docName}`);
  try {
    const text = await extractText(state.filePath, state.fileType);
    if (!text?.trim()) throw new Error('File rỗng');
    return { rawText: text };
  } catch (e) { return { error: e.message }; }
}

function idxChunk(state) {
  if (state.error) return {};
  const chunks = chunkText(state.rawText);
  if (!chunks.length) return { error: 'Không tạo được chunk' };
  console.log(`[Index·chunk] ${chunks.length} chunks`);
  return { chunks };
}

async function idxEmbed(state) {
  if (state.error || !state.chunks?.length) return {};
  console.log('[Index·embed] Embedding...');
  return { vectors: await embedTexts(state.chunks) };
}

async function idxSaveChroma(state) {
  if (state.error || !state.vectors?.length) return {};
  const name   = toColName(state.deptCode);
  const chroma = getChroma();
  const col    = await chroma.getOrCreateCollection({
    name,
    metadata: { department_code: state.deptCode, department_id: state.deptId },
  });
  // Xoá cũ → tránh duplicate khi reindex
  try { await col.delete({ where: { document_id: state.docId } }); } catch {}

  await col.add({
    ids:        state.chunks.map((_, i) => `${state.docId}_chunk_${i}`),
    embeddings: state.vectors,
    documents:  state.chunks,
    metadatas:  state.chunks.map((_, i) => ({
      document_id:     state.docId,
      document_name:   state.docName,
      department_code: state.deptCode,
      department_id:   state.deptId,
      chunk_index:     i,
    })),
  });
  console.log(`[Index·chroma] ✓ ${state.chunks.length} chunks → ${name}`);
  return { chunkCount: state.chunks.length, colName: name };
}

async function idxMongo(state) {
  if (state.error) {
    await KnowledgeDocument.findByIdAndUpdate(state.docId,
      { status: 'failed', errorMessage: state.error });
    return {};
  }
  await KnowledgeDocument.findByIdAndUpdate(state.docId, {
    status:             'indexed',
    chunkCount:         state.chunkCount,
    chromaCollectionId: state.colName,
  });
  console.log('[Index·mongo] ✓ status=indexed');
  return {};
}

let _indexGraph = null;
function getIndexGraph() {
  if (_indexGraph) return _indexGraph;
  _indexGraph = new StateGraph(IndexState)
    .addNode('extract',     idxExtract)
    .addNode('chunk',       idxChunk)
    .addNode('embed',       idxEmbed)
    .addNode('save_chroma', idxSaveChroma)
    .addNode('update_mongo',idxMongo)
    .addEdge(START,         'extract')
.addEdge('extract',     'chunk')
    .addEdge('chunk',       'embed')
    .addEdge('embed',       'save_chroma')
    .addEdge('save_chroma', 'update_mongo')
    .addEdge('update_mongo', END)
    .compile();
  console.log('[LangGraph] ✓ Index: extract→chunk→embed→chroma→mongo');
  return _indexGraph;
}

// ══════════════════════════════════════════════════════════════════════
// GRAPH 3 — PRODUCT GRAPH
// ══════════════════════════════════════════════════════════════════════

const ProductState = Annotation.Root({
  product:  Annotation({ reducer: (_, b) => b ?? _, default: () => ({}) }),
  mode:     Annotation({ reducer: (_, b) => b ?? _, default: () => 'upsert' }),
  prodText: Annotation({ reducer: (_, b) => b ?? _, default: () => '' }),
  vector:   Annotation({ reducer: (_, b) => b ?? _, default: () => [] }),
});

function prodBuild(state) {
  const p   = state.product;
  const fmt = n => (n && n > 0) ? new Intl.NumberFormat('vi-VN').format(n) + ' VNĐ' : 'Liên hệ';
  const text = [
    `Tên sản phẩm: ${p.name || ''}`,
    `Thương hiệu: ${p.brand || ''}`,
    `Danh mục: ${p.category || ''}`,
    `SKU: ${p.sku || ''}`,
    `Giá bán: ${fmt(p.price)}`,
    `Tồn kho: ${p.stock ?? 0} sản phẩm`,
    p.description ? `Mô tả: ${p.description}` : null,
  ].filter(Boolean).join('\n');
  return { prodText: text };
}

async function prodEmbed(state) {
  if (state.mode === 'delete') return {};
  const [vector] = await embedTexts([state.prodText]);
  return { vector };
}

async function prodUpsert(state) {
  const chroma = getChroma();
  const col    = await chroma.getOrCreateCollection({ name: 'tttn_products', metadata: { type: 'products' } });
  const id     = state.product._id?.toString();
  try { await col.delete({ where: { product_id: id } }); } catch {}

  if (state.mode === 'delete') {
    console.log('[ProductGraph] ✓ Deleted from ChromaDB');
    return {};
  }

  const p = state.product;
  await col.add({
    ids:        [`product_${id}`],
    embeddings: [state.vector],
    documents:  [state.prodText],
    metadatas:  [{
      product_id: id,
      name:       p.name     || '',
      brand:      p.brand    || '',
      category:   p.category || '',
      sku:        p.sku      || '',
      price:      Number(p.price) || 0,
      stock:      Number(p.stock) || 0,
    }],
  });
  console.log(`[ProductGraph] ✓ ${p.name} saved to ChromaDB`);
  return {};
}

let _productGraph = null;
function getProductGraph() {
  if (_productGraph) return _productGraph;
  _productGraph = new StateGraph(ProductState)
    .addNode('build', prodBuild)
    .addNode('embed', prodEmbed)
    .addNode('upsert', prodUpsert)
    .addEdge(START,   'build')
.addEdge('build', 'embed')
    .addEdge('embed', 'upsert')
    .addEdge('upsert', END)
    .compile();
  console.log('[LangGraph] ✓ Product: build→embed→upsert');
  return _productGraph;
}

// ══════════════════════════════════════════════════════════════════════
// PUBLIC API — interface giống hệt cũ, routes/ không cần sửa
// ══════════════════════════════════════════════════════════════════════

async function ragQuery({
  question, departmentCode, departmentName,
  systemPrompt, history, isMasterAdmin = false,
}) {
  try {
    const result = await getChatGraph().invoke({
      question,
      deptCode:     isMasterAdmin ? 'ALL' : (departmentCode || ''),
      deptName:     isMasterAdmin ? 'Toàn Hệ Thống' : (departmentName || ''),
      systemPrompt: systemPrompt || '',
      history:      history || [],
      isAdmin:      !!isMasterAdmin,
    });
    return {
      answer:  result.answer  || '',
      sources: result.sources || [],
      tokens:  result.tokens  || 0,
    };
  } catch (e) {
    console.error('[LangGraph] ragQuery error:', e.message);
    return { answer: `Lỗi hệ thống AI: ${e.message}`, sources: [], tokens: 0 };
  }
}

async function indexDocument({
  filePath, fileType, documentId, documentName, departmentCode, departmentId,
}) {
  await KnowledgeDocument.findByIdAndUpdate(documentId, { status: 'processing' });
  const result = await getIndexGraph().invoke({
    filePath, fileType,
    docId:    documentId,
    docName:  documentName,
    deptCode: departmentCode,
    deptId:   departmentId,
  });
  if (result.error) throw new Error(result.error);
  return { chunkCount: result.chunkCount, collectionName: result.colName };
}

async function deleteDocument({ documentId, departmentCode }) {
  try {
    const col = await getChroma().getCollection({ name: toColName(departmentCode) });
    await col.delete({ where: { document_id: documentId } });
    console.log(`[RAG] deleteDocument ✓ ${documentId}`);
  } catch (e) { console.warn('[RAG] deleteDocument:', e.message); }
}

async function indexProduct(product) {
  try { await getProductGraph().invoke({ product, mode: 'upsert' }); }
  catch (e) { console.warn('[LangGraph] indexProduct:', e.message); }
}

async function deleteProduct(productId) {
  try { await getProductGraph().invoke({ product: { _id: productId }, mode: 'delete' }); }
  catch (e) { console.warn('[LangGraph] deleteProduct:', e.message); }
}

// Backward compat
async function retrieveChunks({ question, departmentCode }) {
  const [vec] = await embedTexts([question]);
  return queryCollection(toColName(departmentCode), vec, DOC_DIST, TOP_K);
}

async function retrieveProducts(question) {
  const [vec] = await embedTexts([question]);
return queryCollection('tttn_products', vec, PROD_DIST, TOP_K);
}

module.exports = {
  ragQuery,
  indexDocument,
  deleteDocument,
  indexProduct,
  deleteProduct,
  retrieveChunks,
  retrieveProducts,
  extractText,
  chunkText,
  embedTexts,
};
