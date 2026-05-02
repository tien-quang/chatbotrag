'use strict';
/**
 * backend/src/services/ragService.js
 * RAG Pipeline — không LangGraph, retrieval tốt, citation chuẩn
 */

const fs   = require('fs');
const path = require('path');
const { ChromaClient } = require('chromadb');
const { OpenAI }       = require('openai');
const KnowledgeDocument = require('../models/KnowledgeDocument');
const Department        = require('../models/Department');

const CHUNK_SIZE    = 800;
const CHUNK_OVERLAP = 150;
const TOP_K_DOC     = 8;
const TOP_K_PROD    = 5;
const MAX_DIST      = 2.5; // QUAN TRỌNG: không filter chặt, để rerank quyết định

let _openai = null;
let _chroma = null;

function getOpenAI() {
  if (!_openai) {
    if (!process.env.OPENAI_API_KEY) throw new Error('OPENAI_API_KEY chưa cấu hình');
    _openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  }
  return _openai;
}

function getChroma() {
  if (!_chroma) _chroma = new ChromaClient({ path: process.env.CHROMA_URL || 'http://localhost:8000' });
  return _chroma;
}

const toColName = (code) => `tttn_${code.toLowerCase()}_docs`;

// ── Text Extraction ───────────────────────────────────────────────────
async function extractText(filePath, fileType) {
  const ext = (fileType || path.extname(filePath).slice(1)).toLowerCase();
  if (ext === 'pdf') {
    const pdfParse = require('pdf-parse');
    return (await pdfParse(fs.readFileSync(filePath))).text;
  }
  if (ext === 'docx') {
    const mammoth = require('mammoth');
    return (await mammoth.extractRawText({ path: filePath })).value;
  }
  if (['xlsx', 'xls'].includes(ext)) {
    const XLSX = require('xlsx');
    const wb   = XLSX.readFile(filePath);
    return wb.SheetNames.map(s => `[Sheet: ${s}]\n${XLSX.utils.sheet_to_csv(wb.Sheets[s])}`).join('\n\n');
  }
  if (['txt', 'md', 'csv'].includes(ext)) return fs.readFileSync(filePath, 'utf-8');
  throw new Error(`Định dạng .${ext} chưa được hỗ trợ`);
}

// ── Chunking ──────────────────────────────────────────────────────────
function chunkText(text, size = CHUNK_SIZE, overlap = CHUNK_OVERLAP) {
  const cleaned = text.replace(/\r\n/g, '\n').replace(/\n{3,}/g, '\n\n').replace(/[ \t]{2,}/g, ' ').trim();
  if (!cleaned) return [];
  const chunks = [];
  let start = 0;
  while (start < cleaned.length) {
    let end = start + size;
    if (end < cleaned.length) {
      for (const sep of ['\n\n', '.\n', '. ', '\n', ' ']) {
        const idx = cleaned.lastIndexOf(sep, end);
        if (idx > start + size * 0.5) { end = idx + sep.length; break; }
      }
    }
    const c = cleaned.slice(start, Math.min(end, cleaned.length)).trim();
    if (c.length > 50) chunks.push(c);
    start = end - overlap;
  }
  return chunks;
}

// ── Embedding ─────────────────────────────────────────────────────────
async function embedTexts(texts) {
  const all = [];
  for (let i = 0; i < texts.length; i += 50) {
    const res = await getOpenAI().embeddings.create({ model: 'text-embedding-3-small', input: texts.slice(i, i + 50) });
    all.push(...res.data.map(d => d.embedding));
  }
  return all;
}

// ── ChromaDB query helper ─────────────────────────────────────────────
async function queryChroma(colName, vec, topK) {
  try {
    const col   = await getChroma().getCollection({ name: colName });
    const count = await col.count();
    if (count === 0) return [];
    const res = await col.query({ queryEmbeddings: [vec], nResults: Math.min(topK, count), include: ['documents', 'metadatas', 'distances'] });
    return (res.documents?.[0] || []).map((doc, i) => ({
      text:     doc,
      docName:  res.metadatas[0][i]?.document_name   || colName,
      deptCode: res.metadatas[0][i]?.department_code || '',
      docId:    res.metadatas[0][i]?.document_id     || '',
      dist:     res.distances[0][i],
      isProd:   false,
    })).filter(c => c.dist < MAX_DIST);
  } catch { return []; }
}

// ── Rerank bằng LLM ──────────────────────────────────────────────────
// Lấy nhiều chunks → LLM chọn ra những cái thực sự liên quan
// Thay vì dùng số threshold cứng (gây miss)
async function rerank(question, chunks, topK = 5) {
  if (!chunks.length) return [];
  if (chunks.length <= topK) return chunks;
  const list = chunks.map((c, i) => `[${i}] ${c.isProd ? '[SP]' : '[Doc]'} ${c.docName}: ${c.text.slice(0, 220)}`).join('\n\n');
  try {
    const res = await getOpenAI().chat.completions.create({
      model: 'gpt-3.5-turbo', temperature: 0, max_tokens: 60,
      response_format: { type: 'json_object' },
      messages: [{ role: 'user', content: `Câu hỏi: "${question}"\n\nChọn tối đa ${topK} đoạn liên quan nhất (index 0-based):\n${list}\n\nJSON: {"ids":[0,1,...]}` }],
    });
    const { ids } = JSON.parse(res.choices[0].message.content);
    if (Array.isArray(ids) && ids.length > 0) return ids.filter(i => i >= 0 && i < chunks.length).map(i => chunks[i]);
  } catch {}
  return [...chunks].sort((a, b) => a.dist - b.dist).slice(0, topK);
}

// ── Query expansion (nhớ ngữ cảnh) ───────────────────────────────────
async function expandQuery(question, history) {
  if (!history?.length || question.length > 40) return question;
  const ctx = history.slice(-4).map(m => `${m.role === 'user' ? 'User' : 'AI'}: ${m.content.slice(0, 100)}`).join('\n');
  try {
    const res = await getOpenAI().chat.completions.create({
      model: 'gpt-3.5-turbo', temperature: 0, max_tokens: 60,
      messages: [{ role: 'user', content: `Hội thoại:\n${ctx}\n\nCâu hỏi ngắn: "${question}"\n\nViết lại đầy đủ, rõ nghĩa (1 câu tiếng Việt):` }],
    });
    const expanded = res.choices[0].message.content.trim().replace(/^["']|["']$/g, '');
    if (expanded && expanded !== question) { console.log(`[RAG] expand: "${question}" → "${expanded}"`); return expanded; }
  } catch {}
  return question;
}

// ══════════════════════════════════════════════════════════════════════
// PUBLIC: indexDocument
// ══════════════════════════════════════════════════════════════════════
async function indexDocument({ filePath, fileType, documentId, documentName, departmentCode, departmentId }) {
  console.log(`[RAG] Indexing: ${documentName}`);
  const rawText = await extractText(filePath, fileType);
  if (!rawText?.trim()) throw new Error('File rỗng hoặc không đọc được');
  const chunks = chunkText(rawText);
  if (!chunks.length) throw new Error('Không tạo được chunk');
  console.log(`[RAG] ${chunks.length} chunks`);
  const embeddings = await embedTexts(chunks);
  const colName    = toColName(departmentCode);
  const col        = await getChroma().getOrCreateCollection({ name: colName, metadata: { department_code: departmentCode, department_id: departmentId } });
  try { await col.delete({ where: { document_id: documentId } }); } catch {}
  await col.add({
    ids:        chunks.map((_, i) => `${documentId}_chunk_${i}`),
    embeddings,
    documents:  chunks,
    metadatas:  chunks.map((_, i) => ({ document_id: documentId, document_name: documentName, department_code: departmentCode, department_id: departmentId, chunk_index: i })),
  });
  await KnowledgeDocument.findByIdAndUpdate(documentId, { status: 'indexed', chunkCount: chunks.length, chromaCollectionId: colName });
  console.log(`[RAG] ✓ Indexed ${chunks.length} → ${colName}`);
  return { chunkCount: chunks.length, collectionName: colName };
}

async function deleteDocument({ documentId, departmentCode }) {
  try { const col = await getChroma().getCollection({ name: toColName(departmentCode) }); await col.delete({ where: { document_id: documentId } }); } catch (e) { console.warn('[RAG] deleteDocument:', e.message); }
}

// ══════════════════════════════════════════════════════════════════════
// PUBLIC: indexProduct
// ══════════════════════════════════════════════════════════════════════
async function indexProduct(product) {
  try {
    const fmt  = n => (n > 0) ? new Intl.NumberFormat('vi-VN').format(n) + ' VNĐ' : 'Liên hệ';
    const text = [`Tên sản phẩm: ${product.name}`, `Thương hiệu: ${product.brand || ''}`, `Danh mục: ${product.category || ''}`, `SKU: ${product.sku || ''}`, `Giá: ${fmt(product.price || 0)}`, `Tồn kho: ${product.stock ?? 0} sản phẩm`, product.description ? `Mô tả: ${product.description}` : ''].filter(Boolean).join('\n');
    const [embedding] = await embedTexts([text]);
    const col = await getChroma().getOrCreateCollection({ name: 'tttn_products', metadata: { type: 'products' } });
    const id  = product._id.toString();
    try { await col.delete({ where: { product_id: id } }); } catch {}
    await col.add({ ids: [id], embeddings: [embedding], documents: [text], metadatas: [{ product_id: id, name: product.name || '', brand: product.brand || '', category: product.category || '', sku: product.sku || '', price: Number(product.price) || 0, stock: Number(product.stock) || 0 }] });
    console.log(`[RAG] ✓ Product: ${product.name}`);
  } catch (e) { console.warn('[RAG] indexProduct:', e.message); }
}

async function deleteProduct(productId) {
  try { const col = await getChroma().getCollection({ name: 'tttn_products' }); await col.delete({ where: { product_id: productId.toString() } }); } catch (e) { console.warn('[RAG] deleteProduct:', e.message); }
}

// ══════════════════════════════════════════════════════════════════════
// PUBLIC: ragQuery
// ══════════════════════════════════════════════════════════════════════
async function ragQuery({ question, departmentCode, departmentName, systemPrompt, history, isMasterAdmin = false }) {
  // 1. Mở rộng câu hỏi ngắn từ lịch sử chat
  const q = await expandQuery(question, history);

  // 2. Embed
  const [vec] = await embedTexts([q]);

  // 3. Query tài liệu
  let docChunks = [];
  try {
    if (isMasterAdmin || departmentCode === 'ALL') {
      const depts = await Department.find({ isActive: true }).select('code').lean().catch(() => []);
      const results = await Promise.all(depts.map(d => queryChroma(toColName(d.code), vec, TOP_K_DOC)));
      docChunks = results.flat().sort((a, b) => a.dist - b.dist).slice(0, TOP_K_DOC);
    } else {
      docChunks = await queryChroma(toColName(departmentCode), vec, TOP_K_DOC);
    }
  } catch (e) { console.warn('[RAG] doc query:', e.message); }

  // 4. Query sản phẩm song song
  let prodChunks = [];
  try {
    const raw = await queryChroma('tttn_products', vec, TOP_K_PROD);
    prodChunks = raw.map(c => ({ ...c, isProd: true }));
  } catch {}

  console.log(`[RAG] docs=${docChunks.length} prods=${prodChunks.length} q="${q.slice(0,50)}"`);

  // 5. Rerank — LLM chọn chunks liên quan nhất thay vì filter số cứng
  const bestChunks = await rerank(q, [...docChunks, ...prodChunks], 5);

  if (!bestChunks.length) {
    return { answer: `Hệ thống chưa có thông tin về "${question}" trong tài liệu nội bộ.`, sources: [], tokens: 0 };
  }

  // 6. Build context
  const contextText = bestChunks.map(c => {
    const label = c.isProd ? `[Sản phẩm: ${c.docName}]` : `[Tài liệu: ${c.docName}]`;
    return `${label}\n${c.text}`;
  }).join('\n\n---\n\n');

  const role = systemPrompt || (isMasterAdmin ? 'Bạn là trợ lý AI nội bộ toàn hệ thống TTTN.' : `Bạn là trợ lý AI nội bộ phòng ${departmentName} của công ty TTTN.`);

  const systemMsg = `${role}

Trả lời dựa trên TÀI LIỆU NỘI BỘ bên dưới. Cuối mỗi ý chính ghi nguồn trong ngoặc đơn, ví dụ: (Nguồn: tttn Xác nhận.docx) hoặc (Nguồn: MacBook M7). Nếu không có thông tin thì nói "Không có thông tin trong tài liệu nội bộ." — không bịa thêm.

${contextText}`;

  const messages = [
    { role: 'system', content: systemMsg },
    ...(history || []).slice(-6).map(m => ({ role: m.role === 'assistant' ? 'assistant' : 'user', content: m.content })),
    { role: 'user', content: q },
  ];

  const res = await getOpenAI().chat.completions.create({ model: 'gpt-4o-mini', messages, max_tokens: 1200, temperature: 0.1 });
  const answer = res.choices[0].message.content;
  const tokens = res.usage?.total_tokens || 0;

  const seen    = new Set();
  const sources = [];
  for (const c of bestChunks) {
    if (!seen.has(c.docName)) {
      seen.add(c.docName);
      sources.push({ documentName: c.isProd ? `[Sản phẩm] ${c.docName}` : c.docName, documentId: c.docId || '', departmentCode: c.deptCode || '' });
    }
  }

  console.log(`[RAG] ✓ ${tokens} tokens | ${sources.length} sources`);
  return { answer, sources, tokens };
}

// Backward compat
async function retrieveChunks({ question, departmentCode, topK = 5 }) {
  const [vec] = await embedTexts([question]);
  return queryChroma(toColName(departmentCode), vec, topK);
}
async function retrieveProducts(question, topK = 3) {
  const [vec] = await embedTexts([question]);
  return (await queryChroma('tttn_products', vec, topK)).map(c => ({ text: c.text, meta: { name: c.docName, product_id: c.docId } }));
}
async function retrieveAllChunks(question, topK = 5) {
  const [vec]  = await embedTexts([question]);
  const depts  = await Department.find({ isActive: true }).select('code').lean().catch(() => []);
  const results = await Promise.all(depts.map(d => queryChroma(toColName(d.code), vec, topK)));
  return results.flat().sort((a, b) => a.dist - b.dist).slice(0, topK * 2);
}

module.exports = { ragQuery, indexDocument, deleteDocument, indexProduct, deleteProduct, extractText, chunkText, embedTexts, retrieveChunks, retrieveProducts, retrieveAllChunks };
