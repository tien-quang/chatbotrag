/**
 * RAG Service - Retrieval Augmented Generation Pipeline
 *
 * Flow:
 * 1. INDEXING: File → Extract Text → Chunk → OpenAI Embed → ChromaDB
 * 2. RETRIEVAL: Question → OpenAI Embed → ChromaDB Search → Top K chunks
 * 3. GENERATION: Question + Chunks → OpenAI GPT → Answer
 *
 * Fixes:
 *  ✓ Distance threshold nới từ 1.5 → 1.8 (tránh bỏ sót chunk liên quan)
 *  ✓ Fallback: nếu không chunk nào qua threshold → vẫn lấy top 3 gần nhất
 *  ✓ System prompt fix: GPT PHẢI dùng tài liệu khi có context
 *  ✓ Không dùng LangGraph
 */

const fs   = require('fs')
const path = require('path')
const { OpenAI }       = require('openai')
const { ChromaClient } = require('chromadb')

// ─── Config ────────────────────────────────────────────────────────────────────
const DIST_THRESHOLD = 1.8   // text-embedding-3-small: 0.0 = identical, 2.0 = very different
const FALLBACK_TOP   = 3     // nếu không chunk nào qua threshold → lấy N gần nhất nhất
const TOP_K_DOC      = 6
const TOP_K_PROD     = 3

// ─── Singletons ────────────────────────────────────────────────────────────────
let _openai = null
let _chroma = null

function getOpenAI() {
  if (!_openai) {
    const apiKey = process.env.OPENAI_API_KEY
    if (!apiKey || apiKey.startsWith('sk-your') || apiKey === 'sk-proj-your-key-here') {
      throw new Error('OPENAI_API_KEY chưa được cấu hình trong file .env')
    }
    _openai = new OpenAI({ apiKey })
  }
  return _openai
}

function getChroma() {
  if (!_chroma) {
    _chroma = new ChromaClient({ path: process.env.CHROMA_URL || 'http://localhost:8000' })
  }
  return _chroma
}

function colName(departmentCode) {
  return `tttn_${departmentCode.toLowerCase()}_docs`
}

// ─── TEXT EXTRACTION ────────────────────────────────────────────────────────────
async function extractText(filePath, fileType) {
  const ext = (fileType || path.extname(filePath).slice(1)).toLowerCase()

  if (ext === 'pdf') {
    const pdfParse = require('pdf-parse')
    const result   = await pdfParse(fs.readFileSync(filePath))
    return result.text
  }
  if (ext === 'docx') {
    const mammoth = require('mammoth')
    const result  = await mammoth.extractRawText({ path: filePath })
    return result.value
  }
  if (['txt', 'md', 'csv'].includes(ext)) {
    return fs.readFileSync(filePath, 'utf-8')
  }
  if (['xlsx', 'xls'].includes(ext)) {
    const XLSX = require('xlsx')
    const wb   = XLSX.readFile(filePath)
    return wb.SheetNames
      .map(n => `[Sheet: ${n}]\n${XLSX.utils.sheet_to_csv(wb.Sheets[n])}`)
      .join('\n\n')
  }
  throw new Error(`Định dạng file "${ext}" chưa được hỗ trợ`)
}

// ─── CHUNKING ───────────────────────────────────────────────────────────────────
function chunkText(text, chunkSize = 800, overlap = 150) {
  const cleaned = text
    .replace(/\r\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .replace(/[ \t]{2,}/g, ' ')
    .trim()

  if (!cleaned) return []

  const chunks = []
  let start    = 0

  while (start < cleaned.length) {
    let end = start + chunkSize
    if (end < cleaned.length) {
      for (const bp of ['\n\n', '.\n', '. ', '\n', ' ']) {
        const idx = cleaned.lastIndexOf(bp, end)
        if (idx > start + chunkSize * 0.5) { end = idx + bp.length; break }
      }
    }
    const chunk = cleaned.slice(start, Math.min(end, cleaned.length)).trim()
    if (chunk.length > 50) chunks.push(chunk)
    start = end - overlap
  }
  return chunks
}

// ─── EMBEDDING ──────────────────────────────────────────────────────────────────
async function embedTexts(texts) {
  const ai         = getOpenAI()
  const embeddings = []
  for (let i = 0; i < texts.length; i += 50) {
    const res = await ai.embeddings.create({
      model: 'text-embedding-3-small',
      input: texts.slice(i, i + 50),
    })
    embeddings.push(...res.data.map(e => e.embedding))
  }
  return embeddings
}

// ─── INDEX DOCUMENT ─────────────────────────────────────────────────────────────
async function indexDocument({ filePath, fileType, documentId, documentName, departmentCode, departmentId }) {
  console.log(`[RAG] Indexing: ${documentName}`)

  const rawText = await extractText(filePath, fileType)
  if (!rawText?.trim() || rawText.trim().length < 10) {
    throw new Error('Không trích xuất được nội dung từ file')
  }

  const chunks = chunkText(rawText)
  if (!chunks.length) throw new Error('File không có nội dung có thể xử lý')
  console.log(`[RAG] ${chunks.length} chunks`)

  const embeddings = await embedTexts(chunks)

  const chroma = getChroma()
  const col    = colName(departmentCode)

  let collection
  try {
    collection = await chroma.getOrCreateCollection({
      name:     col,
      metadata: { department_code: departmentCode, department_id: departmentId },
    })
  } catch (e) {
    throw new Error(`Không kết nối ChromaDB: ${e.message}`)
  }

  try { await collection.delete({ where: { document_id: documentId } }) } catch {}

  await collection.add({
    ids:        chunks.map((_, i) => `${documentId}_chunk_${i}`),
    embeddings,
    documents:  chunks,
    metadatas:  chunks.map((_, i) => ({
      document_id:     documentId,
      document_name:   documentName,
      department_code: departmentCode,
      department_id:   departmentId,
      chunk_index:     i,
    })),
  })

  console.log(`[RAG] Indexed ${chunks.length} chunks → ${col}`)
  return { chunkCount: chunks.length, collectionName: col }
}

// ─── DELETE DOCUMENT ────────────────────────────────────────────────────────────
async function deleteDocument({ documentId, departmentCode }) {
  try {
    const collection = await getChroma().getCollection({ name: colName(departmentCode) })
    await collection.delete({ where: { document_id: documentId } })
    console.log(`[RAG] Deleted ${documentId}`)
  } catch (e) {
    console.warn('[RAG] deleteDocument:', e.message)
  }
}

// ─── RETRIEVE CHUNKS ────────────────────────────────────────────────────────────
// KEY FIX: Không hard-filter distance trước.
// Lấy top K, ưu tiên relevant nhưng luôn có fallback top FALLBACK_TOP.
async function retrieveChunks({ question, departmentCode, topK = TOP_K_DOC }) {
  let collection
  try {
    collection = await getChroma().getCollection({ name: colName(departmentCode) })
  } catch {
    return []
  }

  const count = await collection.count()
  if (!count) return []

  const [qEmbed] = await embedTexts([question])
  const results  = await collection.query({
    queryEmbeddings: [qEmbed],
    nResults:        Math.min(topK, count),
    include:         ['documents', 'metadatas', 'distances'],
  })

  if (!results.documents?.[0]?.length) return []

  const chunks = results.documents[0].map((doc, i) => ({
    text:           doc,
    documentName:   results.metadatas[0][i]?.document_name   || 'Tài liệu',
    documentId:     results.metadatas[0][i]?.document_id,
    departmentCode: results.metadatas[0][i]?.department_code || departmentCode,
    distance:       results.distances[0][i] ?? 999,
  }))

  const relevant = chunks.filter(c => c.distance < DIST_THRESHOLD)

  if (relevant.length === 0) {
    // Fallback: vẫn trả về top chunks gần nhất để GPT tự đánh giá
    console.log(`[RAG] Fallback: no chunk under ${DIST_THRESHOLD}, returning top ${FALLBACK_TOP}`)
    return chunks.slice(0, FALLBACK_TOP)
  }

  return relevant
}

// ─── RETRIEVE ALL CHUNKS (Master Admin) ────────────────────────────────────────
async function retrieveAllChunks(question, topK = TOP_K_DOC) {
  const chroma = getChroma()
  let deptCodes = ['HR', 'IT', 'SALES', 'ACCOUNTING', 'GENERAL']

  try {
    const mongoose   = require('mongoose')
    const Department = mongoose.model('Department')
    const depts      = await Department.find({}, 'code').lean()
    if (depts.length) deptCodes = depts.map(d => d.code)
  } catch {}

  let qEmbed
  try {
    ;[qEmbed] = await embedTexts([question])
  } catch (e) {
    console.warn('[RAG] embed failed:', e.message)
    return []
  }

  const all = []
  for (const code of deptCodes) {
    try {
      let col
      try { col = await chroma.getCollection({ name: colName(code) }) }
      catch { continue }

      const count = await col.count()
      if (!count) continue

      const res = await col.query({
        queryEmbeddings: [qEmbed],
        nResults:        Math.min(topK, count),
        include:         ['documents', 'metadatas', 'distances'],
      })

      if (res.documents?.[0]) {
        res.documents[0].forEach((doc, i) => {
          all.push({
            text:           doc,
            documentName:   res.metadatas[0][i]?.document_name || 'Tài liệu',
            documentId:     res.metadatas[0][i]?.document_id,
            departmentCode: code,
            distance:       res.distances[0][i] ?? 999,
          })
        })
      }
    } catch (e) {
      console.warn(`[RAG] Skip dept ${code}:`, e.message)
    }
  }

  all.sort((a, b) => a.distance - b.distance)

  const relevant = all.filter(c => c.distance < DIST_THRESHOLD)
  const result   = relevant.length ? relevant : all.slice(0, FALLBACK_TOP)

  console.log(`[RAG] retrieveAllChunks: ${result.length} chunks / ${deptCodes.length} depts`)
  return result.slice(0, topK * 2)
}

// ─── RETRIEVE PRODUCTS ──────────────────────────────────────────────────────────
async function retrieveProducts(question, topK = TOP_K_PROD) {
  try {
    let col
    try { col = await getChroma().getCollection({ name: 'tttn_products' }) }
    catch { return [] }

    const count = await col.count()
    if (!count) return []

    const [qEmbed] = await embedTexts([question])
    const results  = await col.query({
      queryEmbeddings: [qEmbed],
      nResults:        Math.min(topK, count),
      include:         ['documents', 'metadatas', 'distances'],
    })

    if (!results.documents?.[0]) return []

    return results.documents[0]
      .map((doc, i) => ({
        text:     doc,
        meta:     results.metadatas[0][i],
        distance: results.distances[0][i] ?? 999,
      }))
      .filter(p => p.distance < 2.0)
  } catch (e) {
    console.warn('[RAG] retrieveProducts:', e.message)
    return []
  }
}

// ─── GENERATE ANSWER ────────────────────────────────────────────────────────────
async function generateAnswer({ question, chunks, departmentName, systemPrompt, history = [], isMasterAdmin = false }) {
  const ai = getOpenAI()

  const hasContext  = chunks.length > 0
  const contextText = chunks
    .map((c, i) => {
      const src = c.departmentCode === 'PRODUCTS'
        ? `[Sản phẩm: ${c.documentName}]`
        : `[Tài liệu ${i + 1}: ${c.documentName}${c.departmentCode ? ' — ' + c.departmentCode : ''}]`
      return `${src}\n${c.text}`
    })
    .join('\n\n---\n\n')

  const base = systemPrompt
    || `Bạn là trợ lý AI nội bộ${isMasterAdmin ? ' toàn hệ thống TTTN' : ` của phòng ${departmentName}`}.`

  // KEY FIX: Khi có context → ép GPT phải đọc và dùng tài liệu
  // Không được nói "không có thông tin" nếu tài liệu đã được cung cấp
  const systemMsg = hasContext
    ? `${base}

Dưới đây là nội dung tài liệu nội bộ liên quan đến câu hỏi. Hãy đọc kỹ và trả lời dựa trên tài liệu này.

Quy tắc bắt buộc:
1. Trả lời trực tiếp dựa vào nội dung tài liệu bên dưới.
2. Cuối câu trả lời ghi nguồn: (Nguồn: tên tài liệu).
3. Nếu tài liệu có thông tin một phần — nói rõ phần nào có, phần nào không đề cập.
4. Chỉ nói "không tìm thấy thông tin" khi tài liệu thực sự không nhắc đến chủ đề này.
5. Không được bịa hoặc suy đoán ngoài tài liệu.
6. Trả lời bằng tiếng Việt, tự nhiên, thân thiện.

===== TÀI LIỆU NỘI BỘ =====
${contextText}
===========================`
    : `${base}

Không tìm thấy tài liệu nội bộ nào liên quan đến câu hỏi này.
Hãy thông báo lịch sự cho người dùng và gợi ý liên hệ bộ phận phụ trách hoặc Admin.
Trả lời bằng tiếng Việt.`

  const messages = [
    { role: 'system', content: systemMsg },
    ...history.slice(-6).map(m => ({
      role:    m.role === 'assistant' ? 'assistant' : 'user',
      content: m.content,
    })),
    { role: 'user', content: question },
  ]

  const response = await ai.chat.completions.create({
    model:       'GPT-4o mini',
    messages,
    max_tokens:  1500,
    temperature: 0.1,
  })

  const answer = response.choices[0].message.content
  const tokens = response.usage?.total_tokens || 0

  const seen    = new Set()
  const sources = []
  for (const c of chunks) {
    if (!seen.has(c.documentName)) {
      seen.add(c.documentName)
      sources.push({
        documentName:   c.departmentCode === 'PRODUCTS'
          ? `[Sản phẩm] ${c.documentName}`
          : c.documentName,
        documentId:     c.documentId,
        departmentCode: c.departmentCode,
      })
    }
  }

  console.log(`[RAG] Generated: ${tokens} tokens | ${sources.length} sources`)
  return { answer, sources, tokens }
}

// ─── MAIN RAG QUERY ─────────────────────────────────────────────────────────────
async function ragQuery({ question, departmentCode, departmentName, systemPrompt, history, isMasterAdmin = false }) {
  let docChunks  = []
  let prodChunks = []

  try {
    if (isMasterAdmin || departmentCode === 'ALL') {
      docChunks = await retrieveAllChunks(question, TOP_K_DOC)
      console.log(`[RAG] ALL depts: ${docChunks.length} chunks`)
    } else {
      docChunks = await retrieveChunks({ question, departmentCode, topK: TOP_K_DOC })
      console.log(`[RAG] Dept ${departmentCode}: ${docChunks.length} chunks`)
    }

    prodChunks = await retrieveProducts(question, TOP_K_PROD)
    console.log(`[RAG] Products: ${prodChunks.length} results`)
  } catch (e) {
    console.error('[RAG] Retrieval error:', e.message)
    return {
      answer:  `Lỗi hệ thống tìm kiếm: ${e.message}. Kiểm tra ChromaDB đang chạy.`,
      sources: [],
      tokens:  0,
    }
  }

  const productAsChunks = prodChunks.map(p => ({
    text:           p.text,
    documentName:   p.meta?.name || 'Sản phẩm',
    documentId:     p.meta?.product_id,
    departmentCode: 'PRODUCTS',
    distance:       p.distance,
  }))

  const allChunks = [...docChunks, ...productAsChunks]

  return generateAnswer({ question, chunks: allChunks, departmentName, systemPrompt, history, isMasterAdmin })
}

// ─── INDEX PRODUCT ──────────────────────────────────────────────────────────────
async function indexProduct(product) {
  try {
    const col = await getChroma().getOrCreateCollection({
      name:     'tttn_products',
      metadata: { type: 'products' },
    })

    try { await col.delete({ where: { product_id: product._id.toString() } }) } catch {}

    const fmt  = n => (n && n > 0) ? new Intl.NumberFormat('vi-VN').format(n) + ' VNĐ' : 'Liên hệ'
    const text = [
      `Tên sản phẩm: ${product.name}`,
      `Thương hiệu: ${product.brand || ''}`,
      `Danh mục: ${product.category || ''}`,
      `SKU: ${product.sku || ''}`,
      `Giá: ${fmt(product.price)}`,
      `Tồn kho: ${product.stock ?? 0} sản phẩm`,
      product.description ? `Mô tả: ${product.description}` : null,
    ].filter(Boolean).join('\n')

    const [embedding] = await embedTexts([text])

    await col.add({
      ids:        [product._id.toString()],
      embeddings: [embedding],
      documents:  [text],
      metadatas:  [{
        product_id: product._id.toString(),
        name:       product.name     || '',
        brand:      product.brand    || '',
        category:   product.category || '',
        sku:        product.sku      || '',
        price:      Number(product.price) || 0,
        stock:      Number(product.stock) || 0,
      }],
    })
    console.log(`[RAG] Product indexed: ${product.name}`)
  } catch (e) {
    console.warn('[RAG] indexProduct error:', e.message)
  }
}

// ─── DELETE PRODUCT ─────────────────────────────────────────────────────────────
async function deleteProduct(productId) {
  try {
    const col = await getChroma().getCollection({ name: 'tttn_products' })
    await col.delete({ where: { product_id: productId.toString() } })
    console.log(`[RAG] Product deleted: ${productId}`)
  } catch (e) {
    console.warn('[RAG] deleteProduct error:', e.message)
  }
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
}
