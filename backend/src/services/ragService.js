/**
 * RAG Service - Retrieval Augmented Generation Pipeline
 *
 * Flow:
 * 1. INDEXING  : File → Extract Text → Chunk → OpenAI Embed → ChromaDB
 * 2. RETRIEVAL : Question → OpenAI Embed → ChromaDB Search → Top K chunks
 * 3. GENERATION: Question + Chunks → GPT-4o-mini → Answer
 *
 * Fixes:
 *  ✓ Distance threshold nới 1.5 → 1.8, fallback top-3 nếu không có chunk nào qua threshold
 *  ✓ Chunks tài liệu & sản phẩm được tách biệt rõ ràng trong context → GPT không nhầm nguồn
 *  ✓ System prompt chuẩn: ép GPT ghi đúng tên tài liệu / tên sản phẩm làm nguồn
 *  ✓ Không dùng LangGraph
 */

const fs   = require('fs')
const path = require('path')
const { OpenAI }       = require('openai')
const { ChromaClient } = require('chromadb')

// ─── Config ───────────────────────────────────────────────────────────────────
const DIST_DOC      = 1.8   // threshold tài liệu
const DIST_PROD     = 2.0   // threshold sản phẩm (tên thường ngắn nên xa hơn)
const FALLBACK_TOP  = 3     // lấy top N gần nhất khi không chunk nào qua threshold
const TOP_K_DOC     = 6
const TOP_K_PROD    = 3

// ─── Singletons ───────────────────────────────────────────────────────────────
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

// ─── TEXT EXTRACTION ──────────────────────────────────────────────────────────
async function extractText(filePath, fileType) {
  const ext = (fileType || path.extname(filePath).slice(1)).toLowerCase()

  if (ext === 'pdf') {
    const pdfParse = require('pdf-parse')
    return (await pdfParse(fs.readFileSync(filePath))).text
  }
  if (ext === 'docx') {
    const mammoth = require('mammoth')
    return (await mammoth.extractRawText({ path: filePath })).value
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

// ─── CHUNKING ─────────────────────────────────────────────────────────────────
function chunkText(text, chunkSize = 800, overlap = 150) {
  const cleaned = text
    .replace(/\r\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .replace(/[ \t]{2,}/g, ' ')
    .trim()

  if (!cleaned) return []

  const chunks = []
  let start = 0

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

// ─── EMBEDDING ────────────────────────────────────────────────────────────────
async function embedTexts(texts) {
  const embeddings = []
  for (let i = 0; i < texts.length; i += 50) {
    const res = await getOpenAI().embeddings.create({
      model: 'text-embedding-3-small',
      input: texts.slice(i, i + 50),
    })
    embeddings.push(...res.data.map(e => e.embedding))
  }
  return embeddings
}

// ─── INDEX DOCUMENT ───────────────────────────────────────────────────────────
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

  let collection
  try {
    collection = await getChroma().getOrCreateCollection({
      name:     colName(departmentCode),
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

  console.log(`[RAG] Indexed ${chunks.length} chunks → ${colName(departmentCode)}`)
  return { chunkCount: chunks.length, collectionName: colName(departmentCode) }
}

// ─── DELETE DOCUMENT ──────────────────────────────────────────────────────────
async function deleteDocument({ documentId, departmentCode }) {
  try {
    const col = await getChroma().getCollection({ name: colName(departmentCode) })
    await col.delete({ where: { document_id: documentId } })
    console.log(`[RAG] Deleted ${documentId}`)
  } catch (e) {
    console.warn('[RAG] deleteDocument:', e.message)
  }
}

// ─── RETRIEVE CHUNKS (1 phòng ban) ───────────────────────────────────────────
// FIX: Không hard-filter ngay — lấy topK rồi ưu tiên relevant,
//      fallback top FALLBACK_TOP nếu không có chunk nào đủ gần.
async function retrieveChunks({ question, departmentCode, topK = TOP_K_DOC }) {
  let col
  try { col = await getChroma().getCollection({ name: colName(departmentCode) }) }
  catch { return [] }

  const count = await col.count()
  if (!count) return []

  const [qEmbed] = await embedTexts([question])
  const res = await col.query({
    queryEmbeddings: [qEmbed],
    nResults:        Math.min(topK, count),
    include:         ['documents', 'metadatas', 'distances'],
  })

  if (!res.documents?.[0]?.length) return []

  const chunks = res.documents[0].map((doc, i) => ({
    text:           doc,
    documentName:   res.metadatas[0][i]?.document_name   || 'Tài liệu',
    documentId:     res.metadatas[0][i]?.document_id,
    departmentCode: res.metadatas[0][i]?.department_code || departmentCode,
    distance:       res.distances[0][i] ?? 999,
  }))

  const relevant = chunks.filter(c => c.distance < DIST_DOC)
  if (relevant.length === 0) {
    console.log(`[RAG] Fallback top-${FALLBACK_TOP} (best dist=${chunks[0]?.distance?.toFixed(3)})`)
    return chunks.slice(0, FALLBACK_TOP)
  }
  return relevant
}

// ─── RETRIEVE ALL CHUNKS (Master Admin — tất cả phòng ban) ───────────────────
async function retrieveAllChunks(question, topK = TOP_K_DOC) {
  let deptCodes = ['HR', 'IT', 'SALES', 'ACCOUNTING', 'GENERAL']
  try {
    const mongoose   = require('mongoose')
    const Department = mongoose.model('Department')
    const depts      = await Department.find({}, 'code').lean()
    if (depts.length) deptCodes = depts.map(d => d.code)
  } catch {}

  let qEmbed
  try { ;[qEmbed] = await embedTexts([question]) }
  catch (e) { console.warn('[RAG] embed failed:', e.message); return [] }

  const all = []
  for (const code of deptCodes) {
    try {
      let col
      try { col = await getChroma().getCollection({ name: colName(code) }) }
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
    } catch (e) { console.warn(`[RAG] Skip dept ${code}:`, e.message) }
  }

  all.sort((a, b) => a.distance - b.distance)
  const relevant = all.filter(c => c.distance < DIST_DOC)
  const result   = relevant.length ? relevant : all.slice(0, FALLBACK_TOP)

  console.log(`[RAG] retrieveAllChunks: ${result.length} chunks / ${deptCodes.length} depts`)
  return result.slice(0, topK * 2)
}

// ─── RETRIEVE PRODUCTS ────────────────────────────────────────────────────────
async function retrieveProducts(question, topK = TOP_K_PROD) {
  try {
    let col
    try { col = await getChroma().getCollection({ name: 'tttn_products' }) }
    catch { return [] }

    const count = await col.count()
    if (!count) return []

    const [qEmbed] = await embedTexts([question])
    const res = await col.query({
      queryEmbeddings: [qEmbed],
      nResults:        Math.min(topK, count),
      include:         ['documents', 'metadatas', 'distances'],
    })

    if (!res.documents?.[0]) return []

    return res.documents[0]
      .map((doc, i) => ({
        text:     doc,
        meta:     res.metadatas[0][i],
        distance: res.distances[0][i] ?? 999,
      }))
      .filter(p => p.distance < DIST_PROD)
  } catch (e) {
    console.warn('[RAG] retrieveProducts:', e.message)
    return []
  }
}

// ─── GENERATE ANSWER ──────────────────────────────────────────────────────────
async function generateAnswer({
  question, docChunks = [], productChunks = [],
  departmentName, systemPrompt, history = [], isMasterAdmin = false,
}) {
  const ai = getOpenAI()

  const hasDoc  = docChunks.length > 0
  const hasProd = productChunks.length > 0
  const hasAny  = hasDoc || hasProd

  // ── Xây context tài liệu (tách riêng với sản phẩm để GPT không nhầm nguồn) ──
  let contextBlock = ''

  if (hasDoc) {
    contextBlock += '=== TÀI LIỆU NỘI BỘ ===\n'
    contextBlock += docChunks.map((c, i) => {
      const deptTag = c.departmentCode ? ` [${c.departmentCode}]` : ''
      return `--- Tài liệu: "${c.documentName}"${deptTag} ---\n${c.text}`
    }).join('\n\n')
    contextBlock += '\n\n'
  }

  if (hasProd) {
    contextBlock += '=== DANH MỤC SẢN PHẨM ===\n'
    contextBlock += productChunks.map(c => {
      return `--- Sản phẩm: "${c.documentName}" ---\n${c.text}`
    }).join('\n\n')
    contextBlock += '\n'
  }

  const base = systemPrompt
    || `Bạn là trợ lý AI nội bộ${isMasterAdmin ? ' toàn hệ thống TTTN' : ` của phòng ${departmentName}`}.`

  // ── System prompt ──────────────────────────────────────────────────────────
  // FIX KEY: Tách rõ 2 loại nguồn, ép GPT ghi đúng tên tài liệu / tên sản phẩm
  const systemMsg = hasAny ? `${base}

Dưới đây là dữ liệu nội bộ được truy xuất liên quan đến câu hỏi. Hãy đọc kỹ và trả lời dựa trên nội dung này.

${contextBlock}
=== HƯỚNG DẪN TRẢ LỜI ===
1. Trả lời trực tiếp, rõ ràng dựa vào nội dung bên trên.
2. Cuối mỗi ý quan trọng, ghi nguồn trong ngoặc tròn:
   - Nếu thông tin lấy từ tài liệu → ghi: (Nguồn: tên_tài_liệu)
     Ví dụ: (Nguồn: BÁO CÁO TTTN THẦY HIẾU NHÓM 13.docx)
   - Nếu thông tin lấy từ sản phẩm → ghi: (Nguồn: Sản phẩm tên_sản_phẩm)
     Ví dụ: (Nguồn: Sản phẩm MacBook M7)
3. KHÔNG được nhầm lẫn nguồn: thông tin từ tài liệu KHÔNG được ghi nguồn là sản phẩm và ngược lại.
4. Nếu thông tin một phần có, một phần không — nói rõ phần nào tìm thấy.
5. Chỉ nói "không tìm thấy" khi dữ liệu trên thực sự không đề cập chủ đề này.
6. Không bịa, không suy đoán ngoài dữ liệu được cung cấp.
7. Trả lời bằng tiếng Việt, tự nhiên, thân thiện.`

  : `${base}

Không tìm thấy tài liệu hay sản phẩm nào liên quan trong hệ thống nội bộ.
Hãy thông báo lịch sự và gợi ý người dùng liên hệ bộ phận phụ trách hoặc Admin.
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
    model:       'gpt-4o-mini',
    messages,
    max_tokens:  1500,
    temperature: 0.1,
  })

  const answer = response.choices[0].message.content
  const tokens = response.usage?.total_tokens || 0

  // Dedup sources — tài liệu trước, sản phẩm sau
  const seen    = new Set()
  const sources = []
  for (const c of [...docChunks, ...productChunks]) {
    if (!seen.has(c.documentName)) {
      seen.add(c.documentName)
      sources.push({
        documentName:   c.isProduct ? `[Sản phẩm] ${c.documentName}` : c.documentName,
        documentId:     c.documentId,
        departmentCode: c.departmentCode,
      })
    }
  }

  console.log(`[RAG] Generated: ${tokens} tokens | ${sources.length} sources`)
  return { answer, sources, tokens }
}

// ─── MAIN RAG QUERY ───────────────────────────────────────────────────────────
async function ragQuery({
  question, departmentCode, departmentName,
  systemPrompt, history, isMasterAdmin = false,
}) {
  let docChunks     = []
  let productChunks = []

  try {
    // ── Lấy chunks tài liệu ──
    if (isMasterAdmin || departmentCode === 'ALL') {
      docChunks = await retrieveAllChunks(question, TOP_K_DOC)
      console.log(`[RAG] ALL depts: ${docChunks.length} doc chunks`)
    } else {
      docChunks = await retrieveChunks({ question, departmentCode, topK: TOP_K_DOC })
      console.log(`[RAG] Dept ${departmentCode}: ${docChunks.length} doc chunks`)
    }

    // ── Lấy chunks sản phẩm ──
    const rawProdChunks = await retrieveProducts(question, TOP_K_PROD)
    productChunks = rawProdChunks.map(p => ({
      text:           p.text,
      documentName:   p.meta?.name || 'Sản phẩm',
      documentId:     p.meta?.product_id,
      departmentCode: 'PRODUCTS',
      isProduct:      true,
      distance:       p.distance,
    }))
    console.log(`[RAG] Products: ${productChunks.length} chunks`)

  } catch (e) {
    console.error('[RAG] Retrieval error:', e.message)
    return {
      answer:  `Lỗi hệ thống tìm kiếm: ${e.message}. Kiểm tra ChromaDB đang chạy.`,
      sources: [],
      tokens:  0,
    }
  }

  return generateAnswer({
    question,
    docChunks,
    productChunks,
    departmentName,
    systemPrompt,
    history,
    isMasterAdmin,
  })
}

// ─── INDEX PRODUCT ────────────────────────────────────────────────────────────
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
      `Thương hiệu: ${product.brand     || ''}`,
      `Danh mục: ${product.category     || ''}`,
      `SKU: ${product.sku               || ''}`,
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
        name:       product.name       || '',
        brand:      product.brand      || '',
        category:   product.category   || '',
        sku:        product.sku        || '',
        price:      Number(product.price) || 0,
        stock:      Number(product.stock) || 0,
      }],
    })
    console.log(`[RAG] Product indexed: ${product.name}`)
  } catch (e) {
    console.warn('[RAG] indexProduct error:', e.message)
  }
}

// ─── DELETE PRODUCT ───────────────────────────────────────────────────────────
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
