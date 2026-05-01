/**
 * RAG Service - Retrieval Augmented Generation Pipeline
 *
 * Flow:
 * 1. INDEXING: File → Extract Text → Chunk → OpenAI Embed → ChromaDB
 * 2. RETRIEVAL: Question → OpenAI Embed → ChromaDB Search → Top K chunks
 * 3. GENERATION: Question + Chunks → OpenAI GPT → Answer (chỉ từ tài liệu nội bộ)
 */

const fs = require('fs')
const path = require('path')
const { OpenAI } = require('openai')
const { ChromaClient } = require('chromadb')

// ─── Clients ─────────────────────────────────────────────────────────────────
let openai = null
let chroma = null

function getOpenAI() {
  if (!openai) {
    const apiKey = process.env.OPENAI_API_KEY
    if (!apiKey || apiKey.startsWith('sk-your') || apiKey === 'sk-proj-your-key-here') {
      throw new Error('OPENAI_API_KEY chưa được cấu hình trong file .env')
    }
    openai = new OpenAI({ apiKey })
  }
  return openai
}

function getChroma() {
  if (!chroma) {
    const chromaUrl = process.env.CHROMA_URL || 'http://localhost:8000'
    chroma = new ChromaClient({ path: chromaUrl })
  }
  return chroma
}

// Collection name per department
function collectionName(departmentCode) {
  return `tttn_${departmentCode.toLowerCase()}_docs`
}

// ─── TEXT EXTRACTION ──────────────────────────────────────────────────────────
async function extractText(filePath, fileType) {
  const ext = (fileType || path.extname(filePath).slice(1)).toLowerCase()

  if (ext === 'pdf') {
    const pdfParse = require('pdf-parse')
    const buffer = fs.readFileSync(filePath)
    const result = await pdfParse(buffer)
    return result.text
  }

  if (ext === 'docx') {
    const mammoth = require('mammoth')
    const result = await mammoth.extractRawText({ path: filePath })
    return result.value
  }

  if (['txt', 'md', 'csv'].includes(ext)) {
    return fs.readFileSync(filePath, 'utf-8')
  }

  if (ext === 'xlsx') {
    const XLSX = require('xlsx')
    const wb = XLSX.readFile(filePath)
    let text = ''
    wb.SheetNames.forEach(name => {
      const ws = wb.Sheets[name]
      text += `[Sheet: ${name}]\n` + XLSX.utils.sheet_to_csv(ws) + '\n\n'
    })
    return text
  }

  throw new Error(`Định dạng file ${ext} chưa được hỗ trợ`)
}

// ─── CHUNKING ─────────────────────────────────────────────────────────────────
function chunkText(text, chunkSize = 800, overlap = 150) {
  const cleaned = text
    .replace(/\r\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .replace(/[ \t]{2,}/g, ' ')
    .trim()

  if (cleaned.length === 0) return []

  const chunks = []
  let start = 0

  while (start < cleaned.length) {
    let end = start + chunkSize

    if (end < cleaned.length) {
      const breakPoints = ['\n\n', '.\n', '. ', '\n', ' ']
      for (const bp of breakPoints) {
        const idx = cleaned.lastIndexOf(bp, end)
        if (idx > start + chunkSize * 0.5) {
          end = idx + bp.length
          break
        }
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
  const ai = getOpenAI()
  const embeddings = []
  const batchSize = 50

  for (let i = 0; i < texts.length; i += batchSize) {
    const batch = texts.slice(i, i + batchSize)
    const response = await ai.embeddings.create({
      model: 'text-embedding-3-small',
      input: batch,
    })
    embeddings.push(...response.data.map(e => e.embedding))
  }

  return embeddings
}

// ─── INDEX DOCUMENT ───────────────────────────────────────────────────────────
async function indexDocument({ filePath, fileType, documentId, documentName, departmentCode, departmentId }) {
  console.log(`[RAG] Indexing document: ${documentName}`)

  // 1. Extract text
  const rawText = await extractText(filePath, fileType)
  if (!rawText || rawText.trim().length < 10) {
    throw new Error('Không trích xuất được nội dung từ file')
  }

  // 2. Chunk
  const chunks = chunkText(rawText)
  if (chunks.length === 0) throw new Error('File không có nội dung có thể xử lý')
  console.log(`[RAG] Created ${chunks.length} chunks`)

  // 3. Embed all chunks
  const embeddings = await embedTexts(chunks)
  console.log(`[RAG] Embedded ${embeddings.length} chunks`)

  // 4. Store in ChromaDB
  const chromaClient = getChroma()
  const colName = collectionName(departmentCode)

  let collection
  try {
    collection = await chromaClient.getOrCreateCollection({
      name: colName,
      metadata: { department_code: departmentCode, department_id: departmentId }
    })
  } catch (e) {
    throw new Error(`Không kết nối được ChromaDB: ${e.message}. Hãy đảm bảo ChromaDB đang chạy tại ${process.env.CHROMA_URL || 'http://localhost:8000'}`)
  }

  // Delete old chunks of this document if reindexing
  try {
    await collection.delete({ where: { document_id: documentId } })
  } catch {}

  // FIX: Bỏ field chunk_text khỏi metadata — không dùng ở đâu, tốn storage
  const ids = chunks.map((_, i) => `${documentId}_chunk_${i}`)
  const metadatas = chunks.map((_, i) => ({
    document_id: documentId,
    document_name: documentName,
    department_code: departmentCode,
    department_id: departmentId,
    chunk_index: i,
  }))

  await collection.add({
    ids,
    embeddings,
    documents: chunks,
    metadatas,
  })

  console.log(`[RAG] Indexed ${chunks.length} chunks into ChromaDB collection: ${colName}`)
  return { chunkCount: chunks.length, collectionName: colName }
}

// ─── DELETE DOCUMENT FROM CHROMA ─────────────────────────────────────────────
async function deleteDocument({ documentId, departmentCode }) {
  try {
    const chromaClient = getChroma()
    const colName = collectionName(departmentCode)
    const collection = await chromaClient.getCollection({ name: colName })
    await collection.delete({ where: { document_id: documentId } })
    console.log(`[RAG] Deleted document ${documentId} from ChromaDB`)
  } catch (e) {
    console.warn('[RAG] Could not delete from ChromaDB:', e.message)
  }
}

// ─── RETRIEVE RELEVANT CHUNKS ─────────────────────────────────────────────────
async function retrieveChunks({ question, departmentCode, topK = 5 }) {
  const chromaClient = getChroma()
  const colName = collectionName(departmentCode)

  const [questionEmbedding] = await embedTexts([question])

  let collection
  try {
    collection = await chromaClient.getCollection({ name: colName })
  } catch {
    return []
  }

  const count = await collection.count()
  if (count === 0) return []

  const results = await collection.query({
    queryEmbeddings: [questionEmbedding],
    nResults: Math.min(topK, count),
    include: ['documents', 'metadatas', 'distances'],
  })

  const chunks = []
  if (results.documents?.[0]) {
    results.documents[0].forEach((doc, i) => {
      chunks.push({
        text: doc,
        documentName: results.metadatas[0][i]?.document_name || 'Tài liệu',
        documentId: results.metadatas[0][i]?.document_id,
        distance: results.distances?.[0]?.[i] || 0,
      })
    })
  }

  return chunks.filter(c => c.distance < 1.5)
}

// ─── GENERATE ANSWER (RAG) ────────────────────────────────────────────────────
async function generateAnswer({ question, chunks, departmentName, systemPrompt, history = [], isMasterAdmin = false }) {
  const ai = getOpenAI()

  const hasContext = chunks.length > 0
  const contextText = hasContext
    ? chunks.map((c, i) =>
        '[Nguon ' + (i + 1) + ': ' + c.documentName +
        (c.departmentCode && c.departmentCode !== 'PRODUCTS' ? ' | PB: ' + c.departmentCode : '') +
        ']\n' + c.text
      ).join('\n\n---\n\n')
    : ''

  const baseSystemPrompt = systemPrompt ||
    `Bạn là trợ lý AI nội bộ${isMasterAdmin ? ' toàn hệ thống TTTN' : ` của phòng ${departmentName}`}. Nhiệm vụ của bạn là hỗ trợ tra cứu thông tin từ tài liệu nội bộ công ty.`

  // FIX: Template literal đã được đóng đúng, bỏ comment // gây nhiễu cho GPT
  const ragSystemPrompt = `${baseSystemPrompt}

Bạn là trợ lý AI nội bộ của công ty, hỗ trợ nhân viên tìm kiếm thông tin từ tài liệu và dữ liệu nội bộ. Hãy trả lời như một đồng nghiệp am hiểu — lịch sự, rõ ràng, không rườm rà.

Nguyên tắc trả lời
Chỉ dựa vào TÀI LIỆU NỘI BỘ bên dưới để trả lời. Đọc kỹ toàn bộ nội dung, kể cả những đoạn diễn đạt khác cách trước khi kết luận không tìm thấy thông tin.
Không suy đoán, không thêm thông tin từ bên ngoài.

Xử lý từng tình huống
✦ Tìm thấy thông tin → Trả lời trực tiếp, đúng trọng tâm. Cuối ý ghi nguồn gọn:
  - Thông tin sản phẩm: (Nguồn: Trang sản phẩm)
  - Tài liệu nội bộ: (Nguồn: [Tên phòng ban])
  Mỗi ý chỉ một loại nguồn.

✦ Thiếu thông tin về sản phẩm → Nói thật là hệ thống chưa có dữ liệu này, và gợi ý người dùng liên hệ Admin hoặc bộ phận phụ trách để được hỗ trợ.

✦ Không tìm thấy gì → Thông báo thẳng thắn, lịch sự. Có thể gợi ý thêm: liên hệ phòng ban liên quan, tra cứu website công ty, hoặc tự tìm thêm nếu cần — nhưng không cung cấp nội dung chi tiết từ nguồn ngoài.

✦ Câu hỏi chưa rõ → Hỏi lại để hiểu đúng ý, không tự đoán.

Giọng văn
Tự nhiên, thân thiện, chuyên nghiệp. Không liệt kê cứng nhắc khi không cần thiết. Không lặp lại câu hỏi của người dùng. Trả lời ngắn nếu câu hỏi đơn giản — dài hơn khi thực sự cần thiết.

Tài liệu nội bộ
${hasContext ? contextText : 'Chưa có tài liệu nào được upload cho phòng ban này.'}`

  const messages = [
    { role: 'system', content: ragSystemPrompt },
    // Include recent history (last 6 messages)
    ...history.slice(-6).map(m => ({
      role: m.role === 'assistant' ? 'assistant' : 'user',
      content: m.content
    })),
    { role: 'user', content: question }
  ]

  const response = await ai.chat.completions.create({
    model: 'gpt-3.5-turbo',
    messages,
    max_tokens: 1500,
    temperature: 0.1,
  })

  const answer = response.choices[0].message.content
  const tokens = response.usage?.total_tokens || 0

  const seenNames = new Set()
  const sources = []
  for (const c of chunks) {
    if (!seenNames.has(c.documentName)) {
      seenNames.add(c.documentName)
      const deptLabel = c.departmentCode && c.departmentCode !== 'PRODUCTS' ? ' [' + c.departmentCode + ']' : ''
      sources.push({
        documentName: c.documentName + deptLabel,
        documentId: c.documentId,
        departmentCode: c.departmentCode,
      })
    }
  }

  return { answer, sources, tokens }
}

// ─── FULL RAG QUERY ───────────────────────────────────────────────────────────
async function ragQuery({ question, departmentCode, departmentName, systemPrompt, history, isMasterAdmin = false }) {
  let chunks = []
  let productChunks = []

  try {
    if (isMasterAdmin || departmentCode === 'ALL') {
      chunks = await retrieveAllChunks(question, 5)
      console.log(`[RAG] ALL depts: ${chunks.length} chunks`)
    } else {
      chunks = await retrieveChunks({ question, departmentCode, topK: 5 })
      console.log(`[RAG] Dept ${departmentCode}: ${chunks.length} chunks`)
    }

    productChunks = await retrieveProducts(question, 3)
    console.log(`[RAG] Products: ${productChunks.length} results`)
  } catch (e) {
    console.warn('[RAG] Retrieval failed:', e.message)
    return {
      answer: `Hệ thống tìm kiếm tài liệu chưa sẵn sàng. Vui lòng đảm bảo ChromaDB đang chạy.\n\nLỗi: ${e.message}`,
      sources: [],
      tokens: 0
    }
  }

  const productAsChunks = productChunks.map(p => ({
    text: p.text,
    documentName: `[Sản phẩm] ${p.meta?.name || 'Sản phẩm'}`,
    documentId: p.meta?.product_id,
    departmentCode: 'PRODUCTS',
    distance: 0,
  }))
  const allChunks = [...chunks, ...productAsChunks]

  return await generateAnswer({ question, chunks: allChunks, departmentName, systemPrompt, history, isMasterAdmin })
}

// ─── INDEX PRODUCT INTO CHROMA ────────────────────────────────────────────────
async function indexProduct(product) {
  try {
    const chromaClient = getChroma()
    const colName = 'tttn_products'

    let collection
    try {
      collection = await chromaClient.getOrCreateCollection({
        name: colName,
        metadata: { type: 'products' }
      })
    } catch (e) {
      console.warn('[RAG] Products collection error:', e.message)
      return
    }

    try { await collection.delete({ where: { product_id: product._id.toString() } }) } catch {}

    const fmt = (n) => n ? new Intl.NumberFormat('vi-VN').format(n) + ' VNĐ' : ''

    const text = [
      `Tên sản phẩm: ${product.name}`,
      `Thương hiệu: ${product.brand}`,
      `Danh mục: ${product.category}`,
      `SKU: ${product.sku}`,
      `Giá: ${fmt(product.price)}`,
      `Tồn kho: ${product.stock} sản phẩm`,
      product.description ? `Mô tả: ${product.description}` : '',
    ].filter(Boolean).join('\n')

    const [embedding] = await embedTexts([text])

    await collection.add({
      ids: [product._id.toString()],
      embeddings: [embedding],
      documents: [text],
      metadatas: [{
        product_id: product._id.toString(),
        name: product.name,
        brand: product.brand,
        category: product.category,
        price: product.price || 0,
        stock: product.stock || 0,
        sku: product.sku,
      }]
    })
    console.log(`[RAG] Product indexed: ${product.name}`)
  } catch (e) {
    console.warn('[RAG] Product index error:', e.message)
  }
}

async function deleteProduct(productId) {
  try {
    const chromaClient = getChroma()
    const collection = await chromaClient.getCollection({ name: 'tttn_products' })
    await collection.delete({ where: { product_id: productId.toString() } })
    console.log(`[RAG] Product deleted from chroma: ${productId}`)
  } catch (e) {
    console.warn('[RAG] Product delete error:', e.message)
  }
}

// ─── RETRIEVE ALL CHUNKS (master admin) ───────────────────────────────────────
async function retrieveAllChunks(question, topK = 4) {
  const chromaClient = getChroma()
  const allChunks = []

  let deptCodes = ['HR', 'IT', 'SALES', 'ACCOUNTING', 'GENERAL']
  try {
    const mongoose = require('mongoose')
    const Department = mongoose.model('Department')
    const depts = await Department.find({}, 'code').lean()
    if (depts.length > 0) deptCodes = depts.map(d => d.code)
  } catch {}

  let qEmbed
  try {
    const [e] = await embedTexts([question])
    qEmbed = e
  } catch (e) {
    console.warn('[RAG] Embed failed:', e.message)
    return []
  }

  for (const code of deptCodes) {
    try {
      const colName = collectionName(code)
      let collection
      try {
        collection = await chromaClient.getCollection({ name: colName })
      } catch { continue }

      const count = await collection.count()
      if (count === 0) continue

      const results = await collection.query({
        queryEmbeddings: [qEmbed],
        nResults: Math.min(topK, count),
        include: ['documents', 'metadatas', 'distances'],
      })

      if (results.documents?.[0]) {
        results.documents[0].forEach((doc, i) => {
          const dist = results.distances?.[0]?.[i] || 0
          if (dist < 1.5) {
            allChunks.push({
              text: doc,
              documentName: results.metadatas[0][i]?.document_name || 'Tai lieu',
              documentId: results.metadatas[0][i]?.document_id,
              departmentCode: code,
              distance: dist,
            })
          }
        })
      }
    } catch (e) {
      console.warn('[RAG] Skip collection ' + code + ':', e.message)
    }
  }

  allChunks.sort((a, b) => a.distance - b.distance)
  console.log('[RAG] retrieveAllChunks: found ' + allChunks.length + ' chunks across ' + deptCodes.length + ' depts')
  return allChunks.slice(0, topK * 2)
}

// ─── RETRIEVE PRODUCTS FROM CHROMA ───────────────────────────────────────────
async function retrieveProducts(question, topK = 3) {
  try {
    const chromaClient = getChroma()
    let collection
    try {
      collection = await chromaClient.getCollection({ name: 'tttn_products' })
    } catch { return [] }

    const count = await collection.count()
    if (count === 0) return []

    const [qEmbed] = await embedTexts([question])
    const results = await collection.query({
      queryEmbeddings: [qEmbed],
      nResults: Math.min(topK, count),
      include: ['documents', 'metadatas', 'distances'],
    })

    const items = []
    if (results.documents?.[0]) {
      results.documents[0].forEach((doc, i) => {
        if ((results.distances?.[0]?.[i] || 0) < 1.5) {
          items.push({ text: doc, meta: results.metadatas[0][i] })
        }
      })
    }
    return items
  } catch (e) {
    console.warn('[RAG] Product retrieve error:', e.message)
    return []
  }
}

module.exports = {
  indexDocument,
  deleteDocument,
  ragQuery,
  retrieveChunks,
  extractText,
  chunkText,
  indexProduct,
  deleteProduct,
  retrieveProducts,
}
