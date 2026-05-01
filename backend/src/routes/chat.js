const express = require("express");
const ChatSession = require("../models/ChatSession");
const Department = require("../models/Department");
const KnowledgeDocument = require("../models/KnowledgeDocument");
const AuditLog = require("../models/AuditLog");
const { authenticate } = require("../middleware/auth");
const { ragQuery } = require("../services/ragService");

const router = express.Router();

// GET /api/chat/sessions
router.get("/sessions", authenticate, async (req, res) => {
  try {
    const { page = 1, limit = 50 } = req.query;
    const sessions = await ChatSession.find({ user: req.user._id })
      .populate("department", "name code color icon")
      .select("-messages")
      .sort({ lastMessageAt: -1 })
      .skip((page - 1) * limit)
      .limit(Number(limit));
    const total = await ChatSession.countDocuments({ user: req.user._id });
    res.json({ sessions, total });
  } catch (err) {
    res.status(500).json({ error: "Lỗi server" });
  }
});

// POST /api/chat/sessions
router.post("/sessions", authenticate, async (req, res) => {
  try {
    const { departmentId } = req.body;

    // Master admin: chat toàn hệ thống (ALL departments)
    if (departmentId === "ALL" && req.user.role === "master_admin") {
      // Use GENERAL dept as placeholder, but RAG will search all
      const generalDept = await Department.findOne({ code: "GENERAL" }) || await Department.findOne();
      const session = await ChatSession.create({
        user: req.user._id,
        department: generalDept._id,
        title: `Chat Toàn Hệ Thống - ${new Date().toLocaleDateString("vi-VN")}`,
        isGlobal: true,
      });
      session._doc.isGlobal = true;
      await session.populate("department", "name code color icon welcomeMessage aiSystemPrompt");
      // Override display info
      session._doc.globalChat = true;
      return res.status(201).json({ ...session.toObject(), isGlobal: true, globalTitle: "Toàn Hệ Thống" });
    }

    const dept = await Department.findById(departmentId);
    if (!dept) return res.status(404).json({ error: "Phòng ban không tồn tại" });

    // Permission: employee chỉ chat phòng ban của mình
    if (req.user.role === "employee") {
      const userDeptId = req.user.department?._id?.toString();
      if (userDeptId !== departmentId) {
        return res.status(403).json({ error: "Bạn chỉ có thể chat trong phòng ban của mình" });
      }
    }

    const session = await ChatSession.create({
      user: req.user._id,
      department: departmentId,
      title: `Chat ${dept.name} - ${new Date().toLocaleDateString("vi-VN")}`
    });
    await session.populate("department", "name code color icon welcomeMessage aiSystemPrompt");
    res.status(201).json(session);
  } catch (err) {
    res.status(500).json({ error: "Lỗi server" });
  }
});

// GET /api/chat/sessions/:id
router.get("/sessions/:id", authenticate, async (req, res) => {
  try {
    const session = await ChatSession.findOne({ _id: req.params.id, user: req.user._id })
      .populate("department", "name code color icon aiSystemPrompt welcomeMessage");
    if (!session) return res.status(404).json({ error: "Session không tồn tại" });
    res.json(session);
  } catch (err) {
    res.status(500).json({ error: "Lỗi server" });
  }
});

// POST /api/chat/sessions/:id/message  ← CORE: RAG pipeline
router.post("/sessions/:id/message", authenticate, async (req, res) => {
  try {
    const { message } = req.body;
    if (!message?.trim()) return res.status(400).json({ error: "Tin nhắn không được để trống" });

    const session = await ChatSession.findOne({ _id: req.params.id, user: req.user._id })
      .populate("department");
    if (!session) return res.status(404).json({ error: "Session không tồn tại" });

    const dept = session.department;
    const userMsg = { role: "user", content: message.trim(), timestamp: new Date() };
    session.messages.push(userMsg);

    // ─── RAG Query ─────────────────────────────────────────────────────────
    const isGlobalSession = session.isGlobal === true;
    let aiResult
    try {
      aiResult = await ragQuery({
        question: message.trim(),
        departmentCode: isGlobalSession ? 'ALL' : dept.code,
        departmentName: isGlobalSession ? 'Toàn hệ thống' : dept.name,
        systemPrompt: dept.aiSystemPrompt,
        history: session.messages.slice(-10, -1),
        isMasterAdmin: isGlobalSession,
      })
    } catch (ragErr) {
      console.error("[Chat] RAG error:", ragErr.message)
      // If RAG fails (no OpenAI key etc), give clear message
      aiResult = {
        answer: `⚠️ **Hệ thống AI chưa sẵn sàng**\n\nĐể chatbot hoạt động, cần:\n1. Cấu hình **OPENAI_API_KEY** hợp lệ trong \`backend/.env\`\n2. Khởi động **ChromaDB**: \`docker run -p 8000:8000 chromadb/chroma\`\n\nLỗi: ${ragErr.message}`,
        sources: [],
        tokens: 0
      }
    }

    const assistantMsg = {
      role: "assistant",
      content: aiResult.answer,
      sources: aiResult.sources || [],
      tokens: aiResult.tokens || 0,
      timestamp: new Date()
    };
    session.messages.push(assistantMsg);
    session.totalTokens = (session.totalTokens || 0) + (aiResult.tokens || 0);
    session.lastMessageAt = new Date();

    // Auto title from first question
    if (session.messages.length === 2) {
      session.title = message.trim().substring(0, 60) + (message.length > 60 ? "..." : "");
    }

    await session.save();

    res.json({
      userMessage: session.messages[session.messages.length - 2],
      assistantMessage: assistantMsg,
      sessionId: session._id
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Lỗi server" });
  }
});

// DELETE /api/chat/sessions/:id
router.delete("/sessions/:id", authenticate, async (req, res) => {
  try {
    const session = await ChatSession.findOneAndDelete({ _id: req.params.id, user: req.user._id });
    if (!session) return res.status(404).json({ error: "Session không tồn tại" });
    AuditLog.create({
      actor: req.user._id, actorName: req.user.name, actorRole: req.user.role,
      action: "CHAT_DELETE", targetType: "Chat", targetId: session._id, ipAddress: req.ip
    }).catch(() => {});
    res.json({ message: "Đã xóa" });
  } catch (err) {
    res.status(500).json({ error: "Lỗi server" });
  }
});

// POST /api/chat/sessions/:id/rating
router.post("/sessions/:id/rating", authenticate, async (req, res) => {
  try {
    const { rating } = req.body;
    if (!rating || rating < 1 || rating > 5) return res.status(400).json({ error: "Rating 1-5" });
    await ChatSession.findOneAndUpdate({ _id: req.params.id, user: req.user._id }, { rating });
    res.json({ message: "Cảm ơn phản hồi!" });
  } catch (err) {
    res.status(500).json({ error: "Lỗi server" });
  }
});

// GET /api/chat/sessions/:id/export
router.get("/sessions/:id/export", authenticate, async (req, res) => {
  try {
    const session = await ChatSession.findOne({ _id: req.params.id, user: req.user._id })
      .populate("department", "name").populate("user", "name email");
    if (!session) return res.status(404).json({ error: "Session không tồn tại" });

    let content = `=== LỊCH SỬ CHAT ===\n`;
    content += `Phòng ban: ${session.department?.name}\n`;
    content += `Người dùng: ${session.user?.name} (${session.user?.email})\n`;
    content += `Thời gian: ${session.createdAt.toLocaleString("vi-VN")}\n\n${"=".repeat(50)}\n\n`;
    session.messages.forEach(msg => {
      const role = msg.role === "user" ? `👤 ${session.user?.name}` : "🤖 AI";
      content += `${role}:\n${msg.content}\n`;
      if (msg.sources?.length > 0) content += `📎 Nguồn: ${msg.sources.map(s => s.documentName).join(", ")}\n`;
      content += "\n";
    });

    res.setHeader("Content-Type", "text/plain; charset=utf-8");
    res.setHeader("Content-Disposition", `attachment; filename="chat_${session._id}.txt"`);
    res.send(content);

    AuditLog.create({
      actor: req.user._id, actorName: req.user.name, actorRole: req.user.role,
      action: "CHAT_EXPORT", targetType: "Chat", targetId: session._id, ipAddress: req.ip
    }).catch(() => {});
  } catch (err) {
    res.status(500).json({ error: "Lỗi server" });
  }
});

module.exports = router;
