/**
 * frontend/chat.js
 * ----------------
 * Client-side logic for the AI Customer Support Chatbot v2.
 *
 * New in v2:
 *   - KNN model support in model selector
 *   - Confidence threshold slider — sent with each request
 *   - Conversation context — last 3 exchanges sent for continuity
 *   - Message search — searches Supabase via /api/search
 *   - Export chat — downloads conversation as a .txt file
 */

"use strict";

const API_BASE = "http://localhost:5000/api";

// DOM references
const chatMessages      = document.getElementById("chat-messages");
const userInput         = document.getElementById("user-input");
const sendBtn           = document.getElementById("send-btn");
const typingIndicator   = document.getElementById("typing-indicator");
const sessionDisplay    = document.getElementById("session-display");
const statMessages      = document.getElementById("stat-messages");
const statConfidence    = document.getElementById("stat-confidence");
const clearBtn          = document.getElementById("clear-btn");
const exportBtn         = document.getElementById("export-btn");
const lastIntentDisplay = document.getElementById("last-intent-display");
const welcomeTime       = document.getElementById("welcome-time");
const modelBtns         = document.querySelectorAll(".model-btn");
const confidenceSlider  = document.getElementById("confidence-slider");
const thresholdDisplay  = document.getElementById("threshold-display");
const searchInput       = document.getElementById("search-input");
const searchBtn         = document.getElementById("search-btn");
const searchResults     = document.getElementById("search-results");

// State
let sessionId         = "";
let messageCount      = 0;
let activeModel       = "ann";
let confidenceThreshold = 0.30;

// Conversation context — stores last 3 exchanges for continuity.
// Each entry: { role: "user"|"bot", text: string }
let conversationContext = [];
const MAX_CONTEXT = 6; // 3 user + 3 bot messages


// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function generateUUID() {
  if (crypto.randomUUID) return crypto.randomUUID();
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    return (c === "x" ? r : (r & 0x3) | 0x8).toString(16);
  });
}

function formatTime(date) {
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function escapeHTML(str) {
  return str
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function scrollToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}


// ---------------------------------------------------------------------------
// Initialisation
// ---------------------------------------------------------------------------

function init() {
  sessionId = sessionStorage.getItem("chatSessionId") || generateUUID();
  sessionStorage.setItem("chatSessionId", sessionId);
  sessionDisplay.textContent = sessionId.slice(0, 8) + "...";
  welcomeTime.textContent    = formatTime(new Date());

  // Send button
  sendBtn.addEventListener("click", handleSend);

  // Enter to send
  userInput.addEventListener("input", () => {
    sendBtn.disabled = userInput.value.trim().length === 0;
    userInput.style.height = "auto";
    userInput.style.height = userInput.scrollHeight + "px";
  });
  userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!sendBtn.disabled) handleSend();
    }
  });

  // Model selector
  modelBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      modelBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      activeModel = btn.dataset.model;
    });
  });

  // Confidence slider
  confidenceSlider.addEventListener("input", () => {
    confidenceThreshold = parseInt(confidenceSlider.value) / 100;
    thresholdDisplay.textContent = confidenceSlider.value + "%";
  });

  // Clear and export
  clearBtn.addEventListener("click", clearChat);
  exportBtn.addEventListener("click", exportChat);

  // Search
  searchBtn.addEventListener("click", handleSearch);
  searchInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") handleSearch();
  });
}


// ---------------------------------------------------------------------------
// Message rendering
// ---------------------------------------------------------------------------

function appendUserMessage(text) {
  const div = document.createElement("div");
  div.className = "message user-message";
  div.innerHTML = `
    <div class="avatar user-avatar">👤</div>
    <div class="bubble">
      <p>${escapeHTML(text)}</p>
      <span class="timestamp">${formatTime(new Date())}</span>
    </div>`;
  chatMessages.appendChild(div);
  scrollToBottom();
}

function appendBotMessage(text, intent = null, confidence = 0,
                           modelUsed = "", isError = false) {
  const conf  = (confidence * 100).toFixed(1);
  const label = intent ? intent.replace(/_/g, " ") : "unknown";
  const badge = intent
    ? `<span class="intent-badge">🎯 ${label} · ${conf}%</span>`
    : "";

  const div = document.createElement("div");
  div.className = `message bot-message${isError ? " error-bubble" : ""}`;
  div.innerHTML = `
    <div class="avatar bot-avatar">🤖</div>
    <div class="bubble">
      <p>${escapeHTML(text)}</p>
      ${badge}
      <span class="timestamp">
        ${formatTime(new Date())} · ${modelUsed.toUpperCase() || "—"}
      </span>
    </div>`;
  chatMessages.appendChild(div);
  scrollToBottom();
}


// ---------------------------------------------------------------------------
// API — Chat
// ---------------------------------------------------------------------------

async function handleSend() {
  const text = userInput.value.trim();
  if (!text) return;

  userInput.value        = "";
  userInput.style.height = "auto";
  sendBtn.disabled       = true;

  appendUserMessage(text);
  messageCount++;
  updateStats();

  typingIndicator.style.display = "flex";
  scrollToBottom();

  try {
    const response = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message:    text,
        session_id: sessionId,
        // Send last N messages for context awareness.
        context:    conversationContext.slice(-MAX_CONTEXT),
      }),
    });

    const data = await response.json();
    typingIndicator.style.display = "none";

    if (!response.ok) {
      appendBotMessage(data.error || "Something went wrong.", null, 0, "", true);
      return;
    }

    appendBotMessage(data.response, data.intent, data.confidence, data.model_used);
    updateLastIntent(data.intent, data.confidence);
    updateStats(data.confidence);

    // Update conversation context for next message.
    conversationContext.push({ role: "user", text });
    conversationContext.push({ role: "bot",  text: data.response });
    if (conversationContext.length > MAX_CONTEXT) {
      conversationContext = conversationContext.slice(-MAX_CONTEXT);
    }

  } catch (err) {
    typingIndicator.style.display = "none";
    appendBotMessage(
      "⚠️ Cannot connect to the server. Make sure Flask is running on port 5000.",
      null, 0, "", true
    );
    console.error("API error:", err);
  }
}


// ---------------------------------------------------------------------------
// API — Search
// ---------------------------------------------------------------------------

async function handleSearch() {
  const query = searchInput.value.trim();
  if (!query) return;

  searchResults.style.display = "flex";
  searchResults.innerHTML     = `<p class="search-no-results">Searching...</p>`;

  try {
    const url      = `${API_BASE}/search?session_id=${sessionId}&q=${encodeURIComponent(query)}`;
    const response = await fetch(url);
    const data     = await response.json();

    if (!response.ok || data.count === 0) {
      searchResults.innerHTML = `<p class="search-no-results">No results for "${escapeHTML(query)}"</p>`;
      return;
    }

    searchResults.innerHTML = data.results.map((r) => `
      <div class="search-result-item">
        <div class="result-user">👤 ${escapeHTML(r.user_message)}</div>
        <div class="result-bot">🤖 ${escapeHTML(r.bot_response.slice(0, 80))}...</div>
        <div class="result-meta">
          🎯 ${r.predicted_intent || "unknown"} ·
          ${r.model_used?.toUpperCase() || "—"}
        </div>
      </div>`
    ).join("");

  } catch (err) {
    searchResults.innerHTML = `<p class="search-no-results">Search failed. Is the server running?</p>`;
    console.error("Search error:", err);
  }
}


// ---------------------------------------------------------------------------
// Export chat as .txt file
// ---------------------------------------------------------------------------

function exportChat() {
  const messages = chatMessages.querySelectorAll(".message");
  if (messages.length <= 1) {
    alert("No messages to export yet.");
    return;
  }

  const lines = [`SupportBot — Chat Export`, `Session: ${sessionId}`,
                 `Date: ${new Date().toLocaleString()}`, `${"─".repeat(50)}`];

  messages.forEach((msg) => {
    const isUser  = msg.classList.contains("user-message");
    const bubble  = msg.querySelector(".bubble p");
    const time    = msg.querySelector(".timestamp");
    const intent  = msg.querySelector(".intent-badge");
    const text    = bubble ? bubble.textContent : "";
    const ts      = time   ? time.textContent.split("·")[0].trim() : "";
    const role    = isUser ? "You" : "Bot";

    lines.push(`\n[${ts}] ${role}:`);
    lines.push(text);
    if (intent) lines.push(`Intent: ${intent.textContent}`);
  });

  lines.push(`\n${"─".repeat(50)}`);
  lines.push(`Models used: NB · KNN · ANN | Bahria University Spring 2026`);

  const blob = new Blob([lines.join("\n")], { type: "text/plain" });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href     = url;
  a.download = `supportbot-chat-${sessionId.slice(0, 8)}.txt`;
  a.click();
  URL.revokeObjectURL(url);
}


// ---------------------------------------------------------------------------
// UI updates
// ---------------------------------------------------------------------------

function updateStats(confidence) {
  statMessages.textContent = messageCount;
  if (confidence !== undefined) {
    statConfidence.textContent = (confidence * 100).toFixed(1) + "%";
  }
}

function updateLastIntent(intent, confidence) {
  if (!intent) { lastIntentDisplay.classList.remove("visible"); return; }
  lastIntentDisplay.textContent =
    `🎯 ${intent.replace(/_/g, " ")} · ${(confidence * 100).toFixed(1)}%`;
  lastIntentDisplay.classList.add("visible");
}

function clearChat() {
  chatMessages.querySelectorAll(".message:not(#welcome-message)")
    .forEach((m) => m.remove());
  messageCount        = 0;
  conversationContext = [];
  searchResults.style.display = "none";
  searchInput.value           = "";
  updateStats(undefined);
  statConfidence.textContent = "—";
  lastIntentDisplay.classList.remove("visible");
  sessionId = generateUUID();
  sessionStorage.setItem("chatSessionId", sessionId);
  sessionDisplay.textContent = sessionId.slice(0, 8) + "...";
}


// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", init);
