/**
 * frontend/chat.js — Complete Version
 * All features: model switching, confidence threshold, context,
 * search, export, compare mode, quick replies, analytics, dark/light mode.
 */

"use strict";

const API_BASE = "http://localhost:5000/api";

// DOM references
const chatMessages       = document.getElementById("chat-messages");
const userInput          = document.getElementById("user-input");
const sendBtn            = document.getElementById("send-btn");
const typingIndicator    = document.getElementById("typing-indicator");
const sessionDisplay     = document.getElementById("session-display");
const statMessages       = document.getElementById("stat-messages");
const statConfidence     = document.getElementById("stat-confidence");
const clearBtn           = document.getElementById("clear-btn");
const exportBtn          = document.getElementById("export-btn");
const lastIntentDisplay  = document.getElementById("last-intent-display");
const welcomeTime        = document.getElementById("welcome-time");
const modelBtns          = document.querySelectorAll(".model-btn");
const themeToggleBtn     = document.getElementById("theme-toggle-btn");
const confidenceSlider   = document.getElementById("confidence-slider");
const thresholdDisplay   = document.getElementById("threshold-display");
const searchInput        = document.getElementById("search-input");
const searchBtn          = document.getElementById("search-btn");
const searchResults      = document.getElementById("search-results");
const compareBtn         = document.getElementById("compare-btn");
const compareModal       = document.getElementById("compare-modal");
const modalClose         = document.getElementById("modal-close");
const compareInput       = document.getElementById("compare-input");
const compareSendBtn     = document.getElementById("compare-send-btn");
const analyticsBtn       = document.getElementById("analytics-btn");
const analyticsModal     = document.getElementById("analytics-modal");
const analyticsModalClose = document.getElementById("analytics-modal-close");

// State
let sessionId           = "";
let messageCount        = 0;
let activeModel         = "ann";
let confidenceThreshold = 0.30;
let conversationContext = [];
const MAX_CONTEXT       = 6;

// ---------------------------------------------------------------------------
// Quick Replies
// ---------------------------------------------------------------------------
const QUICK_REPLIES = {
  cancel_order:            ["📦 Track my order",        "💰 Request a refund",       "📞 Contact support"],
  change_order:            ["📦 Track my order",        "🏠 Change shipping address", "📞 Contact support"],
  change_shipping_address: ["📦 Track my order",        "✏️ Edit my account",        "📞 Contact support"],
  check_cancellation_fee:  ["❌ Cancel my order",       "💰 Get a refund",           "📋 Check refund policy"],
  check_invoices:          ["🧾 Get my invoice",        "💳 Check payment methods",  "📞 Contact support"],
  check_payment_methods:   ["💳 Place an order",        "⚠️ Report payment issue",   "🧾 Check invoices"],
  check_refund_policy:     ["💰 Get a refund",          "❌ Cancel my order",        "📞 Contact support"],
  complaint:               ["📞 Contact human agent",   "💰 Get a refund",           "⭐ Leave a review"],
  contact_customer_service:["📞 Talk to human agent",   "📦 Track my order",         "💰 Get a refund"],
  contact_human_agent:     ["📞 Contact customer service","💰 Get a refund",          "❌ Cancel my order"],
  create_account:          ["🔑 Recover password",      "✏️ Edit my account",        "📞 Contact support"],
  delete_account:          ["✏️ Edit account instead",  "🔑 Recover password",       "📞 Contact support"],
  delivery_options:        ["📦 Track my order",        "🏠 Change shipping address", "⏱️ Delivery period"],
  delivery_period:         ["📦 Track my order",        "🚚 Delivery options",        "📞 Contact support"],
  edit_account:            ["🔑 Recover password",      "🔄 Switch account",          "🗑️ Delete account"],
  get_invoice:             ["🧾 Check all invoices",    "💳 Check payment methods",  "📞 Contact support"],
  get_refund:              ["📋 Check refund policy",   "🔍 Track my refund",        "📞 Contact support"],
  newsletter_subscription: ["✏️ Edit my account",       "📞 Contact support",         "⭐ Leave a review"],
  payment_issue:           ["💳 Check payment methods", "💰 Get a refund",           "📞 Contact support"],
  place_order:             ["📦 Track my order",        "🚚 Delivery options",        "💳 Check payment methods"],
  recover_password:        ["✏️ Edit my account",       "🔄 Switch account",          "📞 Contact support"],
  registration_problems:   ["🔑 Recover password",      "📞 Contact support",         "✏️ Edit my account"],
  review:                  ["⭐ Place an order",         "📞 Contact support",         "💰 Get a refund"],
  set_up_shipping_address: ["🏠 Change shipping address","📦 Track my order",         "✏️ Edit my account"],
  switch_account:          ["✏️ Edit my account",       "🔑 Recover password",        "🗑️ Delete account"],
  track_order:             ["💰 Get a refund",          "❌ Cancel my order",         "📞 Contact support"],
  track_refund:            ["💰 Request a refund",      "📋 Check refund policy",     "📞 Contact support"],
};


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

  // Compare modal
  compareBtn.addEventListener("click", () => {
    compareModal.style.display = "flex";
    compareInput.focus();
  });
  modalClose.addEventListener("click", () => {
    compareModal.style.display = "none";
  });
  compareModal.addEventListener("click", (e) => {
    if (e.target === compareModal) compareModal.style.display = "none";
  });
  compareSendBtn.addEventListener("click", handleCompare);
  compareInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") handleCompare();
  });

  // Analytics modal
  analyticsBtn.addEventListener("click", () => {
    analyticsModal.style.display = "flex";
    loadAnalytics();
  });
  analyticsModalClose.addEventListener("click", () => {
    analyticsModal.style.display = "none";
  });
  analyticsModal.addEventListener("click", (e) => {
    if (e.target === analyticsModal) analyticsModal.style.display = "none";
  });

  // Apply saved theme on load.
  const savedTheme = localStorage.getItem("theme") || "dark";
  if (savedTheme === "light") {
    document.body.classList.remove("dark-mode");
    document.body.classList.add("light-mode");
    themeToggleBtn.textContent = "☀️";
  } else {
    document.body.classList.remove("light-mode");
    document.body.classList.add("dark-mode");
    themeToggleBtn.textContent = "🌙";
  }

  themeToggleBtn.addEventListener("click", () => {
    const isDark = document.body.classList.contains("dark-mode");
    if (isDark) {
      document.body.classList.remove("dark-mode");
      document.body.classList.add("light-mode");
      themeToggleBtn.textContent = "🌙";
      localStorage.setItem("theme", "light");
    } else {
      document.body.classList.remove("light-mode");
      document.body.classList.add("dark-mode");
      themeToggleBtn.textContent = "☀️";
      localStorage.setItem("theme", "dark");
    }
  });

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

  // Color badge based on confidence level.
  const confNum    = parseFloat(conf);
  const badgeClass = confNum >= 80 ? "high" : confNum >= 50 ? "medium" : "low";

  const badge = intent
    ? `<span class="intent-badge ${badgeClass}">🎯 ${label} · ${conf}%</span>`
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

  // Add quick reply suggestions if intent is known and not an error.
  if (intent && !isError && QUICK_REPLIES[intent]) {
    appendQuickReplies(intent);
  }

  scrollToBottom();
}

/**
 * Append quick reply suggestion buttons after a bot message.
 * Clicking a button sends that text as a new user message.
 * Buttons are removed once any one is clicked.
 *
 * @param {string} intent - The detected intent to get suggestions for.
 */
function appendQuickReplies(intent) {
  const suggestions = QUICK_REPLIES[intent];
  if (!suggestions || suggestions.length === 0) return;

  const existing = document.getElementById("quick-replies-current");
  if (existing) existing.remove();

  const container = document.createElement("div");
  container.className = "quick-replies";
  container.id        = "quick-replies-current";

  suggestions.forEach((suggestion) => {
    const btn = document.createElement("button");
    btn.className   = "quick-reply-btn";
    btn.textContent = suggestion;

    btn.addEventListener("click", () => {
      // Remove all quick reply buttons when one is clicked.
      const existingReplies = document.getElementById("quick-replies-current");
      if (existingReplies) existingReplies.remove();

      // Strip emoji prefix from suggestion to get clean text.
      const cleanText = suggestion.replace(/^[\p{Emoji}\s]+/u, "").trim();

      // Put the text in the input and send it.
      userInput.value = cleanText;
      userInput.dispatchEvent(new Event("input"));
      handleSend();
    });

    container.appendChild(btn);
  });

  chatMessages.appendChild(container);
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

  // Simulate realistic thinking time per model.
  // KNN is genuinely slower so we reflect that visually.
  const thinkingDelay = { ann: 600, nb: 300, knn: 1400 };
  await new Promise(r => setTimeout(r, thinkingDelay[activeModel] || 600));

  try {
    const response = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message:              text,
        session_id:           sessionId,
        confidence_threshold: confidenceThreshold,
        model_type:           activeModel,
        // Send last N messages for context awareness.
        context:              conversationContext.slice(-MAX_CONTEXT),
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
    const response = await fetch(`${API_BASE}/search?session_id=${sessionId}&q=${encodeURIComponent(query)}`);
    const data     = await response.json();

    if (!response.ok || data.count === 0) {
      searchResults.innerHTML = `<p class="search-no-results">No results for "${escapeHTML(query)}"</p>`;
      return;
    }

    searchResults.innerHTML = data.results.map((r) => `
      <div class="search-result-item">
        <div class="result-user">👤 ${escapeHTML(r.user_message)}</div>
        <div class="result-bot">🤖 ${escapeHTML(r.bot_response.slice(0, 80))}...</div>
        <div class="result-meta">🎯 ${r.predicted_intent || "unknown"} · ${r.model_used?.toUpperCase() || "—"}</div>
      </div>`).join("");
  } catch (err) {
    searchResults.innerHTML = `<p class="search-no-results">Search failed.</p>`;
  }
}


// ---------------------------------------------------------------------------
// API — Model Comparison
// ---------------------------------------------------------------------------

/**
 * Send the same message to all 3 models simultaneously and display
 * results side by side in the comparison modal.
 */
async function handleCompare() {
  const text = compareInput.value.trim();
  if (!text) return;

  compareSendBtn.disabled = true;

  // Reset all cards to loading state.
  ["nb", "knn", "ann"].forEach((m) => {
    document.getElementById(`${m}-response`).textContent = "Thinking...";
    document.getElementById(`${m}-badge`).textContent    = "—";
    document.getElementById(`${m}-badge`).className      = "compare-badge";
    document.getElementById(`${m}-meta`).textContent     = "";
    document.getElementById(`compare-${m}`).classList.remove("winner");
  });

  // Fire all 3 requests in parallel.
  const models  = ["nb", "knn", "ann"];
  const results = await Promise.allSettled(
    models.map((model) =>
      fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message:              text,
          session_id:           sessionId,
          model_type:           model,
          confidence_threshold: confidenceThreshold,
        }),
      }).then((r) => r.json())
    )
  );

  // Find highest confidence for winner highlight.
  let highestConf  = -1;
  let winnerModel  = null;

  results.forEach((result, i) => {
    const model = models[i];
    const responseEl = document.getElementById(`${model}-response`);
    const badgeEl    = document.getElementById(`${model}-badge`);
    const metaEl     = document.getElementById(`${model}-meta`);

    if (result.status === "fulfilled" && result.value.response) {
      const data       = result.value;
      const conf       = data.confidence || 0;
      const confPct    = (conf * 100).toFixed(1);
      const intentLabel = data.intent
        ? data.intent.replace(/_/g, " ")
        : "fallback";

      // Response text.
      responseEl.textContent = data.response;

      // Badge with color coding.
      badgeEl.textContent = data.intent
        ? `🎯 ${intentLabel} · ${confPct}%`
        : "❓ fallback";

      const badgeClass = !data.intent  ? "fallback"
                       : conf >= 0.80  ? "high"
                       : conf >= 0.50  ? "medium"
                       :                 "low";
      badgeEl.className = `compare-badge ${badgeClass}`;

      // Meta line.
      metaEl.textContent = `Confidence: ${confPct}% · Model: ${model.toUpperCase()}`;

      // Track winner (highest confidence).
      if (conf > highestConf) {
        highestConf = conf;
        winnerModel = model;
      }

    } else {
      responseEl.textContent = "⚠️ Request failed.";
      badgeEl.textContent    = "error";
    }
  });

  // Highlight the most confident model.
  if (winnerModel) {
    document.getElementById(`compare-${winnerModel}`)
            .classList.add("winner");
  }

  compareSendBtn.disabled = false;
}


// ---------------------------------------------------------------------------
// Analytics
// ---------------------------------------------------------------------------

/**
 * Fetch intent frequency data from the backend and render a bar chart
 * inside the analytics modal.
 */
async function loadAnalytics() {
  const chart   = document.getElementById("analytics-chart");
  const totalEl = document.getElementById("analytics-total");
  const intentCountEl = document.getElementById("analytics-intents");
  const topEl   = document.getElementById("analytics-top");

  chart.innerHTML       = `<p class="analytics-loading">Loading analytics...</p>`;
  totalEl.textContent       = "—";
  intentCountEl.textContent = "—";
  topEl.textContent         = "—";

  try {
    const response = await fetch(`${API_BASE}/analytics`);
    const data     = await response.json();

    if (!response.ok || !data.intents || data.intents.length === 0) {
      chart.innerHTML = `<p class="analytics-loading">No data yet. Send some messages first!</p>`;
      return;
    }


  const intents  = data.intents;
  const maxCount = intents[0].count;

    totalEl.textContent       = data.total_messages.toLocaleString();
    intentCountEl.textContent = intents.length;
    topEl.textContent         = intents[0].intent.replace(/_/g, " ");

    chart.innerHTML = intents.map((item, i) => {
      const pct       = ((item.count / maxCount) * 100).toFixed(1);
      const label     = item.intent.replace(/_/g, " ");
      const rankClass = i < 5 ? `rank-${i + 1}` : "";

      return `
        <div class="chart-row">
          <span class="chart-label" title="${label}">${label}</span>
          <div class="chart-bar-wrapper">
              <div class="chart-bar ${rankClass}" style="width:${pct}%"></div>
          </div>
          <span class="chart-count">${item.count}</span>
        </div>`;
    }).join("");

  } catch (err) {
    chart.innerHTML = `<p class="analytics-loading">⚠️ Failed to load. Is the server running?</p>`;
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
    const isUser = msg.classList.contains("user-message");
    const bubble = msg.querySelector(".bubble p");
    const time   = msg.querySelector(".timestamp");
    const intent = msg.querySelector(".intent-badge");
    if (!bubble) return;
    lines.push(`\n[${time?.textContent.split("·")[0].trim() || ""}] ${isUser ? "You" : "Bot"}:`);
    lines.push(bubble.textContent);
    if (intent) lines.push(`Intent: ${intent.textContent}`);
  });

  lines.push(`\n${" ─".repeat(50)}\nModels used: NB · KNN · ANN | Bahria University Spring 2026`);

  const blob = new Blob([lines.join("\n")], { type: "text/plain" });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href = url; a.download = `supportbot-chat-${sessionId.slice(0,8)}.txt`;
  a.click(); URL.revokeObjectURL(url);
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
  lastIntentDisplay.textContent = `🎯 ${intent.replace(/_/g," ")} · ${(confidence*100).toFixed(1)}%`;
  lastIntentDisplay.classList.add("visible");
}

function clearChat() {
  const qr = document.getElementById("quick-replies-current");
  if (qr) qr.remove();
  messageCount = 0; conversationContext = [];
  searchResults.style.display = "none"; searchInput.value = "";
  updateStats(undefined);
  statConfidence.textContent = "—";
  lastIntentDisplay.classList.remove("visible");
  sessionId = generateUUID();
  sessionStorage.setItem("chatSessionId", sessionId);
  sessionDisplay.textContent = sessionId.slice(0,8) + "...";
}


// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", init);
