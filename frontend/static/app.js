(() => {
  const transcript = document.getElementById("transcript");
  const turnLabel = document.getElementById("turn-label");
  const progressFill = document.getElementById("progress-fill");
  const completeBanner = document.getElementById("complete-banner");
  const composer = document.getElementById("composer");
  const messageInput = document.getElementById("message-input");
  const sendBtn = document.getElementById("send-btn");
  const startBtn = document.getElementById("start-btn");

  let sessionId = null;
  let maxTurns = 10;

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  function setTurn(turn) {
    turnLabel.textContent = `Turn ${turn} / ${maxTurns}`;
    progressFill.style.width = `${Math.min(100, (turn / maxTurns) * 100)}%`;
  }

  function setComposerEnabled(enabled) {
    messageInput.disabled = !enabled;
    sendBtn.disabled = !enabled;
  }

  function addUserBubble(text) {
    const row = document.createElement("div");
    row.className = "bubble-row user";
    row.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
    transcript.appendChild(row);
  }

  function addAgentBubble(data) {
    const row = document.createElement("div");
    row.className = "bubble-row agent";

    let html = `<div class="bubble">${escapeHtml(data.message || "")}</div>`;
    if (data.ask_attribute) {
      html += `<div class="ask-pill">Asking about: ${escapeHtml(data.ask_attribute)}</div>`;
    }
    if (data.recommendations && data.recommendations.length) {
      html += `<div class="rec-grid">`;
      data.recommendations.forEach((rec, i) => {
        const price = rec.price != null ? `$${Number(rec.price).toFixed(2)}` : "?";
        const rating = rec.average_rating != null ? `★ ${rec.average_rating.toFixed(1)}` : "";
        html += `
          <div class="rec-card">
            <div class="rank">#${i + 1}</div>
            <div class="title">${escapeHtml(rec.title)}</div>
            <div class="meta">${escapeHtml(rec.store || "")}</div>
            <div class="meta">${price} ${rating}</div>
          </div>`;
      });
      html += `</div>`;
    }
    row.innerHTML = html;
    transcript.appendChild(row);
  }

  function scrollToBottom() {
    transcript.scrollTop = transcript.scrollHeight;
  }

  function collectProfile() {
    const tags = document.getElementById("p-tags").value
      .split(",")
      .map((t) => t.trim())
      .filter(Boolean);
    return {
      purchase_frequency: document.getElementById("p-frequency").value,
      rating_style: document.getElementById("p-rating-style").value,
      average_prior_rating: parseFloat(document.getElementById("p-avg-rating").value) || null,
      preference_tags: tags,
      summary: document.getElementById("p-summary").value,
    };
  }

  async function startSession() {
    const res = await fetch("/api/session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ profile: collectProfile() }),
    });
    const data = await res.json();
    sessionId = data.session_id;
    maxTurns = data.max_turns;
    transcript.innerHTML = "";
    completeBanner.classList.add("hidden");
    setTurn(0);
    setComposerEnabled(true);
    messageInput.focus();
  }

  async function sendMessage(text) {
    addUserBubble(text);
    scrollToBottom();
    setComposerEnabled(false);

    const res = await fetch("/api/message", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message: text }),
    });
    const data = await res.json();

    if (!data.session_complete || data.message) {
      if (data.message !== undefined) {
        addAgentBubble(data);
      }
    }
    setTurn(data.turn || maxTurns);
    scrollToBottom();

    if (data.session_complete) {
      completeBanner.classList.remove("hidden");
      setComposerEnabled(false);
    } else {
      setComposerEnabled(true);
      messageInput.focus();
    }
  }

  composer.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = messageInput.value.trim();
    if (!text || !sessionId) return;
    messageInput.value = "";
    sendMessage(text);
  });

  startBtn.addEventListener("click", () => {
    startSession();
  });

  startSession();
})();
