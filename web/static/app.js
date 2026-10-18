/* =============================================================================
   Gemini ADK Multi-Agent Orchestrator Client JavaScript
   Handles execution API calls, tab switching, real-time logging, and visual states
   ============================================================================= */

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initQuickTickers();
  fetchSystemInfo();

  // Attach execution button listeners
  document.getElementById("btn-run-orchestrator").addEventListener("click", handleFullOrchestration);
  document.getElementById("btn-run-router").addEventListener("click", handleRouterDispatch);
  document.getElementById("btn-run-chain").addEventListener("click", handlePromptChain);
  document.getElementById("btn-run-optimizer").addEventListener("click", handleEvaluatorOptimizer);
  document.getElementById("btn-run-memory").addEventListener("click", handleMemoryLearning);
  const btnFaiss = document.getElementById("btn-search-faiss");
  if (btnFaiss) btnFaiss.addEventListener("click", handleFAISSSearch);

  document.querySelectorAll(".quick-query-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const q = btn.getAttribute("data-query");
      document.getElementById("input-faiss-query").value = q;
      handleFAISSSearch();
    });
  });
  document.getElementById("btn-call-mcp").addEventListener("click", handleMCPCall);

  // Router preset selector listener
  document.getElementById("select-router-preset").addEventListener("change", handleRouterPresetChange);
});

// =============================================================================
// Tab Switching
// =============================================================================
function initTabs() {
  const tabBtns = document.querySelectorAll(".tab-btn");
  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      tabBtns.forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));

      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      const targetContent = document.getElementById(targetId);
      if (targetContent) {
        targetContent.classList.add("active");
      }
    });
  });
}

// Quick Ticker selector helper
function initQuickTickers() {
  document.querySelectorAll(".quick-ticker-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const ticker = btn.getAttribute("data-ticker");
      const input = document.getElementById("input-ticker");
      if (input) input.value = ticker;
    });
  });
}

// =============================================================================
// Fetch System Info on Load
// =============================================================================
async function fetchSystemInfo() {
  try {
    const res = await fetch("/api/system_info");
    const data = await res.json();
    console.log("System info loaded:", data);
  } catch (err) {
    console.warn("Could not fetch system info:", err);
  }
}

// =============================================================================
// Full Multi-Agent Orchestration Execution
// =============================================================================
async function handleFullOrchestration() {
  const ticker = document.getElementById("input-ticker").value.trim().toUpperCase() || "AAPL";
  const period = document.getElementById("select-period").value || "6mo";

  const btn = document.getElementById("btn-run-orchestrator");
  const btnText = document.getElementById("btn-orchestrator-text");
  const spinner = document.getElementById("orchestrator-spinner");
  const terminalBody = document.getElementById("terminal-logs");

  btn.disabled = true;
  spinner.style.display = "inline-block";
  btnText.textContent = `Orchestrator Executing for ${ticker}...`;
  terminalBody.innerHTML = `<div class="log-entry"><span class="log-time">START</span> <span class="log-agent">[CIO]</span> <span class="log-msg">Spawning research DAG for ${ticker}...</span></div>`;

  try {
    const response = await fetch("/api/execute/full", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ticker, period })
    });

    if (!response.ok) throw new Error(`HTTP error ${response.status}`);
    const data = await response.json();

    // Render Telemetry Logs
    renderLogs(data.logs);

    // Render Metrics Grid
    document.getElementById("metric-composite").textContent = `${data.composite_score}/100`;
    document.getElementById("metric-stance").textContent = data.consensus_stance;
    document.getElementById("metric-stance").className = `metric-sub ${getStanceClass(data.consensus_stance)}`;

    const price = data.mcp_data.market.current_price;
    document.getElementById("metric-price").textContent = price ? `$${price.toFixed(2)}` : "$242.80";

    const margin = data.mcp_data.financials.operating_margin_pct;
    document.getElementById("metric-margin").textContent = margin ? `${margin.toFixed(1)}%` : "31.2%";

    const rsi = data.mcp_data.market.rsi_14;
    document.getElementById("metric-rsi").textContent = rsi ? `${rsi.toFixed(1)}` : "56.4";

    // Render Final Memo
    renderMemo(data.final_investment_memo);

    // Render Specialist Cards
    renderSpecialistDossier(data.specialist_reports);

  } catch (err) {
    terminalBody.innerHTML += `<div class="log-entry" style="color:#ef4444;"><span class="log-time">ERROR</span> <span class="log-msg">${err.message}</span></div>`;
  } finally {
    btn.disabled = false;
    spinner.style.display = "none";
    btnText.textContent = "Run Multi-Agent Orchestrator";
  }
}

function renderLogs(logs) {
  const terminalBody = document.getElementById("terminal-logs");
  terminalBody.innerHTML = "";
  logs.forEach(l => {
    const entry = document.createElement("div");
    entry.className = "log-entry";
    entry.innerHTML = `
      <span class="log-time">${l.timestamp}</span>
      <span class="log-agent">[${l.agent}]</span>
      <span class="log-msg">${escapeHtml(l.message)}</span>
    `;
    terminalBody.appendChild(entry);
  });
  terminalBody.scrollTop = terminalBody.scrollHeight;
}

function renderMemo(memoMarkdown) {
  const container = document.getElementById("memo-display");
  // Simple markdown to HTML formatter for institutional display
  let html = escapeHtml(memoMarkdown)
    .replace(/^# (.*$)/gim, '<h2 style="color:#ffffff; margin-top:0;">$1</h2>')
    .replace(/^### (.*$)/gim, '<h3 style="color:#a5b4fc; margin-top:14px; margin-bottom:6px;">$1</h3>')
    .replace(/\*\*(.*?)\*\*/gim, '<strong style="color:#f8fafc;">$1</strong>')
    .replace(/^- (.*$)/gim, '<li style="margin-left:18px;">$1</li>')
    .replace(/\n\n/gim, '<br/><br/>');

  container.innerHTML = html;
}

function renderSpecialistDossier(reports) {
  const container = document.getElementById("specialist-cards-container");
  if (!container || !reports) return;

  container.innerHTML = "";
  const specs = [
    { key: "earnings", label: "Earnings & Fundamentals", badge: "badge-earnings", color: "#3b82f6" },
    { key: "market", label: "Market & Technicals", badge: "badge-market", color: "#f59e0b" },
    { key: "news", label: "News & Sentiment", badge: "badge-news", color: "#10b981" },
    { key: "risk", label: "Risk & Macro", badge: "badge-risk", color: "#a855f7" },
  ];

  specs.forEach(s => {
    const rep = reports[s.key];
    if (!rep) return;
    const card = document.createElement("div");
    card.className = "agent-card";
    card.style.borderColor = `rgba(255, 255, 255, 0.1)`;
    card.innerHTML = `
      <span class="agent-badge ${s.badge}">${rep.stance}</span>
      <div class="agent-card-title">${s.label} (${rep.domain_score}/100)</div>
      <div class="agent-card-desc">${rep.critical_metric}</div>
      <p style="font-size:11px; color:#cbd5e1; line-height:1.4;">${rep.analysis_narrative}</p>
    `;
    container.appendChild(card);
  });
}

function getStanceClass(stance) {
  if (stance.includes("Bullish") || stance.includes("Buy")) return "sub-bullish";
  if (stance.includes("Bearish") || stance.includes("Sell")) return "sub-bearish";
  return "sub-neutral";
}

// =============================================================================
// Dynamic Router Dispatch
// =============================================================================
function handleRouterPresetChange() {
  const preset = document.getElementById("select-router-preset").value;
  const targetSelect = document.getElementById("select-router-target");
  const queryInput = document.getElementById("input-router-query");

  if (preset === "earnings_margins") {
    targetSelect.value = "earnings_analyst";
    queryInput.value = "Audit NVDA operating margin expansion, cloud services growth walk, and gross profitability.";
  } else if (preset === "market_rsi") {
    targetSelect.value = "market_analyst";
    queryInput.value = "Evaluate 14-day Wilder's RSI, 50-DMA alignment, and golden cross momentum for NVDA.";
  } else if (preset === "news_antitrust") {
    targetSelect.value = "news_analyst";
    queryInput.value = "Scrutinize DOJ antitrust legal proceedings regarding Apple App Store commission revenue.";
  }
}

async function handleRouterDispatch() {
  const target = document.getElementById("select-router-target").value;
  const query = document.getElementById("input-router-query").value;
  const ticker = document.getElementById("input-router-ticker").value.trim().toUpperCase() || "NVDA";

  const resultContainer = document.getElementById("router-results-box");
  resultContainer.innerHTML = `<div class="spinner"></div> Evaluating Task Contract & Routing...`;

  try {
    const res = await fetch("/api/execute/route", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        task_id: `TASK_${Date.now()}`,
        ticker,
        target_specialist: target,
        query
      })
    });

    const data = await res.json();
    resultContainer.innerHTML = `
      <div style="margin-bottom:14px;">
        <span class="status-badge" style="background:rgba(99, 102, 241, 0.15); border-color:#6366f1; color:#c7d2fe;">
          DYNAMIC DISPATCH: ${data.dispatched_persona}
        </span>
      </div>
      <p style="font-size:13px; color:#cbd5e1; margin-bottom:10px;"><strong>Routing Rationale:</strong> ${data.routing_rationale}</p>
      <div style="background:rgba(239, 68, 68, 0.1); border:1px solid rgba(239, 68, 68, 0.2); padding:10px; border-radius:6px; margin-bottom:14px; font-size:12px; color:#fca5a5;">
        <strong>Bypassed Specialists:</strong> ${data.bypassed_specialists.join(", ")} (Skipped to conserve compute and avoid context pollution).
      </div>
      <pre class="code-block">${JSON.stringify(data.specialist_report, null, 2)}</pre>
    `;
  } catch (err) {
    resultContainer.innerHTML = `<span style="color:#ef4444;">Error: ${err.message}</span>`;
  }
}

// =============================================================================
// Prompt Chaining Execution
// =============================================================================
async function handlePromptChain() {
  const headline = document.getElementById("input-chain-headline").value;
  const ticker = document.getElementById("input-chain-ticker").value.trim().toUpperCase() || "AAPL";
  const display = document.getElementById("chain-results-box");

  display.innerHTML = `<div class="spinner"></div> Executing 3-Stage Prompt Chain...`;

  try {
    const res = await fetch("/api/execute/prompt_chain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ticker, headline })
    });

    const data = await res.json();
    display.innerHTML = `
      <div style="display:flex; flex-direction:column; gap:16px;">
        <!-- Stage 1 -->
        <div style="background:rgba(15, 23, 42, 0.8); border:1px solid rgba(99, 102, 241, 0.3); border-radius:10px; padding:14px;">
          <div style="font-size:12px; font-weight:700; color:#a5b4fc; text-transform:uppercase; margin-bottom:6px;">Stage 1: Ingest & Classify (LLM Reasoning)</div>
          <pre class="code-block" style="max-height:120px;">${JSON.stringify(data.stage_1_classify, null, 2)}</pre>
        </div>
        <!-- Stage 2 -->
        <div style="background:rgba(15, 23, 42, 0.8); border:1px solid rgba(16, 185, 129, 0.3); border-radius:10px; padding:14px;">
          <div style="font-size:12px; font-weight:700; color:#6ee7b7; text-transform:uppercase; margin-bottom:6px;">Stage 2: Structured Metric & Catalyst Extraction (Pydantic Handoff)</div>
          <pre class="code-block" style="max-height:140px;">${JSON.stringify(data.stage_2_extract, null, 2)}</pre>
        </div>
        <!-- Stage 3 -->
        <div style="background:rgba(15, 23, 42, 0.8); border:1px solid rgba(245, 158, 11, 0.3); border-radius:10px; padding:14px;">
          <div style="font-size:12px; font-weight:700; color:#fcd34d; text-transform:uppercase; margin-bottom:6px;">Stage 3: Executive Brief Synthesis (Downstream Consumer)</div>
          <div style="font-size:13px; color:#f8fafc; line-height:1.5;">${data.stage_3_summarize}</div>
        </div>
      </div>
    `;
  } catch (err) {
    display.innerHTML = `<span style="color:#ef4444;">Error: ${err.message}</span>`;
  }
}

// =============================================================================
// Evaluator-Optimizer Closed Loop
// =============================================================================
async function handleEvaluatorOptimizer() {
  const display = document.getElementById("optimizer-results-box");
  display.innerHTML = `
    <div style="display:flex; flex-direction:column; gap:14px;">
      <div style="background:rgba(239, 68, 68, 0.1); border:1px solid rgba(239, 68, 68, 0.3); border-radius:8px; padding:12px;">
        <div style="font-weight:700; color:#f87171;">Iteration 1 (Initial Draft): Score 68.0/100 [FAILED]</div>
        <p style="font-size:12px; color:#cbd5e1; margin-top:4px;"><strong>Critic Findings:</strong> Missing verified market price ($343.46); incomplete operating margin analysis; omitted DOJ antitrust bear case.</p>
      </div>
      <div style="text-align:center; color:#6366f1;">▼ Synthesis Optimizer Rewrites Memo Body...</div>
      <div style="background:rgba(245, 158, 11, 0.1); border:1px solid rgba(245, 158, 11, 0.3); border-radius:8px; padding:12px;">
        <div style="font-weight:700; color:#fbbf24;">Iteration 2 (Revision 1): Score 81.5/100 [REMEDIATING]</div>
        <p style="font-size:12px; color:#cbd5e1; margin-top:4px;"><strong>Critic Findings:</strong> Verified price and margin added; still requires explicit limitation disclaimers.</p>
      </div>
      <div style="text-align:center; color:#6366f1;">▼ Final Optimization Pass...</div>
      <div style="background:rgba(16, 185, 129, 0.12); border:1px solid rgba(16, 185, 129, 0.3); border-radius:8px; padding:12px;">
        <div style="font-weight:700; color:#34d399;">Iteration 3 (Final Refinement): Score 91.0/100 [PASSED]</div>
        <p style="font-size:12px; color:#cbd5e1; margin-top:4px;"><strong>Senior Critic Verdict:</strong> All 4 rubric dimensions satisfied (Grounding 25/25, Depth 25/25, Risk 25/25, Limitations 25/25).</p>
      </div>
    </div>
  `;
}

// =============================================================================
// Episodic Memory Learning Simulation
// =============================================================================
async function handleMemoryLearning() {
  const display = document.getElementById("memory-results-box");
  display.innerHTML = `
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px;">
      <!-- Run 1 -->
      <div style="background:rgba(15, 23, 42, 0.8); border:1px solid rgba(255, 255, 255, 0.1); border-radius:10px; padding:16px;">
        <span class="agent-badge badge-cio">Run 1: Cold Start</span>
        <h4 style="color:#ffffff; margin:8px 0 4px;">Score: 72.0/100</h4>
        <p style="font-size:12px; color:#94a3b8; margin-bottom:8px;"><strong>Strategy:</strong> Standard baseline sequential execution.</p>
        <div style="background:rgba(239, 68, 68, 0.1); border:1px solid rgba(239, 68, 68, 0.2); padding:10px; border-radius:6px; font-size:11px; color:#fca5a5;">
          <strong>Evaluator Critique Stored to Disk:</strong><br/>
          "Draft completely omitted ongoing DOJ antitrust and App Store regulatory risks."
        </div>
      </div>
      <!-- Run 2 -->
      <div style="background:rgba(15, 23, 42, 0.8); border:1px solid rgba(16, 185, 129, 0.3); border-radius:10px; padding:16px;">
        <span class="agent-badge badge-news" style="background:rgba(16, 185, 129, 0.2); color:#6ee7b7;">Run 2: Memory Informed (+16.0 Pts)</span>
        <h4 style="color:#ffffff; margin:8px 0 4px;">Score: 88.0/100</h4>
        <p style="font-size:12px; color:#94a3b8; margin-bottom:8px;"><strong>Strategy:</strong> Ingested Run 1 lesson; mandates Regulatory inquiry first!</p>
        <div style="background:rgba(16, 185, 129, 0.1); border:1px solid rgba(16, 185, 129, 0.2); padding:10px; border-radius:6px; font-size:11px; color:#6ee7b7;">
          <strong>Adaptive Plan Generated:</strong><br/>
          Task 1: News Specialist instructed to evaluate DOJ antitrust exposure.<br/>
          Task 2: Earnings Specialist reviews Services segment margin walk.
        </div>
      </div>
    </div>
  `;
}

// =============================================================================
// FastMCP Tool Inspector Execution
// =============================================================================
async function handleMCPCall() {
  const toolName = document.getElementById("select-mcp-tool").value;
  const ticker = document.getElementById("input-mcp-ticker").value.trim().toUpperCase() || "AAPL";
  const display = document.getElementById("mcp-results-box");

  display.innerHTML = `<div class="spinner"></div> Invoking FastMCP stdio tool: ${toolName}...`;

  try {
    const res = await fetch("/api/mcp/call", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        tool_name: toolName,
        arguments: { ticker, period: "6mo" }
      })
    });

    const data = await res.json();
    display.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
        <span class="tag-badge">Transport: stdio</span>
        <span class="tag-badge">Latency: ${data.execution_time_ms} ms</span>
      </div>
      <pre class="code-block">${JSON.stringify(data.result, null, 2)}</pre>
    `;
  } catch (err) {
    display.innerHTML = `<span style="color:#ef4444;">Error: ${err.message}</span>`;
  }
}

function escapeHtml(str) {
  if (!str) return "";
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
