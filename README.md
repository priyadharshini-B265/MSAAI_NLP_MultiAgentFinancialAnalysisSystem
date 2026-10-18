# Final Team Project: Multi-Agent Financial Analysis System with Orchestrator, Router & FAISS Memory

**Course:** Natural Language Processing (MSAAI) — Final Team Project  
**Team Identification:** `MS_AAI_520_GRP3`  
**Team Members:** Priyadharshini B., Moinuddin K, Ravisha Krishnamurthy  
**GitHub Repository:** [https://github.com/priyadharshini-B265/MSAAI_NLP_MultiAgentFinancialAnalysisSystem](https://github.com/priyadharshini-B265/MSAAI_NLP_MultiAgentFinancialAnalysisSystem)  
**Primary Target Asset:** Apple Inc. (`AAPL`) | **Secondary Assets:** NVIDIA Corporation (`NVDA`), JPMorgan Chase (`JPM`)  
**Technology Stack:** Python 3.10–3.13, Google Agent Development Kit (`google-adk`), Facebook AI Similarity Search (`faiss-cpu`), Anthropic Model Context Protocol (`mcp`), FastAPI, Uvicorn, Pydantic v2, `yfinance`, Pandas, NumPy, Matplotlib, Seaborn

---

## 1. System Overview & Architecture

The deliverable is a modular, multi-agent financial research system built according to the instructor's architecture clarification:
Rather than relying on a single monolithic prompt chain, the system implements an **orchestrator and routing workflow** that delegates distinct tasks to domain-specialist personas.

```text
                                 ┌────────────────────────────────────────────────────────┐
                                 │             USER / PORTFOLIO QUERY (AAPL)              │
                                 └───────────────────────────┬────────────────────────────┘
                                                             │
                                                             ▼
                                 ┌────────────────────────────────────────────────────────┐
                                 │             RESEARCH MANAGER AGENT (PLANNER)           │
                                 │   - Queries In-Memory FAISS Vector Memory for critiques│
                                 │   - Formulates Dynamic 7-Step Execution Plan           │
                                 └───────────────────────────┬────────────────────────────┘
                                                             │
                                                             ▼
                                 ┌────────────────────────────────────────────────────────┐
                                 │               MAIN INVESTMENT ORCHESTRATOR             │
                                 │  - Coordinates sub-agent lifecycles & global state     │
                                 │  - Dispatches FastMCP Tool Calls over stdio            │
                                 └───────────────────────────┬────────────────────────────┘
                                                             │
                    ┌────────────────────────────────────────┼────────────────────────────────────────┐
                    ↓                                        ↓                                        ↓
         ┌───────────────────────┐                ┌───────────────────────┐                ┌───────────────────────┐
         │   MARKET SPECIALIST   │                │  EARNINGS SPECIALIST  │                │    NEWS SPECIALIST    │
         │  (Prices, Vol, 20/50) │                │(P/E, Margins, ROE, D/E│                │(Ingest, Classify, Sum)│
         └───────────────────────┘                └───────────────────────┘                └───────────────────────┘
                    │                                        │                                        │
                    └────────────────────────────────────────┼────────────────────────────────────────┘
                                                             │
                                                             ▼
                                 ┌────────────────────────────────────────────────────────┐
                                 │            INTEGRATED RESEARCH SYNTHESIS               │
                                 └───────────────────────────┬────────────────────────────┘
                                                             │
                                                             ▼
                                 ┌────────────────────────────────────────────────────────┐
                                 │           EVALUATOR–OPTIMIZER CLOSED LOOP              │
                                 │  - Evaluator: Audits 5-point completeness rubric       │
                                 │  - Optimizer: Augments text with risks & limitations   │
                                 └───────────────────────────┬────────────────────────────┘
                                                             │
                                                             ▼
                                 ┌────────────────────────────────────────────────────────┐
                                 │       IN-MEMORY FAISS VECTOR STORE & PERSISTENCE       │
                                 │  - Dense 64-d embeddings | Exact Cosine Similarity     │
                                 │  - Saves lessons for subsequent run adaptation         │
                                 └────────────────────────────────────────────────────────┘
```

---

## 2. Exact Deterministic Agents & 7-Step Lifecycle

The system utilizes specialized deterministic agents with isolated responsibilities:

1. **Research Manager (`ResearchManagerAgent`):**
   - Formulates the 7-step plan before execution begins.
   - Recalls historical critique lessons from prior runs via FAISS vector search.
2. **Market Specialist (`MarketSpecialistAgent`):**
   - Computes 6-month returns, annualized volatility, 20-day & 50-day moving averages, and technical trend signals (`Bullish`/`Bearish`).
3. **Earnings Specialist (`EarningsSpecialistAgent`):**
   - Audits income statements, operating margins, trailing/forward P/E, return on equity (ROE), and debt-to-equity leverage.
4. **News Specialist (`NewsSpecialistAgent`):**
   - Ingests breaking news, preprocesses text, classifies category (`earnings`, `market`, `news`), extracts key topics, and produces bulleted executive summaries.
5. **Evaluator (`EvaluatorAgent`):**
   - Audits synthesis completeness across 5 quantitative checks: `has_market_data`, `has_financial_data`, `has_news_evidence`, `has_risk_discussion`, `has_uncertainty`.
6. **Optimizer (`OptimizerAgent`):**
   - Implements closed-loop iterative text refinement, injecting specific downside risk stress-tests and fiduciary uncertainty disclosures.

### The 7-Step Planning Lifecycle:
- **Step 1:** Collect market price history (`Yahoo Finance` / `FastMCP`)
- **Step 2:** Collect company financial information (`Yahoo Finance` / `FastMCP`)
- **Step 3:** Collect recent financial news (`Yahoo Finance news` / `FastMCP`)
- **Step 4:** Route content to specialist agents (`Router`)
- **Step 5:** Generate integrated research analysis (`Analysis`)
- **Step 6:** Evaluate and refine the analysis (`Evaluator/Optimizer`)
- **Step 7:** Store a learning note (`Memory` / `FAISS FlatIP`)

---

## 3. Key Add-Ons & Enhancements

### A. In-Memory FAISS Vector Store (`faiss.IndexFlatIP`)
- Implements continuous semantic vector search over past run evaluations and critique notes.
- Uses $d = 64$ dense semantic projections with unit-normalized vectors ($\|\mathbf{v}\|_2 = 1.0$).
- Exact Maximum Inner Product Search computes exact **Cosine Similarity**:
  $$\text{Sim}(\mathbf{q}, \mathbf{x}_i) = \langle \hat{\mathbf{q}}, \hat{\mathbf{x}}_i \rangle = \cos(\theta) \in [-1.0, 1.0]$$

### B. FastMCP Server Protocol (`financial_mcp_server.py`)
- Standards-compliant Model Context Protocol server exposing real-time financial tools over `stdio`:
  - `get_market_technicals`: Prices, 20/50/200 DMA, 14-day Wilder's RSI, trend classification.
  - `get_financial_statements`: Operating margins, YoY revenue growth, ROE, debt-to-equity.
  - `get_financial_news`: Verified headlines, publisher attribution, and canonical links.
  - `compute_technical_indicators`: Vectorized mathematical utility.

### C. Google ADK Interactive Web Dashboard (`web/`)
- Interactive, glassmorphism single-page web app running locally at **`http://127.0.0.1:8080`**.
- Features:
  - **Live Orchestrator Execution:** One-click autonomous run with real-time telemetry stream.
  - **Dynamic Specialist Router Dossier:** Inspects contract-based routing to domain specialists.
  - **Prompt Chaining Inspector:** Sequential 3-stage LLM handoffs (Classify $\to$ Extract $\to$ Summarize).
  - **Evaluator-Optimizer Closed Loop:** Interactive score convergence audit.
  - **FAISS Vector Memory Explorer:** Real-time semantic similarity query interface with cosine similarity scoring.
  - **FastMCP Tool Registry:** Direct stdio tool execution.

---

## 4. How to Execute the Project

### Prerequisites
Install the required dependencies using pip:
```bash
pip install -r requirements.txt
```

### Option 1: Run the Jupyter Notebook
Open and run the multi-agent orchestrator notebook in Jupyter Notebook, JupyterLab, or VS Code:
```bash
jupyter notebook notebooks/Assignment_7_1_Multi_Agent_Financial_Analysis_System_Orchestrator.ipynb
```
*Note: The notebook contains all 71 original team cells fully intact, plus the 3 clean Add-On sections, and is pre-executed with all outputs, tables, and charts.*

### Option 2: Run the Interactive Web Dashboard
Launch the FastAPI web server using the provided one-click launcher from the repository root:
```bash
python run_web.py
```
*(Or alternatively: `cd web && python server.py`)*

Open your browser and navigate to:
```
http://127.0.0.1:8080
```
- Click **"Run Full Orchestrator Pipeline"** to watch live multi-agent execution.
- Click **"Episodic Memory"** to execute live FAISS vector semantic queries.
- Click **"Dynamic Router"** to test specialist contract dispatches.

### Option 3: View Standalone HTML Report
The notebook has been pre-compiled to a self-contained HTML report with embedded images:
- Double click or open in any browser:
  - `notebooks/Assignment_7_1_Multi_Agent_Financial_Analysis_System_Orchestrator.html`

---

## 5. Submission File Manifest

```text
finalProject_submission/
├── notebooks/                                                       # Executed Jupyter notebooks & HTML reports
│   ├── Assignment_7_1_Multi_Agent_Financial_Analysis_System_Orchestrator.ipynb # Primary executed multi-agent orchestrator notebook
│   └── Assignment_7_1_Multi_Agent_Financial_Analysis_System_Orchestrator.html  # Standalone self-contained HTML report
├── images/                                                          # High-resolution publication figures (300 DPI)
│   ├── figure_1_market_technicals_rsi.png                           # Figure 1: Technicals, DMA & 14-Day RSI
│   ├── figure_2_specialist_routing_dossier.png                      # Figure 2: Specialist Routing Contracts
│   ├── figure_3_evaluator_optimizer_convergence.png                 # Figure 3: Closed-Loop Convergence
│   └── figure_4_episodic_memory_learning_curve.png                  # Figure 4: In-Memory FAISS Learning Curve
├── src/                                                             # Core Python source packages & tools
│   ├── __init__.py                                                  # Python package initializer
│   ├── financial_mcp_server.py                                      # FastMCP stdio tool server
│   └── generate_figures.py                                          # Figure reproduction script
├── memory/                                                          # Episodic & FAISS vector memory state
│   ├── investment_agent_memory.json                                 # Baseline JSON episodic memory
│   └── orchestrator_agent_memory.json                               # FAISS-indexed vector records (d=64)
├── web/                                                             # Interactive Web Dashboard
│   ├── server.py                                                    # FastAPI backend (6 agents + FAISS)
│   └── static/
│       ├── index.html                                               # UI (4-specialist widget, banner removed)
│       ├── app.js                                                   # Telemetry, FAISS search & router client
│       └── styles.css                                               # Glassmorphism styling & 4-col grid
├── run_web.py                                                       # One-click web server launcher
├── requirements.txt                                                 # Pinned Python package dependencies
├── README.md                                                        # Comprehensive guide & architecture
└── .gitignore                                                       # Git ignore rules
```
