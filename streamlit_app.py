"""TradingAgents Dashboard — Streamlit Web UI."""

import os
import re
import json
import threading
from pathlib import Path
from datetime import date, timedelta

import streamlit as st
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TradingAgents Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
/* ---- overall ---- */
section[data-testid="stSidebar"] {min-width:300px; max-width:340px}

/* ---- decision badge ---- */
.decision-badge {
    display:inline-block; padding:12px 32px; border-radius:12px;
    font-size:2rem; font-weight:800; letter-spacing:2px; color:#fff;
    text-align:center;
}
.badge-buy        {background:linear-gradient(135deg,#00c853,#009624)}
.badge-overweight  {background:linear-gradient(135deg,#66bb6a,#43a047)}
.badge-hold       {background:linear-gradient(135deg,#ff9800,#ef6c00)}
.badge-underweight {background:linear-gradient(135deg,#ff7043,#e64a19)}
.badge-sell       {background:linear-gradient(135deg,#f44336,#b71c1c)}

/* ---- metric card ---- */
.metric-card {
    background:var(--background-secondary, #f8f9fa); border-radius:12px;
    padding:20px; text-align:center; border:1px solid rgba(128,128,128,.15);
}
.metric-card h3 {margin:0; font-size:.85rem; opacity:.6}
.metric-card p  {margin:4px 0 0; font-size:1.5rem; font-weight:700}

/* ---- debate column ---- */
.debate-card {
    border-radius:10px; padding:16px; margin-bottom:12px;
    border:1px solid rgba(128,128,128,.15);
}
.debate-bull {background:rgba(0,200,83,.07)}
.debate-bear {background:rgba(244,67,54,.07)}
.debate-judge {background:rgba(33,150,243,.07)}

/* ---- progress ---- */
.step-done   {color:#00c853}
.step-active {color:#ff9800}
.step-wait   {color:#9e9e9e}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Secrets → env (for Streamlit Cloud deployment)
# ---------------------------------------------------------------------------
for key in ("OPENROUTER_API_KEY", "FRED_API_KEY", "GOOGLE_API_KEY"):
    val = st.secrets.get(key, "") if hasattr(st, "secrets") else ""
    if val and not os.environ.get(key):
        os.environ[key] = val

# Also load from .env for local development
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

AGENT_STEPS = [
    "Market Analyst",
    "Social Analyst",
    "News Analyst",
    "Fundamentals Analyst",
    "Macro Analyst",
    "Bull Researcher",
    "Bear Researcher",
    "Research Manager",
    "Trader",
    "Aggressive Analyst",
    "Conservative Analyst",
    "Neutral Analyst",
    "Portfolio Manager",
]


def badge_class(decision: str) -> str:
    d = decision.strip().upper()
    if "BUY" in d and "OVER" not in d:
        return "badge-buy"
    if "OVERWEIGHT" in d:
        return "badge-overweight"
    if "HOLD" in d:
        return "badge-hold"
    if "UNDERWEIGHT" in d:
        return "badge-underweight"
    if "SELL" in d:
        return "badge-sell"
    return "badge-hold"


def extract_decision(text: str) -> str:
    """Pull the first BUY / HOLD / SELL / etc. from the report."""
    for word in ("BUY", "OVERWEIGHT", "HOLD", "UNDERWEIGHT", "SELL"):
        if word in text.upper():
            return word
    return "N/A"


def extract_metrics(fundamentals: str) -> dict:
    """Best-effort extraction of key numbers from fundamentals report."""
    metrics = {}
    patterns = {
        "PE Ratio": r"PE Ratio.*?:\s*([\d.]+)",
        "ROE": r"ROE.*?:\s*([\d.]+)%",
        "Gross Margin": r"Gross Margin.*?:\s*([\d.]+)%",
        "Profit Margin": r"Profit Margin.*?:\s*([\d.]+)%",
        "Current Ratio": r"Current Ratio.*?:\s*([\d.]+)",
        "Market Cap": r"Market Cap.*?:\s*\$?([\d.,]+\s*(?:Trillion|Billion|Million)?)",
    }
    for label, pat in patterns.items():
        m = re.search(pat, fundamentals, re.IGNORECASE)
        if m:
            metrics[label] = m.group(1)
    return metrics


def extract_allocation(text: str) -> dict:
    """Parse portfolio allocation table from final decision."""
    alloc = {}
    for m in re.finditer(
        r"\|\s*([A-Za-z\s\(\)]+?)\s*\|\s*(\d+)%", text
    ):
        name = m.group(1).strip()
        pct = int(m.group(2))
        if pct > 0:
            alloc[name] = pct
    return alloc


def list_history() -> list[dict]:
    """Scan eval_results/ for past runs."""
    results = []
    base = Path("eval_results")
    if not base.exists():
        return results
    for ticker_dir in sorted(base.iterdir()):
        if not ticker_dir.is_dir() or ticker_dir.name.startswith("."):
            continue
        log_dir = ticker_dir / "TradingAgentsStrategy_logs"
        if not log_dir.exists():
            continue
        for f in sorted(log_dir.glob("full_states_log_*.json")):
            date_str = f.stem.replace("full_states_log_", "")
            results.append(
                {"ticker": ticker_dir.name, "date": date_str, "path": str(f)}
            )
    return results


def load_result(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # JSON has date keys at root; return the first (usually only) entry
        for _date_key, state in data.items():
            return state
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Run analysis in background thread
# ---------------------------------------------------------------------------

def run_analysis_thread(ticker: str, trade_date: str, analysts: list[str]):
    """Execute the full TradingAgents pipeline, updating session_state progress."""
    ss = st.session_state

    try:
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        from tradingagents.default_config import DEFAULT_CONFIG

        config = DEFAULT_CONFIG.copy()
        config["llm_provider"] = "openrouter"
        config["deep_think_llm"] = "openai/gpt-4o-mini"
        config["quick_think_llm"] = "openai/gpt-4o-mini"
        config["max_debate_rounds"] = 1
        config["max_risk_discuss_rounds"] = 1

        ss.progress_step = 0
        ss.progress_label = "Initializing agents..."

        ta = TradingAgentsGraph(
            selected_analysts=analysts, debug=False, config=config
        )

        # We patch the graph to track which node fires
        original_stream = ta.graph.stream

        def tracked_stream(state, **kwargs):
            step_map = {}
            for i, name in enumerate(AGENT_STEPS):
                step_map[name] = i

            for chunk in original_stream(state, **kwargs):
                # chunk is a dict; keys are node names
                if isinstance(chunk, dict):
                    for node_name in chunk:
                        if node_name in step_map:
                            ss.progress_step = step_map[node_name] + 1
                            ss.progress_label = f"{node_name} completed"
                yield chunk

        ta.graph.stream = tracked_stream

        ss.progress_label = "Starting analysis pipeline..."
        final_state, decision = ta.propagate(ticker, trade_date)

        # Build result dict matching JSON log format
        result = {
            "company_of_interest": ticker,
            "trade_date": trade_date,
            "market_report": final_state.get("market_report", ""),
            "sentiment_report": final_state.get("sentiment_report", ""),
            "news_report": final_state.get("news_report", ""),
            "fundamentals_report": final_state.get("fundamentals_report", ""),
            "macro_report": final_state.get("macro_report", ""),
            "investment_debate_state": {
                "bull_history": final_state.get("investment_debate_state", {}).get("bull_history", ""),
                "bear_history": final_state.get("investment_debate_state", {}).get("bear_history", ""),
                "history": final_state.get("investment_debate_state", {}).get("history", ""),
                "current_response": final_state.get("investment_debate_state", {}).get("current_response", ""),
                "judge_decision": final_state.get("investment_debate_state", {}).get("judge_decision", ""),
            },
            "trader_investment_decision": final_state.get("trader_investment_plan", ""),
            "risk_debate_state": {
                "aggressive_history": final_state.get("risk_debate_state", {}).get("aggressive_history", ""),
                "conservative_history": final_state.get("risk_debate_state", {}).get("conservative_history", ""),
                "neutral_history": final_state.get("risk_debate_state", {}).get("neutral_history", ""),
                "history": final_state.get("risk_debate_state", {}).get("history", ""),
                "judge_decision": final_state.get("risk_debate_state", {}).get("judge_decision", ""),
            },
            "investment_plan": final_state.get("investment_plan", ""),
            "final_trade_decision": final_state.get("final_trade_decision", ""),
        }

        ss.run_result = result
        ss.run_decision = decision
        ss.progress_step = len(AGENT_STEPS)
        ss.progress_label = "Analysis complete!"
        ss.run_error = None

    except Exception as e:
        ss.run_error = str(e)
        ss.progress_label = f"Error: {e}"


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
for key, default in {
    "run_result": None,
    "run_decision": None,
    "run_error": None,
    "running": False,
    "progress_step": 0,
    "progress_label": "",
    "view_mode": "new",  # "new" or "history"
    "history_result": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📈 TradingAgents")
    st.caption("Multi-Agent LLM Trading Framework")
    st.divider()

    # Mode selector
    mode = st.radio("Mode", ["🚀 New Analysis", "📂 History"], horizontal=True, label_visibility="collapsed")

    if mode == "🚀 New Analysis":
        st.session_state.view_mode = "new"

        st.markdown("### Settings")
        ticker = st.text_input("Stock Ticker", value="NVDA", max_chars=10).upper()

        # Quick pick buttons
        cols = st.columns(4)
        for i, t in enumerate(["NVDA", "AAPL", "TSLA", "MSFT"]):
            if cols[i].button(t, use_container_width=True, key=f"q_{t}"):
                ticker = t

        trade_date = st.date_input(
            "Analysis Date",
            value=date(2024, 5, 10),
            min_value=date(2020, 1, 1),
            max_value=date.today(),
        )

        st.markdown("### Analysts")
        analyst_options = {
            "market": "📊 Market",
            "social": "💬 Social Media",
            "news": "📰 News",
            "fundamentals": "📋 Fundamentals",
            "macro": "🌍 Macro Economy",
        }
        selected_analysts = []
        for key, label in analyst_options.items():
            if st.checkbox(label, value=True, key=f"cb_{key}"):
                selected_analysts.append(key)

        st.divider()

        # Start button
        if st.button(
            "🚀 Start Analysis",
            use_container_width=True,
            type="primary",
            disabled=st.session_state.running or len(selected_analysts) == 0,
        ):
            st.session_state.running = True
            st.session_state.run_result = None
            st.session_state.run_decision = None
            st.session_state.run_error = None
            st.session_state.progress_step = 0
            st.session_state.progress_label = "Initializing..."

            thread = threading.Thread(
                target=run_analysis_thread,
                args=(ticker, str(trade_date), selected_analysts),
                daemon=True,
            )
            thread.start()

    else:
        st.session_state.view_mode = "history"
        history = list_history()
        if history:
            options = [f"{h['ticker']} — {h['date']}" for h in history]
            choice = st.selectbox("Select past analysis", options)
            idx = options.index(choice)
            if st.button("Load", use_container_width=True, type="primary"):
                st.session_state.history_result = load_result(history[idx]["path"])
        else:
            st.info("No history yet. Run an analysis first!")

# ---------------------------------------------------------------------------
# Determine which result to display
# ---------------------------------------------------------------------------
result = None
if st.session_state.view_mode == "history" and st.session_state.history_result:
    result = st.session_state.history_result
elif st.session_state.run_result:
    result = st.session_state.run_result

# ---------------------------------------------------------------------------
# Main area — header
# ---------------------------------------------------------------------------
st.markdown("# 📈 TradingAgents Dashboard")
st.caption("AI Multi-Agent Collaborative Trading Analysis System")
st.divider()

# ---------------------------------------------------------------------------
# Progress bar (when running)
# ---------------------------------------------------------------------------
if st.session_state.running:
    total = len(AGENT_STEPS)
    current = st.session_state.progress_step
    pct = min(current / total, 1.0)

    st.markdown(f"### 🔄 Analysis in Progress")
    st.progress(pct, text=st.session_state.progress_label)

    # Step indicators
    cols = st.columns(min(total, 6))
    for i, step in enumerate(AGENT_STEPS):
        col = cols[i % len(cols)]
        if i < current:
            col.markdown(f"<span class='step-done'>✅ {step}</span>", unsafe_allow_html=True)
        elif i == current:
            col.markdown(f"<span class='step-active'>🔄 {step}</span>", unsafe_allow_html=True)
        else:
            col.markdown(f"<span class='step-wait'>⬜ {step}</span>", unsafe_allow_html=True)

    if st.session_state.run_error:
        st.error(f"Error: {st.session_state.run_error}")
        st.session_state.running = False
    elif current >= total:
        st.session_state.running = False
        st.rerun()
    else:
        # Auto-refresh every 3 seconds
        import time
        time.sleep(3)
        st.rerun()

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------
elif result:
    ticker = result.get("company_of_interest", "N/A")
    trade_date = result.get("trade_date", "N/A")
    final_decision_text = result.get("final_trade_decision", "")
    decision = extract_decision(final_decision_text)

    # ── Overview row ──
    c1, c2, c3 = st.columns([1, 1, 2])

    with c1:
        st.markdown(
            f"""<div class="metric-card">
            <h3>TICKER</h3>
            <p style="font-size:2rem">{ticker}</p>
            </div>""",
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""<div class="metric-card">
            <h3>ANALYSIS DATE</h3>
            <p>{trade_date}</p>
            </div>""",
            unsafe_allow_html=True,
        )

    with c3:
        badge = badge_class(decision)
        st.markdown(
            f"""<div class="metric-card" style="display:flex;align-items:center;justify-content:center;flex-direction:column">
            <h3>FINAL DECISION</h3>
            <div class="decision-badge {badge}" style="margin-top:8px">{decision}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ── Key metrics + allocation ──
    metrics = extract_metrics(result.get("fundamentals_report", ""))
    allocation = extract_allocation(final_decision_text)

    mc1, mc2 = st.columns([1, 1])

    with mc1:
        if metrics:
            st.markdown("#### 📊 Key Metrics")
            metric_cols = st.columns(min(len(metrics), 3))
            for i, (label, val) in enumerate(metrics.items()):
                metric_cols[i % len(metric_cols)].metric(label, val)
        else:
            st.info("Key metrics not available in this analysis.")

    with mc2:
        if allocation:
            st.markdown("#### 🥧 Portfolio Allocation")
            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=list(allocation.keys()),
                        values=list(allocation.values()),
                        hole=0.45,
                        marker=dict(
                            colors=["#00c853", "#2196f3", "#ff9800", "#9c27b0", "#f44336"]
                        ),
                        textinfo="label+percent",
                        textfont_size=14,
                    )
                ]
            )
            fig.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                height=300,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Portfolio allocation not available.")

    st.divider()

    # ── Tabs ──
    tab_analysts, tab_research, tab_trading, tab_risk, tab_workflow = st.tabs(
        ["📈 Analysts", "⚔️ Research Debate", "💼 Trading Plan", "⚖️ Risk Debate", "🔄 Workflow"]
    )

    # ── Tab: Analysts ──
    with tab_analysts:
        analyst_reports = [
            ("📊 Market Analyst", result.get("market_report", "")),
            ("💬 Social Media Analyst", result.get("sentiment_report", "")),
            ("📰 News Analyst", result.get("news_report", "")),
            ("📋 Fundamentals Analyst", result.get("fundamentals_report", "")),
            ("🌍 Macro Analyst", result.get("macro_report", "")),
        ]
        for title, content in analyst_reports:
            if content:
                with st.expander(title, expanded=False):
                    st.markdown(content)

    # ── Tab: Research Debate ──
    with tab_research:
        debate = result.get("investment_debate_state", {})

        col_bull, col_bear = st.columns(2)
        with col_bull:
            st.markdown("### 🐂 Bull Researcher")
            bull_text = debate.get("bull_history", "").strip()
            if bull_text:
                st.markdown(
                    f'<div class="debate-card debate-bull">{bull_text}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.caption("No bull arguments recorded.")

        with col_bear:
            st.markdown("### 🐻 Bear Researcher")
            bear_text = debate.get("bear_history", "").strip()
            if bear_text:
                st.markdown(
                    f'<div class="debate-card debate-bear">{bear_text}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.caption("No bear arguments recorded.")

        judge = debate.get("judge_decision", "").strip()
        if judge:
            st.markdown("### ⚖️ Research Manager Decision")
            st.markdown(
                f'<div class="debate-card debate-judge">{judge}</div>',
                unsafe_allow_html=True,
            )

    # ── Tab: Trading Plan ──
    with tab_trading:
        trader_text = result.get("trader_investment_decision", "") or result.get("investment_plan", "")
        if trader_text:
            st.markdown(trader_text)
        else:
            st.info("Trading plan not available.")

    # ── Tab: Risk Debate ──
    with tab_risk:
        risk = result.get("risk_debate_state", {})

        c_agg, c_neu, c_con = st.columns(3)

        with c_agg:
            st.markdown("### 🔥 Aggressive")
            agg = risk.get("aggressive_history", "").strip()
            if agg:
                st.markdown(agg)
            else:
                st.caption("N/A")

        with c_neu:
            st.markdown("### ⚖️ Neutral")
            neu = risk.get("neutral_history", "").strip()
            if neu:
                st.markdown(neu)
            else:
                st.caption("N/A")

        with c_con:
            st.markdown("### 🛡️ Conservative")
            con = risk.get("conservative_history", "").strip()
            if con:
                st.markdown(con)
            else:
                st.caption("N/A")

    # ── Tab: Workflow ──
    with tab_workflow:
        st.markdown("### Agent Workflow Graph")

        # Show pre-generated PNG if available, otherwise show text diagram
        graph_img = Path("trading_agents_graph.png")
        if graph_img.exists():
            col_wf1, col_wf2, col_wf3 = st.columns([1, 2, 1])
            with col_wf2:
                st.image(str(graph_img), caption="LangGraph Agent Workflow", use_container_width=True)
        else:
            st.markdown("""
**Pipeline Flow:**

```
START → Market Analyst → Social Analyst → News Analyst
     → Fundamentals Analyst → Macro Analyst
     → Bull Researcher ⇄ Bear Researcher → Research Manager
     → Trader → Aggressive ⇄ Conservative ⇄ Neutral
     → Portfolio Manager → FINAL DECISION
```
""")

    # ── Full report expander ──
    with st.expander("📄 Full Portfolio Manager Report"):
        st.markdown(final_decision_text)

# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------
else:
    st.markdown(
        """
    <div style="text-align:center; padding:80px 20px">
        <h2 style="opacity:.5">👈 Configure and start an analysis</h2>
        <p style="opacity:.4">Select a stock ticker and date from the sidebar, then click Start Analysis</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
