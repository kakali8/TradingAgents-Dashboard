# TradingAgents Dashboard Usage Guide

## Access

Open the link below in any browser (Chrome / Safari / Edge):

**https://tradingagents-dashboard-2ofljywwkf3jpsnnhhlhwv.streamlit.app/**

No installation required. No API key needed. Just open and use.

---

## Step-by-Step Guide

### Step 1: Choose Mode

On the left sidebar, you'll see two modes:

- **New Analysis** — Run a fresh AI analysis on any stock
- **History** — View previously completed analyses

### Step 2: Configure Analysis (New Analysis mode)

1. **Stock Ticker** — Type a ticker symbol (e.g., `NVDA`, `AAPL`, `TSLA`, `MSFT`), or click one of the quick-pick buttons
2. **Analysis Date** — Select the date you want to analyze (historical date)
3. **Analysts** — Check which analysts to include:
   - Market Analyst (technical indicators, RSI backtest)
   - Social Media Analyst (retail sentiment)
   - News Analyst (news & earnings search)
   - Fundamentals Analyst (financials, peer comparison)
   - Macro Analyst (CPI, unemployment, Fed rate, yield curve)

### Step 3: Start Analysis

Click the **Start Analysis** button.

The system will run through 13 AI agents sequentially. A progress bar shows real-time status:

```
━━━━━━━━━━━━━━━━━━░░░░░  70%
News Analyst completed
```

**Note:** A full analysis takes approximately 5-8 minutes. Please be patient.

### Step 4: View Results

Once complete, the dashboard shows:

#### Overview Section
- **Final Decision** — Color-coded badge: BUY (green) / HOLD (orange) / SELL (red)
- **Key Metrics** — PE Ratio, ROE, Gross Margin, Profit Margin
- **Portfolio Allocation** — Recommended asset allocation with percentage bars

#### 5 Detail Tabs

| Tab | What It Shows |
|-----|---------------|
| **Analysts** | Expandable reports from each analyst (market, social, news, fundamentals, macro) |
| **Research Debate** | Bull vs Bear researcher arguments + Research Manager's verdict |
| **Trading Plan** | The Trader agent's investment strategy |
| **Risk Debate** | Three-way debate: Aggressive / Neutral / Conservative risk perspectives |
| **Workflow** | Visual diagram of the AI agent pipeline |

### Step 5: View History (Optional)

Switch to **History** mode in the sidebar to browse past analyses. Select a ticker + date combination and click **Load**.

---

## How It Works

The system uses 13 specialized AI agents that collaborate to make a trading decision:

```
Market Analyst ──> Social Analyst ──> News Analyst
     ──> Fundamentals Analyst ──> Macro Analyst
     ──> Bull Researcher <──> Bear Researcher
     ──> Research Manager (Judge)
     ──> Trader (creates investment plan)
     ──> Aggressive <──> Conservative <──> Neutral (risk debate)
     ──> Portfolio Manager (final decision)
```

Each agent analyzes real market data (via Yahoo Finance and FRED) and uses AI (GPT-4o-mini via OpenRouter) to generate insights. The agents debate and challenge each other before reaching a final consensus.

---

## FAQ

**Q: Is there a cost to use this?**
A: No, it's free for all users.

**Q: Can I trust the trading decisions?**
A: This is an academic research project for IE5604. The results are for educational purposes only and should NOT be used as actual financial advice.

**Q: Why does the analysis take so long?**
A: The system runs 13 AI agents sequentially, each making API calls and analyzing data. The total processing time is typically 5-8 minutes.

**Q: Can I analyze any stock?**
A: Yes, any US-listed stock with a valid ticker symbol (e.g., NVDA, AAPL, GOOGL, AMZN, TSLA, META, etc.)
