# 文件路径: tradingagents/agents/utils/advanced_tools.py
from langchain_core.tools import tool
from typing import Annotated
import yfinance as yf
import pandas as pd
import numpy as np

# 1. 基本面分析师：同行对比工具 (Multi-Asset)
@tool
def get_peer_comparison(ticker: Annotated[str, "Ticker symbol"]) -> str:
    """Retrieve fundamental metrics of the target company and its top competitors for valuation comparison."""
    try:
        # 预设一些常见股票的竞争对手
        peers_map = {
            "NVDA": ["AMD", "INTC"], "AAPL": ["MSFT", "GOOGL"], "MSFT": ["AAPL", "GOOGL"],
            "TSLA": ["F", "GM", "RIVN"], "SPY": ["QQQ", "IWM"]
        }
        peers = peers_map.get(ticker.upper(), ["SPY"]) # 如果没找到，就跟大盘对比
        tickers_to_compare = [ticker.upper()] + peers
        
        report = "### Peer Comparison (Valuation & Profitability)\n\n"
        for t in tickers_to_compare:
            info = yf.Ticker(t).info
            pe = info.get("trailingPE", "N/A")
            margin = info.get("grossMargins", 0) * 100 if info.get("grossMargins") else "N/A"
            roe = info.get("returnOnEquity", 0) * 100 if info.get("returnOnEquity") else "N/A"
            report += f"- **{t}**: PE Ratio: {pe}, Gross Margin: {margin}%, ROE: {roe}%\n"
        return report
    except Exception as e:
        return f"Error fetching peer data: {e}"

# 2. 市场分析师：滚动回测工具 (Roll-forward Validation)
@tool
def get_rsi_backtest(ticker: Annotated[str, "Ticker symbol"]) -> str:
    """Run a 6-month historical backtest on RSI strategy to determine win rate and average return."""
    try:
        data = yf.download(ticker, period="6mo", progress=False)
        if data.empty: return "Not enough data for backtest."
        
        # 计算简单的 RSI (14天)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # 回测逻辑：如果 RSI < 30 (超卖)，买入并持有 5 天
        buy_signals = data[data['RSI'] < 30]
        wins, total_return = 0, 0
        
        for date in buy_signals.index:
            idx = data.index.get_loc(date)
            if idx + 5 < len(data):
                buy_price = data['Close'].iloc[idx].item()
                sell_price = data['Close'].iloc[idx + 5].item()
                ret = (sell_price - buy_price) / buy_price
                if ret > 0: wins += 1
                total_return += ret
                
        total_trades = len(buy_signals)
        if total_trades == 0: return "No RSI < 30 buy signals in the past 6 months."
        
        win_rate = (wins / total_trades) * 100
        avg_return = (total_return / total_trades) * 100
        return f"### 6-Month RSI Strategy Backtest for {ticker}\n- Total Trades Triggered: {total_trades}\n- Win Rate: {win_rate:.2f}%\n- Avg 5-Day Return: {avg_return:.2f}%"
    except Exception as e:
        return f"Backtest failed: {e}"

# 3. 社媒分析师：Reddit散户情绪代理工具 (Alternative Data)
@tool
def get_retail_sentiment(ticker: Annotated[str, "Ticker symbol"]) -> str:
    """Analyze retail sentiment and FOMO levels based on recent abnormal trading volume (Proxy for Reddit/WSB interest)."""
    try:
        data = yf.download(ticker, period="1mo", progress=False)
        avg_vol = data['Volume'].mean().item()
        latest_vol = data['Volume'].iloc[-1].item()
        
        vol_spike = latest_vol / avg_vol
        if vol_spike > 1.5:
            sentiment = "HIGH FOMO / Short Squeeze Risk (Retail interest is surging)"
        elif vol_spike < 0.8:
            sentiment = "Low retail interest (Quiet)"
        else:
            sentiment = "Normal retail activity"
            
        return f"### Retail Sentiment Analysis\n- Volume Spike Ratio: {vol_spike:.2f}x\n- Current Sentiment: {sentiment}"
    except Exception as e:
        return f"Error fetching retail data: {e}"

# 4. 新闻分析师/基本面分析师：RAG 财报与动态检索工具 (使用真实数据源替代硬编码)
@tool
def search_earnings_call(ticker: Annotated[str, "Ticker symbol"], query: Annotated[str, "Keyword to search"]) -> str:
    """RAG Tool: Search the company's latest news, earnings commentary, and real-time updates for specific keywords."""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news:
            return f"No recent context found for {ticker} regarding '{query}'."

        # yfinance news structure: title/provider are nested under 'content'
        def extract(item):
            content = item.get('content', item)
            title = content.get('title', '')
            provider = content.get('provider', {})
            publisher = provider.get('displayName', '') if isinstance(provider, dict) else str(provider)
            summary = content.get('summary', '')
            return title, publisher, summary

        relevant_context = []
        for item in news:
            title, publisher, summary = extract(item)
            text = f"{title} {summary}"
            if query.lower() in text.lower():
                relevant_context.append(f"[{publisher}] {title}")

        if relevant_context:
            context_str = "\n".join(relevant_context[:5])
            return f"### Real-time Context Retrieval for {ticker}\n- Found matches for '{query}':\n{context_str}"
        else:
            general_context = "\n".join([f"[{extract(n)[1]}] {extract(n)[0]}" for n in news[:5]])
            return f"### Real-time Context Retrieval for {ticker}\n- No exact match for '{query}'. Here are the latest real updates:\n{general_context}"
    except Exception as e:
        return f"RAG search failed for {ticker}: {e}"