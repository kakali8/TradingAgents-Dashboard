# 文件路径: tradingagents/agents/utils/macro_data_tools.py
import os
from langchain_core.tools import tool
from typing import Annotated
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta

@tool
def get_macro_indicators(
    curr_date: Annotated[str, "Current date you are trading at, yyyy-mm-dd"],
) -> str:
    """
    Retrieve market-priced macroeconomic indicators including Yields, VIX, USD,
    Commodities (Gold, Oil, Copper), and Credit Spreads to assess the global market regime.
    """
    try:
        tickers = {
            "^TNX": "10-Year Treasury Yield",
            "^VIX": "VIX Volatility Index",
            "DX-Y.NYB": "US Dollar Index",
            "GLD": "Gold ETF",
            "USO": "Crude Oil ETF",
            "CPER": "Copper ETF",
            "HYG": "High Yield Corporate Bond ETF"
        }
        end_date = pd.to_datetime(curr_date)
        start_date = end_date - pd.Timedelta(days=14)
        report = "### Market Pricing Indicators (via yfinance)\n\n"
        for ticker, name in tickers.items():
            data = yf.download(
                ticker,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False
            )
            if not data.empty and len(data) >= 2:
                latest_close = data['Close'].iloc[-1].item()
                prev_close = data['Close'].iloc[0].item()
                pct_change = ((latest_close - prev_close) / prev_close) * 100
                trend = "UP" if latest_close > prev_close else "DOWN"
                report += f"- **{name}**: Latest = {latest_close:.2f} (14-day trend: {trend} by {pct_change:.2f}%)\n"
            else:
                report += f"- **{name}**: Data unavailable.\n"
        return report
    except Exception as e:
        return f"Error fetching macro indicators: {str(e)}"


@tool
def get_economic_data(
    curr_date: Annotated[str, "Current date you are trading at, yyyy-mm-dd"],
) -> str:
    """
    Retrieve official macroeconomic data from FRED including CPI, Unemployment Rate,
    Fed Funds Rate, and Yield Curve spreads.
    """
    try:
        series_dict = {
            'CPIAUCSL': 'CPI (Inflation)',
            'UNRATE': 'Unemployment Rate',
            'FEDFUNDS': 'Federal Funds Effective Rate',
            'T10Y2Y': '10Y-2Y Treasury Yield Spread'
        }
        end = pd.to_datetime(curr_date)
        start = end - timedelta(days=180)
        report = "### Official Economic Indicators (via FRED)\n\n"

        for series_id, name in series_dict.items():
            url = (
                f"https://api.stlouisfed.org/fred/series/observations"
                f"?series_id={series_id}"
                f"&observation_start={start.strftime('%Y-%m-%d')}"
                f"&observation_end={end.strftime('%Y-%m-%d')}"
                f"&api_key={os.environ.get('FRED_API_KEY', '24dde6b9ecf9fb5d50b7f5291eb9e4f3')}"
                f"&file_type=json"
            )
            resp = requests.get(url, timeout=10)
            observations = resp.json().get("observations", [])
            # 过滤掉缺失值
            clean = [
                float(o["value"])
                for o in observations
                if o["value"] != "."
            ]
            if len(clean) >= 2:
                latest_val = clean[-1]
                prev_val = clean[-2]
                trend = (
                    "Rising" if latest_val > prev_val
                    else "Falling" if latest_val < prev_val
                    else "Unchanged"
                )
                report += (
                    f"- **{name}**: Latest = {latest_val:.2f} "
                    f"(Previous: {prev_val:.2f}, Trend: {trend})\n"
                )
            else:
                report += f"- **{name}**: Insufficient data.\n"

        return report
    except Exception as e:
        return f"Error fetching FRED data: {str(e)}"