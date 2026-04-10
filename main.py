import sys
import io
from datetime import datetime
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Create a custom config
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "openrouter"
config["deep_think_llm"] = "openai/gpt-4o-mini"
config["quick_think_llm"] = "openai/gpt-4o-mini"
config["max_debate_rounds"] = 1  # Increase debate rounds

# Configure data vendors (default uses yfinance, no extra API keys needed)
config["data_vendors"] = {
    "core_stock_apis": "yfinance",           # Options: alpha_vantage, yfinance
    "technical_indicators": "yfinance",      # Options: alpha_vantage, yfinance
    "fundamental_data": "yfinance",          # Options: alpha_vantage, yfinance
    "news_data": "yfinance",                 # Options: alpha_vantage, yfinance
}

# Capture all output to both console and buffer
class TeeOutput:
    def __init__(self, original):
        self.original = original
        self.buffer = io.StringIO()
    def write(self, text):
        self.original.write(text)
        self.buffer.write(text)
    def flush(self):
        self.original.flush()

tee_stdout = TeeOutput(sys.stdout)
tee_stderr = TeeOutput(sys.stderr)
sys.stdout = tee_stdout
sys.stderr = tee_stderr

# Initialize with custom config
ta = TradingAgentsGraph(debug=True, config=config)

# forward propagate
_, decision = ta.propagate("NVDA", "2024-05-10")
print(decision)

# Restore original stdout/stderr
sys.stdout = tee_stdout.original
sys.stderr = tee_stderr.original

# Write captured output to markdown
output_file = f"run_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"# TradingAgents Run Output\n\n")
    f.write(f"- **Ticker**: NVDA\n")
    f.write(f"- **Trade Date**: 2024-05-10\n")
    f.write(f"- **Model**: openai/gpt-4o-mini (via OpenRouter)\n")
    f.write(f"- **Run Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"---\n\n")
    f.write("## Console Output\n\n```\n")
    f.write(tee_stdout.buffer.getvalue())
    f.write(tee_stderr.buffer.getvalue())
    f.write("\n```\n\n")
    f.write(f"---\n\n## Final Decision\n\n{decision}\n")

print(f"\nOutput saved to: {output_file}")

# Memorize mistakes and reflect
# ta.reflect_and_remember(1000) # parameter is the position returns
