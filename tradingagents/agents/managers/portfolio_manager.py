from langchain_core.messages import AIMessage
from tradingagents.agents.utils.agent_utils import build_instrument_context, get_language_instruction
import re
import time  #[NEW] 导入 time 模块

def create_portfolio_manager(llm, memory):
    def portfolio_manager_node(state) -> dict:

        instrument_context = build_instrument_context(state["company_of_interest"])

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        #
        macro_report = state.get("macro_report", "Macro report unavailable.")
        market_research_report = state["market_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        sentiment_report = state["sentiment_report"]
        trader_plan = state["investment_plan"]

        curr_situation = f"{macro_report}\n\n{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        # 🚀 [NEW] 提示词全面重构：从“单票评级”升级为“多资产组合分配”
        prompt = f"""As the Chief Multi-Asset Portfolio Manager, your mandate is to synthesize the risk analysts' debate and deliver a final, mathematically sound Asset Allocation strategy. 

You do NOT just trade a single stock in a vacuum. You must construct a balanced portfolio based on the macro environment, peer comparison, and risk parameters.

{instrument_context}

---

[NEW] **Multi-Asset Allocation Mandate**:
You must allocate 100% of the theoretical portfolio capital. You have three asset classes to choose from:
1. **Target Asset**: The primary company of interest.
2. **Peer/Competitor Assets**: Based on the 'Peer Comparison' data found in the Fundamentals Report (e.g., allocating to AMD if NVDA's risk is too high).
3. **Cash / Risk-Free**: If the Macro Report or Risk Analysts warn of high volatility, overfitting, or severe market risk, you MUST allocate a significant portion to Cash.

**Context:**
- Trader's proposed plan: **{trader_plan}**
- Lessons from past decisions: **{past_memory_str}**

**Required Output Structure:**
1. **Portfolio Allocation (Markdown Table)**: Explicitly state the percentage allocation across the Target Asset, Peer Assets, and Cash. The total MUST equal 100%. 
   *(Example: 40% NVDA, 20% AMD, 40% Cash)*
2. **Quantitative Justification**: Explain the allocation using the mathematical probability (Roll-forward win rate) from the Market Analyst and the margin comparisons from the Fundamentals Analyst.
3. **Risk Management Overlay**: Explain how the debate from the Risk Analysts (Aggressive vs. Conservative) influenced your Cash position.

---

**Risk Analysts Debate History:**
{history}

---

Be decisive and ground every conclusion in specific evidence from the analysts.{get_language_instruction()}"""

        response = llm.invoke(prompt)

        new_risk_debate_state = {
            "judge_decision": response.content,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": response.content,
            # [NEW] 将经理的最终裁决推送到公共频道
            "messages": [AIMessage(content=f"Portfolio Manager: {response.content}", name="Portfolio_Manager")]
        }

    return portfolio_manager_node
