from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_indicators,
    get_language_instruction,
    get_stock_data,
)
from tradingagents.dataflows.config import get_config
# [NEW] 导入历史策略回测工具
from tradingagents.agents.utils.advanced_tools import get_rsi_backtest

def create_market_analyst(llm):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        # [NEW] 将回测工具加入可用工具列表
        tools = [
            get_stock_data,
            get_indicators,
            get_rsi_backtest,
        ]

# [NEW] 提示词重构：从“散户看图”升级为“量化概率分析”
        system_message = (
            "You are a Quantitative Technical Analyst tasked with analyzing financial markets. "
            "You may select relevant indicators (like close_50_sma, macd, rsi, boll, etc.) using `get_indicators` to observe current trends, "
            "but you DO NOT just guess future movements based solely on chart patterns.\n\n"
            "[NEW] Roll-forward Validation Focus: You MUST explicitly use the `get_rsi_backtest` tool to check the historical win rate and average return of technical signals over the past 6 months for this specific stock. "
            "Ground your final analysis STRICTLY in these backtested probabilities. "
            "If the backtest shows a win rate below 50%, warn the team that the current technical signals are historically unreliable and likely overfitted. "
            "Provide specific, actionable insights combining both current indicator readings and their historical statistical reliability."
            + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)

        time.sleep(4)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content


        return {
            "messages": [result],
            "market_report": report,
        }

    return market_analyst_node
