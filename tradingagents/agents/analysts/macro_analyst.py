# 文件路径: tradingagents/agents/analysts/macro_analyst.py

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import build_instrument_context, get_language_instruction
from tradingagents.agents.utils.macro_data_tools import get_macro_indicators, get_economic_data
import time

def create_macro_analyst(llm):
    def macro_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [get_macro_indicators, get_economic_data]

        system_message = (
            "You are a Top-Tier Quantitative Macroeconomic Strategist. Your job is to synthesize both "
            "Market Pricing Data (Yields, VIX, Commodities) and Official Economic Data "
            "(CPI, Fed Funds Rate, Unemployment) to determine the macroeconomic regime.\n\n"
            "Use `get_macro_indicators` for real-time market sentiment and `get_economic_data` "
            "for underlying economic reality. Analyze the following:\n"
            "1. Monetary Policy: Is the Fed in a rate hike or cut cycle? How does CPI affect this?\n"
            "2. Yield Curve & Risk: Is the 10Y-2Y curve inverted? Is VIX rising?\n"
            "3. Discrepancies: Is the market pricing in something different from what official data shows?\n\n"
            "Provide a final 'Market Regime' classification and strictly explain how this macro backdrop "
            "impacts the specific target company and its industry.\n"
            + " Make sure to append a Markdown table at the end of the report to organize key points."
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant. Use the provided tools to progress towards answering the question. "
                    "You have access to the following tools: {tool_names}.\n{system_message}\n"
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
            "macro_report": report,
        }

    return macro_analyst_node