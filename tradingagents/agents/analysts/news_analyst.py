from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_global_news,
    get_language_instruction,
    get_news,
)
from tradingagents.dataflows.config import get_config
# 导入 RAG 财报检索工具
from tradingagents.agents.utils.advanced_tools import search_earnings_call

def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_news,
            get_global_news,
            search_earnings_call,
        ]

        system_message = (
            "You are a Deep Research & News Analyst. "
            # 强烈要求模型优先使用 RAG 检索内部讲话，再结合外部新闻
            "First, use the `search_earnings_call` tool (RAG) to retrieve specific context and exact quotes from the company's latest earnings call "
            "(search for keywords like 'Guidance', 'Supply', or 'AI'). "
            "Then, use get_news() and get_global_news() to gather external sentiment. "
            "Combine this internal management perspective with external news to provide a comprehensive qualitative report. "
            "Highlight any discrepancies between what management says and what the news is reporting."
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
            "news_report": report,
        }

    return news_analyst_node
