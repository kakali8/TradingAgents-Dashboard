import functools
import time
import json
import re  # 🚀 [NEW] 导入正则表达式库，用于物理抹除幻觉标签

from langchain_core.messages import AIMessage # 🚀 [NEW] 重新包装清洗后的消息
from tradingagents.agents.utils.agent_utils import build_instrument_context


def create_trader(llm, memory):
    def trader_node(state, name):
        company_name = state["company_of_interest"]
        instrument_context = build_instrument_context(company_name)
        investment_plan = state["investment_plan"]
        
        # 🚀 [NEW] 顺手升级为防御性编程，防止前面取消勾选分析师导致 KeyError 崩溃
        macro_report = state.get("macro_report", "Macro report unavailable.")
        market_research_report = state.get("market_report", "Market report unavailable.")
        sentiment_report = state.get("sentiment_report", "Sentiment report unavailable.")
        news_report = state.get("news_report", "News report unavailable.")
        fundamentals_report = state.get("fundamentals_report", "Fundamentals report unavailable.")

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        if past_memories:
            for i, rec in enumerate(past_memories, 1):
                past_memory_str += rec["recommendation"] + "\n\n"
        else:
            past_memory_str = "No past memories found."

        context = {
            "role": "user",
            "content": f"Based on a comprehensive analysis by a team of analysts, here is an investment plan tailored for {company_name}. {instrument_context} This plan incorporates insights from current technical market trends, macroeconomic indicators, and social media sentiment. Use this plan as a foundation for evaluating your next trading decision.\n\nProposed Investment Plan: {investment_plan}\n\nLeverage these insights to make an informed and strategic decision.",
        }

        # 🚀 [NEW] 第一重保险：在系统提示词末尾加上极其严厉的防幻觉指令
        messages = [
            {
                "role": "system",
                "content": f"""You are a trading agent analyzing market data to make investment decisions. Based on your analysis, provide a specific recommendation to buy, sell, or hold. End with a firm decision and always conclude your response with 'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**' to confirm your recommendation. Apply lessons from past decisions to strengthen your analysis. Here are reflections from similar situations you traded in and the lessons learned: {past_memory_str}

🚨 CRITICAL RULE (ANTI-HALLUCINATION): 
1. ALL required market data, including current prices, is ALREADY provided in the context.
2. You DO NOT have access to any external tools, APIs, or internet search.
3. You are STRICTLY FORBIDDEN from outputting any XML tags, pseudo-code, or tool-calling attempts (e.g., <get_current_price>, <get_market_data>).
4. Output plain conversational text ONLY. Do NOT attempt to fetch data."""
            },
            context,
        ]

        result = llm.invoke(messages)

        # 🚀 [NEW] 第二重保险：正则表达式物理清洗
        # 强行删掉大模型可能生成的形如 <XXX>...</XXX> 或 <XXX/> 的标签
        cleaned_content = re.sub(r'<[^>]+>.*?</[^>]+>', '', result.content, flags=re.DOTALL)
        cleaned_content = re.sub(r'<[^>]+/>', '', cleaned_content)
        cleaned_content = cleaned_content.strip()

        # 🚀 [NEW] 用清洗干净的内容重新构建一条标准的消息对象
        cleaned_msg = AIMessage(content=cleaned_content, name=name)

        return {
            "messages": [cleaned_msg],
            "trader_investment_plan": cleaned_content,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")