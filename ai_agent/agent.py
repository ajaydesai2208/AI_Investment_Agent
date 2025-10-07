from __future__ import annotations

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.yfinance import YFinanceTools
import inspect


__all__ = ["build_agent"]


def build_agent(api_key: str) -> Agent:
    """
    Construct the analysis agent:
    - Uses OpenAI GPT-4o (change the model id here if you want).
    - Includes YFinanceTools for on-demand fundamentals/price lookups.
    - Prints tool calls in Streamlit so you can see its chain of tools.
    - Instructions: decisive, trade-oriented HF PM; minimal fluff.
    """
    model = OpenAIChat(id="gpt-4o", api_key=api_key)
    tools = [YFinanceTools()]
    instructions = (
        "Act as a hedge fund portfolio manager. "
        "Be decisive and trade-oriented. Synthesize fundamentals, momentum, valuation, sentiment and news. "
        "Propose concrete trades with entry zones, stops, targets, and example options structures when suitable. "
        "Explain the thesis, catalysts and risks succinctly. Avoid filler."
    )

    # Backward/forward compatible construction: older agno.Agent may not accept
    # 'show_tool_calls'. Prefer enabling it when supported; otherwise, fall back
    # silently to avoid runtime errors in constrained environments (e.g., Cloud Run).
    try:
        sig = inspect.signature(Agent)
        if "show_tool_calls" in sig.parameters:
            return Agent(model=model, tools=tools, instructions=instructions, show_tool_calls=True)
        # Some versions expose it under 'stream' or not at all — fall back to base kwargs
        return Agent(model=model, tools=tools, instructions=instructions)
    except Exception:
        # As a final safeguard, call without optional flags
        return Agent(model=model, tools=tools, instructions=instructions)
