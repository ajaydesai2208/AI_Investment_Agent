from __future__ import annotations

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.yfinance import YFinanceTools


__all__ = ["build_agent"]


def build_agent(api_key: str) -> Agent:
    """
    Construct the analysis agent:
    - Uses OpenAI GPT-4o (change the model id here if you want).
    - Includes YFinanceTools for on-demand fundamentals/price lookups.
    - Prints tool calls in Streamlit so you can see its chain of tools.
    - Instructions: decisive, trade-oriented HF PM; minimal fluff.
    """
    return Agent(
        model=OpenAIChat(id="gpt-4o", api_key=api_key),
        tools=[YFinanceTools()],
        show_tool_calls=True,
        instructions=(
            "Act as a hedge fund portfolio manager. "
            "Be decisive and trade-oriented. Synthesize fundamentals, momentum, valuation, sentiment and news. "
            "Propose concrete trades with entry zones, stops, targets, and example options structures when suitable. "
            "Explain the thesis, catalysts and risks succinctly. Avoid filler."
        ),
    )
