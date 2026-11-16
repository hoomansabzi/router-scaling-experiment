"""LangChain tool wrappers around tools_impl."""
from __future__ import annotations

from langchain_core.tools import tool

from .tools_impl import IMPLS

TOOLS = {name: tool(fn) for name, fn in IMPLS.items()}


def tools_for(agent):
    return [TOOLS[t] for t in agent.tools]
