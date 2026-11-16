"""Architecture B: LangChain Deep Agents (`deepagents`, built on LangGraph).

The main deep agent plays the router: it delegates with its built-in `task`
tool, task(subagent_type=<name>, ...), to the subagents we register (the same
agents/tools as architecture A). Differences from A that you should discuss:
  * routing is implicit (a tool call), may be repeated, or skipped entirely
    (the main agent answers by itself, or uses the built-in 'general-purpose'
    subagent);
  * a much longer built-in system prompt (planning, filesystem tools), so
    more tokens per task.

Routing metric for B: the FIRST `task` call is the route. No `task` call at
all is logged as route "<self>" (counted as a wrong route).

NOTE: deepagents is evolving quickly. This module follows the subagent dict
spec {name, description, system_prompt, tools}. If your installed version
differs, adapt `build_deep_agent` - everything else stays the same.
"""
from __future__ import annotations

import threading

from langchain_core.callbacks import BaseCallbackHandler

from .graph_router import specialist_prompt
from .scoring import content_to_text
from .tools import tools_for

MAIN_PROMPT = """You coordinate a team of specialist subagents. For every user task, delegate
it to the ONE most suitable subagent with the `task` tool and report its answer.
Do not solve tasks yourself and do not use the filesystem or planning tools for these short tasks.
End your reply with a last line of the form:
FINAL ANSWER: <answer only>"""


class TokenCounter(BaseCallbackHandler):
    def __init__(self):
        self.tin = self.tout = self.calls = 0
        self._lock = threading.Lock()

    def on_llm_end(self, response, **kwargs):
        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                u = getattr(msg, "usage_metadata", None) or {}
                with self._lock:
                    self.tin += int(u.get("input_tokens", 0) or 0)
                    self.tout += int(u.get("output_tokens", 0) or 0)
                    self.calls += 1


def build_deep_agent(pool, main_llm, agent_llm):
    from deepagents import create_deep_agent

    subagents = [{
        "name": a.name,
        "description": a.description,
        "system_prompt": specialist_prompt(a),
        "tools": tools_for(a),
        "model": agent_llm,
    } for a in pool]
    return create_deep_agent(model=main_llm, tools=[], system_prompt=MAIN_PROMPT, subagents=subagents)


def run_one_deep(agent, question: str) -> dict:
    counter = TokenCounter()
    out = agent.invoke({"messages": [{"role": "user", "content": question}]},
                       config={"callbacks": [counter], "recursion_limit": 40})
    msgs = out["messages"]
    routes, main_in, main_out, tool_log = [], 0, 0, []
    for m in msgs:
        for tc in getattr(m, "tool_calls", None) or []:
            tool_log.append({"name": tc["name"], "args": {k: str(v)[:200] for k, v in tc["args"].items()},
                             "ok": True})
            if tc["name"] == "task":
                routes.append(str(tc["args"].get("subagent_type", "")))
        u = getattr(m, "usage_metadata", None) or {}
        main_in += int(u.get("input_tokens", 0) or 0)
        main_out += int(u.get("output_tokens", 0) or 0)
    return {
        "routes": routes or ["<self>"],
        "route_raw": "",
        "route_valid": bool(routes),
        "answer_raw": content_to_text(msgs[-1].content),
        "router_tokens_in": main_in,
        "router_tokens_out": main_out,
        "agent_tokens_in": max(counter.tin - main_in, 0),
        "agent_tokens_out": max(counter.tout - main_out, 0),
        "llm_calls": counter.calls,
        "tool_calls": tool_log,
    }
