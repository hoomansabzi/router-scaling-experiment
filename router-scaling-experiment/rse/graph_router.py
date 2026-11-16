"""Architecture A: explicit router (supervisor) built with LangGraph.

    START --> router --(conditional edge)--> <one specialist> --> END
                    \\--(unparseable / unknown name)--> invalid_route --> END

The router is a single LLM call that must output {"agent": "<name>"}.
Each specialist is a small tool-calling loop (LLM <-> its own tools).
With pool_type == "generalist" the graph is just START --> generalist --> END.

Everything that the analysis needs (route, tokens, LLM calls, tool calls) is
written into the graph state, so a single `graph.invoke` returns a full record.
"""
from __future__ import annotations

import json
import re
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from .scoring import content_to_text
from .tools import tools_for

ROUTER_SYSTEM = """You are the ROUTER of a multi-agent system. You never solve tasks yourself.
Your only job is to pick the ONE agent best able to solve the user's task.

Available agents:
{listing}

Respond with ONLY a JSON object of the form {{"agent": "<agent name>"}} and nothing else."""

SPECIALIST_SYSTEM = """You are {name}. {description}
{tool_line}
Solve the user's task. Be concise.
If the task is outside your abilities, say so and give your best guess.
Always end your reply with a last line of the form:
FINAL ANSWER: <answer only, no units unless asked>"""


class RState(TypedDict, total=False):
    question: str
    agent_order: list
    route: str
    route_raw: str
    route_valid: bool
    router_in: int
    router_out: int
    answer_raw: str
    agent_in: int
    agent_out: int
    agent_llm_calls: int
    tool_calls: list


def _tokens(msg):
    u = getattr(msg, "usage_metadata", None) or {}
    return int(u.get("input_tokens", 0) or 0), int(u.get("output_tokens", 0) or 0)


def parse_route(text: str, names: list[str]):
    """Return (agent_name | None). Tolerant to code fences and extra prose."""
    t = text.strip()
    m = re.search(r"\{.*?\}", t, re.DOTALL)
    if m:
        try:
            cand = str(json.loads(m.group(0)).get("agent", "")).strip()
            if cand in names:
                return cand
        except (json.JSONDecodeError, AttributeError):
            pass
    hits = [n for n in names if re.search(r"\b" + re.escape(n) + r"\b", t)]
    return hits[0] if len(hits) == 1 else None


def specialist_prompt(agent) -> str:
    tool_line = (f"You have these tools: {', '.join(agent.tools)}. Use them; do not compute by hand."
                 if agent.tools else "You have no tools.")
    return SPECIALIST_SYSTEM.format(name=agent.name, description=agent.description, tool_line=tool_line)


def run_tool_agent(llm, agent, question: str, max_steps: int = 6) -> dict:
    """Minimal ReAct-style loop: LLM -> tool calls -> LLM ... -> final text."""
    tools = tools_for(agent)
    by_name = {t.name: t for t in tools}
    model = llm.bind_tools(tools) if tools else llm
    msgs = [SystemMessage(specialist_prompt(agent)), HumanMessage(question)]
    t_in = t_out = calls = 0
    tool_log = []
    text = ""
    for _ in range(max_steps):
        ai = model.invoke(msgs)
        calls += 1
        i, o = _tokens(ai)
        t_in, t_out = t_in + i, t_out + o
        msgs.append(ai)
        text = content_to_text(ai.content)
        tcs = getattr(ai, "tool_calls", None) or []
        if not tcs:
            break
        for tc in tcs:
            tool = by_name.get(tc["name"])
            try:
                out = tool.invoke(tc["args"]) if tool else f"ERROR: unknown tool {tc['name']}"
                ok = tool is not None
            except Exception as e:  # tool errors are fed back to the model
                out, ok = f"ERROR: {e}", False
            tool_log.append({"name": tc["name"], "args": tc["args"], "ok": ok})
            msgs.append(ToolMessage(content=str(out), tool_call_id=tc["id"]))
    else:
        text = text or "MAX_STEPS_REACHED"
    return {"answer_raw": text, "agent_in": t_in, "agent_out": t_out,
            "agent_llm_calls": calls, "tool_calls": tool_log}


def build_router_graph(pool, router_llm, agent_llm):
    g = StateGraph(RState)

    def make_specialist(agent):
        def node(state: RState):
            return run_tool_agent(agent_llm, agent, state["question"])
        return node

    if len(pool) == 1:  # generalist baseline: no router
        g.add_node(pool[0].name, make_specialist(pool[0]))
        g.add_edge(START, pool[0].name)
        g.add_edge(pool[0].name, END)
        return g.compile()

    by_name = {a.name: a for a in pool}

    def router(state: RState):
        order = state.get("agent_order") or list(by_name)
        listing = "\n".join(f"- {n}: {by_name[n].description}" for n in order)
        msgs = [SystemMessage(ROUTER_SYSTEM.format(listing=listing)), HumanMessage(state["question"])]
        ai = router_llm.invoke(msgs)
        i, o = _tokens(ai)
        raw = content_to_text(ai.content)
        name = parse_route(raw, list(by_name))
        return {"route": name or "", "route_raw": raw, "route_valid": name is not None,
                "router_in": i, "router_out": o}

    def invalid(state: RState):
        return {"answer_raw": "ROUTING_FAILED", "agent_in": 0, "agent_out": 0,
                "agent_llm_calls": 0, "tool_calls": []}

    g.add_node("router", router)
    g.add_node("invalid_route", invalid)
    for a in pool:
        g.add_node(a.name, make_specialist(a))
        g.add_edge(a.name, END)
    g.add_edge(START, "router")
    g.add_conditional_edges(
        "router",
        lambda s: s["route"] if s.get("route_valid") else "invalid_route",
        {**{a.name: a.name for a in pool}, "invalid_route": "invalid_route"},
    )
    g.add_edge("invalid_route", END)
    return g.compile()


def run_one(graph, question: str, agent_order: list) -> dict:
    out = graph.invoke({"question": question, "agent_order": agent_order})
    return {
        "routes": [out["route"]] if out.get("route") else [],
        "route_raw": out.get("route_raw", ""),
        "route_valid": out.get("route_valid", True),
        "answer_raw": out.get("answer_raw", ""),
        "router_tokens_in": out.get("router_in", 0),
        "router_tokens_out": out.get("router_out", 0),
        "agent_tokens_in": out.get("agent_in", 0),
        "agent_tokens_out": out.get("agent_out", 0),
        "llm_calls": out.get("agent_llm_calls", 0) + (1 if "router_in" in out else 0),
        "tool_calls": out.get("tool_calls", []),
    }
