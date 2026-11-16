"""Model factory: Azure OpenAI / Microsoft Foundry (v1 API) or an offline mock.

Azure / Foundry
---------------
Uses `langchain_openai.ChatOpenAI` against the Azure OpenAI v1 endpoint
(`https://<resource>.openai.azure.com/openai/v1/` or
`https://<resource>.services.ai.azure.com/openai/v1/`). No api_version needed.

Auth: if AZURE_OPENAI_API_KEY is set it is used; otherwise Microsoft Entra ID
via DefaultAzureCredential (works after `az login`, and is what many company
tenants require because key auth is disabled).

The deployment you use must support tool / function calling
(e.g. gpt-4.1-mini, gpt-4o-mini, gpt-4.1, gpt-5-mini ...).

Mock
----
`MockChatModel` lets you test the whole pipeline offline and for free. It
"cheats" using the task's oracle tool call and makes routing mistakes with a
probability that grows with N and with the number of decoy agents. Its numbers
are NOT results - they only show the code works.
"""
from __future__ import annotations

import functools
import os
import random
import threading
import uuid


# --------------------------------------------------------------------------- #
# Azure
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=1)
def _token_provider():
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    return get_bearer_token_provider(DefaultAzureCredential(),
                                     "https://cognitiveservices.azure.com/.default")


def make_azure_llm(deployment: str):
    from langchain_openai import ChatOpenAI

    base_url = os.environ.get("AZURE_OPENAI_BASE_URL", "").strip()
    if not base_url:
        raise RuntimeError("Set AZURE_OPENAI_BASE_URL in .env (see .env.example)")
    if not base_url.rstrip("/").endswith("/openai/v1"):
        base_url = base_url.rstrip("/") + "/openai/v1/"
    key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
    kwargs = dict(model=deployment, base_url=base_url,
                  api_key=key if key else _token_provider(),
                  max_retries=int(os.environ.get("LLM_MAX_RETRIES", "6")),
                  timeout=float(os.environ.get("LLM_TIMEOUT_S", "120")))
    temp = os.environ.get("LLM_TEMPERATURE", "0").strip()
    if temp:  # leave LLM_TEMPERATURE empty for reasoning models that reject it
        kwargs["temperature"] = float(temp)
    return ChatOpenAI(**kwargs)


# --------------------------------------------------------------------------- #
# Mock
# --------------------------------------------------------------------------- #
def _usage(msgs, out_text):
    n_in = sum(len(str(getattr(m, "content", ""))) for m in msgs) // 4 + 10
    n_out = len(out_text) // 4 + 5
    return {"input_tokens": n_in, "output_tokens": n_out, "total_tokens": n_in + n_out}


class MockChatModel:
    """Duck-types the two methods we use: .invoke(messages) and .bind_tools(tools)."""

    def __init__(self, tasks, seed: int = 0, tool_names=()):
        self.by_question = {t.question: t for t in tasks}
        self.pool = []            # set by the runner per condition
        self.tool_names = tuple(tool_names)
        self._rng = random.Random(seed)
        self._lock = threading.Lock()

    def bind_tools(self, tools, **_):
        m = MockChatModel([], tool_names=[getattr(t, "name", str(t)) for t in tools])
        m.by_question, m.pool, m._rng, m._lock = self.by_question, self.pool, self._rng, self._lock
        return m

    def _task(self, msgs):
        from langchain_core.messages import HumanMessage
        for m in msgs:
            if isinstance(m, HumanMessage) and m.content in self.by_question:
                return self.by_question[m.content]
        raise KeyError("mock: task not found in messages")

    def invoke(self, msgs, config=None, **_):
        from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
        task = self._task(msgs)
        sys = msgs[0].content if isinstance(msgs[0], SystemMessage) else ""

        if "You are the ROUTER" in sys:
            names = [a.name for a in self.pool]
            good = [a.name for a in self.pool if task.required_tool in a.tools]
            n_decoy = sum(a.kind == "decoy" for a in self.pool)
            p_wrong = min(0.9, 0.012 * (len(names) - 1) + 0.035 * n_decoy)
            with self._lock:
                pick = self._rng.choice(names) if self._rng.random() < p_wrong else self._rng.choice(good)
            text = '{"agent": "%s"}' % pick
            return AIMessage(content=text, usage_metadata=_usage(msgs, text))

        if isinstance(msgs[-1], ToolMessage):
            text = f"The tool returned the result.\nFINAL ANSWER: {task.gold}"
            return AIMessage(content=text, usage_metadata=_usage(msgs, text))

        if task.required_tool in self.tool_names:
            call = dict(task.oracle_call, id="call_" + uuid.uuid4().hex[:8], type="tool_call")
            return AIMessage(content="", tool_calls=[call], usage_metadata=_usage(msgs, str(call)))

        with self._lock:
            lucky = self._rng.random() < 0.15   # agent without the tool guesses
        text = f"I don't have a tool for this, my best guess.\nFINAL ANSWER: {task.gold if lucky else 'unknown'}"
        return AIMessage(content=text, usage_metadata=_usage(msgs, text))
