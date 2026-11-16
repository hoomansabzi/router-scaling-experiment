# Router-scaling experiment
### How does a router-based multi-agent system behave as you add more agents?

A small, fully automatic benchmark. A router (supervisor) agent sends each task to one
of N specialist agents. We change **how many** agents there are and **what kind**, then
measure routing accuracy, task accuracy, cost and latency. Two architectures are included:

* **A. LangGraph router** (`--arch langgraph`, default): an explicit router node that
  returns `{"agent": name}`, with a conditional edge to that specialist.
* **B. Deep Agents** (`--arch deep`): LangChain `deepagents`. The main agent delegates
  through its built-in `task` tool to the same specialists, registered as subagents.

The models run on **Azure OpenAI / Microsoft Foundry**.

---

## 1. Design

### Tasks (the controlled part)
There are 5 categories. Each task needs exactly one tool, and only the right specialist
has that tool. A wrong route therefore usually means a wrong answer, so routing quality
directly affects results.

| category | example | tool needed | why an LLM needs the tool |
|---|---|---|---|
| math | compound interest over 26 years, rounded to 2 decimals | `calculator` | exact multi-digit arithmetic |
| units | 876.6 furlongs → km | `convert_units` | uncommon conversion factors |
| records | "Who leads Project Halvard at Veltrano Industries?" | `lookup_record` | **fictional** company, so the fact is in no training data |
| dates | "What weekday is 250 days after 2026-11-02?" | `date_calculator` | calendar arithmetic |
| text | reverse / count letters / sort words | `text_tool` | models are unreliable at character-level operations |

Tasks are generated from a seed (`rse/tasks.py`) and graded automatically
(`rse/scoring.py`). Every agent ends its reply with `FINAL ANSWER: ...`.

### Independent variable: the agent pool (`rse/agents.py`)

| pool type | N | what it tests |
|---|---|---|
| `generalist` | 1 | baseline: one agent holds all 5 tools, no router |
| `granularity` | 2, 3, 5 | the same 5 tools split across more, narrower specialists. Coverage is always complete. |
| `irrelevant` | 5 + k → 8, 12, 16 | adds agents from unrelated domains (translator, poet, SQL, …) |
| `decoy` | 5 + k → 8, 12, 16 | adds agents whose **descriptions overlap** the task domains (finance analyst, calendar assistant, HR assistant, linguist, …) but who lack the tools |

N=5 (the 5 core specialists) is the shared starting point for all three curves.

### Controls
* Temperature 0 (`LLM_TEMPERATURE`) and a fixed seed for the tasks.
* The router sees the agent list **in a random order for every task**. The record
  stores where the correct agent appeared in the list (`gold_position`), so you can
  measure position bias (`position_bias.csv`).
* `--repeats R` runs each condition R times to estimate run-to-run variance. The plots
  show 95% Wilson confidence intervals.
* Router and specialists can use different deployments (`ROUTER_DEPLOYMENT`,
  `AGENT_DEPLOYMENT`). That gives a second axis to vary, e.g. a small versus a large router.

### Metrics, one row per (condition, repeat, task) in `records.jsonl`
* **routing accuracy**: the chosen agent owns the required tool
* **task accuracy**: the final answer is correct
* `acc_if_routed_right` / `acc_if_routed_wrong`: separates *routing errors* from
  *execution errors*. This split is the key to interpreting the curves.
* invalid-route rate (the router output could not be parsed or named an unknown agent)
* tokens (router vs specialists), LLM calls, tool calls, latency, errors

### Hypotheses to test
* **H1:** Splitting tools into narrower agents (granularity) costs little routing
  accuracy when roles are clearly separated, but adds one router call per task in
  tokens and latency compared with the generalist.
* **H2:** Irrelevant agents barely hurt routing; cost grows roughly linearly with N
  because the router prompt gets longer.
* **H3:** Decoy agents hurt much more than irrelevant agents of the same N. What matters
  is overlap between role descriptions, not N on its own.
* **H4:** The router is biased towards agents listed earlier, more so at larger N.

---

## 2. Setup on the company laptop

Requires Python 3.10 or newer.

```bash
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1      macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-deep.txt      # only for --arch deep
copy .env.example .env                    # macOS/Linux: cp .env.example .env
```

### Azure / Microsoft Foundry
1. In the Foundry portal, deploy a chat model that **supports tool calling**, e.g.
   `gpt-4.1-mini` or `gpt-4o-mini`. Small models are cheaper and show the effects more
   clearly.
2. From the deployment page, copy the endpoint into `.env` as `AZURE_OPENAI_BASE_URL`.
   It ends in `/openai/v1/` and looks like
   `https://<resource>.openai.azure.com/openai/v1/` or
   `https://<resource>.services.ai.azure.com/openai/v1/`.
3. Authentication: either paste the key into `AZURE_OPENAI_API_KEY`, or leave it empty
   and run `az login`. Empty means Microsoft Entra ID, which needs the
   *Cognitive Services OpenAI User* or *Azure AI User* role on the resource. Many
   company tenants disable keys, so Entra is often the only option.
4. Put your deployment name(s) in `ROUTER_DEPLOYMENT` / `AGENT_DEPLOYMENT`.

## 3. Running

```bash
# a) free offline test of the plumbing (fake LLM; the numbers mean nothing)
python run_experiment.py --backend mock --out results/mock
python analyze.py results/mock

# b) tiny live check: 2 conditions, 1 task per category
python run_experiment.py --backend azure --conditions generalist:1,decoy:8 --n-per-category 1 --out results/check

# c) pilot: full sweep, 5 tasks/category (10 conditions x 25 tasks = 250 runs)
python run_experiment.py --backend azure --n-per-category 5 --out results/pilot

# d) main run
python run_experiment.py --backend azure --n-per-category 20 --repeats 3 --out results/main
python analyze.py results/main

# e) same sweep with Deep Agents, then compare both architectures on one plot
python run_experiment.py --backend azure --arch deep --n-per-category 20 --out results/deep
python analyze.py results/main results/deep
```

* Runs **resume** automatically. If something crashes or hits rate limits, rerun the
  same command with the same `--out`.
* `--concurrency` (default 4, or `MAX_CONCURRENCY` in `.env`) sets parallel requests.
  Lower it if you get 429 errors.
* Set `RSE_DEBUG=1` to print full tracebacks of failed runs.
* Cost: a run uses about 1–3k tokens with the LangGraph router, and several times more
  with Deep Agents because of its long built-in prompt. The main run (d) is 3,000 runs.
  Check the pilot's `tokens_per_task` before scaling up.

## 4. Outputs (in the results folder)
`summary.csv`, `per_category.csv`, `position_bias.csv`, `fig_accuracy_vs_N.png`,
`fig_routing_vs_N.png`, `fig_tokens_vs_N.png`, `fig_latency_vs_N.png`, and
`fig_confusion_*` (where each task category was routed: the correct agents are the
first 5 columns).

## 5. Code map
```
rse/tools_impl.py   5 tools in plain Python (also used to compute gold answers)
rse/kb.py           fictional Veltrano Industries project database
rse/tasks.py        task generator + gold answers
rse/agents.py       agent definitions and pools (edit here to add agents/conditions)
rse/graph_router.py architecture A: LangGraph router graph + specialist tool loop
rse/graph_deep.py   architecture B: deepagents main agent + subagents
rse/llm.py          Azure/Foundry model factory + offline mock
rse/scoring.py      answer extraction and grading
run_experiment.py   the sweep
analyze.py          tables + figures
```

## 6. Possible extensions
* **Redundant agents:** two agents that both own the calculator. Does the router split
  tasks between them, and does that hurt anything?
* **Description quality:** shorten or blur the core agents' descriptions while keeping N fixed.
* **Router model size:** small vs large `ROUTER_DEPLOYMENT` at fixed specialists.
* **Hierarchical routing:** at N=16, compare a two-level router (domain, then agent)
  with a flat one.
* **Multi-step tasks** that need 2 agents in sequence. This is where Deep Agents'
  planning should help.

## Caveats
* The LangGraph path and the analysis were tested end to end only with the offline mock
  backend. Run step (b) before anything big.
* `deepagents` changes quickly. `rse/graph_deep.py` uses the subagent spec
  `{name, description, system_prompt, tools, model}` and reads routes from the `task`
  tool's `subagent_type` argument. If your installed version differs, that file is the
  only one to adapt.
