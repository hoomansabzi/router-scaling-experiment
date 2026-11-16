#!/usr/bin/env python
"""Run the router-scaling sweep and write one JSON record per (condition, repeat, task).

Examples
--------
  # 1) free offline smoke test (no Azure calls)
  python run_experiment.py --backend mock --out results/mock

  # 2) small real pilot on Azure (~5 tasks/category, all conditions)
  python run_experiment.py --backend azure --n-per-category 5 --out results/pilot

  # 3) full run
  python run_experiment.py --backend azure --n-per-category 20 --repeats 3 --out results/full

  # 4) same sweep with the Deep Agents architecture
  python run_experiment.py --backend azure --arch deep --n-per-category 10 --out results/deep

Re-running with the same --out resumes: finished records are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rse.agents import DEFAULT_CONDITIONS, build_pool, owners
from rse.scoring import extract_final, is_correct
from rse.tasks import generate_tasks, save_tasks


def parse_conditions(s: str | None):
    if not s:
        return DEFAULT_CONDITIONS
    out = []
    for item in s.split(","):
        p, n = item.strip().split(":")
        out.append((p, int(n)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["azure", "mock"], default="mock")
    ap.add_argument("--arch", choices=["langgraph", "deep"], default="langgraph")
    ap.add_argument("--conditions", default=None,
                    help="comma list pool:N, e.g. 'generalist:1,granularity:5,decoy:16' (default: full sweep)")
    ap.add_argument("--n-per-category", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=int(os.environ.get("MAX_CONCURRENCY", "4")))
    ap.add_argument("--out", default="results/run")
    args = ap.parse_args(argv)

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = generate_tasks(args.n_per_category, args.seed)
    save_tasks(tasks, out / "tasks.jsonl")
    conditions = parse_conditions(args.conditions)

    router_dep = os.environ.get("ROUTER_DEPLOYMENT", "mock")
    agent_dep = os.environ.get("AGENT_DEPLOYMENT", router_dep)
    meta = {"args": vars(args), "router_deployment": router_dep, "agent_deployment": agent_dep,
            "conditions": conditions, "n_tasks": len(tasks), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "config.json").write_text(json.dumps(meta, indent=2))

    # ---- models ------------------------------------------------------------
    if args.backend == "mock":
        if args.arch == "deep":
            sys.exit("The mock backend only supports --arch langgraph.")
        from rse.llm import MockChatModel
        mock = MockChatModel(tasks, seed=args.seed)
        router_llm = agent_llm = mock
    else:
        from rse.llm import make_azure_llm
        router_llm = make_azure_llm(router_dep)
        agent_llm = make_azure_llm(agent_dep) if agent_dep != router_dep else router_llm

    # ---- resume --------------------------------------------------------------
    rec_path = out / "records.jsonl"
    done = set()
    if rec_path.exists():
        for line in rec_path.open(encoding="utf-8"):
            r = json.loads(line)
            done.add((r["pool_type"], r["n_agents"], r["repeat"], r["task_id"]))
    lock = threading.Lock()

    from rse.graph_router import build_router_graph, run_one
    for pool_type, n in conditions:
        pool = build_pool(pool_type, n)
        if args.backend == "mock":
            mock.pool[:] = pool
        use_deep = args.arch == "deep" and pool_type != "generalist"
        if use_deep:
            from rse.graph_deep import build_deep_agent, run_one_deep
            system = build_deep_agent(pool, router_llm, agent_llm)
        else:
            system = build_router_graph(pool, router_llm, agent_llm)

        jobs = [(rep, t) for rep in range(args.repeats) for t in tasks
                if (pool_type, n, rep, t.task_id) not in done]
        print(f"[{pool_type:11s} N={n:2d}] {len(jobs)} runs", flush=True)

        def work(rep, t):
            rng = random.Random(f"{args.seed}-{pool_type}-{n}-{rep}-{t.task_id}")
            order = [a.name for a in pool]
            rng.shuffle(order)  # randomise listing order -> controls for position bias
            gold_agents = owners(pool, t.required_tool)
            t0 = time.perf_counter()
            err = None
            try:
                res = run_one_deep(system, t.question) if use_deep else run_one(system, t.question, order)
            except Exception as e:  # keep going; the error is recorded
                err = f"{type(e).__name__}: {e}"
                res = {"routes": [], "route_raw": "", "route_valid": False, "answer_raw": "",
                       "router_tokens_in": 0, "router_tokens_out": 0, "agent_tokens_in": 0,
                       "agent_tokens_out": 0, "llm_calls": 0, "tool_calls": []}
                if os.environ.get("RSE_DEBUG"):
                    traceback.print_exc()
            latency = time.perf_counter() - t0
            answer = extract_final(res["answer_raw"])
            first_route = res["routes"][0] if res["routes"] else None
            rec = {
                "arch": "langgraph" if not use_deep else "deep",
                "pool_type": pool_type, "n_agents": n,
                "n_decoy": sum(a.kind == "decoy" for a in pool),
                "n_irrelevant": sum(a.kind == "irrelevant" for a in pool),
                "repeat": rep, "task_id": t.task_id, "category": t.category,
                "required_tool": t.required_tool, "gold_agents": gold_agents,
                "gold_position": min(order.index(g) for g in gold_agents) / max(len(order) - 1, 1),
                "route": first_route, "n_routes": len(res["routes"]),
                "route_valid": res["route_valid"],
                "route_correct": (None if pool_type == "generalist" else first_route in gold_agents),
                "used_required_tool": any(tc["name"] == t.required_tool and tc.get("ok")
                                          for tc in res["tool_calls"]),
                "answer": answer[:500], "gold": t.gold,
                "correct": (err is None) and is_correct(answer, t.gold, t.answer_type),
                "router_tokens_in": res["router_tokens_in"], "router_tokens_out": res["router_tokens_out"],
                "agent_tokens_in": res["agent_tokens_in"], "agent_tokens_out": res["agent_tokens_out"],
                "llm_calls": res["llm_calls"], "n_tool_calls": len(res["tool_calls"]),
                "latency_s": round(latency, 3), "error": err,
                "route_raw": res["route_raw"][:300], "answer_raw": res["answer_raw"][-1500:],
            }
            with lock, rec_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            return rec

        n_ok = n_err = 0
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
            futs = [ex.submit(work, rep, t) for rep, t in jobs]
            for fut in as_completed(futs):
                r = fut.result()
                n_ok += r["correct"]
                n_err += r["error"] is not None
        if jobs:
            print(f"    accuracy {n_ok}/{len(jobs)}   errors {n_err}", flush=True)

    print(f"\nDone. Records: {rec_path}\nNext: python analyze.py {out}")


if __name__ == "__main__":
    main()
