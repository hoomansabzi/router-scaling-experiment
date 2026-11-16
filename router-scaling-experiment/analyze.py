#!/usr/bin/env python
"""Summarise records.jsonl and draw the plots.

  python analyze.py results/pilot                 # one run
  python analyze.py results/full results/deep     # compare architectures

Writes into the FIRST directory given:
  summary.csv, per_category.csv, position_bias.csv
  fig_accuracy_vs_N.png, fig_routing_vs_N.png, fig_tokens_vs_N.png,
  fig_latency_vs_N.png, fig_confusion_<arch>_<pool>_N<k>.png
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# palette (categorical slots in fixed order) + recessive chrome
POOL_COLORS = {"granularity": "#2a78d6", "irrelevant": "#eb6834", "decoy": "#1baf7a"}
POOL_LABELS = {"granularity": "granularity (split 5 tools over N agents)",
               "irrelevant": "core 5 + irrelevant agents",
               "decoy": "core 5 + decoy (overlapping) agents"}
ARCH_STYLE = {"langgraph": "-", "deep": "--"}
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd", "#86b6ef", "#2a78d6", "#0f3d75"])

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "legend.frameon": False,
})


def wilson(k, n, z=1.96):
    if n == 0:
        return (math.nan, math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def load(dirs):
    rows = []
    for d in dirs:
        for line in (Path(d) / "records.jsonl").open(encoding="utf-8"):
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["tokens_total"] = df[["router_tokens_in", "router_tokens_out", "agent_tokens_in",
                             "agent_tokens_out"]].sum(axis=1)
    df["router_tokens"] = df["router_tokens_in"] + df["router_tokens_out"]
    return df


def summarise(df):
    out = []
    for (arch, pool, n), g in df.groupby(["arch", "pool_type", "n_agents"]):
        acc, lo, hi = wilson(int(g["correct"].sum()), len(g))
        rg = g[g["route_correct"].notna()]
        racc, rlo, rhi = wilson(int(rg["route_correct"].astype(bool).sum()), len(rg)) if len(rg) else (math.nan,) * 3
        # accuracy given a correct / wrong route -> separates routing error from execution error
        good = rg[rg["route_correct"].astype(bool)] if len(rg) else rg
        bad = rg[~rg["route_correct"].astype(bool)] if len(rg) else rg
        out.append({
            "arch": arch, "pool_type": pool, "n_agents": n, "runs": len(g),
            "task_acc": acc, "task_acc_lo": lo, "task_acc_hi": hi,
            "route_acc": racc, "route_acc_lo": rlo, "route_acc_hi": rhi,
            "acc_if_routed_right": good["correct"].mean() if len(good) else math.nan,
            "acc_if_routed_wrong": bad["correct"].mean() if len(bad) else math.nan,
            "invalid_route_rate": 1 - g["route_valid"].astype(bool).mean() if pool != "generalist" else 0.0,
            "used_tool_rate": g["used_required_tool"].mean(),
            "tokens_per_task": g["tokens_total"].mean(),
            "router_tokens_per_task": g["router_tokens"].mean(),
            "llm_calls_per_task": g["llm_calls"].mean(),
            "latency_s_mean": g["latency_s"].mean(),
            "latency_s_p90": g["latency_s"].quantile(0.9),
            "error_rate": g["error"].notna().mean(),
        })
    return pd.DataFrame(out).sort_values(["arch", "pool_type", "n_agents"])


def _curves(summary, arch):
    """granularity curve, and irrelevant/decoy curves anchored at the N=5 core pool."""
    s = summary[summary["arch"] == arch]
    anchor = s[(s["pool_type"] == "granularity") & (s["n_agents"] == 5)]
    curves = {}
    for pool in ["granularity", "irrelevant", "decoy"]:
        c = s[s["pool_type"] == pool]
        if pool != "granularity" and len(c) and len(anchor):
            c = pd.concat([anchor.assign(pool_type=pool), c])
        if len(c):
            curves[pool] = c.sort_values("n_agents")
    return curves


def line_plot(summary, metric, lo, hi, ylabel, title, path, ylim=None, baseline=True):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for arch in summary["arch"].unique():
        for pool, c in _curves(summary, arch).items():
            label = POOL_LABELS[pool] + ("" if summary["arch"].nunique() == 1 else f"  [{arch}]")
            ax.plot(c["n_agents"], c[metric], ARCH_STYLE.get(arch, "-"), color=POOL_COLORS[pool],
                    lw=2, marker="o", ms=6, mec=SURF, mew=1.5, label=label, zorder=3)
            if lo and hi:
                ax.fill_between(c["n_agents"], c[lo], c[hi], color=POOL_COLORS[pool], alpha=0.12, lw=0)
        gen = summary[(summary["arch"] == arch) & (summary["pool_type"] == "generalist")]
        if baseline and len(gen) and not math.isnan(gen[metric].iloc[0]):
            ax.axhline(gen[metric].iloc[0], color=MUTED, lw=1.2, ls=":", zorder=2)
            ax.annotate("single generalist agent (N=1)", xy=(ax.get_xlim()[1], gen[metric].iloc[0]),
                        xytext=(-4, 4), textcoords="offset points", ha="right", color=INK2, fontsize=9)
    ax.set_xlabel("number of agents N")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ns = sorted(summary["n_agents"].unique())
    ax.set_xticks(ns)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(loc="lower left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def confusion(df, arch, pool, n, path):
    g = df[(df["arch"] == arch) & (df["pool_type"] == pool) & (df["n_agents"] == n)].copy()
    g["route"] = g["route"].fillna("<invalid>")
    tab = pd.crosstab(g["category"], g["route"], normalize="index")
    core = ["math_agent", "unit_agent", "records_agent", "date_agent", "text_agent"]
    tab = tab.reindex(["math", "units", "records", "dates", "text"]).dropna(how="all")
    rest = [c for c in tab.sum().sort_values(ascending=False).index if c not in core]
    tab = tab[[c for c in core if c in tab.columns] + rest]  # correct agents form the diagonal
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * tab.shape[1] + 2.5), 3.4))
    im = ax.imshow(tab.values, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(tab.shape[1]), tab.columns, rotation=55, ha="right", fontsize=8.5)
    ax.set_yticks(range(tab.shape[0]), tab.index)
    ax.grid(False)
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = tab.values[i, j]
            if v >= 0.05:
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if v > 0.55 else INK)
    ax.set_title(f"Where each task category was routed - {pool}, N={n} ({arch})", loc="left", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="share of tasks in row")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(dirs):
    out = Path(dirs[0])
    df = load(dirs)
    summary = summarise(df)
    summary.to_csv(out / "summary.csv", index=False)

    per_cat = (df.groupby(["arch", "pool_type", "n_agents", "category"])
                 .agg(task_acc=("correct", "mean"),
                      route_acc=("route_correct", lambda s: s.dropna().astype(bool).mean() if s.notna().any() else math.nan),
                      runs=("correct", "size")).reset_index())
    per_cat.to_csv(out / "per_category.csv", index=False)

    r = df[df["route_correct"].notna()].copy()
    if len(r):
        r["gold_position_bin"] = pd.cut(r["gold_position"], [-0.01, 1 / 3, 2 / 3, 1.0],
                                        labels=["top third", "middle third", "bottom third"])
        pb = (r.groupby(["arch", "pool_type", "n_agents", "gold_position_bin"], observed=True)["route_correct"]
                .apply(lambda s: s.astype(bool).mean()).unstack())
        pb.to_csv(out / "position_bias.csv")

    line_plot(summary, "task_acc", "task_acc_lo", "task_acc_hi", "task accuracy",
              "Task accuracy vs number of agents", out / "fig_accuracy_vs_N.png", ylim=(0, 1.02))
    line_plot(summary[summary["pool_type"] != "generalist"], "route_acc", "route_acc_lo", "route_acc_hi",
              "routing accuracy", "Router picks an agent that owns the needed tool",
              out / "fig_routing_vs_N.png", ylim=(0, 1.02), baseline=False)
    line_plot(summary, "tokens_per_task", None, None, "tokens per task (in + out)",
              "Cost: tokens per task vs number of agents", out / "fig_tokens_vs_N.png")
    line_plot(summary, "latency_s_mean", None, None, "mean latency per task (s)",
              "Latency vs number of agents", out / "fig_latency_vs_N.png")
    for arch in df["arch"].unique():
        for pool in ["decoy", "irrelevant"]:
            sub = df[(df["arch"] == arch) & (df["pool_type"] == pool)]
            if len(sub):
                n = int(sub["n_agents"].max())
                confusion(df, arch, pool, n, out / f"fig_confusion_{arch}_{pool}_N{n}.png")

    cols = ["arch", "pool_type", "n_agents", "runs", "task_acc", "route_acc", "acc_if_routed_right",
            "acc_if_routed_wrong", "invalid_route_rate", "tokens_per_task", "llm_calls_per_task",
            "latency_s_mean", "error_rate"]
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        print(summary[cols].to_string(index=False))
    print(f"\nWrote summary + figures to {out}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python analyze.py RESULTS_DIR [MORE_DIRS...]")
    main(sys.argv[1:])
