"""Experiments for the report.

    python run_experiments.py --exp main                  # E1/E2/E5: methods x instances x seeds
    python run_experiments.py --exp sensitivity           # E3: GA parameter study
    python run_experiments.py --exp ablation              # E4: GA design choices
    python run_experiments.py --exp all                   # everything
    python run_experiments.py --exp all --quick           # fast smoke run (few seeds, small budgets)
    python run_experiments.py --exp main --instances small medium my_real_timetable.xlsx

All methods get the SAME number of cost evaluations per instance (fair, hardware independent).
Results (CSV + PNG) are written to ./results
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ttga.algorithms import run_algorithm
from ttga.data import SIZES, generate_instance, instance_path, load_problem

BASE_BUDGET = {"small": 20000, "medium": 40000, "large": 80000}
DEFAULT_BUDGET = 40000
MAIN_METHODS = ["random", "greedy", "sa", "ga", "memetic"]
NICE = {"random": "Random search", "greedy": "Greedy", "sa": "Simulated Annealing",
        "ga": "Genetic Algorithm", "memetic": "GA + local search"}


# ----------------------------------------------------------------------------- plumbing
@lru_cache(maxsize=16)
def _load(path):
    return load_problem(path)


def _job(job):
    path, label, method, params, seed, budget, tag = job
    r = run_algorithm(method, _load(path), budget, seed, **dict(params))
    return {"instance": label, "method": tag, "seed": seed, "budget": budget,
            "hard": r["hard"], "soft": r["soft"], "cost": r["cost"], "feasible": r["feasible"],
            "evals": r["evals"], "first_feasible": r["first_feasible"], "time": r["time"],
            "history": r["history"]}


def run_jobs(jobs, workers):
    t0 = time.time()
    out = []
    if workers <= 1:
        for k, j in enumerate(jobs, 1):
            out.append(_job(j))
            if k % 10 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} runs done ({time.time() - t0:.0f}s)", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for k, r in enumerate(ex.map(_job, jobs, chunksize=1), 1):
                out.append(r)
                if k % 10 == 0 or k == len(jobs):
                    print(f"  {k}/{len(jobs)} runs done ({time.time() - t0:.0f}s)", flush=True)
    return out


def resolve(name):
    """Return (path, label, base_budget) for 'small|medium|large' or an .xlsx path."""
    if name in SIZES:
        path = instance_path(name)
        if not os.path.exists(path):
            generate_instance(name)
        return path, name, BASE_BUDGET[name]
    if not os.path.exists(name):
        raise SystemExit(f"Instance file not found: {name}")
    return name, os.path.splitext(os.path.basename(name))[0], DEFAULT_BUDGET


def summarise(df):
    g = df.groupby(["instance", "method"], sort=False)
    rows = []
    for (inst, m), d in g:
        feas = d[d["feasible"]]
        rows.append({
            "instance": inst, "method": m, "runs": len(d),
            "success_rate_%": round(100 * d["feasible"].mean(), 1),
            "hard_mean": round(d["hard"].mean(), 2), "hard_std": round(d["hard"].std(ddof=0), 2),
            "soft_mean": round(d["soft"].mean(), 2), "soft_std": round(d["soft"].std(ddof=0), 2),
            "soft_mean_feasible_only": round(feas["soft"].mean(), 2) if len(feas) else np.nan,
            "cost_mean": round(d["cost"].mean(), 1), "cost_median": float(d["cost"].median()),
            "cost_best": int(d["cost"].min()),
            "evals_to_first_feasible_mean": round(feas["first_feasible"].mean(), 0)
            if len(feas) else np.nan,
            "time_s_mean": round(d["time"].mean(), 2),
        })
    return pd.DataFrame(rows)


def wilcoxon_table(df, pairs):
    from scipy.stats import wilcoxon
    rows = []
    for inst, d in df.groupby("instance", sort=False):
        wide = d.pivot(index="seed", columns="method", values="cost")
        for a, b in pairs:
            if a not in wide or b not in wide:
                continue
            x, y = wide[a].values, wide[b].values
            try:
                p = float(wilcoxon(x, y).pvalue)
            except ValueError:                       # all differences are zero
                p = 1.0
            better = a if x.mean() < y.mean() else (b if y.mean() < x.mean() else "tie")
            rows.append({"instance": inst, "A": a, "B": b, "mean_cost_A": round(x.mean(), 1),
                         "mean_cost_B": round(y.mean(), 1), "wilcoxon_p": round(p, 4),
                         "significant_at_0.05": p < 0.05, "lower_cost": better})
    return pd.DataFrame(rows)


def convergence_curves(rows, budget, points=200):
    """Mean best-cost curve over runs on a common evaluation grid."""
    grid = np.unique(np.linspace(1, budget, points).astype(int))
    curves = []
    for r in rows:
        h = r["history"]
        xs = np.array([e for e, _ in h])
        ys = np.array([c for _, c in h], dtype=float)
        idx = np.searchsorted(xs, grid, side="right") - 1
        curve = np.where(idx >= 0, ys[np.clip(idx, 0, None)], ys[0])
        curves.append(curve)
    return grid, np.mean(curves, axis=0)


# ----------------------------------------------------------------------------- experiments
def exp_main(a):
    print("== E1/E2/E5: method comparison ==")
    jobs, budgets = [], {}
    for name in a.instances:
        path, label, base = resolve(name)
        budget = max(500, int(base * a.budget_scale))
        budgets[label] = budget
        for m in MAIN_METHODS:
            for seed in range(1, a.seeds + 1):
                jobs.append((path, label, m, (), seed, budget, m))
    rows = run_jobs(jobs, a.workers)
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "history"} for r in rows])
    df.to_csv(os.path.join(a.out, "main_runs.csv"), index=False)

    summ = summarise(df)
    summ["method"] = summ["method"].map(NICE)
    summ.to_csv(os.path.join(a.out, "main_summary.csv"), index=False)
    print(summ[["instance", "method", "success_rate_%", "hard_mean", "soft_mean",
                "cost_median", "time_s_mean"]].to_string(index=False))

    pairs = [("ga", "sa"), ("memetic", "ga"), ("memetic", "sa"), ("ga", "random"), ("sa", "random")]
    wt = wilcoxon_table(df, pairs)
    wt.to_csv(os.path.join(a.out, "main_wilcoxon.csv"), index=False)
    print("\nWilcoxon signed-rank tests (paired by seed, on total cost):")
    print(wt.to_string(index=False))

    for label, budget in budgets.items():
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for m in MAIN_METHODS:
            sel = [r for r in rows if r["instance"] == label and r["method"] == m]
            grid, mean = convergence_curves(sel, budget)
            ax.plot(grid, mean + 1, label=NICE[m])
        ax.set_yscale("log")
        ax.set_xlabel("cost evaluations")
        ax.set_ylabel("mean best cost + 1 (log scale)")
        ax.set_title(f"Convergence on '{label}' (mean of {a.seeds} runs)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, f"convergence_{label}.png"), dpi=150)
        plt.close(fig)

    labels = list(budgets)
    fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 4), squeeze=False)
    for ax, label in zip(axes[0], labels):
        data = [df[(df.instance == label) & (df.method == m)]["soft"].values for m in MAIN_METHODS]
        ax.boxplot(data)
        ax.set_xticks(range(1, len(MAIN_METHODS) + 1))
        ax.set_xticklabels([NICE[m].replace(" ", "\n", 1) for m in MAIN_METHODS])
        ax.set_title(f"{label}: soft penalty of best solution")
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "boxplot_soft.png"), dpi=150)
    plt.close(fig)


def exp_sensitivity(a):
    print("== E3: GA parameter sensitivity ==")
    path, label, base = resolve(a.sens_instance)
    budget = max(500, int(base * a.budget_scale))
    seeds = range(1, a.sens_seeds + 1)
    pops, pms, pcs = [20, 40, 80], [0.002, 0.005, 0.02], [0.5, 0.8, 0.95]
    jobs = []
    for pop in pops:
        for pm in pms:
            for s in seeds:
                jobs.append((path, label, "ga", (("pop_size", pop), ("pm", pm)), s, budget,
                             f"pop={pop}|pm={pm}"))
    for pc in pcs:
        for s in seeds:
            jobs.append((path, label, "ga", (("pc", pc),), s, budget, f"pc={pc}"))
    rows = run_jobs(jobs, a.workers)
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "history"} for r in rows])
    df.to_csv(os.path.join(a.out, "sensitivity_runs.csv"), index=False)

    grid = df[df.method.str.startswith("pop=")].copy()
    grid["pop"] = grid.method.str.extract(r"pop=(\d+)")[0].astype(int)
    grid["pm"] = grid.method.str.extract(r"pm=([\d.]+)")[0].astype(float)
    piv = grid.pivot_table(index="pop", columns="pm", values="cost", aggfunc="mean")
    piv.to_csv(os.path.join(a.out, "sensitivity_pop_pm_mean_cost.csv"))
    print("Mean cost (rows = population size, columns = mutation rate):")
    print(piv.round(1).to_string())
    pc_tab = df[df.method.str.startswith("pc=")].groupby("method")["cost"].agg(["mean", "std"]).round(1)
    pc_tab.to_csv(os.path.join(a.out, "sensitivity_pc.csv"))
    print("\nCrossover rate (defaults for other parameters):")
    print(pc_tab.to_string())

    fig, ax = plt.subplots(figsize=(5.5, 4))
    im = ax.imshow(piv.values, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(piv.columns)), [str(c) for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), [str(i) for i in piv.index])
    ax.set_xlabel("mutation rate pm")
    ax.set_ylabel("population size")
    ax.set_title(f"GA mean cost on '{label}' (lower is better)")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.0f}", ha="center", va="center", color="white",
                    fontsize=8)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "sensitivity_heatmap.png"), dpi=150)
    plt.close(fig)


def exp_ablation(a):
    print("== E4: ablation of GA design choices ==")
    path, label, base = resolve(a.sens_instance)
    budget = max(500, int(base * a.budget_scale))
    variants = {
        "GA (batch crossover)": ("ga", ()),
        "GA (uniform crossover)": ("ga", (("crossover_kind", "uniform"),)),
        "GA (one-point crossover)": ("ga", (("crossover_kind", "onepoint"),)),
        "GA (no elitism)": ("ga", (("elite", 0),)),
        "GA (no crossover, mutation only)": ("ga", (("pc", 0.0),)),
        "GA + local search": ("memetic", ()),
    }
    jobs = [(path, label, m, p, s, budget, name)
            for name, (m, p) in variants.items() for s in range(1, a.sens_seeds + 1)]
    rows = run_jobs(jobs, a.workers)
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "history"} for r in rows])
    df.to_csv(os.path.join(a.out, "ablation_runs.csv"), index=False)
    summ = summarise(df)
    summ.to_csv(os.path.join(a.out, "ablation_summary.csv"), index=False)
    print(summ[["method", "success_rate_%", "hard_mean", "soft_mean", "cost_mean"]]
          .to_string(index=False))
    fig, ax = plt.subplots(figsize=(7, 4))
    order = list(variants)
    means = [df[df.method == v]["cost"].mean() for v in order]
    stds = [df[df.method == v]["cost"].std(ddof=0) for v in order]
    ax.barh(range(len(order)), means, xerr=stds, color="#4a7fb5")
    ax.set_yticks(range(len(order)), order, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("mean cost (lower is better)")
    ax.set_title(f"Ablation on '{label}'")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "ablation_bar.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--exp", default="main", choices=["main", "sensitivity", "ablation", "all"])
    ap.add_argument("--instances", nargs="+", default=["small", "medium", "large"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--sens-seeds", type=int, default=5)
    ap.add_argument("--sens-instance", default="medium")
    ap.add_argument("--budget-scale", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--quick", action="store_true", help="3 seeds, 15%% budget")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.sens_seeds, a.budget_scale = 3, 2, 0.15
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    if a.exp in ("main", "all"):
        exp_main(a)
    if a.exp in ("sensitivity", "all"):
        exp_sensitivity(a)
    if a.exp in ("ablation", "all"):
        exp_ablation(a)
    print(f"\nDone in {time.time() - t0:.0f}s. Files are in '{a.out}/'.")


if __name__ == "__main__":
    main()
