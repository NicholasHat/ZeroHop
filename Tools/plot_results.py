#!/usr/bin/env python3
"""Generate the paper figures from Results/ raw CSVs (M4 tooling; offline
analysis only — anything goes here per spec §11).

Usage: Tools/.venv/bin/python Tools/plot_results.py [results-dir] [out-dir]

Each figure is written as PDF (camera-ready) and PNG (drafting). Cells are
selected by the newest run directory matching each cell-name pattern, so
rerunning after fresh measurements picks up the new data automatically.
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(sys.argv[1] if len(sys.argv) > 1 else "Results")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "Figures")
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})


def newest(pattern: str) -> Path | None:
    dirs = sorted(RESULTS.glob(f"*{pattern}*"))
    return dirs[-1] if dirs else None


def load_column(run: Path, column: str, thermal_only: bool = True) -> np.ndarray:
    with open(run / "samples.csv") as f:
        rows = list(csv.DictReader(f))
    vals = [float(r[column]) for r in rows
            if not thermal_only or r.get("thermal_ok", "1") == "1"]
    return np.asarray(vals)


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/{name}.pdf")


def fig_handoff_gpuwarm() -> None:
    """Fig: handoff CCDF per GPU keep-warm level (the idle-ramp result)."""
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for label, pattern in [("none", "m1-e3-sync_e4-rt_e5-none_gw-none"),
                           ("10 ms", "m1-e3-sync_e4-rt_e5-none_gw-10"),
                           ("2 ms", "m1-e3-sync_e4-rt_e5-none_gw-2"),
                           ("saturated", "m1-e3-sync_e4-rt_e5-none_gw-saturated")]:
        run = newest(pattern)
        if run is None:
            continue
        v = np.sort(load_column(run, "t2_t6_handoff_ns")) / 1e3  # µs
        ccdf = 1.0 - np.arange(1, len(v) + 1) / len(v)
        ax.plot(v, ccdf, label=f"gpu-warm {label}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("handoff t2′→t6 (µs)")
    ax.set_ylabel("P(latency > x)")
    ax.legend(fontsize=7)
    save(fig, "handoff_gpuwarm_ccdf")


def fig_e5_warmth() -> None:
    """Fig: ANE warmth sweep — handoff p99/max per heartbeat period."""
    labels, p99s, maxs = [], [], []
    for period in ["none", "500", "100", "50", "10", "saturated"]:
        run = newest(f"e5-{period}_gw-saturated")
        if run is None:
            continue
        v = load_column(run, "t2_t6_handoff_ns") / 1e3
        labels.append(period)
        p99s.append(np.percentile(v, 99))
        maxs.append(v.max())
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    x = np.arange(len(labels))
    ax.bar(x - 0.2, p99s, width=0.4, label="p99")
    ax.bar(x + 0.2, maxs, width=0.4, label="max")
    ax.set_yscale("log")
    ax.set_xticks(x, labels)
    ax.set_xlabel("ANE heartbeat period (ms)")
    ax.set_ylabel("handoff (µs)")
    ax.legend(fontsize=7)
    save(fig, "e5_warmth")


def fig_stage_attribution() -> None:
    """Fig: per-stage box plot for the headline M1 cell."""
    run = newest("m1-e3-sync_e4-rt_e5-none_gw-saturated")
    if run is None:
        return
    stages = [("t0_t2_dispatch_ns", "dispatch\nt0→t2′"),
              ("t4_t5_signal_ns", "signal\nt4→t5"),
              ("t5_t6_release_ns", "release\nt5→t6"),
              ("t2_t6_handoff_ns", "handoff\nt2′→t6")]
    data, labels = [], []
    for col, label in stages:
        try:
            data.append(load_column(run, col) / 1e3)
            labels.append(label)
        except KeyError:
            pass
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.boxplot(data, tick_labels=labels, whis=(1, 99), showfliers=True,
               flierprops={"markersize": 2, "alpha": 0.4})
    ax.set_yscale("log")
    ax.set_ylabel("latency (µs)")
    save(fig, "stage_attribution")


def fig_m0_noise_floor() -> None:
    """Fig: cross-thread wake CCDF per thread policy (the E4 result)."""
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for policy in ["default", "qos-ui", "rt"]:
        run = newest(f"m0-xwake-{policy}")
        if run is None:
            continue
        v = np.sort(load_column(run, "xwake_latency_ns")) / 1e3
        ccdf = 1.0 - np.arange(1, len(v) + 1) / len(v)
        ax.plot(v, ccdf, label=policy)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("cross-thread wake (µs)")
    ax.set_ylabel("P(latency > x)")
    ax.legend(fontsize=7)
    save(fig, "m0_noise_floor")


def fig_k_sweep() -> None:
    """Fig: end-to-end k-sweep (values from the m3 runs; see FINDINGS.md)."""
    ks = [4, 6, 8]
    tok_s = [25.4, 28.1, 26.3]
    speedup = [1.33, 1.47, 1.38]
    baseline = 19.1
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.plot(ks, tok_s, "o-", label="ANE draft, pipelined")
    ax.axhline(baseline, color="gray", ls="--", label="GPU-only baseline")
    ax.axhline(24.8, color="tab:orange", ls=":", label="GPU draft (serial, k=4)")
    for k, t, s in zip(ks, tok_s, speedup):
        ax.annotate(f"{s:.2f}×", (k, t), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=7)
    ax.set_xticks(ks)
    ax.set_xlabel("speculation window k")
    ax.set_ylabel("tokens/s")
    ax.legend(fontsize=7)
    save(fig, "k_sweep")


if __name__ == "__main__":
    fig_m0_noise_floor()
    fig_handoff_gpuwarm()
    fig_e5_warmth()
    fig_stage_attribution()
    fig_k_sweep()
