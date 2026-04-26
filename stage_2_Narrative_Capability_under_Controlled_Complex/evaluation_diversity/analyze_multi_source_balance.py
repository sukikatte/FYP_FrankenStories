#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep analysis for multi-story attribution:
within one output, treat per-source B_i/A values as non-negative weights,
normalize them, compute Shannon entropy, then divide by log2(K) to obtain
balance score H_norm in [0,1].

Outputs:
- analysis/analysis_multi_source_balance_depth.json
- visualizations/figure_multi_source_balance_depth.png
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DETAILED_DIR = ROOT / "evaluation_results_diversity"
OUT_ANALYSIS = ROOT / "analysis"
OUT_VIZ = ROOT / "visualizations"

MULTI_DIMS = (
    "dim2_single_char_multi_story",
    "dim4_multi_char_multi_story",
)

MODEL_ORDER = ["gpt_4", "gpt_4o", "deepseek_v3", "deepseek_r1", "llama_3_70b"]
MODEL_LABEL = {
    "gpt_4": "GPT-4.1",
    "gpt_4o": "GPT-4o",
    "deepseek_v3": "DeepSeek-V3.1",
    "deepseek_r1": "DeepSeek-R1",
    "llama_3_70b": "Llama-3.3-70B",
}


def find_latest_detailed() -> Path:
    files = sorted(DETAILED_DIR.glob("detailed_results_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No detailed_results_*.json under {DETAILED_DIR}")
    return files[0]


def story_ids_from_label(label: str) -> list[int]:
    label = label.strip()
    if "_" in label:
        return [int(x) for x in label.split("_")]
    return [int(label)]


def normalized_balance_entropy(ratios: list[float]) -> float:
    """For K sources, p_i ∝ B_i/A; return H(p)/log2(K)."""
    if not ratios:
        return float("nan")
    k = len(ratios)
    s = float(sum(ratios))
    if s <= 0:
        return float("nan")
    ps = [max(0.0, r) / s for r in ratios]
    h = 0.0
    for p in ps:
        if p > 0:
            h -= p * math.log2(p)
    h_max = math.log2(k) if k > 1 else 1.0
    return h / h_max if h_max > 0 else 1.0


def load_records(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for r in data:
        if "error" in r:
            continue
        dim = r.get("dimension")
        if dim not in MULTI_DIMS:
            continue
        us = r.get("diversity", {}).get("unique_source_share")
        if not us:
            continue
        label = str(r.get("story_label", "")).strip()
        ids = [str(s) for s in sorted(story_ids_from_label(label), key=int)]
        ratios = []
        for sid in ids:
            entry = us.get(sid)
            if isinstance(entry, dict) and "ratio" in entry:
                ratios.append(float(entry["ratio"]))
            else:
                ratios.append(0.0)
        h_norm = normalized_balance_entropy(ratios)
        k = len(ids)
        rows.append(
            {
                "file": r.get("file"),
                "model": r.get("model"),
                "dimension": dim,
                "story_label": label,
                "n_sources": k,
                "h_norm": h_norm,
                "ratios": ratios,
                "story_ids": ids,
            }
        )
    return rows


def aggregate(rows: list[dict]) -> dict:
    by_mdn = defaultdict(list)  # (model, dim, n_sources) -> h_norm
    by_mdb = defaultdict(list)  # (model, dim, bundle) -> h_norm
    for row in rows:
        key_n = (row["model"], row["dimension"], row["n_sources"])
        by_mdn[key_n].append(row["h_norm"])
        key_b = (row["model"], row["dimension"], row["story_label"])
        by_mdb[key_b].append(row["h_norm"])

    def stats(vals: list[float]) -> dict:
        a = np.array(vals, dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return {"n": 0, "mean": None, "std": None, "median": None}
        return {
            "n": int(a.size),
            "mean": float(a.mean()),
            "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "median": float(np.median(a)),
        }

    out_mdn = {f"{m}|{d}|{n}": stats(v) for (m, d, n), v in sorted(by_mdn.items())}
    out_mdb = {f"{m}|{d}|{b}": stats(v) for (m, d, b), v in sorted(by_mdb.items())}

    contrast = defaultdict(dict)
    for dim in MULTI_DIMS:
        for m in MODEL_ORDER:
            s2 = by_mdn.get((m, dim, 2), [])
            s3 = by_mdn.get((m, dim, 3), [])
            st2, st3 = stats(s2), stats(s3)
            diff = (
                None
                if st2["mean"] is None or st3["mean"] is None
                else float(st3["mean"] - st2["mean"])
            )
            contrast[dim][m] = {
                "two_source": st2,
                "three_source": st3,
                "mean_diff_three_minus_two": diff,
            }

    bundle_consistency = {}
    for m in MODEL_ORDER:
        for dim in MULTI_DIMS:
            bundle_means = []
            for b in sorted({row["story_label"] for row in rows if row["model"] == m and row["dimension"] == dim}):
                key = (m, dim, b)
                st = stats(by_mdb[key])
                if st["mean"] is not None:
                    bundle_means.append(st["mean"])
            if len(bundle_means) >= 2:
                bundle_consistency[f"{m}|{dim}"] = {
                    "mean_of_bundle_means": float(np.mean(bundle_means)),
                    "std_across_bundles": float(np.std(bundle_means, ddof=1)),
                    "n_bundles": len(bundle_means),
                }

    return {
        "by_model_dim_n_sources": out_mdn,
        "by_model_dim_bundle": out_mdb,
        "contrast_2_vs_3_sources": dict(contrast),
        "bundle_consistency": bundle_consistency,
    }


def build_insights(rows: list[dict], agg: dict) -> list[str]:
    lines = []
    contrast = agg["contrast_2_vs_3_sources"]

    for dim in MULTI_DIMS:
        diffs = []
        for m in MODEL_ORDER:
            d = contrast[dim][m].get("mean_diff_three_minus_two")
            if d is not None:
                diffs.append((d, m))
        if diffs:
            diffs.sort(key=lambda x: x[0], reverse=True)
            hi_m, lo_m = diffs[0], diffs[-1]
            lines.append(
                f"{dim}: largest mean H_norm difference (3-source minus 2-source) is {hi_m[1]} (Δ={hi_m[0]:+.3f}); "
                f"smallest is {lo_m[1]} (Δ={lo_m[0]:+.3f}). Δ>0 indicates more spread attribution in 3-source bundles."
            )

    by_b = defaultdict(list)
    for row in rows:
        by_b[(row["dimension"], row["story_label"])].append(row["h_norm"])
    hardest = []
    for (dim, bundle), vals in by_b.items():
        a = np.array(vals, dtype=float)
        a = a[np.isfinite(a)]
        if a.size:
            hardest.append((float(a.mean()), dim, bundle, int(a.size)))
    hardest.sort()
    if len(hardest) >= 2:
        low = hardest[0]
        lines.append(
            f"After pooling across models, the lowest-balance combination is {low[1]} / bundle {low[2]} "
            f"(mean H_norm≈{low[0]:.3f}, n={low[3]}), while one of the highest is "
            f"{hardest[-1][1]} / {hardest[-1][2]} (≈{hardest[-1][0]:.3f})."
        )

    bc = agg["bundle_consistency"]
    if bc:
        ranked = sorted(
            ((v["std_across_bundles"], k) for k, v in bc.items()),
            reverse=True,
        )
        top = ranked[0]
        lines.append(
            f"The largest across-bundle mean difference is {top[1]} "
            f"(std of bundle-level mean H_norm≈{top[0]:.3f}), indicating inconsistent spread "
            "for the same model under different source-story combinations."
        )

    return lines


def plot_depth(rows: list[dict], dest: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(20, 4.15), sharey=True)
    width = 0.34
    x = np.arange(len(MODEL_ORDER))

    for ax, dim, title in zip(
        axes,
        MULTI_DIMS,
        ["SC-MS (multi-source)", "MC-MS (multi-source)"],
    ):
        means2, err2, means3, err3 = [], [], [], []
        for m in MODEL_ORDER:
            sub2 = [r["h_norm"] for r in rows if r["model"] == m and r["dimension"] == dim and r["n_sources"] == 2]
            sub3 = [r["h_norm"] for r in rows if r["model"] == m and r["dimension"] == dim and r["n_sources"] == 3]
            for sub, means, errs in ((sub2, means2, err2), (sub3, means3, err3)):
                a = np.array(sub, dtype=float)
                a = a[np.isfinite(a)]
                if a.size == 0:
                    means.append(float("nan"))
                    errs.append(0.0)
                else:
                    means.append(float(a.mean()))
                    errs.append(float(a.std(ddof=1) / math.sqrt(a.size)) if a.size > 1 else 0.0)

        ax.bar(x - width / 2, means2, width, yerr=err2, label="2 sources", color="#4c72b0", capsize=3, ecolor="#333")
        ax.bar(x + width / 2, means3, width, yerr=err3, label="3 sources", color="#dd8452", capsize=3, ecolor="#333")
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABEL[m] for m in MODEL_ORDER], rotation=22, ha="right", fontsize=10)
        ax.set_ylabel(r"Balance $H_{\mathrm{norm}}$", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="y", labelsize=10)
        ax.legend(loc="lower right", fontsize=9, framealpha=0.92)

    fig.suptitle(
        r"Within one output: $p_i \propto B_i/A$ over sources in the bundle; "
        r"$H_{\mathrm{norm}} = H(p)/\log_2 K$",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.93])
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=220, bbox_inches="tight", facecolor="white", pad_inches=0.15)
    plt.close(fig)


def main() -> None:
    detailed = find_latest_detailed()
    rows = load_records(detailed)
    agg = aggregate(rows)
    insights = build_insights(rows, agg)
    OUT_ANALYSIS.mkdir(parents=True, exist_ok=True)
    OUT_VIZ.mkdir(parents=True, exist_ok=True)

    report = {
        "source_detailed_json": str(detailed.relative_to(ROOT)),
        "n_multi_story_files": len(rows),
        "records_sample": rows[:3],
        "aggregation": agg,
        "insights_zh": insights,
    }
    out_json = OUT_ANALYSIS / "analysis_multi_source_balance_depth.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    out_png = OUT_VIZ / "figure_multi_source_balance_depth.png"
    plot_depth(rows, out_png)

    print(f"Detailed: {detailed}")
    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_png}")
    for line in insights:
        print(line)


if __name__ == "__main__":
    main()
