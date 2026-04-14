"""Experiment 2: offline coverage (good / neutral / poor)."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from hybrid_fmucb import LeaderFeasibility
from tqdm import tqdm

import experiment_common as ec


def experiment2(
    out_dir: str,
    seeds: Sequence[int],
    n_a: int,
    n_b: int,
    horizon: int,
    gamma_exp3: float,
    n_off_fixed: int,
    progress: bool = True,
    reward_noise_std: float = ec.DEFAULT_REWARD_NOISE_STD,
    follower_elim_eps: Optional[float] = None,
    leader_feasibility: LeaderFeasibility = "pessimistic",
    feasibility_margin: float = 0.0,
) -> None:
    kinds = ("good", "neutral", "poor")
    t_fw: Dict[str, np.ndarray] = {k: np.zeros(len(seeds)) for k in kinds}
    success: Dict[str, np.ndarray] = {k: np.zeros(len(seeds)) for k in kinds}
    sub_at_star: Dict[str, np.ndarray] = {
        k: np.full(len(seeds), np.nan, dtype=np.float64) for k in kinds
    }

    for kind in kinds:
        seed_bar = tqdm(
            seeds,
            desc=f"Exp 2 ({kind})",
            unit="seed",
            disable=not progress,
        )
        for si, seed in enumerate(seed_bar):
            rng = np.random.default_rng(seed)
            env = ec.StackelbergBandit.sample(n_a, n_b, rng)
            rng_off = np.random.default_rng(seed + 91_000)
            rng_on = np.random.default_rng(seed + 92_000)
            if kind == "good":
                off = ec.build_offline_good_coverage(
                    env, n_off_fixed, rng_off, reward_noise_std=reward_noise_std
                )
            elif kind == "neutral":
                off = ec.build_offline_uniform(
                    env, n_off_fixed, rng_off, reward_noise_std=reward_noise_std
                )
            else:
                off = ec.build_offline_poor_coverage(
                    env, n_off_fixed, rng_off, reward_noise_std=reward_noise_std
                )
            tr = ec.simulate_run(
                env,
                horizon,
                rng_on,
                gamma_exp3=gamma_exp3,
                offline_init=off,
                reward_noise_std=reward_noise_std,
                follower_elim_eps=follower_elim_eps,
                leader_feasibility=leader_feasibility,
                feasibility_margin=feasibility_margin,
            )
            t_fw[kind][si] = tr["subopt"].sum()
            w = min(500, horizon)
            success[kind][si] = 1.0 - tr["subopt"][-w:].mean()
            sub_at_star[kind][si] = ec.follower_subopt_rate_when_leader_a_star(tr, env)

    labels = [
        "Good (includes\n$(a^*, b^*)$)",
        "Neutral\n(uniform)",
        "Poor (never\n$(a^*, b^*)$)",
    ]
    colors_b = [ec.COLORS["good"], ec.COLORS["neutral"], ec.COLORS["poor"]]
    means_t = [float(t_fw[k].mean()) for k in kinds]
    errs_t = [
        1.96 * t_fw[k].std(ddof=1) / np.sqrt(len(seeds)) for k in kinds
    ]
    means_s = [float(success[k].mean()) for k in kinds]
    errs_s = [
        1.96 * success[k].std(ddof=1) / np.sqrt(len(seeds)) for k in kinds
    ]
    means_star: List[float] = []
    errs_star: List[float] = []
    for k in kinds:
        m, e = ec.ci_mean_nan_1d(sub_at_star[k])
        means_star.append(m)
        errs_star.append(e)

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 4.8))
    xpos = np.arange(3)
    bars = axes[0].bar(
        xpos,
        means_t,
        yerr=errs_t,
        color=colors_b,
        edgecolor="white",
        linewidth=1.2,
        capsize=6,
        error_kw={"linewidth": 1.5},
    )
    for rect, m in zip(bars, means_t):
        axes[0].text(
            rect.get_x() + rect.get_width() / 2,
            m + max(errs_t) * 0.15,
            f"{m:.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    axes[0].set_xticks(xpos)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel(r"$T_{f,w}$ (vs.\ $F^{fm}$)")
    axes[0].set_title(rf"Global: total $T_{{f,w}}$ ($N_{{\mathrm{{off}}}}={n_off_fixed}$)")

    bars_m = axes[1].bar(
        xpos,
        means_star,
        yerr=errs_star,
        color=colors_b,
        edgecolor="white",
        linewidth=1.2,
        capsize=6,
        error_kw={"linewidth": 1.5},
    )
    emax = max(errs_star) if errs_star else 1.0
    for rect, m in zip(bars_m, means_star):
        if not np.isfinite(m):
            continue
        axes[1].text(
            rect.get_x() + rect.get_width() / 2,
            min(1.02, m + emax * 0.2 + 0.02),
            f"{m:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel(r"Subopt. rate when $a_t=a^*$")
    axes[1].set_title(r"Conditional on Stackelberg leader action (lower is better)")

    bars2 = axes[2].bar(
        xpos,
        means_s,
        yerr=errs_s,
        color=colors_b,
        edgecolor="white",
        linewidth=1.2,
        capsize=6,
        error_kw={"linewidth": 1.5},
    )
    for rect, m in zip(bars2, means_s):
        axes[2].text(
            rect.get_x() + rect.get_width() / 2,
            min(1.02, m + max(errs_s) * 0.2 + 0.02),
            f"{m:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    axes[2].set_xticks(xpos)
    axes[2].set_xticklabels(labels)
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("Success rate (final window)")
    axes[2].set_title("Global: final-window optimality (higher is better)")
    fig.suptitle(
        "Experiment 2: Global vs $a=a^*$ conditional metrics",
        fontsize=14,
        y=1.02,
    )
    ec._style_axis(axes[0])
    ec._style_axis(axes[1])
    ec._style_axis(axes[2])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "exp2_coverage_bars.png"), bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(out_dir, "exp2_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_off_fixed": n_off_fixed,
                "good_coverage": "80pct_stackelberg_pair_10pct_br_uniform_a_10pct_uniform_ab",
                "neutral_coverage": "uniform_ab",
                "poor_coverage": "never_a_star_b_star; half_skew_argmin_follower_payoff_when_a_not_a_star",
                "offline_rng_seed_offset": 91_000,
                "online_rng_seed_offset": 92_000,
                **{f"tfw_{k}_mean": float(t_fw[k].mean()) for k in kinds},
                **{
                    f"subopt_rate_at_a_star_{k}_mean": float(np.nanmean(sub_at_star[k]))
                    for k in kinds
                },
                **{f"success_{k}_mean": float(success[k].mean()) for k in kinds},
            },
            f,
            indent=2,
        )
