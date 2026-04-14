"""Experiment 1: offline dataset size vs T_{f,w} (+ optional learning curves)."""

from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from hybrid_fmucb import LeaderFeasibility
from tqdm import tqdm

import experiment_common as ec


def experiment1(
    out_dir: str,
    seeds: Sequence[int],
    n_a: int,
    n_b: int,
    horizon: int,
    gamma_exp3: float,
    n_off_grid: Sequence[int],
    cum_n_off: Optional[int] = None,
    progress: bool = True,
    reward_noise_std: float = ec.DEFAULT_REWARD_NOISE_STD,
    follower_elim_eps: Optional[float] = None,
    leader_feasibility: LeaderFeasibility = "pessimistic",
    feasibility_margin: float = 0.0,
) -> None:
    n_off_list = list(n_off_grid)
    cum_target = max(n_off_list) if cum_n_off is None else cum_n_off
    if cum_target not in n_off_list:
        cum_target = max(n_off_list)

    conv_win = 200
    conv_thr = 0.2
    conv_k = 3

    t_fw_base = np.zeros((len(seeds), len(n_off_list)))
    t_fw_hyb = np.zeros((len(seeds), len(n_off_list)))
    conv_base = np.zeros((len(seeds), len(n_off_list)))
    conv_hyb = np.zeros((len(seeds), len(n_off_list)))
    cum_base_rows: List[np.ndarray] = []
    cum_hyb_rows: List[np.ndarray] = []

    for si, seed in enumerate(
        tqdm(seeds, desc="Exp 1 (offline size × seeds)", unit="seed", disable=not progress)
    ):
        rng = np.random.default_rng(seed)
        env = ec.StackelbergBandit.sample(n_a, n_b, rng)

        for j, n_off in enumerate(n_off_list):
            rng_b = np.random.default_rng(seed * 100_000 + j + 7)
            rng_h = np.random.default_rng(seed * 100_000 + j + 13)

            off = (
                ec.build_offline_uniform(env, n_off, rng_b, reward_noise_std=reward_noise_std)
                if n_off > 0
                else None
            )

            tr_b = ec.simulate_run(
                env,
                horizon,
                rng_b,
                gamma_exp3=gamma_exp3,
                offline_init=None,
                reward_noise_std=reward_noise_std,
                follower_elim_eps=follower_elim_eps,
                leader_feasibility=leader_feasibility,
                feasibility_margin=feasibility_margin,
            )
            tr_h = ec.simulate_run(
                env,
                horizon,
                rng_h,
                gamma_exp3=gamma_exp3,
                offline_init=off,
                reward_noise_std=reward_noise_std,
                follower_elim_eps=follower_elim_eps,
                leader_feasibility=leader_feasibility,
                feasibility_margin=feasibility_margin,
            )

            t_fw_base[si, j] = tr_b["subopt"].sum()
            t_fw_hyb[si, j] = tr_h["subopt"].sum()
            conv_base[si, j] = ec.convergence_round(
                tr_b["subopt"], window=conv_win, threshold=conv_thr, k_sustain=conv_k
            )
            conv_hyb[si, j] = ec.convergence_round(
                tr_h["subopt"], window=conv_win, threshold=conv_thr, k_sustain=conv_k
            )

            if n_off == cum_target:
                cum_base_rows.append(np.cumsum(tr_b["subopt"].astype(np.float64)))
                cum_hyb_rows.append(np.cumsum(tr_h["subopt"].astype(np.float64)))

    x_labels = [str(n) for n in n_off_list]
    x_plot = np.array(n_off_list, dtype=np.float64) + 1.0

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)
    mb, eb = ec.ci_mean(t_fw_base, axis=0)
    mh, eh = ec.ci_mean(t_fw_hyb, axis=0)
    ax.plot(
        x_plot,
        mb,
        "o-",
        color=ec.COLORS["baseline"],
        lw=3,
        ms=8,
        label="FMUCB (online only)",
    )
    ax.fill_between(x_plot, mb - eb, mb + eb, color=ec.COLORS["baseline"], alpha=0.18)
    ax.plot(
        x_plot,
        mh,
        "s-",
        color=ec.COLORS["hybrid"],
        lw=3,
        ms=8,
        label="Hybrid-FMUCB",
    )
    ax.fill_between(x_plot, mh - eh, mh + eh, color=ec.COLORS["hybrid"], alpha=0.18)
    ax.set_xticks(x_plot)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel(r"Offline dataset size $N_{\mathrm{off}}$")
    ax.set_ylabel(r"$T_{f,w}$ (mistakes vs.\ true best manipulation $F^{fm}$)")
    ax.set_title(r"Experiment 1: Offline data size vs $T_{f,w}$")
    ax.legend(frameon=True, fancybox=True, shadow=True, loc="upper right")
    ec._style_axis(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "exp1_noff_vs_tfw.png"), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)
    cb, cbe = ec.ci_mean(conv_base, axis=0)
    ch, che = ec.ci_mean(conv_hyb, axis=0)
    ax.plot(x_plot, cb, "o-", color=ec.COLORS["baseline"], lw=3, ms=8, label="FMUCB")
    ax.fill_between(x_plot, cb - cbe, cb + cbe, color=ec.COLORS["baseline"], alpha=0.18)
    ax.plot(x_plot, ch, "s-", color=ec.COLORS["hybrid"], lw=3, ms=8, label="Hybrid-FMUCB")
    ax.fill_between(x_plot, ch - che, ch + che, color=ec.COLORS["hybrid"], alpha=0.18)
    ax.set_xticks(x_plot)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel(r"$N_{\mathrm{off}}$")
    ax.set_ylabel("Rounds to reach low rolling error rate (lower is better)")
    ax.set_title(
        rf"Experiment 1: Sustained low subopt. rate ($\leq {conv_thr}$, "
        rf"{conv_k} consecutive windows of ${conv_win}$ rounds)"
    )
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ec._style_axis(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "exp1_noff_vs_convergence.png"), bbox_inches="tight")
    plt.close(fig)

    stack_cb = np.stack(cum_base_rows, axis=0)
    stack_ch = np.stack(cum_hyb_rows, axis=0)
    t_axis = np.arange(1, horizon + 1)
    mcb, ecb = ec.ci_mean(stack_cb, axis=0)
    mch, ech = ec.ci_mean(stack_ch, axis=0)
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.grid(True, which="both", alpha=0.3)
    ax.plot(t_axis, mcb, "-", color=ec.COLORS["baseline"], lw=3, label="FMUCB (online only)")
    ax.fill_between(t_axis, mcb - ecb, mcb + ecb, color=ec.COLORS["baseline"], alpha=0.18)
    ax.plot(t_axis, mch, "-", color=ec.COLORS["hybrid"], lw=3, label="Hybrid-FMUCB")
    ax.fill_between(t_axis, mch - ech, mch + ech, color=ec.COLORS["hybrid"], alpha=0.18)
    ax.set_xlabel("Round $t$")
    ax.set_ylabel(r"Cumulative manipulation mistakes (vs.\ $F^{fm}$)")
    ax.set_title(rf"Experiment 1: Cumulative mistakes ($N_{{\mathrm{{off}}}}={cum_target}$)")
    ax.legend(frameon=True, fancybox=True, shadow=True, loc="lower right")
    ec._style_axis(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "exp1_cumulative_mistakes.png"), bbox_inches="tight")
    plt.close(fig)

    summary = {
        "n_off": n_off_list,
        "x_axis_log_shift_plus_1": True,
        "convergence_metric": "first_t_where_rolling_subopt_rate_le_threshold_for_k_consecutive_window_ends",
        "convergence_window": conv_win,
        "convergence_threshold": conv_thr,
        "convergence_k_sustain": conv_k,
        "cum_mistakes_n_off": cum_target,
        "tfw_baseline_mean": t_fw_base.mean(0).tolist(),
        "tfw_hybrid_mean": t_fw_hyb.mean(0).tolist(),
        "convergence_baseline_mean": conv_base.mean(0).tolist(),
        "convergence_hybrid_mean": conv_hyb.mean(0).tolist(),
    }
    with open(os.path.join(out_dir, "exp1_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def learning_curve_figure(
    out_dir: str,
    seeds: Sequence[int],
    n_a: int,
    n_b: int,
    horizon: int,
    gamma_exp3: float,
    n_off: int,
    progress: bool = True,
    reward_noise_std: float = ec.DEFAULT_REWARD_NOISE_STD,
    follower_elim_eps: Optional[float] = None,
    leader_feasibility: LeaderFeasibility = "pessimistic",
    feasibility_margin: float = 0.0,
) -> None:
    curves_b: List[np.ndarray] = []
    curves_h: List[np.ndarray] = []
    curves_b_as: List[np.ndarray] = []
    curves_h_as: List[np.ndarray] = []
    win = max(50, horizon // 100)

    for seed in tqdm(
        seeds,
        desc="Learning curves (optional)",
        unit="seed",
        disable=not progress,
    ):
        rng = np.random.default_rng(seed)
        env = ec.StackelbergBandit.sample(n_a, n_b, rng)
        a_star_env = int(env.stackelberg_leader_action())
        rng_b = np.random.default_rng(seed + 3)
        rng_h = np.random.default_rng(seed + 5)
        off = (
            ec.build_offline_uniform(env, n_off, rng_h, reward_noise_std=reward_noise_std)
            if n_off > 0
            else None
        )
        tr_b = ec.simulate_run(
            env,
            horizon,
            rng_b,
            gamma_exp3=gamma_exp3,
            offline_init=None,
            reward_noise_std=reward_noise_std,
            follower_elim_eps=follower_elim_eps,
            leader_feasibility=leader_feasibility,
            feasibility_margin=feasibility_margin,
        )
        tr_h = ec.simulate_run(
            env,
            horizon,
            rng_h,
            gamma_exp3=gamma_exp3,
            offline_init=off,
            reward_noise_std=reward_noise_std,
            follower_elim_eps=follower_elim_eps,
            leader_feasibility=leader_feasibility,
            feasibility_margin=feasibility_margin,
        )
        sub_b = tr_b["subopt"].astype(np.float64)
        sub_h = tr_h["subopt"].astype(np.float64)
        roll_b = np.convolve(sub_b, np.ones(win) / win, mode="valid")
        roll_h = np.convolve(sub_h, np.ones(win) / win, mode="valid")
        curves_b.append(roll_b)
        curves_h.append(roll_h)
        curves_b_as.append(
            ec.rolling_subopt_rate_at_a_star(sub_b, tr_b["a_hist"], a_star_env, win)
        )
        curves_h_as.append(
            ec.rolling_subopt_rate_at_a_star(sub_h, tr_h["a_hist"], a_star_env, win)
        )

    stack_b = np.stack(curves_b, axis=0)
    stack_h = np.stack(curves_h, axis=0)
    t_axis = np.arange(win, horizon + 1)
    mb, eb = ec.ci_mean(stack_b, axis=0)
    mh, eh = ec.ci_mean(stack_h, axis=0)

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.plot(t_axis, mb, color=ec.COLORS["baseline"], lw=2.0, label="FMUCB")
    ax.fill_between(t_axis, mb - eb, mb + eb, color=ec.COLORS["baseline"], alpha=0.2)
    ax.plot(t_axis, mh, color=ec.COLORS["hybrid"], lw=2.0, label="Hybrid-FMUCB")
    ax.fill_between(t_axis, mh - eh, mh + eh, color=ec.COLORS["hybrid"], alpha=0.2)
    ax.set_xlabel("Round $t$")
    ax.set_ylabel(f"Rolling mean manipulation-mistake rate (window={win})")
    ax.set_title(rf"Learning curves (optional): $N_{{\mathrm{{off}}}}={n_off}$")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.0)
    ec._style_axis(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "exp_optional_learning_curves.png"), bbox_inches="tight")
    plt.close(fig)

    stack_b_as = np.stack(curves_b_as, axis=0)
    stack_h_as = np.stack(curves_h_as, axis=0)
    mb_a, eb_a = ec.ci_mean_nan(stack_b_as, axis=0)
    mh_a, eh_a = ec.ci_mean_nan(stack_h_as, axis=0)
    fig_a, ax_a = plt.subplots(figsize=(9.0, 5.0))
    ax_a.plot(t_axis, mb_a, color=ec.COLORS["baseline"], lw=2.0, label="FMUCB")
    ax_a.fill_between(t_axis, mb_a - eb_a, mb_a + eb_a, color=ec.COLORS["baseline"], alpha=0.2)
    ax_a.plot(t_axis, mh_a, color=ec.COLORS["hybrid"], lw=2.0, label="Hybrid-FMUCB")
    ax_a.fill_between(t_axis, mh_a - eh_a, mh_a + eh_a, color=ec.COLORS["hybrid"], alpha=0.2)
    ax_a.set_xlabel("Round $t$")
    ax_a.set_ylabel(f"Rolling mistake rate at $a_t=a^*$ (window={win})")
    ax_a.set_title(
        rf"Learning curves (conditional on $a^*$): $N_{{\mathrm{{off}}}}={n_off}$"
    )
    ax_a.legend(loc="upper right")
    ax_a.set_ylim(0, 1.0)
    ec._style_axis(ax_a)
    fig_a.tight_layout()
    fig_a.savefig(
        os.path.join(out_dir, "exp_optional_learning_curves_at_a_star.png"),
        bbox_inches="tight",
    )
    plt.close(fig_a)
