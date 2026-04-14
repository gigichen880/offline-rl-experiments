#!/usr/bin/env python3
"""
Hybrid Offline-Online Follower Manipulation — experiments for the Stackelberg-bandit paper.

- Leader: EXP3 on observed leader rewards (same as the paper).
- Follower: tabular Hybrid-FMUCB (Algorithm 1, tabular reduction): worst-response rules
  F_{a,b}, manipulation contrast ∆_{F,a}(·), pooled offline+online means for μ_ℓ and μ_f
  (see hybrid_fmucb.py and Sec. 2 of the paper).
- T_{f,w}: counts rounds where the follower deviates from the true best manipulation rule F^{fm},
  not mere best-response error (paper Sec. 2 / Theorem 1).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from hybrid_fmucb import (
    OfflineRewardStats,
    hybrid_fmucb_pick,
    true_best_manipulation,
)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
for _style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
    try:
        plt.style.use(_style)
        break
    except OSError:
        continue

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 220,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "axes.facecolor": "#fafafa",
        "figure.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
    }
)

COLORS = {
    "baseline": "#C73E1D",
    "hybrid": "#2E86AB",
    "good": "#3A7D44",
    "poor": "#A23B72",
    "neutral": "#7D8590",
    "tabular": "#6B4E71",
    "contextual": "#2A9D8F",
}


def _style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


@dataclass
class StackelbergBandit:
    """Tabular rewards μ_ℓ(a,b), μ_f(a,b) ~ Uniform(0,1)."""

    mu_leader: np.ndarray  # shape (n_a, n_b)
    mu_follower: np.ndarray

    @classmethod
    def sample(cls, n_a: int, n_b: int, rng: np.random.Generator) -> "StackelbergBandit":
        mu_l = rng.uniform(0.0, 1.0, size=(n_a, n_b))
        mu_f = rng.uniform(0.0, 1.0, size=(n_a, n_b))
        return cls(mu_l, mu_f)

    @property
    def n_a(self) -> int:
        return self.mu_leader.shape[0]

    @property
    def n_b(self) -> int:
        return self.mu_leader.shape[1]

    # Follower best response
    def follower_br(self, a: int) -> int:
        return int(np.argmax(self.mu_follower[a]))

    def stackelberg_leader_action(self) -> int:
        br = np.argmax(self.mu_follower, axis=1)
        vals = np.array([self.mu_leader[a, br[a]] for a in range(self.n_a)])
        return int(np.argmax(vals))

    # Leader reward at the follower's best response
    def leader_reward_at_br(self, a: int) -> float:
        b = self.follower_br(a)
        return float(self.mu_leader[a, b])


def exp3_sample(weights: np.ndarray, gamma: float, rng: np.random.Generator) -> int:
    n = len(weights)
    w = np.maximum(weights, 1e-18)
    p = (1.0 - gamma) * (w / w.sum()) + gamma / n
    return int(rng.choice(n, p=p))


def exp3_update(
    weights: np.ndarray,
    a: int,
    reward: float,
    gamma: float,
) -> None:
    n = len(weights)
    w = np.maximum(weights, 1e-18)
    p = (1.0 - gamma) * (w / w.sum()) + gamma / n
    r_hat = reward / max(p[a], 1e-18)
    weights[a] *= np.exp(gamma * r_hat / n)


def _unpack_offline_init(
    offline_init: Optional[Union[OfflineRewardStats, Tuple[np.ndarray, ...]]],
    n_a: int,
    n_b: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pooled offline counts and per-player reward sums (paper Sec. 2)."""
    if offline_init is None:
        return (
            np.zeros((n_a, n_b), dtype=np.int64),
            np.zeros((n_a, n_b), dtype=np.float64),
            np.zeros((n_a, n_b), dtype=np.float64),
        )
    if isinstance(offline_init, OfflineRewardStats):
        return offline_init.n_visits, offline_init.sum_r_f, offline_init.sum_r_l
    n_visits = offline_init[0].astype(np.int64)
    sum_r_f = offline_init[1].astype(np.float64)
    sum_r_l = (
        offline_init[2].astype(np.float64)
        if len(offline_init) > 2
        else np.zeros((n_a, n_b), dtype=np.float64)
    )
    return n_visits, sum_r_f, sum_r_l


def simulate_run(
    env: StackelbergBandit,
    horizon: int,
    rng: np.random.Generator,
    *,
    gamma_exp3: float,
    offline_init: Optional[Union[OfflineRewardStats, Tuple[np.ndarray, ...]]] = None,
    reward_noise_std: float = 0.1,
) -> Dict[str, np.ndarray]:
    """
    One trajectory: leader EXP3; follower tabular Hybrid-FMUCB (Algorithm 1, tabular).

    Pooled means use N_off(a,b)+N_on_t(a,b) in numerator and denominator for both μ_ℓ and μ_f.
    Offline init may be OfflineRewardStats or legacy (n_visits, sum_r_f) only — in the latter
    case leader sums start at zero (online-only μ̂_ℓ).

    ``subopt`` marks deviation from the true best manipulation rule F^{fm} (paper T_{f,w}).
    """
    n_a, n_b = env.n_a, env.n_b
    n_visits, sum_r_f, sum_r_l = _unpack_offline_init(offline_init, n_a, n_b)

    weights = np.ones(n_a, dtype=np.float64)

    a_hist = np.zeros(horizon, dtype=np.int32)
    b_hist = np.zeros(horizon, dtype=np.int32)
    subopt = np.zeros(horizon, dtype=np.bool_)
    leader_regret_inst = np.zeros(horizon, dtype=np.float64)

    a_star = env.stackelberg_leader_action()
    opt_leader_payoff = env.leader_reward_at_br(a_star)
    F_true, _ = true_best_manipulation(env.mu_leader, env.mu_follower)

    for t in range(1, horizon + 1):
        a = exp3_sample(weights, gamma_exp3, rng)
        b = hybrid_fmucb_pick(a, n_visits, sum_r_f, sum_r_l, t, n_a, n_b, rng)

        r_l = env.mu_leader[a, b] + rng.normal(0.0, reward_noise_std)
        r_f = env.mu_follower[a, b] + rng.normal(0.0, reward_noise_std)

        exp3_update(weights, a, float(np.clip(r_l, 0.0, 1.0)), gamma_exp3)

        n_visits[a, b] += 1
        sum_r_f[a, b] += r_f
        sum_r_l[a, b] += r_l

        subopt[t - 1] = b != F_true[a]

        a_hist[t - 1] = a
        b_hist[t - 1] = b
        leader_regret_inst[t - 1] = opt_leader_payoff - float(env.mu_leader[a, b])

    return {
        "subopt": subopt,
        "a_hist": a_hist,
        "b_hist": b_hist,
        "leader_regret_inst": leader_regret_inst,
    }


def build_offline_uniform(
    env: StackelbergBandit,
    n_off: int,
    rng: np.random.Generator,
) -> OfflineRewardStats:
    n_a, n_b = env.n_a, env.n_b
    n_visits = np.zeros((n_a, n_b), dtype=np.int64)
    sum_r_f = np.zeros((n_a, n_b), dtype=np.float64)
    sum_r_l = np.zeros((n_a, n_b), dtype=np.float64)
    for _ in range(n_off):
        a = int(rng.integers(0, n_a))
        b = int(rng.integers(0, n_b))
        r_f = env.mu_follower[a, b] + rng.normal(0.0, 0.1)
        r_l = env.mu_leader[a, b] + rng.normal(0.0, 0.1)
        n_visits[a, b] += 1
        sum_r_f[a, b] += r_f
        sum_r_l[a, b] += r_l
    return OfflineRewardStats(n_visits=n_visits, sum_r_f=sum_r_f, sum_r_l=sum_r_l)


def build_offline_good_coverage(
    env: StackelbergBandit,
    n_off: int,
    rng: np.random.Generator,
) -> OfflineRewardStats:
    """
    Includes the Stackelberg pair (a*, b*) often, but also many (a, BR(a)) for
    random leader actions and uniform (a,b) draws. Heavy mass on only (a*, b*)
    under-covers other rows and hurts the follower when EXP3 visits diverse a;
    this mix matches “optimal region + broad coverage over leader actions.”
    """
    n_a, n_b = env.n_a, env.n_b
    n_visits = np.zeros((n_a, n_b), dtype=np.int64)
    sum_r_f = np.zeros((n_a, n_b), dtype=np.float64)
    sum_r_l = np.zeros((n_a, n_b), dtype=np.float64)
    a_star = env.stackelberg_leader_action()
    b_star = env.follower_br(a_star)

    for _ in range(n_off):
        u = rng.random()
        if u < 0.4:
            a = a_star
            b = b_star
        elif u < 0.8:
            a = int(rng.integers(0, n_a))
            b = env.follower_br(a)
        else:
            a = int(rng.integers(0, n_a))
            b = int(rng.integers(0, n_b))
        r_f = env.mu_follower[a, b] + rng.normal(0.0, 0.1)
        r_l = env.mu_leader[a, b] + rng.normal(0.0, 0.1)
        n_visits[a, b] += 1
        sum_r_f[a, b] += r_f
        sum_r_l[a, b] += r_l
    return OfflineRewardStats(n_visits=n_visits, sum_r_f=sum_r_f, sum_r_l=sum_r_l)


def build_offline_poor_coverage(
    env: StackelbergBandit,
    n_off: int,
    rng: np.random.Generator,
) -> OfflineRewardStats:
    """
    Offline data that **never** contains the Stackelberg pair (a*, b*): when
    a = a*, sample b uniformly from B \\ {b*}; otherwise sample (a, b) uniformly.
    This is a missing-optimal-region condition, not merely “negative examples”
    (which can still be highly informative for elimination).
    """
    n_a, n_b = env.n_a, env.n_b
    n_visits = np.zeros((n_a, n_b), dtype=np.int64)
    sum_r_f = np.zeros((n_a, n_b), dtype=np.float64)
    sum_r_l = np.zeros((n_a, n_b), dtype=np.float64)
    a_star = env.stackelberg_leader_action()
    b_star = env.follower_br(a_star)

    for _ in range(n_off):
        a = int(rng.integers(0, n_a))
        if a == a_star:
            others = [bb for bb in range(n_b) if bb != b_star]
            if others:
                b = int(rng.choice(others))
            else:
                # n_b == 1: cannot exclude b*; resample a so the pair is not (a*, b*)
                while a == a_star:
                    a = int(rng.integers(0, n_a))
                b = int(rng.integers(0, n_b))
        else:
            b = int(rng.integers(0, n_b))
        r_f = env.mu_follower[a, b] + rng.normal(0.0, 0.1)
        r_l = env.mu_leader[a, b] + rng.normal(0.0, 0.1)
        n_visits[a, b] += 1
        sum_r_f[a, b] += r_f
        sum_r_l[a, b] += r_l
    return OfflineRewardStats(n_visits=n_visits, sum_r_f=sum_r_f, sum_r_l=sum_r_l)


def convergence_round(
    subopt: np.ndarray,
    window: int = 200,
    threshold: float = 0.2,
    k_sustain: int = 3,
) -> int:
    """
    First round index t (end of an online window) such that the rolling
    suboptimal-play rate is at most `threshold` for `k_sustain` consecutive
    window ends. Uses an absolute threshold so methods that already start
    with a low rate are not penalized (unlike "halve the initial window").
    """
    h = len(subopt)
    min_t = window + k_sustain - 1
    if h < min_t:
        return h
    for t in range(min_t, h + 1):
        if all(
            float(subopt[t - s - window : t - s].mean()) <= threshold
            for s in range(k_sustain)
        ):
            return t
    return h


def ci_mean(
    data: np.ndarray,
    axis: int = 0,
    z: float = 1.96,
) -> Tuple[np.ndarray, np.ndarray]:
    m = data.mean(axis=axis)
    s = data.std(axis=axis, ddof=1) / np.sqrt(max(1, data.shape[axis]))
    return m, z * s


def follower_subopt_rate_when_leader_a_star(
    tr: Dict[str, np.ndarray],
    env: StackelbergBandit,
) -> float:
    """
    Fraction of rounds with manipulation mistakes (vs. true F^fm), among rounds with
    leader action a* (Stackelberg leader action). Aligns with theory about
    identifying optimal manipulation at a*; NaN if a* is never played.
    """
    a_star = int(env.stackelberg_leader_action())
    a_hist = tr["a_hist"]
    subopt = tr["subopt"]
    mask = a_hist == a_star
    if not np.any(mask):
        return float("nan")
    return float(subopt[mask].mean())


def ci_mean_nan_1d(data: np.ndarray, z: float = 1.96) -> Tuple[float, float]:
    """Mean and 95% CI half-width, ignoring non-finite values."""
    x = data[np.isfinite(data)]
    if x.size == 0:
        return float("nan"), float("nan")
    m = float(x.mean())
    if x.size <= 1:
        return m, 0.0
    s = float(x.std(ddof=1)) / np.sqrt(x.size)
    return m, z * s


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
) -> None:
    n_off_list = list(n_off_grid)
    cum_target = max(n_off_list) if cum_n_off is None else cum_n_off
    if cum_target not in n_off_list:
        cum_target = max(n_off_list)

    # We assume it converged if error rate ≤ 20% for 3 consecutive windows of 200 rounds
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
        env = StackelbergBandit.sample(n_a, n_b, rng)

        for j, n_off in enumerate(n_off_list):
            rng_b = np.random.default_rng(seed * 100_000 + j + 7)
            rng_h = np.random.default_rng(seed * 100_000 + j + 13)

            off = build_offline_uniform(env, n_off, rng_b) if n_off > 0 else None

            tr_b = simulate_run(env, horizon, rng_b, gamma_exp3=gamma_exp3, offline_init=None)
            tr_h = simulate_run(env, horizon, rng_h, gamma_exp3=gamma_exp3, offline_init=off)

            t_fw_base[si, j] = tr_b["subopt"].sum()
            t_fw_hyb[si, j] = tr_h["subopt"].sum()
            conv_base[si, j] = convergence_round(
                tr_b["subopt"], window=conv_win, threshold=conv_thr, k_sustain=conv_k
            )
            conv_hyb[si, j] = convergence_round(
                tr_h["subopt"], window=conv_win, threshold=conv_thr, k_sustain=conv_k
            )

            if n_off == cum_target:
                cum_base_rows.append(np.cumsum(tr_b["subopt"].astype(np.float64)))
                cum_hyb_rows.append(np.cumsum(tr_h["subopt"].astype(np.float64)))

    x_labels = [str(n) for n in n_off_list]
    x_plot = np.array(n_off_list, dtype=np.float64) + 1.0

    # --- Plot 1: N_off vs T_{f,w}
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)
    mb, eb = ci_mean(t_fw_base, axis=0)
    mh, eh = ci_mean(t_fw_hyb, axis=0)
    ax.plot(
        x_plot,
        mb,
        "o-",
        color=COLORS["baseline"],
        lw=3,
        ms=8,
        label="FMUCB (online only)",
    )
    ax.fill_between(x_plot, mb - eb, mb + eb, color=COLORS["baseline"], alpha=0.18)
    ax.plot(
        x_plot,
        mh,
        "s-",
        color=COLORS["hybrid"],
        lw=3,
        ms=8,
        label="Hybrid-FMUCB",
    )
    ax.fill_between(x_plot, mh - eh, mh + eh, color=COLORS["hybrid"], alpha=0.18)
    ax.set_xticks(x_plot)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel(r"Offline dataset size $N_{\mathrm{off}}$")
    ax.set_ylabel(r"$T_{f,w}$ (mistakes vs.\ true best manipulation $F^{fm}$)")
    ax.set_title(r"Experiment 1: Offline data size vs $T_{f,w}$")
    ax.legend(frameon=True, fancybox=True, shadow=True, loc="upper right")
    _style_axis(ax)
    fig.tight_layout()
    p1 = os.path.join(out_dir, "exp1_noff_vs_tfw.png")
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)

    # --- Plot 2: rounds to sustained low rolling subopt. rate (absolute threshold)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)
    cb, cbe = ci_mean(conv_base, axis=0)
    ch, che = ci_mean(conv_hyb, axis=0)
    ax.plot(x_plot, cb, "o-", color=COLORS["baseline"], lw=3, ms=8, label="FMUCB")
    ax.fill_between(x_plot, cb - cbe, cb + cbe, color=COLORS["baseline"], alpha=0.18)
    ax.plot(x_plot, ch, "s-", color=COLORS["hybrid"], lw=3, ms=8, label="Hybrid-FMUCB")
    ax.fill_between(x_plot, ch - che, ch + che, color=COLORS["hybrid"], alpha=0.18)
    ax.set_xticks(x_plot)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel(r"$N_{\mathrm{off}}$")
    ax.set_ylabel("Rounds to reach low rolling error rate (lower is better)")
    ax.set_title(
        rf"Experiment 1: Sustained low subopt. rate ($\leq {conv_thr}$, "
        rf"{conv_k} consecutive windows of ${conv_win}$ rounds)"
    )
    ax.legend(frameon=True, fancybox=True, shadow=True)
    _style_axis(ax)
    fig.tight_layout()
    p2 = os.path.join(out_dir, "exp1_noff_vs_convergence.png")
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)

    # --- Plot 3: cumulative suboptimal plays vs time (fixed N_off for comparison)
    stack_cb = np.stack(cum_base_rows, axis=0)
    stack_ch = np.stack(cum_hyb_rows, axis=0)
    t_axis = np.arange(1, horizon + 1)
    mcb, ecb = ci_mean(stack_cb, axis=0)
    mch, ech = ci_mean(stack_ch, axis=0)
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.grid(True, which="both", alpha=0.3)
    ax.plot(t_axis, mcb, "-", color=COLORS["baseline"], lw=3, label="FMUCB (online only)")
    ax.fill_between(t_axis, mcb - ecb, mcb + ecb, color=COLORS["baseline"], alpha=0.18)
    ax.plot(t_axis, mch, "-", color=COLORS["hybrid"], lw=3, label="Hybrid-FMUCB")
    ax.fill_between(t_axis, mch - ech, mch + ech, color=COLORS["hybrid"], alpha=0.18)
    ax.set_xlabel("Round $t$")
    ax.set_ylabel(r"Cumulative manipulation mistakes (vs.\ $F^{fm}$)")
    ax.set_title(
        rf"Experiment 1: Cumulative mistakes ($N_{{\mathrm{{off}}}}={cum_target}$)"
    )
    ax.legend(frameon=True, fancybox=True, shadow=True, loc="lower right")
    _style_axis(ax)
    fig.tight_layout()
    p3 = os.path.join(out_dir, "exp1_cumulative_mistakes.png")
    fig.savefig(p3, bbox_inches="tight")
    plt.close(fig)

    # Save numbers
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


def experiment2(
    out_dir: str,
    seeds: Sequence[int],
    n_a: int,
    n_b: int,
    horizon: int,
    gamma_exp3: float,
    n_off_fixed: int,
    progress: bool = True,
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
            env = StackelbergBandit.sample(n_a, n_b, rng)
            rng_off = np.random.default_rng(seed + 91_000)
            rng_on = np.random.default_rng(seed + 92_000)
            if kind == "good":
                off = build_offline_good_coverage(env, n_off_fixed, rng_off)
            elif kind == "neutral":
                off = build_offline_uniform(env, n_off_fixed, rng_off)
            else:
                off = build_offline_poor_coverage(env, n_off_fixed, rng_off)
            tr = simulate_run(env, horizon, rng_on, gamma_exp3=gamma_exp3, offline_init=off)
            t_fw[kind][si] = tr["subopt"].sum()
            w = min(500, horizon)
            success[kind][si] = 1.0 - tr["subopt"][-w:].mean()
            sub_at_star[kind][si] = follower_subopt_rate_when_leader_a_star(tr, env)

    labels = [
        "Good (includes\n$(a^*, b^*)$)",
        "Neutral\n(uniform)",
        "Poor (never\n$(a^*, b^*)$)",
    ]
    colors_b = [COLORS["good"], COLORS["neutral"], COLORS["poor"]]
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
        m, e = ci_mean_nan_1d(sub_at_star[k])
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
    _style_axis(axes[0])
    _style_axis(axes[1])
    _style_axis(axes[2])
    fig.tight_layout()
    p = os.path.join(out_dir, "exp2_coverage_bars.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(out_dir, "exp2_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_off_fixed": n_off_fixed,
                "good_coverage": "40pct_stackelberg_pair_40pct_br_uniform_a_20pct_uniform_ab",
                "neutral_coverage": "uniform_ab",
                "poor_coverage": "never_sample_stackelberg_pair_a_star_b_star_else_uniform",
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


def learning_curve_figure(
    out_dir: str,
    seeds: Sequence[int],
    n_a: int,
    n_b: int,
    horizon: int,
    gamma_exp3: float,
    n_off: int,
    progress: bool = True,
) -> None:
    """Optional: rolling suboptimality rate over time (baseline vs hybrid)."""
    curves_b: List[np.ndarray] = []
    curves_h: List[np.ndarray] = []
    win = max(50, horizon // 100)

    for seed in tqdm(
        seeds,
        desc="Learning curves (optional)",
        unit="seed",
        disable=not progress,
    ):
        rng = np.random.default_rng(seed)
        env = StackelbergBandit.sample(n_a, n_b, rng)
        rng_b = np.random.default_rng(seed + 3)
        rng_h = np.random.default_rng(seed + 5)
        off = build_offline_uniform(env, n_off, rng_h) if n_off > 0 else None
        tr_b = simulate_run(env, horizon, rng_b, gamma_exp3=gamma_exp3, offline_init=None)
        tr_h = simulate_run(env, horizon, rng_h, gamma_exp3=gamma_exp3, offline_init=off)
        sub_b = tr_b["subopt"].astype(np.float64)
        sub_h = tr_h["subopt"].astype(np.float64)
        roll_b = np.convolve(sub_b, np.ones(win) / win, mode="valid")
        roll_h = np.convolve(sub_h, np.ones(win) / win, mode="valid")
        curves_b.append(roll_b)
        curves_h.append(roll_h)

    stack_b = np.stack(curves_b, axis=0)
    stack_h = np.stack(curves_h, axis=0)
    t_axis = np.arange(win, horizon + 1)
    mb, eb = ci_mean(stack_b, axis=0)
    mh, eh = ci_mean(stack_h, axis=0)

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.plot(t_axis, mb, color=COLORS["baseline"], lw=2.0, label="FMUCB")
    ax.fill_between(t_axis, mb - eb, mb + eb, color=COLORS["baseline"], alpha=0.2)
    ax.plot(t_axis, mh, color=COLORS["hybrid"], lw=2.0, label="Hybrid-FMUCB")
    ax.fill_between(t_axis, mh - eh, mh + eh, color=COLORS["hybrid"], alpha=0.2)
    ax.set_xlabel("Round $t$")
    ax.set_ylabel(f"Rolling mean manipulation-mistake rate (window={win})")
    ax.set_title(rf"Learning curves (optional): $N_{{\mathrm{{off}}}}={n_off}$")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.0)
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "exp_optional_learning_curves.png"), bbox_inches="tight")
    plt.close(fig)


# --- Experiment 3: contextual extension (linear φ vs tabular per context) -----------------


def phi_vec(x: int, a: int, b: int, n_x: int, n_a: int, n_b: int) -> np.ndarray:
    d = n_x + n_a + n_b
    v = np.zeros(d)
    v[x] = 1.0
    v[n_x + a] = 1.0
    v[n_x + n_a + b] = 1.0
    return v


def _mu_from_theta(theta: np.ndarray, x: int, a: int, b: int, n_x: int, n_a: int, n_b: int) -> float:
    return float(np.clip(theta @ phi_vec(x, a, b, n_x, n_a, n_b), 0.0, 1.0))


def build_br_table(theta_f: np.ndarray, n_x: int, n_a: int, n_b: int) -> np.ndarray:
    br = np.zeros((n_x, n_a), dtype=np.int32)
    for x in range(n_x):
        for a in range(n_a):
            vals = np.array(
                [_mu_from_theta(theta_f, x, a, b, n_x, n_a, n_b) for b in range(n_b)],
                dtype=np.float64,
            )
            br[x, a] = int(np.argmax(vals))
    return br


def build_offline_contextual_tabular(
    theta_f: np.ndarray,
    n_off: int,
    rng: np.random.Generator,
    n_x: int,
    n_a: int,
    n_b: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n_visits = np.zeros((n_x, n_a, n_b), dtype=np.int64)
    sum_r = np.zeros((n_x, n_a, n_b), dtype=np.float64)
    for _ in range(n_off):
        x = int(rng.integers(0, n_x))
        a = int(rng.integers(0, n_a))
        b = int(rng.integers(0, n_b))
        r_f = _mu_from_theta(theta_f, x, a, b, n_x, n_a, n_b) + rng.normal(0.0, 0.1)
        n_visits[x, a, b] += 1
        sum_r[x, a, b] += r_f
    return n_visits, sum_r


def build_offline_contextual_linear(
    theta_f: np.ndarray,
    n_off: int,
    rng: np.random.Generator,
    n_x: int,
    n_a: int,
    n_b: int,
    ridge_reg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    d = n_x + n_a + n_b
    a_mat = ridge_reg * np.eye(d)
    b_vec = np.zeros(d)
    for _ in range(n_off):
        x = int(rng.integers(0, n_x))
        a = int(rng.integers(0, n_a))
        b = int(rng.integers(0, n_b))
        ph = phi_vec(x, a, b, n_x, n_a, n_b)
        r_f = _mu_from_theta(theta_f, x, a, b, n_x, n_a, n_b) + rng.normal(0.0, 0.1)
        a_mat += np.outer(ph, ph)
        b_vec += ph * r_f
    return a_mat, b_vec


def simulate_contextual_tabular_hybrid(
    theta_l: np.ndarray,
    theta_f: np.ndarray,
    n_x: int,
    n_a: int,
    n_b: int,
    horizon: int,
    rng: np.random.Generator,
    gamma_exp3: float,
    reward_noise_std: float,
    offline_init: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> np.ndarray:
    """
    Per-context UCB on μ_f (not full contextual Hybrid-FMUCB / Alg. 2).
    ``subopt`` is best-response error, not manipulation-track F^{fm}.
    """
    br = build_br_table(theta_f, n_x, n_a, n_b)
    n_visits = np.zeros((n_x, n_a, n_b), dtype=np.int64)
    sum_r = np.zeros((n_x, n_a, n_b), dtype=np.float64)
    if offline_init is not None:
        n_visits += offline_init[0]
        sum_r += offline_init[1]
    weights = np.ones((n_x, n_a), dtype=np.float64)
    subopt = np.zeros(horizon, dtype=np.bool_)

    for t in range(1, horizon + 1):
        x = int(rng.integers(0, n_x))
        a = exp3_sample(weights[x], gamma_exp3, rng)
        na = np.maximum(1, n_visits[x, a].astype(np.float64))
        mean = sum_r[x, a] / na
        bonus = np.sqrt(2.0 * np.log(max(1, t)) / na)
        ucb = mean + bonus
        maxv = ucb.max()
        cands = np.flatnonzero(np.isclose(ucb, maxv))
        b = int(rng.choice(cands))

        r_l = _mu_from_theta(theta_l, x, a, b, n_x, n_a, n_b) + rng.normal(0.0, reward_noise_std)
        r_f = _mu_from_theta(theta_f, x, a, b, n_x, n_a, n_b) + rng.normal(0.0, reward_noise_std)
        exp3_update(weights[x], a, float(np.clip(r_l, 0.0, 1.0)), gamma_exp3)
        n_visits[x, a, b] += 1
        sum_r[x, a, b] += r_f
        subopt[t - 1] = b != br[x, a]
    return subopt


def simulate_contextual_linucb_hybrid(
    theta_l: np.ndarray,
    theta_f: np.ndarray,
    n_x: int,
    n_a: int,
    n_b: int,
    horizon: int,
    rng: np.random.Generator,
    gamma_exp3: float,
    reward_noise_std: float,
    offline_init: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    linucb_alpha: float = 0.85,
    ridge_reg: float = 1.0,
) -> np.ndarray:
    d = n_x + n_a + n_b
    br = build_br_table(theta_f, n_x, n_a, n_b)
    if offline_init is not None:
        a_mat, b_vec = offline_init
    else:
        a_mat = ridge_reg * np.eye(d)
        b_vec = np.zeros(d)
    weights = np.ones((n_x, n_a), dtype=np.float64)
    subopt = np.zeros(horizon, dtype=np.bool_)

    for t in range(1, horizon + 1):
        x = int(rng.integers(0, n_x))
        a = exp3_sample(weights[x], gamma_exp3, rng)
        inv_a = np.linalg.inv(a_mat)
        theta_hat = inv_a @ b_vec
        best_b = 0
        best_score = -np.inf
        for bb in range(n_b):
            ph = phi_vec(x, a, bb, n_x, n_a, n_b)
            pred = float(theta_hat @ ph)
            conf = float(linucb_alpha * np.sqrt(max(0.0, ph @ inv_a @ ph)))
            score = pred + conf
            if score > best_score:
                best_score = score
                best_b = bb
        b = best_b

        r_l = _mu_from_theta(theta_l, x, a, b, n_x, n_a, n_b) + rng.normal(0.0, reward_noise_std)
        r_f = _mu_from_theta(theta_f, x, a, b, n_x, n_a, n_b) + rng.normal(0.0, reward_noise_std)
        exp3_update(weights[x], a, float(np.clip(r_l, 0.0, 1.0)), gamma_exp3)
        ph = phi_vec(x, a, b, n_x, n_a, n_b)
        a_mat = a_mat + np.outer(ph, ph)
        b_vec = b_vec + ph * r_f
        subopt[t - 1] = b != br[x, a]
    return subopt


def experiment3_contextual(
    out_dir: str,
    seeds: Sequence[int],
    horizon: int,
    gamma_exp3: float,
    n_off: int,
    n_x: int,
    n_a: int,
    n_b: int,
    progress: bool = True,
) -> None:
    d = n_x + n_a + n_b
    tfw_tab: List[float] = []
    tfw_ctx: List[float] = []
    gen_tab: List[float] = []
    gen_ctx: List[float] = []

    for seed in tqdm(seeds, desc="Exp 3 (contextual)", unit="seed", disable=not progress):
        rng = np.random.default_rng(seed)
        theta_l = rng.uniform(-1.0, 1.0, d)
        theta_f = rng.uniform(-1.0, 1.0, d)
        rng_tab = np.random.default_rng(seed + 17)
        rng_ctx = np.random.default_rng(seed + 29)
        off_tab = build_offline_contextual_tabular(theta_f, n_off, rng_tab, n_x, n_a, n_b)
        off_ctx = build_offline_contextual_linear(theta_f, n_off, rng_ctx, n_x, n_a, n_b, ridge_reg=1.0)
        sub_tab = simulate_contextual_tabular_hybrid(
            theta_l,
            theta_f,
            n_x,
            n_a,
            n_b,
            horizon,
            rng_tab,
            gamma_exp3,
            0.1,
            offline_init=off_tab,
        )
        sub_ctx = simulate_contextual_linucb_hybrid(
            theta_l,
            theta_f,
            n_x,
            n_a,
            n_b,
            horizon,
            rng_ctx,
            gamma_exp3,
            0.1,
            offline_init=off_ctx,
        )
        tfw_tab.append(float(sub_tab.sum()))
        tfw_ctx.append(float(sub_ctx.sum()))
        w = min(500, horizon)
        gen_tab.append(float(1.0 - sub_tab[-w:].mean()))
        gen_ctx.append(float(1.0 - sub_ctx[-w:].mean()))

    arr_tab = np.array(tfw_tab)
    arr_ctx = np.array(tfw_ctx)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    xpos = np.arange(2)
    means = [arr_tab.mean(), arr_ctx.mean()]
    errs = [
        1.96 * arr_tab.std(ddof=1) / np.sqrt(len(seeds)),
        1.96 * arr_ctx.std(ddof=1) / np.sqrt(len(seeds)),
    ]
    bars = ax.bar(
        xpos,
        means,
        yerr=errs,
        color=[COLORS["tabular"], COLORS["contextual"]],
        edgecolor="white",
        linewidth=1.2,
        capsize=6,
        error_kw={"linewidth": 1.5},
        width=0.55,
    )
    ax.set_xticks(xpos)
    ax.set_xticklabels(["Tabular Hybrid-FMUCB", "Contextual Hybrid-FMUCB (LinUCB)"])
    ax.set_ylabel(r"Follower error count (not $F^{fm}$; BR subopt.)")
    ax.set_title(
        rf"Experiment 3 (optional): contextual extension ($N_{{\mathrm{{off}}}}={n_off}$, "
        rf"$|X|={n_x}$, $|A|={n_a}$, $|B|={n_b}$)"
    )
    for rect, m in zip(bars, means):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            m + max(errs) * 0.12,
            f"{m:.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "exp3_contextual_vs_tabular.png"), bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(out_dir, "exp3_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_off": n_off,
                "n_x": n_x,
                "n_a": n_a,
                "n_b": n_b,
                "tfw_tabular_mean": float(arr_tab.mean()),
                "tfw_contextual_mean": float(arr_ctx.mean()),
                "final_window_optimality_tabular_mean": float(np.mean(gen_tab)),
                "final_window_optimality_contextual_mean": float(np.mean(gen_ctx)),
            },
            f,
            indent=2,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stackelberg bandit offline-online experiments")
    parser.add_argument("--out-dir", type=str, default="figures", help="Output directory for plots")
    parser.add_argument("--seeds", type=int, default=24, help="Number of random seeds")
    parser.add_argument("--base-seed", type=int, default=0, help="Base seed offset")
    parser.add_argument("--n-a", type=int, default=8, help="|A| (leader actions)")
    parser.add_argument("--n-b", type=int, default=8, help="|B| (follower actions)")
    parser.add_argument("--horizon", type=int, default=8000, help="Online rounds")
    parser.add_argument("--gamma-exp3", type=float, default=0.05, help=r"EXP3 exploration $\gamma$")
    parser.add_argument(
        "--exp1-cum-n-off",
        type=int,
        default=None,
        help="Offline size for cumulative-mistakes curve (must be in --n-off-grid; default: max grid value)",
    )
    parser.add_argument("--exp2-n-off", type=int, default=1000, help="Fixed offline size for Exp. 2")
    parser.add_argument("--skip-learning-curves", action="store_true")
    parser.add_argument(
        "--exp3",
        action="store_true",
        help="Run optional Experiment 3 (contextual vs tabular hybrid)",
    )
    parser.add_argument("--exp3-n-off", type=int, default=1500, help="Offline size for Exp. 3")
    parser.add_argument("--n-x", type=int, default=5, help="Number of contexts |X| (Exp. 3)")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars (e.g. for logs or CI)",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seeds = [args.base_seed + i for i in range(args.seeds)]
    n_off_grid = [0, 100, 500, 1000, 5000]
    show_p = not args.no_progress

    experiment1(
        args.out_dir,
        seeds,
        args.n_a,
        args.n_b,
        args.horizon,
        args.gamma_exp3,
        n_off_grid,
        cum_n_off=args.exp1_cum_n_off,
        progress=show_p,
    )
    experiment2(
        args.out_dir,
        seeds,
        args.n_a,
        args.n_b,
        args.horizon,
        args.gamma_exp3,
        args.exp2_n_off,
        progress=show_p,
    )
    if not args.skip_learning_curves:
        learning_curve_figure(
            args.out_dir,
            seeds,
            args.n_a,
            args.n_b,
            args.horizon,
            args.gamma_exp3,
            n_off=1000,
            progress=show_p,
        )

    if args.exp3:
        experiment3_contextual(
            args.out_dir,
            seeds,
            args.horizon,
            args.gamma_exp3,
            args.exp3_n_off,
            args.n_x,
            args.n_a,
            args.n_b,
            progress=show_p,
        )

    print(f"Wrote figures to {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
