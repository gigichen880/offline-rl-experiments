#!/usr/bin/env python3
"""CLI: default runs Exp 1 + Exp 2 + learning curves; use --exp1-only / --exp2-only / --learning-curves-only to restrict."""

from __future__ import annotations

import argparse
import json
import os
import sys

# stderr is unbuffered on most platforms; stdout may wait until first newline
print("run_experiments: importing numpy/matplotlib…", file=sys.stderr, flush=True)
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from hybrid_fmucb import (
    OfflineRewardStats,
    build_rule_F,
    hybrid_fmucb_pick,
    manipulation_contrast,
    offline_candidate_manipulation,
    pooled_mean,
    regression_confidence_radius,
    theorem1_offline_transfer_check,
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

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 220,
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.grid": True, "grid.alpha": 0.35, "grid.linestyle": "--",
    "axes.facecolor": "#fafafa", "figure.facecolor": "white",
    "axes.edgecolor": "#333333", "axes.linewidth": 1.0,
})

COLORS = {
    "baseline": "#C73E1D", "hybrid": "#2E86AB",
    "good": "#3A7D44", "poor": "#A23B72", "neutral": "#7D8590",
    "tabular": "#6B4E71", "contextual": "#2A9D8F",
}


def _style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

@dataclass
class StackelbergBandit:
    """Tabular rewards mu_l(a,b), mu_f(a,b) ~ Uniform(0,1)."""

    mu_leader: np.ndarray    # shape (n_a, n_b)
    mu_follower: np.ndarray  # shape (n_a, n_b)

    @classmethod
    def sample(cls, n_a: int, n_b: int, rng: np.random.Generator) -> "StackelbergBandit":
        return cls(
            rng.uniform(0.0, 1.0, (n_a, n_b)),
            rng.uniform(0.0, 1.0, (n_a, n_b)),
        )

    @property
    def n_a(self) -> int:
        return self.mu_leader.shape[0]

    @property
    def n_b(self) -> int:
        return self.mu_leader.shape[1]

    def follower_br(self, a: int) -> int:
        return int(np.argmax(self.mu_follower[a]))

    def stackelberg_leader_action(self) -> int:
        br = np.argmax(self.mu_follower, axis=1)
        vals = np.array([self.mu_leader[a, br[a]] for a in range(self.n_a)])
        return int(np.argmax(vals))

    def leader_reward_at_br(self, a: int) -> float:
        return float(self.mu_leader[a, self.follower_br(a)])


# ---------------------------------------------------------------------------
# Bernoulli sampling helper (Deviation 1)
# ---------------------------------------------------------------------------

def bernoulli_sample(mu: float, rng: np.random.Generator) -> float:
    """r ~ Ber(mu) as in paper Sec. 2.  Returns 0.0 or 1.0."""
    return float(rng.random() < np.clip(mu, 0.0, 1.0))


# ---------------------------------------------------------------------------
# EXP3
# ---------------------------------------------------------------------------

def exp3_sample(weights: np.ndarray, gamma: float, rng: np.random.Generator) -> int:
    n = len(weights)
    w = np.maximum(weights, 1e-18)
    p = (1.0 - gamma) * (w / w.sum()) + gamma / n
    return int(rng.choice(n, p=p))


def exp3_update(weights: np.ndarray, a: int, reward: float, gamma: float) -> None:
    n = len(weights)
    w = np.maximum(weights, 1e-18)
    p = (1.0 - gamma) * (w / w.sum()) + gamma / n
    r_hat = reward / max(p[a], 1e-18)
    weights[a] *= np.exp(gamma * r_hat / n)


# ---------------------------------------------------------------------------
# Offline dataset builders -- all use Bernoulli rewards (Deviation 1)
# ---------------------------------------------------------------------------

def _sample_bernoulli_pair(
    mu_l: float, mu_f: float, rng: np.random.Generator
) -> Tuple[float, float]:
    return bernoulli_sample(mu_l, rng), bernoulli_sample(mu_f, rng)


def build_offline_uniform(
    env: StackelbergBandit, n_off: int, rng: np.random.Generator
) -> OfflineRewardStats:
    n_a, n_b = env.n_a, env.n_b
    nv = np.zeros((n_a, n_b), dtype=np.int64)
    sf = np.zeros((n_a, n_b)); sl = np.zeros((n_a, n_b))
    for _ in range(n_off):
        a = int(rng.integers(0, n_a)); b = int(rng.integers(0, n_b))
        rl, rf = _sample_bernoulli_pair(env.mu_leader[a, b], env.mu_follower[a, b], rng)
        nv[a, b] += 1; sf[a, b] += rf; sl[a, b] += rl
    return OfflineRewardStats(nv, sf, sl)


def build_offline_good_coverage(
    env: StackelbergBandit, n_off: int, rng: np.random.Generator
) -> OfflineRewardStats:
    n_a, n_b = env.n_a, env.n_b
    nv = np.zeros((n_a, n_b), dtype=np.int64)
    sf = np.zeros((n_a, n_b)); sl = np.zeros((n_a, n_b))
    a_star = env.stackelberg_leader_action(); b_star = env.follower_br(a_star)
    for _ in range(n_off):
        u = rng.random()
        if u < 0.4:
            a, b = a_star, b_star
        elif u < 0.8:
            a = int(rng.integers(0, n_a)); b = env.follower_br(a)
        else:
            a = int(rng.integers(0, n_a)); b = int(rng.integers(0, n_b))
        rl, rf = _sample_bernoulli_pair(env.mu_leader[a, b], env.mu_follower[a, b], rng)
        nv[a, b] += 1; sf[a, b] += rf; sl[a, b] += rl
    return OfflineRewardStats(nv, sf, sl)


def build_offline_poor_coverage(
    env: StackelbergBandit, n_off: int, rng: np.random.Generator
) -> OfflineRewardStats:
    n_a, n_b = env.n_a, env.n_b
    nv = np.zeros((n_a, n_b), dtype=np.int64)
    sf = np.zeros((n_a, n_b)); sl = np.zeros((n_a, n_b))
    a_star = env.stackelberg_leader_action(); b_star = env.follower_br(a_star)
    for _ in range(n_off):
        a = int(rng.integers(0, n_a))
        if a == a_star:
            others = [b for b in range(n_b) if b != b_star]
            if others:
                b = int(rng.choice(others))
            else:
                while a == a_star:
                    a = int(rng.integers(0, n_a))
                b = int(rng.integers(0, n_b))
        else:
            b = int(rng.integers(0, n_b))
        rl, rf = _sample_bernoulli_pair(env.mu_leader[a, b], env.mu_follower[a, b], rng)
        nv[a, b] += 1; sf[a, b] += rf; sl[a, b] += rl
    return OfflineRewardStats(nv, sf, sl)


# ---------------------------------------------------------------------------
# Main simulation loop (Bernoulli rewards, regression confidence sets)
# ---------------------------------------------------------------------------

def _unpack_offline(
    offline_init: Optional[OfflineRewardStats],
    n_a: int,
    n_b: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if offline_init is None:
        return (
            np.zeros((n_a, n_b), dtype=np.int64),
            np.zeros((n_a, n_b)),
            np.zeros((n_a, n_b)),
        )
    return offline_init.n_visits, offline_init.sum_r_f, offline_init.sum_r_l


def simulate_run(
    env: StackelbergBandit,
    horizon: int,
    rng: np.random.Generator,
    *,
    gamma_exp3: float,
    offline_init: Optional[OfflineRewardStats] = None,
    delta: float = 0.05,
    progress_rounds: bool = False,
    progress_desc: str = "round",
) -> Dict[str, np.ndarray]:
    """
    One trajectory: leader EXP3, follower tabular Hybrid-FMUCB (Algorithm 1).

    Rewards are Bernoulli: r_l ~ Ber(mu_l(a,b)), r_f ~ Ber(mu_f(a,b)).
    Confidence sets use regression_confidence_radius (Deviation 2).
    T_{f,w} = sum_t 1{b_t != F^fm(a_t)}.

    Set ``progress_rounds=True`` for a per-round tqdm bar (e.g. learning-curve runs).
    """
    n_a, n_b = env.n_a, env.n_b
    n_visits, sum_r_f, sum_r_l = _unpack_offline(offline_init, n_a, n_b)
    weights = np.ones(n_a)
    F_true, _ = true_best_manipulation(env.mu_leader, env.mu_follower)
    opt_payoff = env.leader_reward_at_br(env.stackelberg_leader_action())

    offline_m0_nonempty = np.array([np.nan])
    theorem1_transfer_ok = np.array([np.nan])
    theorem1_transfer_lhs = np.array([np.nan])
    theorem1_transfer_threshold = np.array([np.nan])
    theorem1_delta3 = np.array([np.nan])
    if offline_init is not None:
        cand_F, _, _, _ = offline_candidate_manipulation(
            sum_r_l, sum_r_f, n_visits, n_a, n_b, delta
        )
        offline_m0_nonempty = np.array([1.0 if cand_F is not None else 0.0])
        ok, lhs, thr, d3 = theorem1_offline_transfer_check(
            env.mu_leader, env.mu_follower, n_visits, sum_r_l, n_a, n_b
        )
        theorem1_transfer_ok = np.array([1.0 if ok else 0.0])
        theorem1_transfer_lhs = np.array([lhs])
        theorem1_transfer_threshold = np.array([thr])
        theorem1_delta3 = np.array([d3])

    a_hist = np.zeros(horizon, dtype=np.int32)
    b_hist = np.zeros(horizon, dtype=np.int32)
    subopt = np.zeros(horizon, dtype=np.bool_)
    leader_regret = np.zeros(horizon)
    fallback_count = 0

    if progress_rounds:
        round_iter = tqdm(
            range(1, horizon + 1),
            desc=progress_desc,
            leave=False,
            unit="t",
            total=horizon,
            mininterval=0.25,
        )
    else:
        round_iter = range(1, horizon + 1)
    for t in round_iter:
        a = exp3_sample(weights, gamma_exp3, rng)
        b, used_fallback = hybrid_fmucb_pick(
            a, n_visits, sum_r_f, sum_r_l, t, n_a, n_b, rng, delta=delta
        )
        if used_fallback:
            fallback_count += 1

        # Bernoulli rewards (Deviation 1)
        r_l = bernoulli_sample(env.mu_leader[a, b], rng)
        r_f = bernoulli_sample(env.mu_follower[a, b], rng)

        exp3_update(weights, a, r_l, gamma_exp3)
        n_visits[a, b] += 1
        sum_r_f[a, b] += r_f
        sum_r_l[a, b] += r_l

        subopt[t - 1] = (b != F_true[a])
        a_hist[t - 1] = a
        b_hist[t - 1] = b
        leader_regret[t - 1] = opt_payoff - env.mu_leader[a, b]

    return {
        "subopt": subopt,
        "a_hist": a_hist,
        "b_hist": b_hist,
        "leader_regret": leader_regret,
        "fallback_count": fallback_count,
        "offline_m0_nonempty": offline_m0_nonempty,
        "theorem1_transfer_ok": theorem1_transfer_ok,
        "theorem1_transfer_lhs": theorem1_transfer_lhs,
        "theorem1_transfer_threshold": theorem1_transfer_threshold,
        "theorem1_delta3": theorem1_delta3,
    }


# ---------------------------------------------------------------------------
# Convergence metric and CI helpers
# ---------------------------------------------------------------------------

def convergence_round(
    subopt: np.ndarray,
    window: int = 200,
    threshold: float = 0.2,
    k_sustain: int = 3,
) -> int:
    h = len(subopt)
    min_t = window + k_sustain - 1
    if h < min_t:
        return h
    for t in range(min_t, h + 1):
        if all(
            float(subopt[t - s - window: t - s].mean()) <= threshold
            for s in range(k_sustain)
        ):
            return t
    return h


def ci_mean(
    data: np.ndarray, axis: int = 0, z: float = 1.96
) -> Tuple[np.ndarray, np.ndarray]:
    m = data.mean(axis=axis)
    s = data.std(axis=axis, ddof=1) / np.sqrt(max(1, data.shape[axis]))
    return m, z * s


def ci_mean_nan_1d(data: np.ndarray, z: float = 1.96) -> Tuple[float, float]:
    x = data[np.isfinite(data)]
    if x.size == 0:
        return float("nan"), float("nan")
    m = float(x.mean())
    if x.size <= 1:
        return m, 0.0
    return m, z * float(x.std(ddof=1)) / np.sqrt(x.size)


def ci_mean_nan(
    data: np.ndarray, axis: int = 0, z: float = 1.96
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and ~95% CI row-wise, ignoring NaNs (effective n varies)."""
    m = np.nanmean(data, axis=axis)
    valid = np.sum(np.isfinite(data), axis=axis).astype(np.float64)
    valid = np.maximum(valid, 1.0)
    std = np.nanstd(data, axis=axis, ddof=1)
    std = np.where(np.isfinite(std), std, 0.0)
    se = std / np.sqrt(valid)
    return m, z * se


def rolling_subopt_rate_at_a_star(
    subopt: np.ndarray,
    a_hist: np.ndarray,
    a_star: int,
    window: int,
) -> np.ndarray:
    """Rolling mean subopt over rounds in each window with `a_t=a^*` (NaN if none)."""
    n = len(subopt)
    if n < window:
        return np.array([], dtype=np.float64)
    out = np.empty(n - window + 1, dtype=np.float64)
    for i in range(n - window + 1):
        sub = subopt[i : i + window]
        m = a_hist[i : i + window] == a_star
        out[i] = np.nan if not np.any(m) else float(sub[m].mean())
    return out


def follower_subopt_rate_at_a_star(
    tr: Dict[str, np.ndarray], env: StackelbergBandit
) -> float:
    a_star = env.stackelberg_leader_action()
    mask = tr["a_hist"] == a_star
    return float(tr["subopt"][mask].mean()) if np.any(mask) else float("nan")


# ---------------------------------------------------------------------------
# Experiment 1: Offline dataset size vs T_{f,w}
# ---------------------------------------------------------------------------

def experiment1(
    out_dir: str, seeds: Sequence[int], n_a: int, n_b: int,
    horizon: int, gamma_exp3: float, n_off_grid: Sequence[int],
    cum_n_off: Optional[int] = None, progress: bool = True,
    delta: float = 0.05,
) -> None:
    n_off_list = list(n_off_grid)
    cum_target = (max(n_off_list) if cum_n_off is None else cum_n_off)
    if cum_target not in n_off_list:
        cum_target = max(n_off_list)
    conv_win, conv_thr, conv_k = 200, 0.2, 3

    t_fw_base = np.zeros((len(seeds), len(n_off_list)))
    t_fw_hyb = np.zeros((len(seeds), len(n_off_list)))
    conv_base = np.zeros((len(seeds), len(n_off_list)))
    conv_hyb = np.zeros((len(seeds), len(n_off_list)))
    cum_base_rows: List[np.ndarray] = []
    cum_hyb_rows: List[np.ndarray] = []
    thm1_transfer_hybrid = np.full((len(seeds), len(n_off_list)), np.nan)
    offline_m0_hybrid = np.full((len(seeds), len(n_off_list)), np.nan)

    for si, seed in enumerate(
        tqdm(seeds, desc="Exp 1", unit="seed", disable=not progress)
    ):
        rng = np.random.default_rng(seed)
        env = StackelbergBandit.sample(n_a, n_b, rng)
        # Inner bar: each cell is 2×horizon (baseline + hybrid); outer "Exp 1" seed bar
        # only moves after all N_off grid points for that seed finish.
        n_off_indices = range(len(n_off_list))
        if progress:
            n_off_indices = tqdm(
                n_off_indices,
                desc=f"Exp1 seed {seed} N_off",
                leave=False,
                unit="cell",
            )
        for j in n_off_indices:
            n_off = n_off_list[j]
            rng_b = np.random.default_rng(seed * 100_000 + j + 7)
            rng_h = np.random.default_rng(seed * 100_000 + j + 13)
            off = build_offline_uniform(env, n_off, rng_b) if n_off > 0 else None
            tr_b = simulate_run(env, horizon, rng_b, gamma_exp3=gamma_exp3, delta=delta)
            tr_h = simulate_run(
                env, horizon, rng_h, gamma_exp3=gamma_exp3, offline_init=off, delta=delta
            )
            if n_off > 0:
                thm1_transfer_hybrid[si, j] = float(tr_h["theorem1_transfer_ok"][0])
                offline_m0_hybrid[si, j] = float(tr_h["offline_m0_nonempty"][0])
            t_fw_base[si, j] = tr_b["subopt"].sum()
            t_fw_hyb[si, j] = tr_h["subopt"].sum()
            conv_base[si, j] = convergence_round(tr_b["subopt"], conv_win, conv_thr, conv_k)
            conv_hyb[si, j] = convergence_round(tr_h["subopt"], conv_win, conv_thr, conv_k)
            if n_off == cum_target:
                cum_base_rows.append(np.cumsum(tr_b["subopt"].astype(float)))
                cum_hyb_rows.append(np.cumsum(tr_h["subopt"].astype(float)))

    x_labels = [str(n) for n in n_off_list]
    x_plot = np.array(n_off_list, dtype=float) + 1.0

    # Plot 1: N_off vs T_{f,w}
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.set_xscale("log"); ax.grid(True, which="both", alpha=0.3)
    mb, eb = ci_mean(t_fw_base); mh, eh = ci_mean(t_fw_hyb)
    ax.plot(x_plot, mb, "o-", color=COLORS["baseline"], lw=3, ms=8, label="FMUCB (online only)")
    ax.fill_between(x_plot, mb - eb, mb + eb, color=COLORS["baseline"], alpha=0.18)
    ax.plot(x_plot, mh, "s-", color=COLORS["hybrid"], lw=3, ms=8, label="Hybrid-FMUCB")
    ax.fill_between(x_plot, mh - eh, mh + eh, color=COLORS["hybrid"], alpha=0.18)
    ax.set_xticks(x_plot); ax.set_xticklabels(x_labels)
    ax.set_xlabel(r"Offline dataset size $N_{\mathrm{off}}$")
    ax.set_ylabel(r"$T_{f,w}$ (mistakes vs.\ true best manipulation $F^{fm}$)")
    ax.set_title(r"Experiment 1: Offline data size vs $T_{f,w}$")
    ax.legend(frameon=True, fancybox=True, shadow=True)
    _style_axis(ax); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "exp1_noff_vs_tfw.png"), bbox_inches="tight")
    plt.close(fig)

    # Plot 2: convergence rounds
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.set_xscale("log"); ax.grid(True, which="both", alpha=0.3)
    cb, cbe = ci_mean(conv_base); ch, che = ci_mean(conv_hyb)
    ax.plot(x_plot, cb, "o-", color=COLORS["baseline"], lw=3, ms=8, label="FMUCB")
    ax.fill_between(x_plot, cb - cbe, cb + cbe, color=COLORS["baseline"], alpha=0.18)
    ax.plot(x_plot, ch, "s-", color=COLORS["hybrid"], lw=3, ms=8, label="Hybrid-FMUCB")
    ax.fill_between(x_plot, ch - che, ch + che, color=COLORS["hybrid"], alpha=0.18)
    ax.set_xticks(x_plot); ax.set_xticklabels(x_labels)
    ax.set_xlabel(r"$N_{\mathrm{off}}$")
    ax.set_ylabel("Rounds to reach sustained low error rate")
    ax.set_title(
        rf"Experiment 1: Convergence ($\leq{conv_thr}$, "
        rf"{conv_k} windows of {conv_win})"
    )
    ax.legend(frameon=True, fancybox=True, shadow=True)
    _style_axis(ax); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "exp1_noff_vs_convergence.png"), bbox_inches="tight")
    plt.close(fig)

    # Plot 3: cumulative mistakes
    t_axis = np.arange(1, horizon + 1)
    mcb, ecb = ci_mean(np.stack(cum_base_rows))
    mch, ech = ci_mean(np.stack(cum_hyb_rows))
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.plot(t_axis, mcb, "-", color=COLORS["baseline"], lw=3, label="FMUCB")
    ax.fill_between(t_axis, mcb - ecb, mcb + ecb, color=COLORS["baseline"], alpha=0.18)
    ax.plot(t_axis, mch, "-", color=COLORS["hybrid"], lw=3, label="Hybrid-FMUCB")
    ax.fill_between(t_axis, mch - ech, mch + ech, color=COLORS["hybrid"], alpha=0.18)
    ax.set_xlabel("Round $t$")
    ax.set_ylabel(r"Cumulative $T_{f,w}$ mistakes")
    ax.set_title(rf"Experiment 1: Cumulative mistakes ($N_{{\mathrm{{off}}}}={cum_target}$)")
    ax.legend(frameon=True, fancybox=True, shadow=True, loc="lower right")
    _style_axis(ax); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "exp1_cumulative_mistakes.png"), bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(out_dir, "exp1_summary.json"), "w") as f:
        json.dump({
            "n_off": n_off_list,
            "reward_model": "Bernoulli",
            "confidence_sets": "regression_Hoeffding",
            "confidence_delta": delta,
            "gamma_exp3": gamma_exp3,
            "tfw_baseline_mean": t_fw_base.mean(0).tolist(),
            "tfw_hybrid_mean": t_fw_hyb.mean(0).tolist(),
            "convergence_baseline_mean": conv_base.mean(0).tolist(),
            "convergence_hybrid_mean": conv_hyb.mean(0).tolist(),
            "theorem1_offline_transfer_rate_hybrid_mean": np.nanmean(
                thm1_transfer_hybrid, axis=0
            ).tolist(),
            "offline_m0_nonempty_rate_hybrid_mean": np.nanmean(
                offline_m0_hybrid, axis=0
            ).tolist(),
        }, f, indent=2)


# ---------------------------------------------------------------------------
# Experiment 2: Coverage quality
# ---------------------------------------------------------------------------

def experiment2(
    out_dir: str, seeds: Sequence[int], n_a: int, n_b: int,
    horizon: int, gamma_exp3: float, n_off_fixed: int, progress: bool = True,
    delta: float = 0.05,
) -> None:
    kinds = ("good", "neutral", "poor")
    t_fw = {k: np.zeros(len(seeds)) for k in kinds}
    success = {k: np.zeros(len(seeds)) for k in kinds}
    sub_at_star = {k: np.full(len(seeds), np.nan) for k in kinds}

    for kind in kinds:
        for si, seed in enumerate(
            tqdm(seeds, desc=f"Exp 2 ({kind})", unit="seed", disable=not progress)
        ):
            rng = np.random.default_rng(seed)
            env = StackelbergBandit.sample(n_a, n_b, rng)
            rng_off = np.random.default_rng(seed + 91_000)
            rng_on = np.random.default_rng(seed + 92_000)
            builders = {
                "good": build_offline_good_coverage,
                "neutral": build_offline_uniform,
                "poor": build_offline_poor_coverage,
            }
            off = builders[kind](env, n_off_fixed, rng_off)
            tr = simulate_run(
                env,
                horizon,
                rng_on,
                gamma_exp3=gamma_exp3,
                offline_init=off,
                delta=delta,
                progress_rounds=progress,
                progress_desc=f"Exp2 {kind} {seed}",
            )
            t_fw[kind][si] = tr["subopt"].sum()
            success[kind][si] = 1.0 - tr["subopt"][-min(500, horizon):].mean()
            sub_at_star[kind][si] = follower_subopt_rate_at_a_star(tr, env)

    labels = ["Good (includes\n$(a^*, b^*)$)", "Neutral\n(uniform)", "Poor (never\n$(a^*, b^*)$)"]
    colors_b = [COLORS["good"], COLORS["neutral"], COLORS["poor"]]

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 4.8))
    xpos = np.arange(3)
    panels = [
        ([float(t_fw[k].mean()) for k in kinds],
         [1.96 * t_fw[k].std(ddof=1) / np.sqrt(len(seeds)) for k in kinds],
         r"$T_{f,w}$ (vs.\ $F^{fm}$)",
         rf"Global $T_{{f,w}}$ ($N_{{\mathrm{{off}}}}={n_off_fixed}$)"),
        ([ci_mean_nan_1d(sub_at_star[k])[0] for k in kinds],
         [ci_mean_nan_1d(sub_at_star[k])[1] for k in kinds],
         r"Subopt. rate when $a_t=a^*$",
         r"Conditional on Stackelberg action (lower is better)"),
        ([float(success[k].mean()) for k in kinds],
         [1.96 * success[k].std(ddof=1) / np.sqrt(len(seeds)) for k in kinds],
         "Success rate (final window)",
         "Final-window optimality (higher is better)"),
    ]
    for ax, (means, errs, ylabel, title) in zip(axes, panels):
        bars = ax.bar(xpos, means, yerr=errs, color=colors_b, edgecolor="white",
                      linewidth=1.2, capsize=6, error_kw={"linewidth": 1.5})
        for rect, m in zip(bars, means):
            if np.isfinite(m):
                ax.text(rect.get_x() + rect.get_width() / 2,
                        m + max((e for e in errs if np.isfinite(e)), default=0) * 0.15,
                        f"{m:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_xticks(xpos); ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel); ax.set_title(title); _style_axis(ax)
    fig.suptitle("Experiment 2: Coverage quality comparison", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "exp2_coverage_bars.png"), bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(out_dir, "exp2_summary.json"), "w") as f:
        json.dump({
            "n_off_fixed": n_off_fixed,
            **{f"tfw_{k}_mean": float(t_fw[k].mean()) for k in kinds},
            **{f"subopt_at_a_star_{k}_mean": float(np.nanmean(sub_at_star[k])) for k in kinds},
            **{f"success_{k}_mean": float(success[k].mean()) for k in kinds},
        }, f, indent=2)


# ---------------------------------------------------------------------------
# Optional: learning curves
# ---------------------------------------------------------------------------

def learning_curve_figure(
    out_dir: str,
    seeds: Sequence[int],
    n_a: int,
    n_b: int,
    horizon: int,
    gamma_exp3: float,
    n_off: int,
    progress: bool = True,
    rolling_window: Optional[int] = None,
    delta: float = 0.05,
    caption_suffix: str = "",
) -> None:
    """
    Rolling **global** mistake rate vs time (subopt averaged over all rounds in the window).

    Paper context (*Hybrid Offline-Online Follower Manipulation...*): $T_{f,w}$ aggregates
    rounds where $b_t \\neq F^{fm}(a_t)$. The theory emphasizes **qualified manipulation**
    at the target and hybrid **sample complexity** (via $C_{\\mathrm{man}}$), not a monotone
    global error curve. EXP3 explores all leader rows, so the global rolling rate can stay high
    even when behavior at the Stackelberg row $a^*$ improves — see
    ``exp_optional_learning_curves_at_a_star.png``. For stronger global trends without changing
    Alg. 1, use smaller $\\gamma$, longer $T$, more offline data (CLI: ``--learning-curve-paper-profile``).
    """
    curves_b: List[np.ndarray] = []
    curves_h: List[np.ndarray] = []
    curves_b_as: List[np.ndarray] = []
    curves_h_as: List[np.ndarray] = []
    win = rolling_window if rolling_window is not None else max(50, horizon // 100)
    win = min(int(win), max(1, horizon))
    fb_hybrid_total = 0
    if progress:
        print(
            f"Learning curves: {len(seeds)} seeds, horizon={horizon}, N_off={n_off} "
            f"(per-round tqdm inside each simulate_run; outer bar = seeds).",
            flush=True,
        )
    for seed in tqdm(seeds, desc="Learning curves", unit="seed", disable=not progress):
        rng = np.random.default_rng(seed)
        env = StackelbergBandit.sample(n_a, n_b, rng)
        a_star_env = int(env.stackelberg_leader_action())
        rng_b = np.random.default_rng(seed + 3)
        rng_h = np.random.default_rng(seed + 5)
        off = build_offline_uniform(env, n_off, rng_h) if n_off > 0 else None
        tr_b = simulate_run(
            env,
            horizon,
            rng_b,
            gamma_exp3=gamma_exp3,
            delta=delta,
            progress_rounds=progress,
            progress_desc=f"LC seed {seed} baseline",
        )
        tr_h = simulate_run(
            env,
            horizon,
            rng_h,
            gamma_exp3=gamma_exp3,
            offline_init=off,
            delta=delta,
            progress_rounds=progress,
            progress_desc=f"LC seed {seed} hybrid",
        )
        fb_hybrid_total += int(tr_h.get("fallback_count", 0))

        sub_b = tr_b["subopt"].astype(float)
        sub_h = tr_h["subopt"].astype(float)
        curves_b.append(np.convolve(sub_b, np.ones(win) / win, mode="valid"))
        curves_h.append(np.convolve(sub_h, np.ones(win) / win, mode="valid"))
        curves_b_as.append(
            rolling_subopt_rate_at_a_star(sub_b, tr_b["a_hist"], a_star_env, win)
        )
        curves_h_as.append(
            rolling_subopt_rate_at_a_star(sub_h, tr_h["a_hist"], a_star_env, win)
        )

    t_axis = np.arange(win, horizon + 1)
    mb, eb = ci_mean(np.stack(curves_b))
    mh, eh = ci_mean(np.stack(curves_h))
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.plot(t_axis, mb, color=COLORS["baseline"], lw=2, label="FMUCB")
    ax.fill_between(t_axis, mb - eb, mb + eb, color=COLORS["baseline"], alpha=0.2)
    ax.plot(t_axis, mh, color=COLORS["hybrid"], lw=2, label="Hybrid-FMUCB")
    ax.fill_between(t_axis, mh - eh, mh + eh, color=COLORS["hybrid"], alpha=0.2)
    ax.set_xlabel("Round $t$")
    ax.set_ylabel(f"Rolling subopt. rate (window={win})")
    fb_rate = fb_hybrid_total / max(1, len(seeds) * horizon)
    ax.set_title(
        rf"Learning curves (global): $N_{{\mathrm{{off}}}}={n_off}$ — "
        f"hybrid row-fallback fraction ≈ {fb_rate:.3f} per round (mean over seeds)"
        + caption_suffix
        + "\n"
        + r"(EXP3 visits all leader rows; global rolling error often stays high — compare the $a_t=a^*$ plot.)"
    )
    ax.legend()
    ax.set_ylim(0, 1.0)
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "exp_optional_learning_curves.png"), bbox_inches="tight")
    plt.close(fig)

    stack_b_as = np.stack(curves_b_as, axis=0)
    stack_h_as = np.stack(curves_h_as, axis=0)
    mb_a, eb_a = ci_mean_nan(stack_b_as, axis=0)
    mh_a, eh_a = ci_mean_nan(stack_h_as, axis=0)
    fig_a, ax_a = plt.subplots(figsize=(9.0, 5.0))
    ax_a.plot(t_axis, mb_a, color=COLORS["baseline"], lw=2, label="FMUCB")
    ax_a.fill_between(t_axis, mb_a - eb_a, mb_a + eb_a, color=COLORS["baseline"], alpha=0.2)
    ax_a.plot(t_axis, mh_a, color=COLORS["hybrid"], lw=2, label="Hybrid-FMUCB")
    ax_a.fill_between(t_axis, mh_a - eh_a, mh_a + eh_a, color=COLORS["hybrid"], alpha=0.2)
    ax_a.set_xlabel("Round $t$")
    ax_a.set_ylabel(f"Rolling subopt. at $a_t=a^*$ (window={win})")
    ax_a.set_title(
        rf"Learning curves (paper-relevant): conditional on Stackelberg leader row "
        rf"($N_{{\mathrm{{off}}}}={n_off}$)"
        + caption_suffix
    )
    ax_a.legend()
    ax_a.set_ylim(0, 1.0)
    _style_axis(ax_a)
    fig_a.tight_layout()
    fig_a.savefig(
        os.path.join(out_dir, "exp_optional_learning_curves_at_a_star.png"),
        bbox_inches="tight",
    )
    plt.close(fig_a)


# ===========================================================================
# Experiment 3: Full contextual Hybrid-FMUCB -- Algorithm 2 (Gap fix)
# ===========================================================================

def phi_vec(x: int, a: int, b: int, n_x: int, n_a: int, n_b: int) -> np.ndarray:
    """One-hot feature map phi(x,a,b) in R^{n_x + n_a + n_b}."""
    d = n_x + n_a + n_b
    v = np.zeros(d)
    v[x] = 1.0; v[n_x + a] = 1.0; v[n_x + n_a + b] = 1.0
    return v


def _mu_ctx(theta: np.ndarray, x: int, a: int, b: int,
            n_x: int, n_a: int, n_b: int) -> float:
    return float(np.clip(theta @ phi_vec(x, a, b, n_x, n_a, n_b), 0.0, 1.0))


def true_best_contextual_manipulation(
    theta_l: np.ndarray,
    theta_f: np.ndarray,
    n_x: int, n_a: int, n_b: int,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Compute the true best contextual manipulation rule F^fm : X x A -> B.

    For each context x, enumerate all (a*, b*) candidate response rules
    F_{a*,b*}^x where F(x,a*)=b* and F(x,a)=argmin_b mu_l(x,a,b) for a!=a*.
    Among qualified ones (contextual contrast > 0), pick the one maximising
    mu_f(x, a*, b*).  Return an array F_fm of shape (n_x, n_a) giving
    F^fm(x, a) for each (x, a).
    """
    # Precompute mu_l and mu_f tables
    mu_l = np.array([[[_mu_ctx(theta_l, x, a, b, n_x, n_a, n_b)
                        for b in range(n_b)]
                       for a in range(n_a)]
                      for x in range(n_x)])  # (n_x, n_a, n_b)
    mu_f = np.array([[[_mu_ctx(theta_f, x, a, b, n_x, n_a, n_b)
                        for b in range(n_b)]
                       for a in range(n_a)]
                      for x in range(n_x)])  # (n_x, n_a, n_b)

    F_fm = np.zeros((n_x, n_a), dtype=np.int32)

    for x in range(n_x):
        # Worst-response for each a under mu_l[x]
        wr = np.array([int(np.argmin(mu_l[x, a])) for a in range(n_a)])
        # Default: myopic BR
        F_fm[x] = np.array([int(np.argmax(mu_f[x, a])) for a in range(n_a)])

        best_val = -np.inf
        for a_star in range(n_a):
            for b_star in range(n_b):
                # F_{a*,b*}^x
                F_x = wr.copy(); F_x[a_star] = b_star
                # Contextual contrast
                left = mu_l[x, a_star, b_star]
                others = [mu_l[x, a, F_x[a]] for a in range(n_a) if a != a_star]
                contrast = left - max(others) if others else left
                if contrast <= eps:
                    continue
                v = mu_f[x, a_star, b_star]
                if v > best_val:
                    best_val = v
                    F_fm[x] = F_x.copy()

    return F_fm  # shape (n_x, n_a)


def _linucb_confidence(
    a_mat_inv: np.ndarray,
    b_vec: np.ndarray,
    ph: np.ndarray,
    alpha: float,
) -> Tuple[float, float]:
    """
    Returns (predicted_value, confidence_width) for a LinUCB/LCB arm.
    pred = theta_hat @ ph = (A^{-1} b) @ ph
    conf = alpha * sqrt(ph^T A^{-1} ph)
    """
    theta_hat = a_mat_inv @ b_vec
    pred = float(theta_hat @ ph)
    conf = float(alpha * np.sqrt(max(0.0, ph @ a_mat_inv @ ph)))
    return pred, conf


def _build_offline_contextual(
    theta_l: np.ndarray,
    theta_f: np.ndarray,
    n_off: int,
    rng: np.random.Generator,
    n_x: int, n_a: int, n_b: int,
    ridge_reg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build offline confidence sets for leader and follower from D_off.
    Returns (A_l, b_l, A_f, b_f) ridge-regression matrices.
    Rewards are Bernoulli (Deviation 1).
    """
    d = n_x + n_a + n_b
    A_l = ridge_reg * np.eye(d); b_l = np.zeros(d)
    A_f = ridge_reg * np.eye(d); b_f = np.zeros(d)
    for _ in range(n_off):
        x = int(rng.integers(0, n_x))
        a = int(rng.integers(0, n_a))
        b = int(rng.integers(0, n_b))
        ph = phi_vec(x, a, b, n_x, n_a, n_b)
        mu_l_ab = _mu_ctx(theta_l, x, a, b, n_x, n_a, n_b)
        mu_f_ab = _mu_ctx(theta_f, x, a, b, n_x, n_a, n_b)
        r_l = bernoulli_sample(mu_l_ab, rng)
        r_f = bernoulli_sample(mu_f_ab, rng)
        A_l += np.outer(ph, ph); b_l += ph * r_l
        A_f += np.outer(ph, ph); b_f += ph * r_f
    return A_l, b_l, A_f, b_f


def simulate_contextual_hybrid_fmucb(
    theta_l: np.ndarray,
    theta_f: np.ndarray,
    n_x: int, n_a: int, n_b: int,
    horizon: int,
    rng: np.random.Generator,
    gamma_exp3: float,
    offline_init: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
    alpha: float = 1.0,
    ridge_reg: float = 1.0,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Full contextual Hybrid-FMUCB -- Algorithm 2 (paper Sec. 3).

    Maintains separate LinUCB confidence sets F_{l,t} and F_{f,t} over
    feature map phi(x,a,b).  At each round t and context x_t:

      1. EXP3 selects a_t (exploration over leader actions).
      2. Feasible set M_t(x_t):
           (F, a*) in M_t(x) iff
           inf_{g in F_{l,t}} Delta_{F,a*}(g; x) > 0
           <=> LCB_l(x, a*, F(x,a*)) - max_{a!=a*} UCB_l(x, a, F(x,a)) > eps
      3. Optimistic follower:
           (F_hat, a_hat) = argmax_{(F,a) in M_t(x_t)}
                            sup_{h in F_{f,t}} h(x_t, a, F(x_t,a))
                          = mu_hat_f(x,a,F(x,a)) + conf_f(x,a,F(x,a))
      4. b_t = F_hat(x_t, a_t).
      5. Update F_{l,t} and F_{f,t} with (x_t, a_t, b_t, r_l, r_f).

    T_{f,w} counts b_t != F^fm(x_t, a_t) where F^fm is the true contextual
    best manipulation (maximises mu_f among qualified response rules per context).

    Rewards are Bernoulli (Deviation 1).
    """
    d = n_x + n_a + n_b
    F_fm = true_best_contextual_manipulation(theta_l, theta_f, n_x, n_a, n_b)

    if offline_init is not None:
        A_l, b_l, A_f, b_f = [m.copy() for m in offline_init]
    else:
        A_l = ridge_reg * np.eye(d); b_l = np.zeros(d)
        A_f = ridge_reg * np.eye(d); b_f = np.zeros(d)

    weights = np.ones((n_x, n_a))
    subopt = np.zeros(horizon, dtype=np.bool_)

    # Precompute worst-response tables from true mu_l (used only for
    # building tabular response rules; in reality the follower uses
    # the estimated theta_hat to compute these -- we re-derive each round)
    for t in range(1, horizon + 1):
        x = int(rng.integers(0, n_x))
        a_t = exp3_sample(weights[x], gamma_exp3, rng)

        # Invert confidence matrices once per round
        A_l_inv = np.linalg.inv(A_l)
        A_f_inv = np.linalg.inv(A_f)
        theta_l_hat = A_l_inv @ b_l
        theta_f_hat = A_f_inv @ b_f

        # Worst-response under current theta_l_hat for this context
        def wr_hat(a: int) -> int:
            vals = [float(np.clip(theta_l_hat @ phi_vec(x, a, b, n_x, n_a, n_b), 0, 1))
                    for b in range(n_b)]
            return int(np.argmin(vals))

        # Search M_t(x) and pick optimistic follower action
        best_ucb_f = -np.inf
        best_F_x: Optional[np.ndarray] = None  # shape (n_a,)

        for a_star in range(n_a):
            for b_star in range(n_b):
                # Build F_{a*,b*}^x under theta_l_hat
                F_x = np.array([wr_hat(a) for a in range(n_a)], dtype=np.int32)
                F_x[a_star] = b_star

                # Pessimistic leader contrast (inf over F_{l,t})
                ph_target = phi_vec(x, a_star, b_star, n_x, n_a, n_b)
                pred_lt, conf_lt = _linucb_confidence(A_l_inv, b_l, ph_target, alpha)
                lcb_target = pred_lt - conf_lt

                max_ucb_other = -np.inf
                for a in range(n_a):
                    if a == a_star:
                        continue
                    b_other = int(F_x[a])
                    ph_other = phi_vec(x, a, b_other, n_x, n_a, n_b)
                    pred_lo, conf_lo = _linucb_confidence(A_l_inv, b_l, ph_other, alpha)
                    ucb_other = pred_lo + conf_lo
                    if ucb_other > max_ucb_other:
                        max_ucb_other = ucb_other

                pess_contrast = lcb_target - max_ucb_other
                if pess_contrast <= eps:
                    continue  # not in M_t(x)

                # Optimistic follower value at (x, a*, b*)
                ph_f = phi_vec(x, a_star, b_star, n_x, n_a, n_b)
                pred_f, conf_f = _linucb_confidence(A_f_inv, b_f, ph_f, alpha)
                ucb_f = pred_f + conf_f
                if ucb_f > best_ucb_f:
                    best_ucb_f = ucb_f
                    best_F_x = F_x.copy()

        # Play b_t = F_hat(x_t, a_t), or UCB-BR fallback if M_t = {}
        if best_F_x is not None:
            b_t = int(best_F_x[a_t])
        else:
            # Fallback: UCB best-response on follower reward for this context+action
            ucb_vals = np.array([
                float(np.clip(theta_f_hat @ phi_vec(x, a_t, b, n_x, n_a, n_b), 0, 1))
                + float(alpha * np.sqrt(max(0.0, phi_vec(x, a_t, b, n_x, n_a, n_b)
                                            @ A_f_inv
                                            @ phi_vec(x, a_t, b, n_x, n_a, n_b))))
                for b in range(n_b)
            ])
            b_t = int(np.argmax(ucb_vals))

        # Observe Bernoulli rewards (Deviation 1)
        r_l = bernoulli_sample(_mu_ctx(theta_l, x, a_t, b_t, n_x, n_a, n_b), rng)
        r_f = bernoulli_sample(_mu_ctx(theta_f, x, a_t, b_t, n_x, n_a, n_b), rng)

        # EXP3 update for leader
        exp3_update(weights[x], a_t, r_l, gamma_exp3)

        # Update confidence sets (Alg 2, line 12)
        ph = phi_vec(x, a_t, b_t, n_x, n_a, n_b)
        A_l += np.outer(ph, ph); b_l += ph * r_l
        A_f += np.outer(ph, ph); b_f += ph * r_f

        # T_{f,w}: compare to true best contextual manipulation
        subopt[t - 1] = (b_t != F_fm[x, a_t])

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
    """
    Experiment 3: full contextual Hybrid-FMUCB (Algorithm 2) vs non-contextual
    Hybrid-FMUCB (Algorithm 1 applied context-blind, treating each context as a
    separate bandit but using a shared tabular representation).

    Both algorithms use:
    - Bernoulli rewards (Deviation 1)
    - Offline data D_off (uniform over X x A x B) for warm-start
    - T_{f,w} vs the true contextual best manipulation F^fm

    The contextual algorithm (Alg 2) shares statistical strength across contexts
    via the linear feature map phi(x,a,b); the non-contextual baseline fits a
    separate table per context.
    """
    d = n_x + n_a + n_b
    tfw_ctx: List[float] = []
    tfw_tab: List[float] = []  # non-contextual: per-context tabular Hybrid-FMUCB

    for seed in tqdm(seeds, desc="Exp 3 (contextual)", unit="seed", disable=not progress):
        rng = np.random.default_rng(seed)
        theta_l = rng.uniform(-0.5, 0.5, d)
        theta_f = rng.uniform(-0.5, 0.5, d)

        # ---- Contextual Algorithm 2 ----
        rng_ctx = np.random.default_rng(seed + 29)
        off_ctx = _build_offline_contextual(
            theta_l, theta_f, n_off, rng_ctx, n_x, n_a, n_b, ridge_reg=1.0
        )
        sub_ctx = simulate_contextual_hybrid_fmucb(
            theta_l, theta_f, n_x, n_a, n_b, horizon,
            rng_ctx, gamma_exp3, offline_init=off_ctx,
        )
        tfw_ctx.append(float(sub_ctx.sum()))

        # ---- Non-contextual Algorithm 1 (per-context tabular) ----
        # Build a separate OfflineRewardStats per context using the same theta
        rng_tab = np.random.default_rng(seed + 17)
        # Aggregate offline data into per-context tabular stats
        ctx_stats: Dict[int, OfflineRewardStats] = {}
        for x in range(n_x):
            ctx_stats[x] = OfflineRewardStats(
                np.zeros((n_a, n_b), dtype=np.int64),
                np.zeros((n_a, n_b)),
                np.zeros((n_a, n_b)),
            )
        n_off_per_ctx = n_off // n_x
        for x in range(n_x):
            for _ in range(n_off_per_ctx):
                a = int(rng_tab.integers(0, n_a))
                b = int(rng_tab.integers(0, n_b))
                r_l = bernoulli_sample(_mu_ctx(theta_l, x, a, b, n_x, n_a, n_b), rng_tab)
                r_f = bernoulli_sample(_mu_ctx(theta_f, x, a, b, n_x, n_a, n_b), rng_tab)
                ctx_stats[x].n_visits[a, b] += 1
                ctx_stats[x].sum_r_l[a, b] += r_l
                ctx_stats[x].sum_r_f[a, b] += r_f

        # Simulate: per-context tabular Hybrid-FMUCB
        # We need F^fm per context to evaluate T_{f,w}
        F_fm_tab = true_best_contextual_manipulation(theta_l, theta_f, n_x, n_a, n_b)
        weights_tab = np.ones((n_x, n_a))
        subopt_tab = np.zeros(horizon, dtype=np.bool_)
        # Carry per-context counts
        ctx_nv = {x: ctx_stats[x].n_visits.copy() for x in range(n_x)}
        ctx_sf = {x: ctx_stats[x].sum_r_f.copy() for x in range(n_x)}
        ctx_sl = {x: ctx_stats[x].sum_r_l.copy() for x in range(n_x)}

        for t in range(1, horizon + 1):
            x = int(rng_tab.integers(0, n_x))
            a_t = exp3_sample(weights_tab[x], gamma_exp3, rng_tab)
            b_t, _ = hybrid_fmucb_pick(
                a_t, ctx_nv[x], ctx_sf[x], ctx_sl[x], t, n_a, n_b, rng_tab
            )
            r_l = bernoulli_sample(_mu_ctx(theta_l, x, a_t, b_t, n_x, n_a, n_b), rng_tab)
            r_f = bernoulli_sample(_mu_ctx(theta_f, x, a_t, b_t, n_x, n_a, n_b), rng_tab)
            exp3_update(weights_tab[x], a_t, r_l, gamma_exp3)
            ctx_nv[x][a_t, b_t] += 1
            ctx_sf[x][a_t, b_t] += r_f
            ctx_sl[x][a_t, b_t] += r_l
            subopt_tab[t - 1] = (b_t != F_fm_tab[x, a_t])
        tfw_tab.append(float(subopt_tab.sum()))

    arr_ctx = np.array(tfw_ctx)
    arr_tab = np.array(tfw_tab)
    means = [arr_tab.mean(), arr_ctx.mean()]
    errs = [
        1.96 * arr_tab.std(ddof=1) / np.sqrt(len(seeds)),
        1.96 * arr_ctx.std(ddof=1) / np.sqrt(len(seeds)),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    xpos = np.arange(2)
    bars = ax.bar(xpos, means, yerr=errs,
                  color=[COLORS["tabular"], COLORS["contextual"]],
                  edgecolor="white", linewidth=1.2, capsize=6,
                  error_kw={"linewidth": 1.5}, width=0.55)
    ax.set_xticks(xpos)
    ax.set_xticklabels([
        "Non-contextual\nHybrid-FMUCB (Alg 1)",
        "Contextual\nHybrid-FMUCB (Alg 2)",
    ])
    ax.set_ylabel(r"$T_{f,w}$ (vs.\ true contextual $F^{fm}$)")
    ax.set_title(
        rf"Experiment 3: Alg 1 vs Alg 2 ($N_{{\mathrm{{off}}}}={n_off}$, "
        rf"$|X|={n_x}$, $|A|={n_a}$, $|B|={n_b}$)"
    )
    for rect, m in zip(bars, means):
        ax.text(rect.get_x() + rect.get_width() / 2,
                m + max(errs) * 0.12, f"{m:.0f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    _style_axis(ax); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "exp3_contextual_vs_tabular.png"), bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(out_dir, "exp3_summary.json"), "w") as f:
        json.dump({
            "algorithm_1": "non_contextual_per_context_tabular_hybrid_fmucb",
            "algorithm_2": "full_contextual_hybrid_fmucb_linucb",
            "reward_model": "Bernoulli",
            "n_off": n_off, "n_x": n_x, "n_a": n_a, "n_b": n_b,
            "tfw_non_contextual_mean": float(arr_tab.mean()),
            "tfw_contextual_mean": float(arr_ctx.mean()),
        }, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)  # py3.7+: show prints immediately
    except (AttributeError, OSError):
        pass
    parser = argparse.ArgumentParser(
        description="Stackelberg bandit offline-online experiments"
    )
    parser.add_argument("--out-dir", type=str, default="figures")
    parser.add_argument("--seeds", type=int, default=24)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--n-a", type=int, default=8)
    parser.add_argument("--n-b", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=8000)
    parser.add_argument(
        "--gamma-exp3",
        type=float,
        default=0.01,
        help=r"EXP3 exploration $\gamma$ (smaller ⇒ more weight on leader actions with high empirical payoff; typical 0.05–0.01)",
    )
    parser.add_argument(
        "--confidence-delta",
        type=float,
        default=0.05,
        help="High-probability parameter for regression confidence sets in Alg. 1 (finite-sample bounds; unchanged from implementation default)",
    )
    parser.add_argument("--exp1-cum-n-off", type=int, default=None)
    parser.add_argument("--exp2-n-off", type=int, default=1000)
    parser.add_argument("--skip-learning-curves", action="store_true")
    parser.add_argument(
        "--learning-curves-only",
        action="store_true",
        help="Run only exp_optional_learning_curves.png (skips Exp 1–2; overrides --exp1-only / --exp2-only)",
    )
    parser.add_argument(
        "--exp1-only",
        action="store_true",
        help="Run only Experiment 1 (no Exp 2, no learning curves unless combined with --exp2-only)",
    )
    parser.add_argument(
        "--exp2-only",
        action="store_true",
        help="Run only Experiment 2 (no Exp 1, no learning curves unless combined with --exp1-only)",
    )
    parser.add_argument("--exp3", action="store_true")
    parser.add_argument("--exp3-n-off", type=int, default=1500)
    parser.add_argument("--n-x", type=int, default=5)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--learning-curve-horizon",
        type=int,
        default=None,
        help="Learning-curve online length (default: same as --horizon; use e.g. 1500 for quick tests)",
    )
    parser.add_argument(
        "--learning-curve-seeds",
        type=int,
        default=None,
        help="Number of seeds for learning curves only (default: same as --seeds)",
    )
    parser.add_argument(
        "--learning-curve-n-off",
        type=int,
        default=1000,
        help="Offline size for learning-curve runs",
    )
    parser.add_argument(
        "--learning-curve-window",
        type=int,
        default=None,
        help="Rolling window length (default: max(50, horizon_lc//100))",
    )
    parser.add_argument(
        "--learning-curve-gamma-exp3",
        type=float,
        default=None,
        help="EXP3 gamma for learning curves only (default: same as --gamma-exp3). Use e.g. 0.005 for more visits to a* in LC plots.",
    )
    parser.add_argument(
        "--learning-curve-paper-profile",
        action="store_true",
        help="Learning curves: use longer online horizon (max 20k vs --horizon), cap gamma at 0.01, floor N_off at 3000 — same Alg 1, stronger global signal",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seeds = [args.base_seed + i for i in range(args.seeds)]
    n_off_grid = [0, 100, 500, 1000, 5000]
    show_p = not args.no_progress

    if args.learning_curves_only:
        run_exp1 = False
        run_exp2 = False
        run_lc = not args.skip_learning_curves
    elif args.exp1_only and args.exp2_only:
        run_exp1 = True
        run_exp2 = True
        run_lc = False
    elif args.exp1_only:
        run_exp1 = True
        run_exp2 = False
        run_lc = False
    elif args.exp2_only:
        run_exp1 = False
        run_exp2 = True
        run_lc = False
    else:
        run_exp1 = True
        run_exp2 = True
        run_lc = not args.skip_learning_curves

    if show_p:
        parts: List[str] = []
        if run_exp1:
            parts.append("exp1")
        if run_exp2:
            parts.append("exp2")
        if run_lc:
            parts.append("learning curves")
        if args.exp3:
            parts.append("exp3")
        msg = (
            f"Starting: {', '.join(parts)} | seeds={args.seeds} horizon={args.horizon} "
            f"n_a={args.n_a} n_b={args.n_b} — exp1: {len(n_off_grid)} N_off cells × "
            f"2 rollouts × {args.horizon} steps (per-round tqdm when progress is on)."
        )
        print(msg, file=sys.stderr, flush=True)

    if run_exp1:
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
            delta=args.confidence_delta,
        )
    if run_exp2:
        experiment2(
            args.out_dir,
            seeds,
            args.n_a,
            args.n_b,
            args.horizon,
            args.gamma_exp3,
            args.exp2_n_off,
            progress=show_p,
            delta=args.confidence_delta,
        )
    if run_lc:
        lc_horizon = (
            args.learning_curve_horizon
            if args.learning_curve_horizon is not None
            else args.horizon
        )
        lc_n = (
            args.learning_curve_seeds
            if args.learning_curve_seeds is not None
            else args.seeds
        )
        lc_seeds = [args.base_seed + i for i in range(lc_n)]
        lc_gamma = (
            args.learning_curve_gamma_exp3
            if args.learning_curve_gamma_exp3 is not None
            else args.gamma_exp3
        )
        lc_n_off = args.learning_curve_n_off
        lc_caption = ""
        if args.learning_curve_paper_profile:
            if args.learning_curve_horizon is None:
                lc_horizon = max(20000, args.horizon)
            lc_gamma = min(lc_gamma, 0.01)
            lc_n_off = max(lc_n_off, 3000)
            lc_caption = (
                rf" — paper-profile: $T={lc_horizon}$, "
                rf"$\gamma={lc_gamma}$, $N_{{\mathrm{{off}}}}={lc_n_off}$"
            )
        learning_curve_figure(
            args.out_dir,
            lc_seeds,
            args.n_a,
            args.n_b,
            lc_horizon,
            lc_gamma,
            n_off=lc_n_off,
            progress=show_p,
            rolling_window=args.learning_curve_window,
            delta=args.confidence_delta,
            caption_suffix=lc_caption,
        )
    if args.exp3:
        experiment3_contextual(args.out_dir, seeds, args.horizon, args.gamma_exp3,
                               args.exp3_n_off, args.n_x, args.n_a, args.n_b,
                               progress=show_p)
    print(f"Wrote figures to {os.path.abspath(args.out_dir)}", flush=True)


if __name__ == "__main__":
    main()