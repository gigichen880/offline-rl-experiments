"""
Shared Stackelberg bandit environment, EXP3 leader, offline data, simulation, and plotting style.
Used by experiment_1 / experiment_2 / experiment_3 and hybrid_fmucb.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from hybrid_fmucb import (
    LeaderFeasibility,
    OfflineRewardStats,
    hybrid_fmucb_pick,
    offline_candidate_manipulation,
    theorem1_offline_transfer_check,
    true_best_manipulation,
)

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

# Tighter default than earlier runs: comparable to reward scale Uniform(0,1).
DEFAULT_REWARD_NOISE_STD = 0.01

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

    mu_leader: np.ndarray
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

    def follower_br(self, a: int) -> int:
        return int(np.argmax(self.mu_follower[a]))

    def stackelberg_leader_action(self) -> int:
        br = np.argmax(self.mu_follower, axis=1)
        vals = np.array([self.mu_leader[a, br[a]] for a in range(self.n_a)])
        return int(np.argmax(vals))

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
    reward_noise_std: float = DEFAULT_REWARD_NOISE_STD,
    follower_elim_eps: Optional[float] = None,
    leader_feasibility: LeaderFeasibility = "pessimistic",
    feasibility_margin: float = 0.0,
) -> Dict[str, np.ndarray]:
    n_a, n_b = env.n_a, env.n_b
    n_visits, sum_r_f, sum_r_l = _unpack_offline_init(offline_init, n_a, n_b)

    offline_m0_nonempty = np.array([np.nan])
    theorem1_transfer_ok = np.array([np.nan])
    theorem1_transfer_lhs = np.array([np.nan])
    theorem1_transfer_threshold = np.array([np.nan])
    theorem1_delta3 = np.array([np.nan])
    if offline_init is not None:
        eps_base = follower_elim_eps if follower_elim_eps is not None else 1e-9
        eps_eff = float(eps_base) + float(feasibility_margin)
        cand_F, _, _, _ = offline_candidate_manipulation(
            sum_r_l, sum_r_f, n_visits, n_a, n_b, 0.05, eps=eps_eff
        )
        offline_m0_nonempty = np.array([1.0 if cand_F is not None else 0.0])
        ok, lhs, thr, d3 = theorem1_offline_transfer_check(
            env.mu_leader, env.mu_follower, n_visits, sum_r_l, n_a, n_b
        )
        theorem1_transfer_ok = np.array([1.0 if ok else 0.0])
        theorem1_transfer_lhs = np.array([lhs])
        theorem1_transfer_threshold = np.array([thr])
        theorem1_delta3 = np.array([d3])

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
        b, _ = hybrid_fmucb_pick(
            a,
            n_visits,
            sum_r_f,
            sum_r_l,
            t,
            n_a,
            n_b,
            rng,
            follower_elim_eps=follower_elim_eps,
            leader_feasibility=leader_feasibility,
            feasibility_margin=feasibility_margin,
        )

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
        "offline_m0_nonempty": offline_m0_nonempty,
        "theorem1_transfer_ok": theorem1_transfer_ok,
        "theorem1_transfer_lhs": theorem1_transfer_lhs,
        "theorem1_transfer_threshold": theorem1_transfer_threshold,
        "theorem1_delta3": theorem1_delta3,
    }


def build_offline_uniform(
    env: StackelbergBandit,
    n_off: int,
    rng: np.random.Generator,
    reward_noise_std: float = DEFAULT_REWARD_NOISE_STD,
) -> OfflineRewardStats:
    n_a, n_b = env.n_a, env.n_b
    n_visits = np.zeros((n_a, n_b), dtype=np.int64)
    sum_r_f = np.zeros((n_a, n_b), dtype=np.float64)
    sum_r_l = np.zeros((n_a, n_b), dtype=np.float64)
    for _ in range(n_off):
        a = int(rng.integers(0, n_a))
        b = int(rng.integers(0, n_b))
        r_f = env.mu_follower[a, b] + rng.normal(0.0, reward_noise_std)
        r_l = env.mu_leader[a, b] + rng.normal(0.0, reward_noise_std)
        n_visits[a, b] += 1
        sum_r_f[a, b] += r_f
        sum_r_l[a, b] += r_l
    return OfflineRewardStats(n_visits=n_visits, sum_r_f=sum_r_f, sum_r_l=sum_r_l)


def build_offline_good_coverage(
    env: StackelbergBandit,
    n_off: int,
    rng: np.random.Generator,
    reward_noise_std: float = DEFAULT_REWARD_NOISE_STD,
) -> OfflineRewardStats:
    n_a, n_b = env.n_a, env.n_b
    n_visits = np.zeros((n_a, n_b), dtype=np.int64)
    sum_r_f = np.zeros((n_a, n_b), dtype=np.float64)
    sum_r_l = np.zeros((n_a, n_b), dtype=np.float64)
    a_star = env.stackelberg_leader_action()
    b_star = env.follower_br(a_star)

    for _ in range(n_off):
        u = rng.random()
        if u < 0.8:
            a = a_star
            b = b_star
        elif u < 0.9:
            a = int(rng.integers(0, n_a))
            b = env.follower_br(a)
        else:
            a = int(rng.integers(0, n_a))
            b = int(rng.integers(0, n_b))
        r_f = env.mu_follower[a, b] + rng.normal(0.0, reward_noise_std)
        r_l = env.mu_leader[a, b] + rng.normal(0.0, reward_noise_std)
        n_visits[a, b] += 1
        sum_r_f[a, b] += r_f
        sum_r_l[a, b] += r_l
    return OfflineRewardStats(n_visits=n_visits, sum_r_f=sum_r_f, sum_r_l=sum_r_l)


def build_offline_poor_coverage(
    env: StackelbergBandit,
    n_off: int,
    rng: np.random.Generator,
    reward_noise_std: float = DEFAULT_REWARD_NOISE_STD,
) -> OfflineRewardStats:
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
                while a == a_star:
                    a = int(rng.integers(0, n_a))
                b = int(rng.integers(0, n_b))
        else:
            # Skew toward misleading (low follower payoff) arms to stress bad priors.
            if rng.random() < 0.5:
                b = int(np.argmin(env.mu_follower[a]))
            else:
                b = int(rng.integers(0, n_b))
        r_f = env.mu_follower[a, b] + rng.normal(0.0, reward_noise_std)
        r_l = env.mu_leader[a, b] + rng.normal(0.0, reward_noise_std)
        n_visits[a, b] += 1
        sum_r_f[a, b] += r_f
        sum_r_l[a, b] += r_l
    return OfflineRewardStats(n_visits=n_visits, sum_r_f=sum_r_f, sum_r_l=sum_r_l)


def rolling_subopt_rate_at_a_star(
    subopt: np.ndarray,
    a_hist: np.ndarray,
    a_star: int,
    window: int,
) -> np.ndarray:
    """
    Rolling empirical manipulation-mistake rate using only rounds with `a_t=a^*` in each window.
    Length ``len(subopt) - window + 1`` (same alignment as ``np.convolve(..., mode='valid')``).
    NaN when no such rounds occur in the window.
    """
    n = len(subopt)
    if n < window:
        return np.array([], dtype=np.float64)
    out = np.empty(n - window + 1, dtype=np.float64)
    for i in range(n - window + 1):
        sub = subopt[i : i + window]
        m = a_hist[i : i + window] == a_star
        if not np.any(m):
            out[i] = np.nan
        else:
            out[i] = float(sub[m].mean())
    return out


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


def ci_mean_nan(
    data: np.ndarray,
    axis: int = 0,
    z: float = 1.96,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and approximate 95% CI ignoring NaNs per slice (effective n varies)."""
    m = np.nanmean(data, axis=axis)
    valid = np.sum(np.isfinite(data), axis=axis).astype(np.float64)
    valid = np.maximum(valid, 1.0)
    std = np.nanstd(data, axis=axis, ddof=1)
    std = np.where(np.isfinite(std), std, 0.0)
    se = std / np.sqrt(valid)
    return m, z * se


def follower_subopt_rate_when_leader_a_star(
    tr: Dict[str, np.ndarray],
    env: StackelbergBandit,
) -> float:
    a_star = int(env.stackelberg_leader_action())
    a_hist = tr["a_hist"]
    subopt = tr["subopt"]
    mask = a_hist == a_star
    if not np.any(mask):
        return float("nan")
    return float(subopt[mask].mean())


def ci_mean_nan_1d(data: np.ndarray, z: float = 1.96) -> Tuple[float, float]:
    x = data[np.isfinite(data)]
    if x.size == 0:
        return float("nan"), float("nan")
    m = float(x.mean())
    if x.size <= 1:
        return m, 0.0
    s = float(x.std(ddof=1)) / np.sqrt(x.size)
    return m, z * s
