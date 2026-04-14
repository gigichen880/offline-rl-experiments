"""Experiment 3 (optional): contextual tabular vs LinUCB-style contextual follower."""

from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import experiment_common as ec


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
    reward_noise_std: float = ec.DEFAULT_REWARD_NOISE_STD,
) -> Tuple[np.ndarray, np.ndarray]:
    n_visits = np.zeros((n_x, n_a, n_b), dtype=np.int64)
    sum_r = np.zeros((n_x, n_a, n_b), dtype=np.float64)
    for _ in range(n_off):
        x = int(rng.integers(0, n_x))
        a = int(rng.integers(0, n_a))
        b = int(rng.integers(0, n_b))
        r_f = _mu_from_theta(theta_f, x, a, b, n_x, n_a, n_b) + rng.normal(
            0.0, reward_noise_std
        )
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
    reward_noise_std: float = ec.DEFAULT_REWARD_NOISE_STD,
) -> Tuple[np.ndarray, np.ndarray]:
    d = n_x + n_a + n_b
    a_mat = ridge_reg * np.eye(d)
    b_vec = np.zeros(d)
    for _ in range(n_off):
        x = int(rng.integers(0, n_x))
        a = int(rng.integers(0, n_a))
        b = int(rng.integers(0, n_b))
        ph = phi_vec(x, a, b, n_x, n_a, n_b)
        r_f = _mu_from_theta(theta_f, x, a, b, n_x, n_a, n_b) + rng.normal(
            0.0, reward_noise_std
        )
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
        a = ec.exp3_sample(weights[x], gamma_exp3, rng)
        na = np.maximum(1, n_visits[x, a].astype(np.float64))
        mean = sum_r[x, a] / na
        bonus = np.sqrt(2.0 * np.log(max(1, t)) / na)
        ucb = mean + bonus
        maxv = ucb.max()
        cands = np.flatnonzero(np.isclose(ucb, maxv))
        b = int(rng.choice(cands))

        r_l = _mu_from_theta(theta_l, x, a, b, n_x, n_a, n_b) + rng.normal(0.0, reward_noise_std)
        r_f = _mu_from_theta(theta_f, x, a, b, n_x, n_a, n_b) + rng.normal(0.0, reward_noise_std)
        ec.exp3_update(weights[x], a, float(np.clip(r_l, 0.0, 1.0)), gamma_exp3)
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
        a = ec.exp3_sample(weights[x], gamma_exp3, rng)
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
        ec.exp3_update(weights[x], a, float(np.clip(r_l, 0.0, 1.0)), gamma_exp3)
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
    reward_noise_std: float = ec.DEFAULT_REWARD_NOISE_STD,
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
        off_tab = build_offline_contextual_tabular(
            theta_f, n_off, rng_tab, n_x, n_a, n_b, reward_noise_std=reward_noise_std
        )
        off_ctx = build_offline_contextual_linear(
            theta_f,
            n_off,
            rng_ctx,
            n_x,
            n_a,
            n_b,
            ridge_reg=1.0,
            reward_noise_std=reward_noise_std,
        )
        sub_tab = simulate_contextual_tabular_hybrid(
            theta_l,
            theta_f,
            n_x,
            n_a,
            n_b,
            horizon,
            rng_tab,
            gamma_exp3,
            reward_noise_std,
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
            reward_noise_std,
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
        color=[ec.COLORS["tabular"], ec.COLORS["contextual"]],
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
    ec._style_axis(ax)
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
