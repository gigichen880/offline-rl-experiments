"""
Tabular Hybrid-FMUCB helpers aligned with:
  "Hybrid Offline-Online Follower Manipulation in General-Sum Stackelberg Games"

- Worst-response rule F_{a*,b} (Sec. 2, tabular special case): F(a*)=b, and for a≠a*,
  F(a) ∈ argmin_{b'} μ_ℓ(a,b') (leader-side worst response).
- Manipulation contrast ∆_{F,a*}(g) from Eq. (2).
- Offline+online pooled means (same numerator/denominator structure as the paper).
- True best qualified manipulation (F^fm, a^fm) for the T_{f,w} evaluation in the paper
  (rounds where the follower does not play the true best manipulation strategy).

**Tabular reduction:** full Algorithm~1 uses regression sets G_{ℓ,t}, G_{f,t}. Leader feasibility
uses a **pessimistic** contrast: LCB on μ_ℓ at (a*, F(a*)) minus max UCB on μ_ℓ at (a', F(a')),
a'≠a*, which lower-bounds ∆_{F,a*} under entrywise uncertainty (finite analogue of
inf_{gℓ ∈ G_{ℓ,t}} ∆_{F,a*}(gℓ) > 0). Follower optimality still uses UCB on μ_f(a*, b*).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np


def worst_response_leader_row(mu_l_row: np.ndarray) -> int:
    """argmin_b μ_ℓ(a, b) for a fixed row (ties → smallest index)."""
    return int(np.argmin(mu_l_row))


def build_rule_F(a_star: int, b_star: int, mu_l_hat: np.ndarray) -> np.ndarray:
    """
    Tabular rule F_{a*,b*} from the paper: F(a*) = b*; for a ≠ a*, F(a) worst for leader.
    """
    n_a = mu_l_hat.shape[0]
    F = np.zeros(n_a, dtype=np.int32)
    for a in range(n_a):
        if a == a_star:
            F[a] = b_star
        else:
            F[a] = worst_response_leader_row(mu_l_hat[a].astype(np.float64))
    return F


def manipulation_contrast(mu_l: np.ndarray, F: np.ndarray, a_star: int) -> float:
    """
    ∆_{F,a*}(g) = g(a*, F(a*)) - max_{a≠a*} g(a, F(a)) with g tabular on A×B.
    """
    n_a = mu_l.shape[0]
    left = float(mu_l[a_star, F[a_star]])
    others = [float(mu_l[a, F[a]]) for a in range(n_a) if a != a_star]
    if not others:
        return left
    return left - max(others)


def _ucb1_radius(num_visits: int, t: int) -> float:
    """Bonus / one-sided confidence half-width (same scaling as UCB1 in run_experiments)."""
    n = max(1, int(num_visits))
    return float(np.sqrt(2.0 * np.log(max(1.0, float(t))) / n))


def manipulation_contrast_leader_pessimistic(
    sum_r_l: np.ndarray,
    n_visits: np.ndarray,
    F: np.ndarray,
    a_star: int,
    t: int,
) -> float:
    """
    Pessimistic lower bound on ∆_{F,a*}(μ_ℓ):

        LCB(a*, F(a*)) - max_{a'≠a*} UCB(a', F(a')),

    using pooled counts for each visited cell. This targets robust feasibility
    (paper: inf_{g in G_{ℓ,t}} ∆_{F,a*}(g) > 0) in a tabular, entrywise form.
    """
    n_a = len(F)
    at, bt = a_star, int(F[a_star])
    nt = max(1, int(n_visits[at, bt]))
    mean_t = float(sum_r_l[at, bt]) / nt
    lcb_target = mean_t - _ucb1_radius(nt, t)

    others: list[float] = []
    for a in range(n_a):
        if a == a_star:
            continue
        b = int(F[a])
        n_c = max(1, int(n_visits[a, b]))
        mean_c = float(sum_r_l[a, b]) / n_c
        others.append(mean_c + _ucb1_radius(n_c, t))

    if not others:
        return lcb_target
    return float(lcb_target - max(others))


def manipulation_contrast_leader_mean(
    sum_r_l: np.ndarray,
    n_visits: np.ndarray,
    F: np.ndarray,
    a_star: int,
) -> float:
    """Empirical ∆ using pooled means only (no LCB/UCB); looser, activates hybrid earlier in ablations."""
    mu_hat = pooled_mean(sum_r_l, n_visits)
    return float(manipulation_contrast(mu_hat, F, a_star))


def true_best_manipulation(
    mu_l: np.ndarray,
    mu_f: np.ndarray,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, int]:
    """
    Among qualified manipulations (∆>0), maximize follower payoff μ_f(a*, b*) at the target.
    Returns (F^fm, a^fm) with F^fm(a^fm)=b^fm encoded in F.

    If none qualify, falls back to myopic best-response mapping (identity of BR per row).
    """
    n_a, n_b = mu_f.shape
    best_val = -np.inf
    best_F: Optional[np.ndarray] = None
    best_a_fm = 0

    for a_star in range(n_a):
        for b_star in range(n_b):
            F = build_rule_F(a_star, b_star, mu_l)
            if manipulation_contrast(mu_l, F, a_star) <= eps:
                continue
            v = float(mu_f[a_star, b_star])
            if v > best_val:
                best_val = v
                best_F = F.copy()
                best_a_fm = a_star

    if best_F is None:
        F_br = np.array([worst_response_leader_row(mu_l[a]) for a in range(n_a)], dtype=np.int32)
        return F_br, int(np.argmax([mu_f[a, F_br[a]] for a in range(n_a)]))

    return best_F, best_a_fm


def pooled_mean(sum_r: np.ndarray, n_visits: np.ndarray) -> np.ndarray:
    """Entrywise sum_r / max(1, n_visits)."""
    n = np.maximum(1, n_visits.astype(np.float64))
    return sum_r.astype(np.float64) / n


LeaderFeasibility = Literal["pessimistic", "mean"]


def hybrid_fmucb_pick(
    a_t: int,
    n_visits: np.ndarray,
    sum_r_f: np.ndarray,
    sum_r_l: np.ndarray,
    t: int,
    n_a: int,
    n_b: int,
    rng: np.random.Generator,
    eps: float = 1e-9,
    follower_elim_eps: Optional[float] = None,
    leader_feasibility: LeaderFeasibility = "pessimistic",
    feasibility_margin: float = 0.0,
) -> int:
    """
    Tabular Hybrid-FMUCB step (Algorithm 1, tabular reduction):

    * **Feasibility:** leader-side contrast ``> eps - feasibility_margin`` (default margin 0;
      increase margin to soften strict pessimism when the feasible set is rarely non-empty).
      Use ``leader_feasibility="mean"`` for ablations with pooled means only.
    * **Follower objective:** among feasible (F, a*), maximize UCB on μ_f(a*, b*).

    Then b_t = F(a_t). If no feasible candidate, fall back to successive-elimination style
    UCB on the row μ_f(a_t,·): keep arms with LCB ≥ max UCB − ε (default ε = 2 max bonus),
    then play among survivors with max UCB.
    """
    mu_l_hat = pooled_mean(sum_r_l, n_visits)

    best_ucb = -np.inf
    best_F: Optional[np.ndarray] = None
    feas_threshold = float(eps) - float(feasibility_margin)

    for a_star in range(n_a):
        for b_star in range(n_b):
            F = build_rule_F(a_star, b_star, mu_l_hat)
            if leader_feasibility == "mean":
                contr = manipulation_contrast_leader_mean(sum_r_l, n_visits, F, a_star)
            else:
                contr = manipulation_contrast_leader_pessimistic(sum_r_l, n_visits, F, a_star, t)
            if contr <= feas_threshold:
                continue
            n = max(1, int(n_visits[a_star, b_star]))
            mean_f = float(sum_r_f[a_star, b_star]) / n
            bonus = np.sqrt(2.0 * np.log(max(1, t)) / n)
            ucb = mean_f + bonus
            if ucb > best_ucb:
                best_ucb = ucb
                best_F = F

    if best_F is None:
        return _ucb_best_response_row(
            a_t, n_visits, sum_r_f, t, n_b, rng, elim_eps=follower_elim_eps
        )

    return int(best_F[a_t])


def _ucb_best_response_row(
    a: int,
    n_visits: np.ndarray,
    sum_r: np.ndarray,
    t: int,
    n_b: int,
    rng: np.random.Generator,
    elim_eps: Optional[float] = None,
) -> int:
    """
    FMUCB-style fallback on μ_f(a,·) when the hybrid feasible set is empty:
    keep b iff LCB_b ≥ max UCB − ε, then play among survivors with max UCB.
    If ``elim_eps is None``, ε = 2·max bonus so the UCB leader always survives.
    """
    na = np.maximum(1, n_visits[a].astype(np.float64))
    mean = sum_r[a] / na
    bonus = np.sqrt(2.0 * np.log(max(1, t)) / na)
    lcb = mean - bonus
    ucb = mean + bonus
    ucb_max = float(np.max(ucb))
    eps = float(elim_eps) if elim_eps is not None else float(2.0 * np.max(bonus))
    survive = lcb >= ucb_max - eps
    if not np.any(survive):
        maxv = float(np.max(ucb))
        cands = np.flatnonzero(np.isclose(ucb, maxv))
        return int(rng.choice(cands))
    masked = np.where(survive, ucb, -np.inf)
    maxv = float(np.max(masked))
    cands = np.flatnonzero(np.isclose(masked, maxv))
    return int(rng.choice(cands))


@dataclass
class OfflineRewardStats:
    """Pooled offline counts and reward sums for both players (paper Sec. 2)."""

    n_visits: np.ndarray
    sum_r_f: np.ndarray
    sum_r_l: np.ndarray
