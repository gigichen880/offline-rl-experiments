"""
Tabular Hybrid-FMUCB helpers aligned with:
  "Hybrid Offline-Online Follower Manipulation in General-Sum Stackelberg Games"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def worst_response_leader_row(mu_l_row: np.ndarray) -> int:
    """argmin_b mu_l(a, b) for a fixed row (ties -> smallest index)."""
    return int(np.argmin(mu_l_row))


def build_rule_F(a_star: int, b_star: int, mu_l_hat: np.ndarray) -> np.ndarray:
    """
    Tabular rule F_{a*,b*} (paper Sec. 2 tabular special case):
      F(a*) = b*
      F(a)  = argmin_b mu_l_hat(a, b)   for a != a*
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
    Delta_{F,a*}(g) = g(a*, F(a*)) - max_{a != a*} g(a, F(a))  [Eq. (2)].
    """
    n_a = mu_l.shape[0]
    left = float(mu_l[a_star, F[a_star]])
    others = [float(mu_l[a, F[a]]) for a in range(n_a) if a != a_star]
    if not others:
        return left
    return left - max(others)


# ---------------------------------------------------------------------------
# Deviation 2: Regression-based confidence sets  (Alg 1 lines 5-8)
# ---------------------------------------------------------------------------

def regression_confidence_radius(
    n_visits: int,
    t: int,
    n_a: int,
    n_b: int,
    delta: float = 0.05,
) -> float:
    """
    Per-cell confidence half-width for regression confidence sets.

    For tabular [0,1]-bounded Bernoulli rewards the ERM is the empirical mean.
    A union-bounded Hoeffding inequality over |A|*|B| cells and t^2 rounds gives:

        beta_t = 2 * log(2 * |A| * |B| * t^2 / delta)
        w_t(a,b) = sqrt(beta_t / max(1, N_t(a,b)))

    This is the tabular special case of the regression confidence set radius
    beta_t in Algorithm 1 (paper Sec. 2 and Appendix D).
    """
    beta_t = 2.0 * np.log(2.0 * n_a * n_b * max(1, t) ** 2 / delta)
    n = max(1, int(n_visits))
    return float(np.sqrt(beta_t / n))


# ---------------------------------------------------------------------------
# Pessimistic feasibility using regression confidence sets
# ---------------------------------------------------------------------------

def manipulation_contrast_leader_pessimistic(
    sum_r_l: np.ndarray,
    n_visits: np.ndarray,
    F: np.ndarray,
    a_star: int,
    t: int,
    n_a: int,
    n_b: int,
    delta: float = 0.05,
) -> float:
    """
    Pessimistic lower bound on Delta_{F,a*}(mu_l) using regression confidence sets:

        LCB_l(a*, F(a*)) - max_{a'!=a*} UCB_l(a', F(a'))

    where
        LCB_l(a,b) = mu_hat(a,b) - w_t(a,b)
        UCB_l(a,b) = mu_hat(a,b) + w_t(a,b)

    This is the tabular implementation of
        inf_{g in G_{l,t}} Delta_{F,a*}(g) > 0   (Alg 1 line 6).
    """
    at, bt = a_star, int(F[a_star])
    nt = int(n_visits[at, bt])
    mean_t = float(sum_r_l[at, bt]) / max(1, nt)
    w_target = regression_confidence_radius(nt, t, n_a, n_b, delta)
    lcb_target = mean_t - w_target

    others: list[float] = []
    for a in range(n_a):
        if a == a_star:
            continue
        b = int(F[a])
        n_c = int(n_visits[a, b])
        mean_c = float(sum_r_l[a, b]) / max(1, n_c)
        w_c = regression_confidence_radius(n_c, t, n_a, n_b, delta)
        others.append(mean_c + w_c)

    if not others:
        return lcb_target
    return float(lcb_target - max(others))


# ---------------------------------------------------------------------------
# Deviation 3: Explicit C_man coefficient (Eq. 3) and offline certification
# ---------------------------------------------------------------------------

def compute_c_man(
    F: np.ndarray,
    a_star: int,
    n_a: int,
    n_b: int,
    nu_counts: np.ndarray,
) -> float:
    """
    Qualified-manipulation transfer coefficient C_man(F, a*) from Eq. (3):

        C_man(F, a*) = max(0,
            sup_{g in G, h=g-mu_l}  Delta_{F,a*}(h)
            / sqrt(E_{(a,b)~nu}[h(a,b)^2])
        )

    For the tabular function class G with uniform pointwise radius eps around
    mu_l, h(a,b) = g(a,b) - mu_l(a,b) in [-eps, eps].

    The numerator Delta_{F,a*}(h) = h(a*, F(a*)) - max_{a!=a*} h(a, F(a)).
    Worst-case sup over G: set h(a*, F(a*)) = +eps and h(a, F(a)) = -eps for
    the one a' that achieves the max.  So sup numerator = 2*eps (two relevant cells).

    Denominator: sqrt(E_nu[h^2]).  With h concentrated on the two relevant cells
    and h(a,b)=eps there:
        E_nu[h^2] = eps^2 * (nu(a*, F(a*)) + nu(a', F(a'))) / nu_total

    After dividing out eps:
        C_man = 2 / sqrt((nu(a*, F(a*)) + nu(a_argmax, F(a_argmax))) / nu_total)

    We sum over all a' != a* in the denominator (conservative: all could be argmax).

    Parameters
    ----------
    F         : response rule, shape (n_a,)
    a_star    : target leader action
    n_a, n_b  : action space sizes
    nu_counts : offline visit counts proxy for nu, shape (n_a, n_b)
    """
    nu_total = float(nu_counts.sum())
    if nu_total <= 0.0:
        return float("inf")

    # Cells relevant to Delta_{F,a*}: (a*, F(a*)) and all (a, F(a)) for a != a*
    nu_relevant = float(nu_counts[a_star, int(F[a_star])])
    for a in range(n_a):
        if a != a_star:
            nu_relevant += float(nu_counts[a, int(F[a])])

    if nu_relevant <= 0.0:
        return float("inf")

    # C_man = (1 + 1) / sqrt(nu_relevant / nu_total)  [target + 1 worst other]
    # More conservatively, count all n_a-1 other cells as potential argmax:
    # numerator = 2 (target gets +eps, one other gets -eps -> contrast = 2eps)
    denom_factor = float(np.sqrt(nu_relevant / nu_total))
    c_man = 2.0 / denom_factor
    return float(max(0.0, c_man))


def certify_offline(
    F: np.ndarray,
    a_star: int,
    mu_l_hat: np.ndarray,
    mu_l_true: np.ndarray,
    nu_counts: np.ndarray,
    delta3: float,
    n_a: int,
    n_b: int,
) -> Tuple[bool, float, float]:
    """
    Check Theorem 1 condition 4:
        C_man(F, a*) * sqrt(E_nu[(mu_l_hat - mu_l)^2]) <= delta3 / 4

    Returns (certified, lhs, threshold).
    'certified' is True when the offline data alone guarantees (F, a*) is a
    feasible manipulation before any online interaction (paper Sec. 2, Thm 1).

    Parameters
    ----------
    mu_l_hat  : offline leader reward estimate (ERM on D_off)
    mu_l_true : ground-truth mu_l
    nu_counts : offline visit counts, shape (n_a, n_b)
    delta3    : true manipulation contrast Delta_{F,a*}(mu_l)
    """
    c_man = compute_c_man(F, a_star, n_a, n_b, nu_counts)
    nu_total = float(nu_counts.sum())
    if nu_total <= 0:
        return False, float("inf"), delta3 / 4.0

    nu_prob = nu_counts.astype(np.float64) / nu_total
    mse_offline = float(np.sum(nu_prob * (mu_l_hat - mu_l_true) ** 2))
    lhs = c_man * float(np.sqrt(mse_offline))
    threshold = delta3 / 4.0
    return (lhs <= threshold), lhs, threshold


# ---------------------------------------------------------------------------
# Core algorithm: pooled mean and Hybrid-FMUCB pick
# ---------------------------------------------------------------------------

def pooled_mean(sum_r: np.ndarray, n_visits: np.ndarray) -> np.ndarray:
    """Entrywise sum_r / max(1, n_visits)  (paper Sec. 2 aggregated estimator)."""
    n = np.maximum(1, n_visits.astype(np.float64))
    return sum_r.astype(np.float64) / n


def hybrid_fmucb_pick(
    a_t: int,
    n_visits: np.ndarray,
    sum_r_f: np.ndarray,
    sum_r_l: np.ndarray,
    t: int,
    n_a: int,
    n_b: int,
    rng: np.random.Generator,
    delta: float = 0.05,
    eps: float = 1e-9,
) -> Tuple[int, bool]:
    """
    Tabular Hybrid-FMUCB step (Algorithm 1, tabular reduction).

    Feasibility (pessimistic, Alg 1 line 6):
        inf_{g in G_{l,t}} Delta_{F,a*}(g) > 0
        implemented as: LCB_l(a*, F(a*)) - max_{a'!=a*} UCB_l(a', F(a')) > eps
        using regression_confidence_radius (calibrated to Bernoulli rewards).

    Follower objective (optimistic, Alg 1 line 7):
        sup_{gf in G_{f,t}} gf(a*, F(a*))
        = mu_hat_f(a*, b*) + w_t(a*, b*)

    Returns (b_t, fallback_used).
    fallback_used=True when M_t={} and UCB best-response is played instead.
    """
    mu_l_hat = pooled_mean(sum_r_l, n_visits)

    best_ucb_f = -np.inf
    best_F: Optional[np.ndarray] = None

    for a_star in range(n_a):
        for b_star in range(n_b):
            F = build_rule_F(a_star, b_star, mu_l_hat)
            pess = manipulation_contrast_leader_pessimistic(
                sum_r_l, n_visits, F, a_star, t, n_a, n_b, delta
            )
            if pess <= eps:
                continue
            n_fb = max(1, int(n_visits[a_star, b_star]))
            mean_f = float(sum_r_f[a_star, b_star]) / n_fb
            w_f = regression_confidence_radius(n_fb, t, n_a, n_b, delta)
            ucb_f = mean_f + w_f
            if ucb_f > best_ucb_f:
                best_ucb_f = ucb_f
                best_F = F

    if best_F is None:
        # Deviation 4 fallback: UCB best-response on mu_f(a_t, .)
        b_fb = _ucb_best_response_row(a_t, n_visits, sum_r_f, t, n_a, n_b, delta, rng)
        return b_fb, True

    return int(best_F[a_t]), False


def _ucb_best_response_row(
    a: int,
    n_visits: np.ndarray,
    sum_r: np.ndarray,
    t: int,
    n_a: int,
    n_b: int,
    delta: float,
    rng: np.random.Generator,
) -> int:
    """UCB best-response on mu_f(a, .) -- fallback when M_t = {}."""
    n_b_local = sum_r.shape[1]
    ucb = np.array([
        float(sum_r[a, b]) / max(1, int(n_visits[a, b]))
        + regression_confidence_radius(int(n_visits[a, b]), t, n_a, n_b, delta)
        for b in range(n_b_local)
    ])
    maxv = ucb.max()
    cands = np.flatnonzero(np.isclose(ucb, maxv))
    return int(rng.choice(cands))


# ---------------------------------------------------------------------------
# Ground-truth best manipulation (for T_{f,w} evaluation)
# ---------------------------------------------------------------------------

def true_best_manipulation(
    mu_l: np.ndarray,
    mu_f: np.ndarray,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, int]:
    """
    Among all qualified manipulations (Delta_{F,a*}(mu_l) > 0), return
    (F^fm, a^fm) that maximises mu_f(a^fm, F^fm(a^fm)).

    Falls back to myopic BR mapping if no manipulation is qualified.
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
        F_br = np.array(
            [int(np.argmax(mu_f[a])) for a in range(n_a)], dtype=np.int32
        )
        return F_br, int(np.argmax([mu_f[a, F_br[a]] for a in range(n_a)]))

    return best_F, best_a_fm


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class OfflineRewardStats:
    """Pooled offline counts and reward sums for both players (paper Sec. 2)."""
    n_visits: np.ndarray   # shape (n_a, n_b), dtype int64
    sum_r_f: np.ndarray    # shape (n_a, n_b)
    sum_r_l: np.ndarray    # shape (n_a, n_b)