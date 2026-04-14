# Experiments: Hybrid Offline-Online Follower Manipulation

This file describes the **implemented** synthetic experiments in `run_experiments.py` and how they match the **tabular reduction** of Hybrid-FMUCB (Algorithm 1). It should be read together with `hybrid_fmucb.py` and the paper.

---

## Implementation (what the code actually does)

### Environment

- Synthetic **general-sum Stackelberg bandit**: mean rewards \(\mu_\ell(a,b),\mu_f(a,b)\sim\mathrm{Uniform}(0,1)\) per pair (same sampled game for each seed in an experiment).
- Observed rewards are **Gaussian** around the means (not Bernoulli); algorithms use pooled empirical means as in Sec. 2 of the paper.
- **Leader:** EXP3 on observed leader rewards (`gamma_exp3`, default `0.05`).
- **Follower:** tabular **Hybrid-FMUCB** (not per-round myopic UCB only):
  - Response rules \(F_{a^\*,b}\): \(F(a^\*)=b^\*\); for \(a\neq a^\*\), \(F(a)\) is **worst for the leader** under current \(\hat\mu_\ell\) (tabular special case in Sec. 2).
  - **Feasibility:** manipulation contrast \(\Delta_{F,a^\*}(\hat\mu_\ell)\) must be **positive in a pessimistic sense**: LCB on \(\mu_\ell(a^\*,F(a^\*))\) minus max UCB on \(\mu_\ell(a',F(a'))\) over \(a'\neq a^\*\) (entrywise surrogate for \(\inf_{g_\ell\in\mathcal G_{\ell,t}}\Delta>0\)).
  - **Objective:** among feasible \((F,a^\*)\), maximize **UCB** on follower payoff \(\mu_f\) at the **target** cell \((a^\*,b^\*)\); then play **\(b_t=F(a_t)\)**.
  - **Offline + online:** counts and reward sums for **both** \(\mu_\ell\) and \(\mu_f\) are **pooled** in each cell \((a,b)\) as in the paper’s aggregated means.
- If **no** feasible manipulation exists at time \(t\), the follower falls back to **UCB best-response** on \(\mu_f\) for the current leader action (exploration fallback).

### Metric \(T_{f,w}\)

- **Paper-consistent definition in code:** count rounds where the follower’s action differs from the **true best qualified manipulation rule** \(F^{\mathrm{fm}}\):  
  \(b_t \neq F^{\mathrm{fm}}(a_t)\).
- \(F^{\mathrm{fm}}\) is computed from **true** \(\mu_\ell,\mu_f\) by enumerating rules \(F_{a^\*,b}\), requiring **plug-in** \(\Delta_{F,a^\*}(\mu_\ell)>0\), and maximizing \(\mu_f(a^\*,b^\*)\) among qualified pairs (see `true_best_manipulation` in `hybrid_fmucb.py`).
- This is **not** “follower did not best-respond to \(a_t\)”.

### Baseline vs hybrid in Experiment 1

- **Same** follower algorithm (tabular Hybrid-FMUCB with pooled estimates).
- **Baseline:** \(N_{\mathrm{off}}=0\) (no offline init).
- **Hybrid:** \(N_{\mathrm{off}}>0\) with offline data built as specified below.

---

## Experiment 1: Offline dataset size

### Objective

Compare online learning cost (\(T_{f,w}\)) as **offline dataset size** \(N_{\mathrm{off}}\) varies.

### Defaults (CLI)

- \(|A|,|B|\): default `8` (override `--n-a`, `--n-b`).
- Online horizon: default `8000` (`--horizon`).
- \(N_{\mathrm{off}}\in\{0,100,500,1000,5000\}\) (fixed in code).
- Optional cumulative-mistakes curve at one \(N_{\mathrm{off}}\) (`--exp1-cum-n-off`, default max grid).

### Outputs

- `exp1_noff_vs_tfw.png`: mean \(T_{f,w}\) vs \(N_{\mathrm{off}}\) (± ~95% CI over seeds).
- `exp1_noff_vs_convergence.png`: rounds until rolling **manipulation-mistake** rate is **sustainably** below an **absolute** threshold (default: rate \(\le 0.2\) for 3 consecutive window ends; window 200). This avoids the misleading “halve initial rate” scaling.
- `exp1_cumulative_mistakes.png`: cumulative \(T_{f,w}\)-style mistakes at fixed \(N_{\mathrm{off}}\).
- `exp1_summary.json`.

### Offline data (hybrid arms)

- **Uniform** over \((a,b)\) with both \(r_\ell\) and \(r_f\) logged (`OfflineRewardStats`).

### Expected qualitative behavior

- Hybrid curve often improves as \(N_{\mathrm{off}}\) increases when offline coverage helps identify \(\mu_\ell,\mu_f\) along manipulation-relevant cells; exact ordering can depend on seeds and noise.

---

## Experiment 2: Coverage of the optimal region

### Objective

Contrast offline designs that **include** frequent coverage of the Stackelberg-optimal pair \((a^\*,b^\*)\), **uniform** mixing, or **never** observe \((a^\*,b^\*)\).

### Defaults

- Fixed \(N_{\mathrm{off}}\): default `1000` (`--exp2-n-off`).
- Three conditions:
  - **Good:** 40% \((a^\*,b^\*)\), 40% \((a,\mathrm{BR}(a))\), 20% uniform \((a,b)\) (both rewards logged).
  - **Neutral:** uniform \((a,b)\).
  - **Poor:** never sample \((a^\*,b^\*)\); if \(a=a^\*\), sample \(b\neq b^\*\) (both rewards logged).
- Separate RNG streams: offline build vs online play (paired across conditions per seed).

### Outputs

- `exp2_coverage_bars.png`: three panels — global \(T_{f,w}\); conditional manipulation-mistake rate when \(a_t=a^\*\) (Stackelberg leader action); final-window “success” rate (1 − mean mistake in last window).
- `exp2_summary.json`.

### Note

- Global \(T_{f,w}\) and conditional metrics can differ: **uniform** can look strong on **global** follower behavior while **good** may win on **target** \((a^\*,\cdot)\) — leader uses EXP3 and visits all \(a\).

---

## Experiment 3 (optional, `--exp3`)

### Objective

Illustrate a **contextual** bandit variant (not the full function-class Algorithm 2 from the paper).

### What is implemented

- **Tabular contextual hybrid:** per-context UCB on \(\mu_f\) with offline warm start; \(T_{f,w}\) in the figure legend is **best-response–style** subopt from that simulator (see docstring on `simulate_contextual_tabular_hybrid`).
- **Linear / LinUCB-style** contextual baseline: `simulate_contextual_linucb_hybrid`.

### Defaults

- `--exp3-n-off` (default 1500), `--n-x` (default 5), same `--n-a`, `--n-b`, `--horizon` unless overridden.

### Output

- `exp3_contextual_vs_tabular.png`, `exp3_summary.json`.

---

## Visualization summary

| Artifact | Content |
| -------- | ------- |
| Exp 1 | Offline size vs \(T_{f,w}\); sustained low mistake rate; cumulative mistakes |
| Exp 2 | Good / neutral / poor coverage (three metrics) |
| Learning curves | Optional rolling manipulation-mistake rate (`--skip-learning-curves` to disable) |
| Exp 3 | Tabular vs contextual bar chart (optional) |

---

## Reproducibility

- Run: `python run_experiments.py` (see **README.md**).
- Use multiple seeds (`--seeds`, `--base-seed`); plots use normal-approximation error bars on the mean.

---

## Relation to the paper

- This codebase implements a **tabular reduction** of Hybrid-FMUCB with **pessimistic leader feasibility** and **optimistic follower selection**, **pooled offline–online means**, and **\(T_{f,w}\)** vs. **\(F^{\mathrm{fm}}\)**.
- **Not** implemented here: full regression confidence sets \(\mathcal G_{\ell,t},\mathcal G_{f,t}\), explicit \(C_{\mathrm{man}}\) certification checks, or the contextual Algorithm 2 in full generality (Exp 3 is a lighter comparison).

