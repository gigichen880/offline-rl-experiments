# Offline RL / Stackelberg manipulation experiments

Synthetic experiments for **hybrid offline–online follower manipulation** in Stackelberg bandits. The default entry point is a single script that runs tabular Hybrid-FMUCB (Algorithm 1), optional learning-curve figures, and an optional contextual extension (Algorithm 2). Rewards are **Bernoulli**; leader confidence sets use the regression-style radii in `hybrid_fmucb.py`.

---

## How to run

**Setup**

```bash
pip install -r requirements.txt
```

**Full default (Experiment 1 + Experiment 2 + learning curves)**  
Writes figures under `figures/` (change with `--out-dir`).

```bash
python run_experiments.py
```

**Run only part of the pipeline**

| Goal | Command |
|------|---------|
| Only Exp 1 | `python run_experiments.py --exp1-only` |
| Only Exp 2 | `python run_experiments.py --exp2-only` |
| Exp 1 and 2, no learning curves | `python run_experiments.py --exp1-only --exp2-only` |
| Only learning-curve plots | `python run_experiments.py --learning-curves-only` |
| Skip learning curves (keep Exp 1–2) | `python run_experiments.py --skip-learning-curves` |
| **Optional** Experiment 3 (contextual) | `python run_experiments.py --exp3` |

**Faster or stronger learning-curve runs** (same algorithms; tuning only)

```bash
# Quick smoke test
python run_experiments.py --learning-curves-only \
  --learning-curve-horizon 1500 --learning-curve-seeds 4 --learning-curve-n-off 500

# Longer horizon, more offline, capped γ for clearer curves
python run_experiments.py --learning-curves-only --learning-curve-paper-profile
```

**Useful flags** (see `python run_experiments.py --help` for all)

| Flag | Default | Role |
|------|---------|------|
| `--out-dir` | `figures` | Output directory |
| `--seeds` | `24` | Random seeds |
| `--horizon` | `8000` | Online rounds (Exp 1–2 and Exp 3) |
| `--gamma-exp3` | `0.01` | EXP3 exploration \(\gamma\) |
| `--confidence-delta` | `0.05` | High-probability parameter for regression confidence sets (Alg. 1) |
| `--skip-learning-curves` | off | Do not generate learning-curve PNGs |
| `--exp3` | off | Run optional Exp 3 |

Learning-curve–specific: `--learning-curve-horizon`, `--learning-curve-seeds`, `--learning-curve-n-off`, `--learning-curve-gamma-exp3`, `--learning-curve-paper-profile`, etc.

---

## What the three experiment sets do

### Experiment 1 (default on)

**Question:** How does **offline dataset size** \(N_{\mathrm{off}}\) affect total manipulation mistakes \(T_{f,w}\) and a simple convergence proxy?

**What it does:** For each seed, samples a random tabular game. For several values of \(N_{\mathrm{off}}\) (grid includes 0, 100, 500, 1000, 5000), builds a **uniform** offline dataset, then runs **online-only FMUCB** vs **hybrid** (offline init + online). Compares mistakes vs the true best manipulation rule \(F^{\mathrm{fm}}\).

**Outputs:** `exp1_noff_vs_tfw.png`, `exp1_noff_vs_convergence.png`, `exp1_cumulative_mistakes.png`, `exp1_summary.json`.

---

### Experiment 2 (default on)

**Question:** Does **offline coverage quality** (whether \((a^*,b^*)\) is well represented) change hybrid performance?

**What it does:** Fixes \(N_{\mathrm{off}}\) and compares three offline builders: **good** (biased toward Stackelberg pair), **neutral** (uniform), **poor** (avoids \((a^*,b^*)\)). Same online horizon and metrics as elsewhere.

**Outputs:** `exp2_coverage_bars.png`, `exp2_summary.json`.

---

### Experiment 3 (**optional**, `--exp3`)

**Question:** How does a **contextual** extension (features \(\phi(x,a,b)\), Ridge / LinUCB-style sets) compare to a simpler per-context tabular hybrid on the same horizon?

**What it does:** Only runs if you pass **`--exp3`**. Not part of the default run. Uses contextual simulators inside `run_experiments.py`; metrics are documented as contextual best-manipulation / LinUCB-style (see figure title and `exp3_summary.json`).

**Outputs:** `exp3_contextual_vs_tabular.png`, `exp3_summary.json`.

---

### Learning curves (default on unless `--skip-learning-curves`)

**What they do:** Rolling **global** mistake rate vs time, plus a **conditional** curve restricted to rounds with \(a_t = a^*\) (Stackelberg leader row), which is often easier to interpret than the global average when EXP3 visits many rows.

**Outputs:** `exp_optional_learning_curves.png`, `exp_optional_learning_curves_at_a_star.png`.

---

## File structure (what matters)

| File | Role |
|------|------|
| **`run_experiments.py`** | **Main entry:** CLI, tabular environment, EXP3 leader, `simulate_run`, Experiment 1–2, optional learning curves, optional Experiment 3, plotting. |
| **`hybrid_fmucb.py`** | Core **Algorithm 1** pieces: manipulation contrast, pessimistic leader feasibility, `hybrid_fmucb_pick`, `true_best_manipulation` (\(F^{\mathrm{fm}}\)), offline certification helpers, confidence radii. |
| **`experiment_common.py`**, **`experiment_1.py`**, **`experiment_2.py`**, **`experiment_3.py`** | Modular versions of experiments + shared helpers; useful for reuse or reading structure. The **`python run_experiments.py`** workflow uses the self-contained code in `run_experiments.py` plus imports from **`hybrid_fmucb.py`** only. |
| **`experiment.md`** | Extra detail on metrics and paper alignment (if present). |
| **`requirements.txt`** | `numpy`, `matplotlib`, `tqdm`. |

---

## Metric note

**\(T_{f,w}\)** counts rounds where \(b_t \neq F^{\mathrm{fm}}(a_t)\), with \(F^{\mathrm{fm}}\) the **true** best *qualified* manipulation rule from the ground-truth game (not plain per-round best-response error unless noted for Exp 3).

---

## Citation / theory

See your paper PDF and `experiment.md` for definitions (pooled means, pessimistic leader contrast, \(C_{\mathrm{man}}\), tabular vs contextual algorithms).
