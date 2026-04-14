# Offline RL / Stackelberg manipulation experiments

Synthetic experiments for **hybrid offline–online follower manipulation** in tabular Stackelberg bandits. The implementation follows the **tabular reduction** of Hybrid-FMUCB (Algorithm 1): explicit response rules \(F:A\to B\), manipulation contrast \(\Delta_{F,a}\) on the **leader** model, pessimistic feasibility (LCB/UCB), follower UCB among feasible rules, and pooled offline–online reward estimates.

## Quick start

```bash
pip install -r requirements.txt
python run_experiments.py
```

Figures and JSON summaries are written to `figures/` (override with `--out-dir`).

### Common options

| Flag | Default | Meaning |
| ---- | ------- | ------- |
| `--out-dir` | `figures` | Output directory |
| `--seeds` | `24` | Number of random seeds |
| `--n-a`, `--n-b` | `8` | \(\|A\|,\|B\|\) |
| `--horizon` | `8000` | Online rounds |
| `--gamma-exp3` | `0.05` | EXP3 exploration |
| `--no-progress` | off | Disable tqdm bars |
| `--skip-learning-curves` | off | Skip optional learning-curve figure |
| `--exp3` | off | Also run contextual comparison (Exp 3) |

Experiment-specific: `--exp1-cum-n-off`, `--exp2-n-off`, `--exp3-n-off`, `--n-x`.

## What gets generated

| File | Description |
| ---- | ----------- |
| `exp1_noff_vs_tfw.png` | \(N_{\mathrm{off}}\) vs \(T_{f,w}\) (mistakes vs true \(F^{\mathrm{fm}}\)) |
| `exp1_noff_vs_convergence.png` | Time to sustain low rolling mistake rate (absolute threshold) |
| `exp1_cumulative_mistakes.png` | Cumulative mistakes at large \(N_{\mathrm{off}}\) |
| `exp1_summary.json` | Means and metric definitions |
| `exp2_coverage_bars.png` | Good / neutral / poor offline coverage |
| `exp2_summary.json` | |
| `exp_optional_learning_curves.png` | Rolling mistake rate over time (unless skipped) |
| `exp3_contextual_vs_tabular.png` | Only if `--exp3` |
| `exp3_summary.json` | Only if `--exp3` |

Experiment 3 counts **best-response** mistakes per round (contextual UCB / LinUCB), not paper-\(T_{f,w}\) vs \(F^{\mathrm{fm}}\) — see `experiment.md`.

## Code layout

| Path | Role |
| ---- | ---- |
| `run_experiments.py` | CLI only: runs Exp 1–2 (and optional learning curve) by default; `--exp3` for contextual script |
| `experiment_common.py` | `StackelbergBandit`, EXP3, offline builders, `simulate_run`, style, CI helpers |
| `experiment_1.py` | Exp 1 figures + `learning_curve_figure` |
| `experiment_2.py` | Exp 2 figures |
| `experiment_3.py` | Optional contextual simulators + Exp 3 figure |
| `hybrid_fmucb.py` | \(\Delta_{F,a}\), worst-response rules \(F_{a,b}\), pessimistic feasibility, `hybrid_fmucb_pick`, `true_best_manipulation` (\(F^{\mathrm{fm}}\)) |
| `experiment.md` | Detailed alignment between experiments and implementation |

## Metrics (important)

- **\(T_{f,w}\):** counts rounds with \(b_t \neq F^{\mathrm{fm}}(a_t)\), where \(F^{\mathrm{fm}}\) is the **true** best qualified manipulation rule from the ground-truth game (not mere best-response error).

## Requirements

See `requirements.txt` (NumPy, Matplotlib, tqdm).

## Citation / theory

See your paper PDF and `experiment.md` for the mapping between code and definitions (pooled means, pessimistic leader contrast, tabular vs full function-class algorithms).

