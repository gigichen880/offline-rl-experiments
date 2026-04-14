#!/usr/bin/env python3
"""
Entry point: runs Experiment 1 and 2 by default; optional learning curves; optional Exp 3.

Selective runs:
  --exp1-only           only Exp 1
  --exp2-only           only Exp 2
  --exp1-only --exp2-only   Exp 1 + Exp 2, no learning curves
  --learning-curves-only    only exp_optional_learning_curves.png (--learning-curves-only wins over --exp*-only)

Core simulation lives in experiment_common.py and hybrid_fmucb.py.
"""

from __future__ import annotations

import argparse
import os

from experiment_1 import experiment1, learning_curve_figure
from experiment_2 import experiment2
from experiment_3 import experiment3_contextual


def main() -> None:
    parser = argparse.ArgumentParser(description="Stackelberg bandit offline-online experiments")
    parser.add_argument("--out-dir", type=str, default="figures", help="Output directory for plots")
    parser.add_argument("--seeds", type=int, default=24, help="Number of random seeds")
    parser.add_argument("--base-seed", type=int, default=0, help="Base seed offset")
    parser.add_argument("--n-a", type=int, default=8, help="|A| (leader actions)")
    parser.add_argument("--n-b", type=int, default=8, help="|B| (follower actions)")
    parser.add_argument("--horizon", type=int, default=8000, help="Online rounds")
    parser.add_argument(
        "--gamma-exp3",
        type=float,
        default=0.01,
        help=r"EXP3 exploration $\gamma$ (lower ⇒ leader visits good rows more often)",
    )
    parser.add_argument(
        "--reward-noise-std",
        type=float,
        default=0.01,
        help="Gaussian reward noise std (offline + online)",
    )
    parser.add_argument(
        "--follower-elim-eps",
        type=float,
        default=None,
        help="FMUCB fallback elimination slack (default: 2× max UCB bonus in the row)",
    )
    parser.add_argument(
        "--leader-feasibility",
        choices=["pessimistic", "mean"],
        default="pessimistic",
        help="Leader-side feasibility: pessimistic LCB/UCB (paper) or pooled means only (ablation, activates hybrid sooner)",
    )
    parser.add_argument(
        "--feasibility-margin",
        type=float,
        default=0.0,
        help="Relax feasibility threshold: require contrast > eps - margin (e.g. 0.02–0.05 if the feasible set is rarely non-empty)",
    )
    parser.add_argument(
        "--exp1-cum-n-off",
        type=int,
        default=None,
        help="Offline size for cumulative-mistakes curve (must be in --n-off-grid; default: max grid value)",
    )
    parser.add_argument("--exp2-n-off", type=int, default=1000, help="Fixed offline size for Exp. 2")
    parser.add_argument("--skip-learning-curves", action="store_true")
    parser.add_argument(
        "--learning-curves-only",
        action="store_true",
        help="Skip Exp 1 and Exp 2; run only the optional learning-curve figure (--learning-curves-only overrides --exp1-only / --exp2-only)",
    )
    parser.add_argument(
        "--exp1-only",
        action="store_true",
        help="Run only Experiment 1 (no Exp 2, no learning curves, unless combined with --exp2-only)",
    )
    parser.add_argument(
        "--exp2-only",
        action="store_true",
        help="Run only Experiment 2 (no Exp 1, no learning curves, unless combined with --exp1-only)",
    )
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

    elim = args.follower_elim_eps
    noise = args.reward_noise_std
    feas_mode = args.leader_feasibility
    feas_margin = args.feasibility_margin

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
            reward_noise_std=noise,
            follower_elim_eps=elim,
            leader_feasibility=feas_mode,
            feasibility_margin=feas_margin,
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
            reward_noise_std=noise,
            follower_elim_eps=elim,
            leader_feasibility=feas_mode,
            feasibility_margin=feas_margin,
        )
    if run_lc:
        learning_curve_figure(
            args.out_dir,
            seeds,
            args.n_a,
            args.n_b,
            args.horizon,
            args.gamma_exp3,
            n_off=1000,
            progress=show_p,
            reward_noise_std=noise,
            follower_elim_eps=elim,
            leader_feasibility=feas_mode,
            feasibility_margin=feas_margin,
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
            reward_noise_std=noise,
        )

    print(f"Wrote figures to {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
