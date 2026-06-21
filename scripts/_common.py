"""Common CLI helpers shared by ``run_cet.py`` / ``run_alpha_procet.py`` /
``run_beta_procet.py``.

Each method-specific script calls ``build_parser(method_cls)`` to get an
``argparse`` parser pre-populated with method-specific defaults, then invokes
``run_repair`` with the parsed args.
"""

import argparse
import os
import sys

# Make ``procet`` importable whether the script is launched from the repo
# root or from inside ``scripts/``.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from procet.core.systems import DYNAMICS_SYSTEMS, SYSTEM_DEPTH  # noqa: E402
from procet.methods.base import MethodConfig  # noqa: E402
from procet.runner import run_repair  # noqa: E402


def build_parser(description, method_cls):
    """Build an argparse parser for a specific repair method.

    Args:
        description: Help string for the script.
        method_cls:  The ``RepairMethod`` subclass — used to set
            ``choices`` for ``--activation`` and any method-specific
            argument defaults.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--activation", "-a", type=str, required=True,
        choices=list(method_cls.supported_activations),
        help="Barrier network activation function",
    )
    parser.add_argument(
        "--system", "-s", type=str, required=True,
        choices=list(DYNAMICS_SYSTEMS.keys()),
        help="Dynamical system",
    )
    parser.add_argument(
        "--num-inner-steps", type=int, default=5,
        help="Gradient / SOCP updates per outer iteration (default: 5)",
    )
    parser.add_argument("--lr", type=float, default=5e-3,
                        help="Base learning rate (depth-aware overrides apply)")
    parser.add_argument("--target-pass-rate", type=float, default=100.0)
    parser.add_argument("--patience", type=int, default=3,
                        help="Stop after N iterations without improvement")
    parser.add_argument("--max-total-iterations", type=int, default=10)
    parser.add_argument(
        "--lambda", dest="lambda_", type=float, default=2.0,
        help="Weight for definitive-fail regions vs uncertain regions",
    )
    parser.add_argument("--seed", type=int, default=2026)

    # Methods that need Top-N selection accept these extra knobs.
    if method_cls.name in ("alpha-procet", "beta-procet"):
        parser.add_argument(
            "--top-n-protect", type=int,
            default=50,
            help="Top-N most vulnerable regions to protect / audit",
        )
        parser.add_argument(
            "--delta-theta-norm-bound", type=float,
            default= 0.05,
            help="L2 trust-region radius ζ for the SOCP update",
        )

    return parser


def main_for(method_cls):
    """Return a ``main()`` function bound to a specific method class."""
    def main():
        parser = build_parser(
            description=f"Neural CBF Iterative Repair — {method_cls.display_name}",
            method_cls=method_cls,
        )
        args = parser.parse_args()

        cfg = MethodConfig(
            system=args.system,
            activation=args.activation,
            num_inner_steps=args.num_inner_steps,
            lr=args.lr,
            target_pass_rate=args.target_pass_rate,
            patience=args.patience,
            max_total_iterations=args.max_total_iterations,
            lambda_=args.lambda_,
            top_n_protect=getattr(args, "top_n_protect", 100),
            delta_theta_norm_bound=getattr(args, "delta_theta_norm_bound", 0.01),
            seed=args.seed,
            max_depth_limit=SYSTEM_DEPTH[args.system],
        )
        method = method_cls(cfg)
        run_repair(method, cfg)
    return main
