"""Entry point for the α-ProCeT method.

Paper name: ``\\textsc{$\\alpha$-ProCeT}`` — SOCP-protected repair with
Jacobian-based Top-N protection and per-step audit.

Example:
    python scripts/run_alpha_procet.py -a Tanh -s barr1 \\
        --lambda 2.0 --delta-theta-norm-bound 0.05 --num-inner-steps 5
"""

from _common import main_for
from procet.methods.alpha_procet import AlphaProCeTMethod


if __name__ == "__main__":
    main_for(AlphaProCeTMethod)()
