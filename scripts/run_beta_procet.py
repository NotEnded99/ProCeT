"""Entry point for the β-ProCeT method.

Paper name: ``\\textsc{$\\beta$-ProCeT}`` — adaptive repair that starts in
CeT mode and escalates to α-ProCeT (with backtracking) on the first plateau.

Example:
    python scripts/run_beta_procet.py -a Sigmoid -s barr1 \\
        --lambda 2.0 --delta-theta-norm-bound 0.05 --num-inner-steps 5
"""

from _common import main_for
from procet.methods.beta_procet import BetaProCeTMethod


if __name__ == "__main__":
    main_for(BetaProCeTMethod)()
