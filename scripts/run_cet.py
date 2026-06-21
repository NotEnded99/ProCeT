"""Entry point for the CeT method.

Paper name: ``\\textsc{CeT}`` — plain gradient descent on the LBP repair loss.

Example:
    python scripts/run_cet.py -a Tanh -s barr1
    python scripts/run_cet.py -a Relu -s simple2d --num-inner-steps 5
"""

from _common import main_for
from procet.methods.cet import CeTMethod


if __name__ == "__main__":
    main_for(CeTMethod)()
