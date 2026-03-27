from .backup import BackupLinearization
from .crown import CrownLinearization
from .taylor import TaylorLinearization
from .linear_derivative_bounds import CrownPartialLinearization


def default_linearization(dynamics):
    """
    Returns the default linearization method.
    """
    return BackupLinearization(TaylorLinearization(dynamics), CrownLinearization(dynamics))


__all__ = [
    "TaylorLinearization",
    "CrownLinearization",
    "BackupLinearization",
    "CrownPartialLinearization",
]
