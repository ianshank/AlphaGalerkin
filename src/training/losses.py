"""Loss functions for neural operator training.

Backwards-compatibility wrapper. All implementations now live in
``src.training.losses.operator``.
"""

from src.training.losses import get_loss  # noqa: F401
from src.training.losses.operator import H1Loss  # noqa: F401
from src.training.losses.operator import L2RelativeLoss  # noqa: F401
from src.training.losses.operator import MSELoss  # noqa: F401
