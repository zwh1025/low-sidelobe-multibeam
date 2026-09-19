"""Training: differentiable physics losses and training loop."""

from .losses import PhysicsLoss
from .trainer import pregenerate, fast_val_sll, train_run

__all__ = ["PhysicsLoss", "pregenerate", "fast_val_sll", "train_run"]
