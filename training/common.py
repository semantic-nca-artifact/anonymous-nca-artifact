import numpy as np
import torch
import torch.optim as optim

from config import LR, LR_DECAY_STEP, STATE_CLIP_VALUE


def finite_clip_state(x: torch.Tensor) -> torch.Tensor:
    x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
    return x.clamp(-STATE_CLIP_VALUE, STATE_CLIP_VALUE)


def validate_loss_weights(loss_weights):
    validated = []
    for w1, w2 in loss_weights:
        w1, w2 = float(w1), float(w2)
        if w1 < 0.0 or w2 < 0.0:
            raise ValueError('Loss weights must be non-negative, got (%s, %s)' % (w1, w2))
        if not np.isclose(w1 + w2, 1.0):
            raise ValueError('Loss weights must sum to 1, got (%s, %s)' % (w1, w2))
        validated.append((w1, w2))
    return validated


def weight_run_name(w1, w2):
    return 'w1_%.2f_w2_%.2f' % (w1, w2)


def make_optimizer(model: torch.nn.Module, train_steps: int = None):
    optimizer = optim.Adam(model.parameters(), lr=LR)
    if train_steps is not None:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_steps, eta_min=LR * 0.01)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_DECAY_STEP, gamma=0.1)
    return optimizer, scheduler
