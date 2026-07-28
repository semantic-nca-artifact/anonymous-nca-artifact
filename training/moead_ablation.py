"""Composable cooperative MOEA/D ablations on one shared CA backbone."""

import os

import numpy as np

from clip_loss import CLIPLoss
from training.moead_ca import train_moead_core
from training.moead_rta import RegionArchive


def configure_fixed_moead_strategy(
        *, variant_name, replacement_policy, pool_policy, use_rta,
        gradient_policy):
    """Apply the immutable policy contract of one named ablation script."""
    import config as _cfg

    _cfg.MOEAD_VARIANT_NAME = str(variant_name)
    _cfg.MOEAD_REPLACEMENT_POLICY = str(replacement_policy)
    _cfg.MOEAD_POOL_POLICY = str(pool_policy)
    _cfg.MOEAD_USE_RTA = bool(use_rta)
    _cfg.MOEAD_GRADIENT_POLICY = str(gradient_policy)
    _cfg.MOEAD_MAX_REPLACEMENTS = 1
    _cfg.MOEAD_COOPERATION_INTERVAL = 50
    _cfg.RTA_WARMUP_STEPS = 0
    _cfg.RTA_TRANSFER_INTERVAL = 50
    _cfg.POOL_SIZE = 1024
    _cfg.MOEAD_EVAL_INTERVAL = 10
    _cfg.MOEAD_POOL_FIGURE_INTERVAL = 100
    _cfg.MOEAD_CHECKPOINT_INTERVAL = 100
    _cfg.EVAL_STEPS = 64
    _cfg.FINAL_EVAL_STEPS = 96
    _cfg.FINAL_EVAL_REPEATS = 4


def train_fixed_moead_strategy(
        clip_loss, seed, *, variant_name, replacement_policy, pool_policy,
        use_rta, gradient_policy):
    """Run one explicitly named strategy through the shared MOEA/D core."""
    configure_fixed_moead_strategy(
        variant_name=variant_name,
        replacement_policy=replacement_policy,
        pool_policy=pool_policy,
        use_rta=use_rta,
        gradient_policy=gradient_policy,
    )
    return train_moead_ablation(clip_loss, seed)


def train_moead_ablation(clip_loss: CLIPLoss, seed: np.ndarray) -> dict:
    import config as _cfg

    variant_name = str(_cfg.MOEAD_VARIANT_NAME).strip().lower()
    if not variant_name or any(ch not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for ch in variant_name):
        raise ValueError(f'Unsafe MOEAD_VARIANT_NAME: {variant_name!r}')

    controller = None
    if _cfg.MOEAD_USE_RTA:
        controller = RegionArchive(enable_transfer=True)
        controller.archive_log_dir = os.path.join(
            _cfg.TRAIN_LOG_ROOT, variant_name, 'archive'
        )

    result = train_moead_core(
        clip_loss,
        seed,
        variant_name=variant_name,
        archive_controller=controller,
    )
    if controller is not None:
        controller.write_logs()
    return {
        **result['summary'],
        'results': result['results'],
        'active_solutions': result['active_solutions'],
        'archive_controller': controller,
        'summary_dir': result['summary_dir'],
    }
