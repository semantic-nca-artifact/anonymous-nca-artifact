"""Standalone MOEA/D-MRTA ablation: MGDA + region-wise RTA."""

from training.moead_ablation import (
    configure_fixed_moead_strategy,
    train_fixed_moead_strategy,
)


POLICY = {
    'variant_name': 'mrta',
    'replacement_policy': 'none',
    'pool_policy': 'preference_reset',
    'use_rta': True,
    'gradient_policy': 'mgda',
}


def configure_strategy():
    configure_fixed_moead_strategy(**POLICY)
    import config as _cfg

    _cfg.RTA_TRANSFER_POOL = False


def train_moead_mrta(clip_loss, seed):
    configure_strategy()
    return train_fixed_moead_strategy(clip_loss, seed, **POLICY)


train_moead_mrta_archive = train_moead_mrta
