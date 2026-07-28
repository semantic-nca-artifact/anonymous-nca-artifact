"""Standalone MOEA/D-MCA ablation."""
from training.moead_ablation import configure_fixed_moead_strategy, train_fixed_moead_strategy

POLICY = dict(variant_name='mca', replacement_policy='neighborhood', pool_policy='preference_reset', use_rta=False, gradient_policy='mgda')

def configure_strategy():
    configure_fixed_moead_strategy(**POLICY)

def train_moead_mca(clip_loss, seed):
    return train_fixed_moead_strategy(clip_loss, seed, **POLICY)
