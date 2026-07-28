import os
import numpy as np
import torch
import torch.nn.utils as nn_utils

from config import (
    BATCH_SIZE, POOL_SIZE, TRAIN_STEPS,
    STEPS_MIN, STEPS_MAX,
    USE_PATTERN_POOL, DAMAGE_N,
    WEIGHTED_LOSS_WEIGHTS, TRAIN_LOG_ROOT,
)
from model import CAModel
from utils import SamplePool, make_circle_masks, generate_pool_figures, visualize_batch, plot_loss
from clip_loss import CLIPLoss
from training.common import finite_clip_state, validate_loss_weights, weight_run_name, make_optimizer


def _make_rank_fn(clip_loss_fn, text_embeddings, w, device):
    """Return pool indices sorted from best to worst by weighted objective loss."""
    def rank_fn(x_np):
        x_t = finite_clip_state(torch.tensor(x_np, device=device))
        with torch.no_grad():
            losses = clip_loss_fn.compute_objective_losses(x_t, text_embeddings)
            loss_stack = torch.stack(losses, dim=0)
            w_t = torch.tensor(w, dtype=torch.float32, device=device)
            combined = (w_t[:, None] * loss_stack).sum(dim=0)
            combined = torch.where(
                torch.isfinite(combined),
                combined,
                torch.full_like(combined, float('inf')),
            )
        return combined.cpu().numpy().argsort()
    return rank_fn


def _train_step(ca, optimizer, scheduler, clip_loss_fn, x0, w, text_embeddings, device):
    x = finite_clip_state(torch.tensor(x0, device=device))
    iter_n = int(torch.randint(STEPS_MIN, STEPS_MAX, ()).item())

    optimizer.zero_grad()
    for _ in range(iter_n):
        x = ca(x)
        x = finite_clip_state(x)
    losses = clip_loss_fn.compute_objective_losses(x, text_embeddings)
    loss_stack = torch.stack(losses, dim=0)
    w_t = torch.tensor(w, dtype=torch.float32, device=device)
    weighted = (w_t[:, None] * loss_stack).sum(dim=0)
    loss = weighted.mean()
    if not torch.isfinite(loss):
        loss = torch.tensor(0.0, device=device, requires_grad=True)

    loss.backward()
    grad_norm = nn_utils.clip_grad_norm_(ca.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    per_sim = torch.stack([-l.mean() for l in losses])
    return finite_clip_state(x).detach(), loss.detach(), per_sim.detach(), grad_norm


def _pool_rank(x0_np, clip_loss_fn, text_embeddings, w, seed, h, w_size, device):
    x0_t = finite_clip_state(torch.tensor(x0_np, device=device))
    with torch.no_grad():
        losses = clip_loss_fn.compute_objective_losses(x0_t, text_embeddings)
        loss_stack = torch.stack(losses, dim=0)
        w_t = torch.tensor(w, dtype=torch.float32, device=device)
        combined = (w_t[:, None] * loss_stack).sum(dim=0)
        # Treat non-finite values as worst-case losses.
        combined = torch.where(torch.isfinite(combined), combined, torch.full_like(combined, float('inf')))
    loss_rank = combined.cpu().numpy().argsort()[::-1]
    x0_np = x0_np[loss_rank]
    x0_np[:1] = seed  # Replace the worst sample with a fresh seed.
    if DAMAGE_N:
        damage = 1.0 - make_circle_masks(DAMAGE_N, h, w_size)
        x0_np[-DAMAGE_N:] *= damage
    return x0_np


def run_weighted_training(w1, w2, clip_loss: CLIPLoss, seed: np.ndarray) -> dict:
    """Two-objective convenience wrapper around the general trainer."""
    return _run_weighted_training_for_weights([w1, w2], clip_loss, seed)


def _run_weighted_training_for_weights(w: list, clip_loss: CLIPLoss, seed: np.ndarray) -> dict:
    device = next(clip_loss.model.parameters()).device
    ca = CAModel().to(device)
    optimizer, scheduler = make_optimizer(ca)

    text_embeddings = clip_loss.embed_objective_prompts()

    _, _, h, w_size = seed.shape  # NCHW
    pool = SamplePool(x=np.repeat(seed, POOL_SIZE, 0))
    loss_log = []
    run_name = weight_run_name(*w) if len(w) == 2 else 'w_' + '_'.join('%.2f' % wi for wi in w)
    log_dir = os.path.join(TRAIN_LOG_ROOT, 'weighted_sum', run_name)
    os.makedirs(log_dir, exist_ok=True)

    w_str = ' + '.join('%.2f*l%d' % (wi, i + 1) for i, wi in enumerate(w))
    print('\n=== Training %s: loss = %s ===' % (run_name, w_str))
    last_per_sim = last_grad_norm = None

    for step in range(TRAIN_STEPS + 1):
        if USE_PATTERN_POOL:
            batch = pool.sample(BATCH_SIZE)
            x0 = _pool_rank(batch.x.copy(), clip_loss, text_embeddings, w, seed, h, w_size, device)
        else:
            x0 = np.repeat(seed, BATCH_SIZE, 0)

        x, loss, per_sim, grad_norm = _train_step(
            ca, optimizer, scheduler, clip_loss, x0, w, text_embeddings, device
        )
        last_per_sim, last_grad_norm = per_sim, grad_norm

        if USE_PATTERN_POOL:
            batch.x[:] = x.cpu().numpy()
            batch.commit()

        loss_log.append(float(loss.item()))

        if step % 10 == 0:
            def rank_fn(x_pool, _w=w, _te=text_embeddings):
                x_t = finite_clip_state(torch.tensor(x_pool, device=device))
                with torch.no_grad():
                    losses = clip_loss.compute_objective_losses(x_t, _te)
                    loss_stack = torch.stack(losses, dim=0)
                    w_t = torch.tensor(_w, dtype=torch.float32, device=device)
                    combined = (w_t[:, None] * loss_stack).sum(dim=0)
                    combined = torch.where(torch.isfinite(combined), combined, torch.full_like(combined, float('inf')))
                return combined.cpu().numpy().argsort()
            generate_pool_figures(pool, step, log_dir, rank_fn=rank_fn, highlight_first=True)
        if step % 100 == 0:
            visualize_batch(x0, x, step, log_dir)
            plot_loss(loss_log, save_path=os.path.join(log_dir, 'loss.png'))
            torch.save(ca.state_dict(), os.path.join(log_dir, '%04d.pt' % step))

        sim_str = ' | '.join('sim%d %.4f' % (k, s.item()) for k, s in enumerate(per_sim))
        print('\r  step %d | loss %.4f | %s | grad_norm %.3e' % (
            step, loss.item(), sim_str, grad_norm), end='')

    print()
    result = {
        'name':            run_name,
        'weights':         tuple(w),
        'log_dir':         log_dir,
        'loss_log':        list(loss_log),
        'final_loss':      float(loss_log[-1]),
        'final_per_sim':   last_per_sim.cpu().numpy().tolist(),
        'final_grad_norm': float(last_grad_norm),
        'ca':              ca,
    }
    return result


def run_loss_sweep(clip_loss: CLIPLoss, seed: np.ndarray) -> dict:
    weights = validate_loss_weights(WEIGHTED_LOSS_WEIGHTS)
    results = {}
    for w1, w2 in weights:
        result = run_weighted_training(w1, w2, clip_loss, seed)
        results[result['name']] = result
    print('\nWeighted-sum sweep completed:')
    for name, r in results.items():
        print('  %s -> final_loss=%.4f  per_sim=%s' % (name, r['final_loss'], r['final_per_sim']))
    return results

def train_weighted_baseline(clip_loss: CLIPLoss, seed: np.ndarray) -> dict:
    weights_list = validate_loss_weights(WEIGHTED_LOSS_WEIGHTS)
    results = {}
    for w_tuple in weights_list:
        result = _run_weighted_training_for_weights(list(w_tuple), clip_loss, seed)
        results[result['name']] = result
    print('\nWeighted baseline completed:')
    for name, r in results.items():
        print('  %s -> final_loss=%.4f  per_sim=%s' % (name, r['final_loss'], r['final_per_sim']))
    return results
