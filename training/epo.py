"""Preference-conditioned EPO baseline."""

import json
import os
import numpy as np
import torch

from config import (
    BATCH_SIZE, POOL_SIZE, TRAIN_STEPS,
    STEPS_MIN, STEPS_MAX,
    USE_PATTERN_POOL, DAMAGE_N,
    TRAIN_LOG_ROOT,
    EPO_N_PREFS,
    GRAD_CLIP_NORM,
    POOL_FIGURE_INTERVAL, CHECKPOINT_INTERVAL,
)
from model import CAModel
from utils import SamplePool, make_circle_masks, generate_pool_figures, visualize_batch, plot_loss
from clip_loss import CLIPLoss
from training.common import cadence_due, checkpoint_due, finite_clip_state, make_optimizer

try:
    import cvxpy as cp
    HAS_CVXPY = True
except Exception:
    cp = None
    HAS_CVXPY = False

def _solver_status_str():
    if not HAS_CVXPY:
        return "cvxpy=NO clarabel=NO"
    try:
        installed = set(cp.installed_solvers())
    except Exception as exc:
        return f"cvxpy=yes solver_discovery_error={type(exc).__name__}"
    return (
        "cvxpy=yes "
        f"clarabel={'yes' if 'CLARABEL' in installed else 'NO'}"
    )


def validate_epo_solver(allow_fallback=False):
    """Verify that the solver required by the reported EPO runs is usable."""
    problem = None
    try:
        if not HAS_CVXPY:
            raise RuntimeError("CVXPY is not importable")
        if "CLARABEL" not in set(cp.installed_solvers()):
            raise RuntimeError("CLARABEL is not registered with CVXPY")
        probe = cp.Variable()
        problem = cp.Problem(cp.Minimize(cp.square(probe - 1.0)))
        problem.solve(solver=cp.CLARABEL, warm_start=False)
        if problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"solver probe returned {problem.status!r}")
    except Exception as exc:
        message = (
            "The EPO baseline requires a working CVXPY/CLARABEL installation; "
            f"environment check failed: {exc}. [{_solver_status_str()}]"
        )
        if allow_fallback:
            print(f"[EPO] WARNING: {message}")
            return False
        raise RuntimeError(message) from exc
    return True


def flatten_grads(grads, params):
    """Flatten a gradient list into a one-dimensional NumPy array."""
    result = []
    for g, p in zip(grads, params):
        if g is None:
            result.append(np.zeros(p.numel(), dtype=np.float32))
        else:
            arr = g.detach().cpu().numpy().ravel()
            result.append(np.where(np.isfinite(arr), arr, 0.0))
    return np.concatenate(result)


# Module-level fallback statistics shared across training steps.
_epo_stats = {'calls': 0, 'fallbacks': 0}


def _adjustments(losses, pref_vec):
    """LibMOON definition: mu_i = (L_i/r_i) / sum_j(L_j/r_j)."""
    r = np.array(pref_vec, dtype=np.float64)
    L = np.array(losses, dtype=np.float64)
    ratio = L / (r + 1e-12)
    return ratio / (ratio.sum() + 1e-12)


def EPO_LP(losses, grads_k, pref_vec):
    """
    Solve the original LibMOON EPO program with CVXPY.
    Balance move: QP  min ||sum_i alpha_i * mu_i * grad_i||^2  s.t. sum=1, alpha>=0
    Dominance move: LP  min sum_i mu_i*L_i*alpha_i + grad_term  s.t. sum=1, alpha>=0
    Return ``(alpha, status_string)``.
    """
    if not HAS_CVXPY:
        raise RuntimeError(
            f"EPO requires CVXPY with CLARABEL. [{_solver_status_str()}]"
        )

    K = len(losses)
    L = np.array(losses, dtype=np.float64)
    r = np.array(pref_vec, dtype=np.float64)
    G = np.stack(grads_k, axis=0).astype(np.float64)  # [K, D]

    mu_vec = _adjustments(L, r)
    # Dominated case: the loss distribution deviates from the preference.
    is_dominated = np.any(mu_vec > 1.0 / K + 1e-4)

    alpha = cp.Variable(K)
    constraints = [cp.sum(alpha) == 1, alpha >= 0]

    try:
        if is_dominated:
            # Dominance move: min  (mu * L) @ alpha  +  0.5 * ||G.T @ alpha||^2
            muL = mu_vec * L                          # numpy [K], elementwise
            obj = muL @ alpha + 0.5 * cp.sum_squares(G.T @ alpha)
        else:
            # Balance move: min  0.5 * ||diag(mu) @ M @ alpha||^2
            M = G @ G.T                               # numpy [K,K]
            muM = np.diag(mu_vec) @ M                 # numpy [K,K]
            obj = 0.5 * cp.sum_squares(muM @ alpha)

        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve(solver=cp.CLARABEL, warm_start=False)

        status = prob.status
        if status in ('optimal', 'optimal_inaccurate') and alpha.value is not None:
            a = np.maximum(alpha.value, 0.0)
            a /= a.sum() + 1e-12
            return a.astype(np.float32), status
        else:
            print(f"    [EPO_LP] status={status!r} move={'dom' if is_dominated else 'bal'}"
                  f" losses={np.round(L,4)} pref={np.round(r,4)} mu={np.round(mu_vec,4)}")
            return None, status
    except Exception as exc:
        print(f"    [EPO_LP] exception={exc!r} move={'dom' if is_dominated else 'bal'}"
              f" losses={np.round(L,4)} pref={np.round(r,4)} mu={np.round(mu_vec,4)}")
        return None, f'exception:{exc}'


def solve_epo(losses, grads_k, pref_vec):
    """Solve EPO and track fallbacks caused by failed or infeasible programs."""
    _epo_stats['calls'] += 1
    r = np.array(pref_vec, dtype=np.float64)

    try:
        alpha, status = EPO_LP(losses, grads_k, pref_vec)
    except Exception as exc:
        alpha, status = None, f"exception:{exc}"
    if alpha is not None:
        return alpha

    import config as _cfg
    if not _cfg.EPO_ALLOW_FALLBACK:
        raise RuntimeError(
            "EPO coefficient optimization failed during a formal run: "
            f"status={status!r}. Re-run with --allow-epo-fallback only for "
            "diagnostic smoke tests."
        )

    _epo_stats['fallbacks'] += 1
    fb_rate = _epo_stats['fallbacks'] / _epo_stats['calls']
    print(f"    [EPO] fallback #{_epo_stats['fallbacks']} "
          f"(rate={fb_rate:.1%}) | status={status!r}"
          f" | alpha=pref/sum(pref)")
    alpha = r / (r.sum() + 1e-12)
    return alpha.astype(np.float32)


def _compute_per_obj_grads(ca, clip_loss_obj, text_embeddings, x0_np, device):
    """
    Compute one gradient vector per objective.
    Returns: x_out (detached), per_losses [K], grads_k (list of K flat np arrays)
    """
    params = list(ca.parameters())
    x = finite_clip_state(torch.tensor(x0_np, device=device))
    iter_n = int(np.random.randint(STEPS_MIN, STEPS_MAX))

    for _ in range(iter_n):
        x = ca(x)
        x = finite_clip_state(x)

    losses = clip_loss_obj.compute_objective_losses(x, text_embeddings)  # list[K][B]
    per_losses = np.array([float(l.mean().item()) for l in losses])

    grads_k = []
    n_params = sum(p.numel() for p in params)
    for k, lk in enumerate(losses):
        loss_k = lk.mean()
        if not torch.isfinite(loss_k):
            grads_k.append(np.zeros(n_params, dtype=np.float32))
            continue
        gs = torch.autograd.grad(
            loss_k, params,
            retain_graph=(k < len(losses) - 1),
            allow_unused=True,
        )
        grads_k.append(flatten_grads(gs, params))

    return finite_clip_state(x).detach(), per_losses, grads_k


def _apply_flat_grad(ca, flat_g, optimizer, scheduler):
    """Apply a flattened gradient vector to the CA parameters."""
    optimizer.zero_grad()
    offset = 0
    for p in ca.parameters():
        size = p.numel()
        g = torch.tensor(
            flat_g[offset:offset + size].reshape(p.shape),
            dtype=p.dtype, device=p.device,
        )
        p.grad = g
        offset += size
    optimizer.step()
    scheduler.step()


def _pool_rank(x0_np, clip_loss_obj, text_embeddings, seed, h, w_size, device):
    """Rank pattern-pool samples by total objective loss."""
    x0_t = finite_clip_state(torch.tensor(x0_np, device=device))
    with torch.no_grad():
        losses = clip_loss_obj.compute_objective_losses(x0_t, text_embeddings)
        combined = torch.stack(losses, dim=0).sum(0)
        combined = torch.where(torch.isfinite(combined), combined, torch.zeros_like(combined))
    loss_rank = combined.cpu().numpy().argsort()[::-1]
    x0_np = x0_np[loss_rank]
    x0_np[:1] = seed
    if DAMAGE_N:
        damage = 1.0 - make_circle_masks(DAMAGE_N, h, w_size)
        x0_np[-DAMAGE_N:] *= damage
    return x0_np


def train_epo(clip_loss: CLIPLoss, seed: np.ndarray) -> dict:
    """
    EPO (Exact Pareto Optimal) training:
      1. Train one independent CAModel per preference vector.
      2. Roll out, compute objective gradients, solve for EPO coefficients,
         combine the gradients, and update the model.
      3. Use losses and gradients to select balance or dominance moves.
    """
    import config as _cfg
    _epo_stats.update(calls=0, fallbacks=0)
    validate_epo_solver(allow_fallback=bool(_cfg.EPO_ALLOW_FALLBACK))
    print(f'[EPO] solver status: {_solver_status_str()}')
    device = next(clip_loss.model.parameters()).device
    n_prefs = EPO_N_PREFS
    pref_vectors = _cfg.EPO_PREF_VECTORS

    _, _, h, w_size = seed.shape  # NCHW
    text_embeddings = clip_loss.embed_objective_prompts()
    K_obj = len(text_embeddings)

    # Initialize one independent model per preference vector.
    models = []
    for i in range(n_prefs):
        pref = pref_vectors[i]
        if len(pref) != K_obj:
            raise ValueError(
                f"Preference vector {i} has length {len(pref)}, "
                f"but the run has {K_obj} objectives."
            )

        # Format the preference vector for a stable directory name.
        pref_str = '_'.join([f'{p:.2f}' for p in pref])
        log_dir = os.path.join(TRAIN_LOG_ROOT, 'epo', f'pref_{i:02d}_r[{pref_str}]')
        os.makedirs(log_dir, exist_ok=True)

        ca = CAModel().to(device)
        optimizer, scheduler = make_optimizer(ca)
        pool = SamplePool(x=np.repeat(seed, POOL_SIZE, 0))

        models.append({
            'ca': ca,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'pool': pool,
            'pref_vec': pref,
            'log_dir': log_dir,
            'loss_log': [],
        })

    print(f'\n=== EPO: {n_prefs} preference vectors, {K_obj} objectives ===')
    for i, m in enumerate(models):
        print(f"  pref_{i:02d}: r = {m['pref_vec']}")
    print()

    for step in range(TRAIN_STEPS + 1):
        loss_vectors = np.zeros((n_prefs, K_obj), dtype=np.float32)
        scalar_losses = []

        # Update every preference-specific model once per outer step.
        for i, m in enumerate(models):
            if USE_PATTERN_POOL:
                batch = m['pool'].sample(BATCH_SIZE)
                x0 = _pool_rank(batch.x.copy(), clip_loss, text_embeddings, seed, h, w_size, device)
            else:
                batch = None
                x0 = np.repeat(seed, BATCH_SIZE, 0)

            # Compute objective-specific gradients.
            x_out, per_losses, grads_k = _compute_per_obj_grads(
                m['ca'], clip_loss, text_embeddings, x0, device,
            )
            loss_vectors[i] = per_losses
            scalar_losses.append(float(per_losses.mean()))

            # Solve for EPO coefficients from losses and gradients.
            alpha = solve_epo(per_losses, grads_k, m['pref_vec'])

            # Combine objective gradients.
            g_epo = sum(float(alpha[k]) * grads_k[k] for k in range(len(grads_k)))

            # Clip the combined gradient.
            norm = np.linalg.norm(g_epo)
            if norm > GRAD_CLIP_NORM:
                g_epo = g_epo * (GRAD_CLIP_NORM / (norm + 1e-8))

            # Apply the update.
            _apply_flat_grad(m['ca'], g_epo, m['optimizer'], m['scheduler'])

            # Update the pattern pool.
            if USE_PATTERN_POOL and batch is not None:
                batch.x[:] = x_out.cpu().numpy()
                batch.commit()

            m['loss_log'].append(scalar_losses[-1])

            # Save diagnostics and checkpoints.
            if cadence_due(step, POOL_FIGURE_INTERVAL):
                generate_pool_figures(m['pool'], step, m['log_dir'])

            if checkpoint_due(step, CHECKPOINT_INTERVAL, TRAIN_STEPS):
                visualize_batch(x0, x_out, step, m['log_dir'])
                plot_loss(m['loss_log'], save_path=os.path.join(m['log_dir'], 'loss.png'))
                torch.save(m['ca'].state_dict(), os.path.join(m['log_dir'], f'{step:04d}.pt'))

        # Progress logging.
        if step % 100 == 0:
            print(f'  step {step} | loss_vec_mean {np.array2string(loss_vectors.mean(0), precision=4, separator=", ")}')
            for i in range(n_prefs):
                pref_str = str(models[i]['pref_vec'])
                loss_str = np.array2string(loss_vectors[i], precision=4, separator=', ')
                print(f'    pref_{i:02d} r={pref_str} | loss={scalar_losses[i]:.4f} | per_obj={loss_str}')

    print()
    results = {}
    for i, m in enumerate(models):
        key = f'pref_{i:02d}'
        results[key] = {
            'log_dir': m['log_dir'],
            'loss_log': m['loss_log'],
            'final_loss': m['loss_log'][-1] if m['loss_log'] else float('nan'),
            'pref_vec': m['pref_vec'],
            'ca': m['ca'],
        }

    fb_rate = _epo_stats['fallbacks'] / max(_epo_stats['calls'], 1)
    print(f'\nEPO completed | LP calls={_epo_stats["calls"]} fallbacks={_epo_stats["fallbacks"]} rate={fb_rate:.1%}')
    diagnostics_dir = os.path.join(TRAIN_LOG_ROOT, 'epo')
    os.makedirs(diagnostics_dir, exist_ok=True)
    with open(os.path.join(diagnostics_dir, 'solver_diagnostics.json'), 'w', encoding='utf-8') as handle:
        json.dump({
            'solver': 'CLARABEL',
            'calls': int(_epo_stats['calls']),
            'fallbacks': int(_epo_stats['fallbacks']),
            'fallback_allowed': bool(_cfg.EPO_ALLOW_FALLBACK),
        }, handle, indent=2)
        handle.write('\n')
    for key, r in results.items():
        print(f"  {key} (r={r['pref_vec']}) -> final_loss={r['final_loss']:.4f}")
    return results
