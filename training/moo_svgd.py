import os
import numpy as np
import torch

from config import (
    BATCH_SIZE, POOL_SIZE, TRAIN_STEPS,
    STEPS_MIN, STEPS_MAX,
    USE_PATTERN_POOL, DAMAGE_N,
    TRAIN_LOG_ROOT, OBJECTIVE_PROMPTS,
    MOO_SVGD_PARTICLES, MOO_SVGD_BANDWIDTH,
    MOO_SVGD_REPULSION_COEF,
    GRAD_CLIP_NORM,
    POOL_FIGURE_INTERVAL, CHECKPOINT_INTERVAL,
)
from model import CAModel
from utils import SamplePool, make_circle_masks, generate_pool_figures, visualize_batch, plot_loss
from clip_loss import CLIPLoss
from training.common import cadence_due, checkpoint_due, finite_clip_state, make_optimizer


def flatten_grads(grads, params):
    result = []
    for g, p in zip(grads, params):
        if g is None:
            result.append(np.zeros(p.numel(), dtype=np.float32))
        else:
            arr = g.detach().cpu().numpy().ravel()
            result.append(np.where(np.isfinite(arr), arr, 0.0))
    return np.concatenate(result)


def flatten_weights(model):
    return np.concatenate([p.detach().cpu().numpy().ravel() for p in model.parameters()])


def frank_wolfe_pareto(grads_k):
    """min_{alpha>=0, sumalpha=1} ||sum_k alpha_k g_k||^2  via Frank-Wolfe."""
    K = len(grads_k)
    G = np.stack(grads_k, axis=0)   # [K, D]
    M = (G @ G.T).astype(np.float64)

    alpha = np.ones(K, dtype=np.float64) / K
    for _ in range(200):
        grad_f = M @ alpha
        k_star = int(np.argmin(grad_f))
        e_k = np.zeros(K, dtype=np.float64)
        e_k[k_star] = 1.0
        d = e_k - alpha
        dMd = float(d @ M @ d)
        if dMd <= 1e-12:
            break
        step = max(0.0, min(1.0, -float(d @ grad_f) / dMd))
        alpha += step * d
        if step < 1e-9:
            break

    alpha = np.maximum(alpha, 0.0)
    alpha /= alpha.sum() + 1e-12
    return alpha.astype(np.float32)


def rbf_kernel_theta(theta_particles, bandwidth):
    diff = theta_particles[:, None, :] - theta_particles[None, :, :]  # [N, N, D]
    sq_dist = np.sum(diff ** 2, axis=-1)                               # [N, N]
    if bandwidth <= 0:  # median heuristic
        med = np.median(sq_dist[sq_dist > 0])
        N = theta_particles.shape[0]
        bandwidth = float(np.sqrt(med / (2.0 * np.log(N + 1)))) + 1e-8
    K_mat = np.exp(-sq_dist / (2.0 * bandwidth ** 2))
    grad_K = K_mat[:, :, None] * (-diff) / (bandwidth ** 2)           # [N, N, D]
    return K_mat, grad_K


def compute_svgd_repulsion(theta_particles, bandwidth, repulsion_coef):
    """Returns repulsion term [N, D] to subtract from gradient."""
    K_mat, grad_K = rbf_kernel_theta(theta_particles, bandwidth)
    N = theta_particles.shape[0]
    # repulsion pushes particles apart: subtract from descent direction
    return repulsion_coef * np.sum(grad_K, axis=1) / N  # [N, D]


def _compute_per_obj_grads(ca, clip_loss_obj, text_embeddings, x0_np, device):
    """
    Returns x_out (detached), per_losses [K], grads_k (list of K flat np arrays).
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
    """Write flat_g into .grad of each parameter, then optimizer.step()."""
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


def train_true_moo_svgd(clip_loss: CLIPLoss, seed: np.ndarray) -> dict:
    """
    theta-space MOO-SVGD:
      1. Per-objective BPTT -> g_k
      2. Frank-Wolfe -> alpha,  g_pareto = sum alpha_k g_k
      3. SVGD RBF repulsion added to g_pareto
      4. Adam.step() with combined gradient
    """
    device = next(clip_loss.model.parameters()).device
    n_particles = MOO_SVGD_PARTICLES
    _, _, h, w_size = seed.shape  # NCHW

    text_embeddings = clip_loss.embed_objective_prompts()

    particles = []
    for i in range(n_particles):
        log_dir = os.path.join(TRAIN_LOG_ROOT, 'moo_svgd', 'particle_%02d' % i)
        os.makedirs(log_dir, exist_ok=True)
        ca = CAModel().to(device)
        optimizer, scheduler = make_optimizer(ca)
        pool = SamplePool(x=np.repeat(seed, POOL_SIZE, 0))
        particles.append({
            'ca': ca, 'optimizer': optimizer, 'scheduler': scheduler,
            'pool': pool, 'log_dir': log_dir, 'loss_log': [],
        })

    print('\n=== MOO-SVGD (Frank-Wolfe + Adam): %d particles, %d objectives ===' % (
        n_particles, len(OBJECTIVE_PROMPTS)))

    for step in range(TRAIN_STEPS + 1):
        pareto_grads, theta_all, xs_out = [], [], []
        loss_vectors = np.zeros((n_particles, len(OBJECTIVE_PROMPTS)), dtype=np.float32)
        scalar_losses = []

        # -- 1. Per-objective BPTT + Frank-Wolfe ----------------------
        for i, p in enumerate(particles):
            if USE_PATTERN_POOL:
                batch = p['pool'].sample(BATCH_SIZE)
                x0 = _pool_rank(batch.x.copy(), clip_loss, text_embeddings, seed, h, w_size, device)
            else:
                batch = None
                x0 = np.repeat(seed, BATCH_SIZE, 0)

            x_out, per_losses, grads_k = _compute_per_obj_grads(
                p['ca'], clip_loss, text_embeddings, x0, device,
            )
            xs_out.append((x_out, x0, batch))
            loss_vectors[i] = per_losses
            scalar_losses.append(float(per_losses.mean()))

            alpha = frank_wolfe_pareto(grads_k)
            g_pareto = sum(float(alpha[k]) * grads_k[k] for k in range(len(grads_k)))

            norm = np.linalg.norm(g_pareto)
            if norm > GRAD_CLIP_NORM:
                g_pareto = g_pareto * (GRAD_CLIP_NORM / (norm + 1e-8))

            pareto_grads.append(g_pareto)
            theta_all.append(flatten_weights(p['ca']))

        # -- 2. SVGD repulsion in theta-space -----------------------------
        theta_mat = np.stack(theta_all, axis=0)       # [N, D]
        repulsion = compute_svgd_repulsion(theta_mat, MOO_SVGD_BANDWIDTH, MOO_SVGD_REPULSION_COEF)

        # -- 3. Adam step with (g_pareto - repulsion) -----------------
        # repulsion pushes particles apart -> subtract from descent direction
        for i, p in enumerate(particles):
            combined = pareto_grads[i] - repulsion[i]
            _apply_flat_grad(p['ca'], combined, p['optimizer'], p['scheduler'])

        # -- 4. Pool update + logging ----------------------------------
        for i, p in enumerate(particles):
            x_out, x0, batch = xs_out[i]
            if USE_PATTERN_POOL and batch is not None:
                batch.x[:] = x_out.cpu().numpy()
                batch.commit()
            p['loss_log'].append(scalar_losses[i])

            if cadence_due(step, POOL_FIGURE_INTERVAL):
                generate_pool_figures(p['pool'], step, p['log_dir'])
            if checkpoint_due(step, CHECKPOINT_INTERVAL, TRAIN_STEPS):
                visualize_batch(x0, x_out, step, p['log_dir'])
                plot_loss(p['loss_log'], save_path=os.path.join(p['log_dir'], 'loss.png'))
                torch.save(p['ca'].state_dict(), os.path.join(p['log_dir'], '%04d.pt' % step))

        if step % 100 == 0:
            print('  step %d | loss_vec_mean %s' % (
                step, np.array2string(loss_vectors.mean(0), precision=4, separator=', ')))
            for i in range(n_particles):
                print('    particle_%02d | loss=%.4f' % (i, scalar_losses[i]))
            if n_particles > 1:
                dists = [np.linalg.norm(theta_all[i] - theta_all[j])
                         for i in range(n_particles) for j in range(i + 1, n_particles)]
                print('    [SVGD] avg pairwise param dist: %.4f' % np.mean(dists))
                print('    [SVGD] avg repulsion norm: %.6f' % np.mean(
                    [np.linalg.norm(repulsion[i]) for i in range(n_particles)]))

    print()
    results = {}
    for i, p in enumerate(particles):
        key = 'particle_%02d' % i
        results[key] = {
            'log_dir':    p['log_dir'],
            'loss_log':   p['loss_log'],
            'final_loss': p['loss_log'][-1] if p['loss_log'] else float('nan'),
            'ca':         p['ca'],
        }
    print('\nMOO-SVGD (Frank-Wolfe) completed:')
    for key, r in results.items():
        print('  %s -> final_loss=%.4f' % (key, r['final_loss']))
    return results
