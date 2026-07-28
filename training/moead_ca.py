import os
import time
import copy
import json
import numpy as np
import torch
import torch.nn.utils as nn_utils
from scipy.optimize import linear_sum_assignment

from config import (
    STEPS_MIN, STEPS_MAX,
    GRAD_CLIP_NORM,
)
from model import CAModel
from utils import SamplePool, make_circle_masks, generate_pool_figures, visualize_batch, plot_loss
from clip_loss import CLIPLoss
from training.common import finite_clip_state, make_optimizer
from training.moo_svgd import (
    _apply_flat_grad,
    flatten_grads,
    flatten_weights,
    frank_wolfe_pareto,
)


def generate_weight_vectors(n_subproblems, n_objectives, seed=42, eps=1e-3):
    if n_objectives == 2:
        t = np.linspace(eps, 1 - eps, n_subproblems)
        return np.stack([t, 1 - t], axis=1).astype(np.float32)
    rng = np.random.default_rng(seed)
    w = rng.dirichlet(np.ones(n_objectives), size=n_subproblems).astype(np.float32)
    w = np.clip(w, eps, None)
    w /= w.sum(axis=1, keepdims=True)
    return w


def build_neighborhood(weights, neighbor_size):
    dists = np.sum((weights[:, None] - weights[None]) ** 2, axis=2)
    return np.argsort(dists, axis=1)[:, :neighbor_size]


def decomposition_value(obj_vec, weight_vec, ideal_point=None,
                        method='tchebycheff', theta=5.0, eps=1e-6):
    f = np.asarray(obj_vec, dtype=np.float64)
    w = np.asarray(weight_vec, dtype=np.float64)
    w_safe = np.where(w < eps, eps, w)
    if method == 'tchebycheff':
        if ideal_point is None:
            raise ValueError('Tchebycheff decomposition requires an ideal_point')
        z = np.asarray(ideal_point, dtype=np.float64)
        if z.shape != f.shape:
            raise ValueError(
                f'ideal_point shape {z.shape} does not match objective shape {f.shape}'
            )
        return float(np.max(w_safe * np.abs(f - z)))
    return float(np.dot(w_safe, f))  # weighted sum fallback


def evaluate_objectives(ca, clip_loss, text_embeddings, x0_np, device, rollout_steps=None):
    steps = rollout_steps or int(np.random.randint(STEPS_MIN, STEPS_MAX))
    x = finite_clip_state(torch.tensor(x0_np, device=device))
    with torch.no_grad():
        for _ in range(steps):
            x = ca(x)
            x = finite_clip_state(x)
        losses = clip_loss.compute_objective_losses(x, text_embeddings)
    obj = np.array([float(l.mean().item()) for l in losses], dtype=np.float32)
    return obj, finite_clip_state(x).detach()


def evaluate_solution_state(ca, clip_loss, text_embeddings, evaluation_seed,
                            device, rollout_steps=None, rng_seed=None):
    """Evaluate every solution on the same seed and fixed rollout horizon."""
    import config as _cfg

    steps = _cfg.EVAL_STEPS if rollout_steps is None else int(rollout_steps)
    if rng_seed is None:
        return evaluate_objectives(
            ca, clip_loss, text_embeddings, evaluation_seed, device,
            rollout_steps=steps,
        )

    devices = []
    if device.type == 'cuda':
        devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(rng_seed))
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(int(rng_seed))
        return evaluate_objectives(
            ca, clip_loss, text_embeddings, evaluation_seed, device,
            rollout_steps=steps,
        )


def empirical_ideal_point(objective_vectors, previous=None):
    """Component-wise best observed objective, with monotonic history."""
    values = np.asarray(objective_vectors, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f'Expected a non-empty objective matrix, got {values.shape}')
    finite_values = np.where(np.isfinite(values), values, np.inf)
    current = finite_values.min(axis=0)
    if not np.all(np.isfinite(current)):
        raise ValueError(f'Cannot construct empirical ideal from {values}')
    if previous is not None:
        current = np.minimum(current, np.asarray(previous, dtype=np.float64))
    return current.astype(np.float32)


def local_train_step(sub, clip_loss, text_embeddings, device, ideal_point):
    import config as _cfg
    ca, optimizer, scheduler = sub['ca'], sub['optimizer'], sub['scheduler']
    w = sub['weight_vec']
    x0 = sub['x0_batch']

    x = finite_clip_state(torch.tensor(x0, device=device))
    iter_n = int(np.random.randint(STEPS_MIN, STEPS_MAX))
    optimizer.zero_grad()
    for _ in range(iter_n):
        x = ca(x)
        x = finite_clip_state(x)

    losses = clip_loss.compute_objective_losses(x, text_embeddings)

    w_t = torch.tensor(w, dtype=torch.float32, device=device)
    loss_stack = torch.stack([l.mean() for l in losses])
    eps = _cfg.MOEAD_EPS
    w_safe = torch.where(w_t < eps, torch.full_like(w_t, eps), w_t)

    z_t = torch.as_tensor(
        ideal_point,
        dtype=loss_stack.dtype,
        device=device,
    )

    if _cfg.MOEAD_GRADIENT_POLICY == 'tchebycheff':
        scalar_loss = torch.max(w_safe * torch.abs(loss_stack - z_t))
    else:
        scalar_loss = (w_safe * loss_stack).sum()

    if not torch.isfinite(scalar_loss):
        scalar_loss = torch.tensor(0.0, device=device, requires_grad=True)

    scalar_loss.backward()
    grad_norm = nn_utils.clip_grad_norm_(ca.parameters(), GRAD_CLIP_NORM)
    optimizer.step()
    scheduler.step()
    return finite_clip_state(x).detach(), float(scalar_loss.item()), float(grad_norm)


def local_mgda_train_step(sub, clip_loss, text_embeddings, device,
                          repulsion=None):
    """Apply a Frank-Wolfe MGDA update, optionally with RBF repulsion."""
    ca, optimizer, scheduler = sub['ca'], sub['optimizer'], sub['scheduler']
    x = finite_clip_state(torch.tensor(sub['x0_batch'], device=device))
    iter_n = int(np.random.randint(STEPS_MIN, STEPS_MAX))
    for _ in range(iter_n):
        x = finite_clip_state(ca(x))

    losses = clip_loss.compute_objective_losses(x, text_embeddings)
    params = list(ca.parameters())
    n_params = sum(p.numel() for p in params)
    grads_k = []
    per_losses = []
    for k, loss_vec in enumerate(losses):
        loss_k = loss_vec.mean()
        per_losses.append(float(loss_k.detach().item()))
        if not torch.isfinite(loss_k):
            grads_k.append(np.zeros(n_params, dtype=np.float32))
            continue
        grads = torch.autograd.grad(
            loss_k,
            params,
            retain_graph=(k < len(losses) - 1),
            allow_unused=True,
        )
        grads_k.append(flatten_grads(grads, params).astype(np.float32, copy=False))

    alpha = frank_wolfe_pareto(grads_k)
    combined = sum(float(alpha[k]) * grads_k[k] for k in range(len(grads_k)))
    combined = np.where(np.isfinite(combined), combined, 0.0).astype(np.float32, copy=False)
    repulsion_norm = 0.0
    if repulsion is not None:
        repulsion = np.where(np.isfinite(repulsion), repulsion, 0.0).astype(np.float32, copy=False)
        repulsion_norm = float(np.linalg.norm(repulsion))
        combined = combined - repulsion

    grad_norm = float(np.linalg.norm(combined))
    if grad_norm > GRAD_CLIP_NORM:
        combined = combined * (GRAD_CLIP_NORM / (grad_norm + 1e-8))
        grad_norm = float(np.linalg.norm(combined))
    _apply_flat_grad(ca, combined, optimizer, scheduler)
    return (
        finite_clip_state(x).detach(),
        float(np.mean(per_losses)),
        grad_norm,
        alpha,
        repulsion_norm,
    )


def copy_solution(src_ca, dst_sub):
    """Copy parameters and restart local optimization for renewed exploration."""
    copy_solution_state(src_ca.state_dict(), dst_sub)


def copy_solution_state(src_state, dst_sub):
    """Install a synchronized child snapshot and restart local optimization."""
    import config as _cfg

    dst_sub['ca'].load_state_dict(copy.deepcopy(src_state))
    dst_sub['optimizer'], dst_sub['scheduler'] = make_optimizer(
        dst_sub['ca'], _cfg.TRAIN_STEPS
    )


def _prepare_pool_batch(x0_np, seed, H, W):
    """Apply random seed reset and damage without fitness-based sorting."""
    import config as _cfg

    x0_np[:1] = seed
    if _cfg.DAMAGE_N:
        x0_np[-_cfg.DAMAGE_N:] *= 1.0 - make_circle_masks(_cfg.DAMAGE_N, H, W)
    return x0_np


def _dominates(a, b):
    return bool(np.all(a <= b) and np.any(a < b))


def _non_dominated_fronts(values):
    """Small-candidate NSGA-II fronts for Pareto-aware pool sampling."""
    values = np.asarray(values, dtype=np.float64)
    remaining = list(range(len(values)))
    fronts = []
    while remaining:
        front = [
            i for i in remaining
            if not any(_dominates(values[j], values[i]) for j in remaining if j != i)
        ]
        if not front:  # Defensive fallback for pathological non-finite input.
            front = [remaining[0]]
        fronts.append(front)
        front_set = set(front)
        remaining = [i for i in remaining if i not in front_set]
    return fronts


def _crowding_distance(values, indices):
    indices = list(indices)
    if len(indices) <= 2:
        return {idx: float('inf') for idx in indices}
    values = np.asarray(values, dtype=np.float64)
    distance = {idx: 0.0 for idx in indices}
    for objective in range(values.shape[1]):
        ordered = sorted(indices, key=lambda idx: (values[idx, objective], idx))
        distance[ordered[0]] = float('inf')
        distance[ordered[-1]] = float('inf')
        span = values[ordered[-1], objective] - values[ordered[0], objective]
        if not np.isfinite(span) or span <= 1e-12:
            continue
        for pos in range(1, len(ordered) - 1):
            if np.isfinite(distance[ordered[pos]]):
                prev_value = values[ordered[pos - 1], objective]
                next_value = values[ordered[pos + 1], objective]
                distance[ordered[pos]] += float((next_value - prev_value) / span)
    return distance


def _pareto_aware_pool_sample(pool, batch_size, clip_loss, text_embeddings,
                              device):
    """Sample without reordering the pool: uniform exploration + Pareto/crowding."""
    import config as _cfg

    multiplier = max(1, int(_cfg.MOEAD_PARETO_CANDIDATE_MULTIPLIER))
    candidate_n = min(pool._size, max(batch_size, batch_size * multiplier))
    candidate_indices = np.random.choice(pool._size, candidate_n, replace=False)
    x_candidates = finite_clip_state(torch.tensor(pool.x[candidate_indices], device=device))
    with torch.no_grad():
        losses = clip_loss.compute_objective_losses(x_candidates, text_embeddings)
        values = torch.stack(losses, dim=1).detach().cpu().numpy()
    values = np.where(np.isfinite(values), values, np.inf)

    exploration = float(np.clip(_cfg.MOEAD_PARETO_EXPLORATION, 0.0, 1.0))
    explore_n = min(batch_size, int(round(batch_size * exploration)))
    selected = []
    if explore_n:
        selected.extend(np.random.choice(candidate_n, explore_n, replace=False).tolist())

    selected_set = set(selected)
    available = [i for i in range(candidate_n) if i not in selected_set]
    need = batch_size - len(selected)
    if need:
        local_values = values[available]
        for front_local in _non_dominated_fronts(local_values):
            front = [available[i] for i in front_local]
            remaining_need = batch_size - len(selected)
            if len(front) <= remaining_need:
                selected.extend(front)
            else:
                crowding = _crowding_distance(values, front)
                ordered = sorted(front, key=lambda idx: (-crowding[idx], idx))
                selected.extend(ordered[:remaining_need])
            if len(selected) >= batch_size:
                break

    return pool.take(candidate_indices[np.asarray(selected[:batch_size], dtype=np.int64)])


def _preference_reset_batch(x0_np, clip_loss, text_embeddings, weight_vec,
                            seed, H, W, device):
    """Match the old pool rule: rank a uniform batch and reset its worst state."""
    import config as _cfg

    x0_t = finite_clip_state(torch.tensor(x0_np, device=device))
    with torch.no_grad():
        losses = clip_loss.compute_objective_losses(x0_t, text_embeddings)
        loss_matrix = torch.stack(losses, dim=0)
        weight = torch.as_tensor(
            weight_vec, dtype=loss_matrix.dtype, device=device
        )[:, None]
        scalar = (weight * loss_matrix).sum(dim=0)
        scalar = torch.where(
            torch.isfinite(scalar),
            scalar,
            torch.full_like(scalar, float('inf')),
        )
        rank = scalar.argsort(descending=True).cpu().numpy()

    x0_np = x0_np[rank]
    x0_np[:1] = seed
    if _cfg.DAMAGE_N:
        x0_np[-_cfg.DAMAGE_N:] *= 1.0 - make_circle_masks(
            _cfg.DAMAGE_N, H, W
        )
    return x0_np


def _sample_training_batch(sub, clip_loss, text_embeddings, device):
    import config as _cfg

    if _cfg.MOEAD_POOL_POLICY == 'pareto':
        return _pareto_aware_pool_sample(
            sub['pool'], _cfg.BATCH_SIZE, clip_loss, text_embeddings, device
        )
    return sub['pool'].sample(_cfg.BATCH_SIZE)


def _repulsion_for_solution(index, subproblems, bandwidth, coefficient):
    """Memory-bounded row of the current implementation's RBF repulsion."""
    if coefficient == 0.0 or len(subproblems) <= 1:
        return None
    theta_all = np.stack(
        [flatten_weights(sub['ca']).astype(np.float32, copy=False) for sub in subproblems],
        axis=0,
    )
    diff = theta_all[index][None, :] - theta_all
    sq_dist = np.sum(diff ** 2, axis=1)
    h = float(bandwidth)
    if h <= 0:
        positive = sq_dist[sq_dist > 0]
        if positive.size == 0:
            return np.zeros_like(theta_all[index])
        h = float(np.sqrt(np.median(positive) / (2.0 * np.log(len(subproblems) + 1)))) + 1e-8
    kernel = np.exp(-sq_dist / (2.0 * h ** 2))
    grad_kernel = kernel[:, None] * (-diff) / (h ** 2)
    return (float(coefficient) * np.sum(grad_kernel, axis=0) / len(subproblems)).astype(np.float32)


def relative_scalar_improvement(candidate_value, current_value, eps=1e-12):
    """Return the normalized decrease of a scalar minimization objective."""
    candidate_value = float(candidate_value)
    current_value = float(current_value)
    if not np.isfinite(candidate_value) or not np.isfinite(current_value):
        return float('-inf')
    return (current_value - candidate_value) / max(abs(current_value), eps)


def maximum_weight_one_to_one_matching(source_ids, target_ids, edge_benefits):
    """Globally maximize positive edge benefit with optional unmatched sources.

    ``edge_benefits`` maps ``(source_id, target_id)`` to a positive benefit.
    Dummy targets with zero benefit let any source remain unmatched.  Sorting
    IDs before constructing the cost matrix makes repeated runs deterministic.
    """
    source_ids = sorted(set(int(source_id) for source_id in source_ids))
    target_ids = sorted(set(int(target_id) for target_id in target_ids))
    if not source_ids or not target_ids or not edge_benefits:
        return []

    source_pos = {source_id: row for row, source_id in enumerate(source_ids)}
    target_pos = {target_id: col for col, target_id in enumerate(target_ids)}
    n_sources = len(source_ids)
    n_targets = len(target_ids)
    invalid_cost = 1e100
    costs = np.full(
        (n_sources, n_targets + n_sources), invalid_cost, dtype=np.float64
    )
    costs[:, n_targets:] = 0.0

    for (source_id, target_id), benefit in edge_benefits.items():
        source_id = int(source_id)
        target_id = int(target_id)
        benefit = float(benefit)
        if (source_id not in source_pos or target_id not in target_pos
                or not np.isfinite(benefit) or benefit <= 0.0):
            continue
        costs[source_pos[source_id], target_pos[target_id]] = -benefit

    row_indices, col_indices = linear_sum_assignment(costs)
    matches = []
    for row, col in zip(row_indices, col_indices):
        if col >= n_targets or costs[row, col] >= 0.0:
            continue
        source_id = source_ids[row]
        target_id = target_ids[col]
        matches.append((source_id, target_id, -float(costs[row, col])))
    return sorted(matches, key=lambda match: (match[0], match[1]))


def plan_global_replacements(candidate_objectives, subproblems, neighborhoods,
                             replacement_policy, method, ideal_point,
                             improvement_threshold):
    """Build eligible propagation edges and solve one global 1:1 matching."""
    source_ids = list(range(len(candidate_objectives)))
    target_ids = list(range(len(subproblems)))
    edge_benefits = {}
    for source_id, candidate_obj in enumerate(candidate_objectives):
        if replacement_policy == 'global':
            eligible_targets = target_ids
        elif replacement_policy == 'neighborhood':
            eligible_targets = neighborhoods[source_id]
        else:
            raise ValueError(
                f'Global replacement matching does not support '
                f'{replacement_policy!r}'
            )
        for raw_target_id in eligible_targets:
            target_id = int(raw_target_id)
            if target_id == source_id:
                continue
            target = subproblems[target_id]
            candidate_value = decomposition_value(
                candidate_obj,
                target['weight_vec'],
                ideal_point,
                method=method,
            )
            current_value = decomposition_value(
                target['obj_vec'],
                target['weight_vec'],
                ideal_point,
                method=method,
            )
            benefit = relative_scalar_improvement(
                candidate_value, current_value
            )
            if benefit > improvement_threshold:
                edge_benefits[(source_id, target_id)] = benefit
    return maximum_weight_one_to_one_matching(
        source_ids, target_ids, edge_benefits
    )


def _stable_matching_assignment(child_objectives, weights, method,
                                ideal_point):
    """Children propose to subproblems; each subproblem prefers lower g."""
    child_objectives = np.asarray(child_objectives)
    n_children = len(child_objectives)
    costs = np.empty((n_children, len(weights)), dtype=np.float64)
    for child in range(n_children):
        for subproblem in range(len(weights)):
            costs[child, subproblem] = decomposition_value(
                child_objectives[child], weights[subproblem], ideal_point,
                method=method,
            )
    preferences = np.argsort(costs, axis=1, kind='stable')
    next_choice = np.zeros(n_children, dtype=np.int64)
    matched_child = [None] * len(weights)
    free = list(range(n_children))
    while free:
        child = free.pop(0)
        if next_choice[child] >= len(weights):
            raise RuntimeError('STM failed: a child exhausted all subproblem preferences')
        target = int(preferences[child, next_choice[child]])
        next_choice[child] += 1
        incumbent = matched_child[target]
        if incumbent is None:
            matched_child[target] = child
            continue
        child_key = (costs[child, target], child)
        incumbent_key = (costs[incumbent, target], incumbent)
        if child_key < incumbent_key:
            matched_child[target] = child
            free.append(incumbent)
        else:
            free.append(child)
    return matched_child, costs


def _visualization_rank_indices(x_pool, clip_loss, text_embeddings, weight_vec,
                                ideal_point, device):
    """Rank a read-only pool snapshot for display; never mutates training data."""
    import config as _cfg

    x_t = finite_clip_state(torch.tensor(x_pool, device=device))
    with torch.no_grad():
        losses = clip_loss.compute_objective_losses(x_t, text_embeddings)
        loss_matrix = torch.stack(losses, dim=0)
        w_t = torch.tensor(weight_vec, dtype=loss_matrix.dtype, device=device)
        w_safe = torch.where(
            w_t < _cfg.MOEAD_EPS,
            torch.full_like(w_t, _cfg.MOEAD_EPS),
            w_t,
        )
        if _cfg.MOEAD_DECOMPOSITION == 'tchebycheff':
            z_t = torch.as_tensor(
                ideal_point,
                dtype=loss_matrix.dtype,
                device=device,
            )[:, None]
            scalar = torch.max(
                w_safe[:, None] * torch.abs(loss_matrix - z_t), dim=0
            ).values
        else:
            scalar = (w_safe[:, None] * loss_matrix).sum(dim=0)
        scalar = torch.where(
            torch.isfinite(scalar), scalar, torch.full_like(scalar, float('inf'))
        )
    return scalar.cpu().numpy().argsort()  # ascending: low loss first


def train_moead_core(clip_loss: CLIPLoss, seed: np.ndarray,
                     variant_name='moead_ca', archive_controller=None) -> dict:
    """Shared cooperative MOEA/D-CA backbone with explicit policy factors."""
    import config as _cfg
    device = next(clip_loss.model.parameters()).device
    text_embeddings = clip_loss.embed_objective_prompts()
    K = len(text_embeddings)
    _, _, H, W = seed.shape

    N = _cfg.MOEAD_N_SUBPROBLEMS
    T = _cfg.MOEAD_NEIGHBOR_SIZE
    max_rep = _cfg.MOEAD_MAX_REPLACEMENTS
    cooperation_interval = int(_cfg.MOEAD_COOPERATION_INTERVAL)
    pool_figure_interval = max(0, int(_cfg.MOEAD_POOL_FIGURE_INTERVAL))
    checkpoint_interval = max(0, int(_cfg.MOEAD_CHECKPOINT_INTERVAL))
    eval_interval = max(1, int(_cfg.MOEAD_EVAL_INTERVAL))
    cooperation_threshold = float(_cfg.COOPERATION_THRESHOLD)
    replacement_policy = str(_cfg.MOEAD_REPLACEMENT_POLICY)
    pool_policy = str(_cfg.MOEAD_POOL_POLICY)
    gradient_policy = str(_cfg.MOEAD_GRADIENT_POLICY)
    if replacement_policy not in {'none', 'neighborhood', 'global', 'stm'}:
        raise ValueError(f'Unknown replacement policy: {replacement_policy}')
    if cooperation_interval <= 0:
        raise ValueError(
            'MOEAD_COOPERATION_INTERVAL must be positive, '
            f'got {cooperation_interval}'
        )
    if pool_policy not in {'preference_reset', 'uniform', 'pareto'}:
        raise ValueError(f'Unknown pool policy: {pool_policy}')
    if gradient_policy not in {'weighted_sum', 'tchebycheff', 'mgda', 'mgda_svgd'}:
        raise ValueError(f'Unknown gradient policy: {gradient_policy}')

    weights = generate_weight_vectors(N, K, seed=_cfg.MOEAD_WEIGHT_SEED)
    neighborhoods = build_neighborhood(weights, T)

    print(f'\n=== {variant_name.upper()}: N={N} subproblems, K={K} objectives, T={T} neighbors ===')
    print(f'    decomposition={_cfg.MOEAD_DECOMPOSITION}  max_replacements={max_rep}')
    print(f'    replacement={replacement_policy}  pool={pool_policy}  '
          f'gradient={gradient_policy}  eval_interval={eval_interval}  '
          f'rta={archive_controller is not None}\n')

    subproblems = []
    for i in range(N):
        w = weights[i]
        w_str = '_'.join('%.2f' % wi for wi in w)
        log_dir = os.path.join(_cfg.TRAIN_LOG_ROOT, variant_name, f'sub_{i:02d}_w[{w_str}]')
        os.makedirs(log_dir, exist_ok=True)
        ca = CAModel().to(device)
        optimizer, scheduler = make_optimizer(ca, _cfg.TRAIN_STEPS)
        pool = SamplePool(x=np.repeat(seed, _cfg.POOL_SIZE, 0))
        subproblems.append({
            'ca': ca, 'optimizer': optimizer, 'scheduler': scheduler,
            'pool': pool, 'weight_vec': w, 'log_dir': log_dir,
            'loss_log': [], 'obj_vec': None, 'x0_batch': None,
            'alpha_log': [], 'grad_norm_log': [], 'repulsion_norm_log': [],
            'replacement_count': 0,
            'solution_id': i, 'region_id': i, 'update_count': 0,
            'archive_transfer_count': 0, 'last_parent_solution_id': None,
        })

    print('Initializing objective vectors...')
    initialization_rng_seed = int(getattr(_cfg, 'RUN_SEED', 0)) + 1000003
    for sub in subproblems:
        obj, _ = evaluate_solution_state(
            sub['ca'], clip_loss, text_embeddings, seed, device,
            rng_seed=initialization_rng_seed,
        )
        sub['obj_vec'] = obj
    ideal_point = empirical_ideal_point(
        [sub['obj_vec'] for sub in subproblems]
    )
    ideal_point_evolution = [ideal_point.tolist()]
    print(
        '  initial synchronized empirical ideal_point: '
        f'{ideal_point.tolist()}\n'
    )

    if archive_controller is not None:
        archive_controller.initialize(subproblems, weights, ideal_point)

    training_started = time.perf_counter()
    for step in range(1, _cfg.TRAIN_STEPS + 1):
        step_ideal_point = ideal_point.copy()
        evaluation_step = (step % eval_interval == 0 or step == _cfg.TRAIN_STEPS)
        cooperation_step = (step % cooperation_interval == 0)
        if step == 1 or step % 10 == 0:
            print(
                f'  starting outer step {step}/{_cfg.TRAIN_STEPS} '
                f'({N} CA updates in this step)',
                flush=True,
            )
        if archive_controller is not None:
            archive_controller.begin_step(step, step_ideal_point)
            before_updates = getattr(archive_controller, 'before_updates', None)
            if callable(before_updates):
                before_updates(subproblems, step)
            else:
                before_updates = None

        step_records = []
        for i, sub in enumerate(subproblems):
            if archive_controller is not None and before_updates is None:
                archive_controller.before_update(sub, step)

            if _cfg.USE_PATTERN_POOL:
                batch = _sample_training_batch(
                    sub, clip_loss, text_embeddings, device
                )
                if pool_policy == 'preference_reset':
                    x0 = _preference_reset_batch(
                        batch.x.copy(), clip_loss, text_embeddings,
                        sub['weight_vec'], seed, H, W, device,
                    )
                else:
                    x0 = _prepare_pool_batch(batch.x.copy(), seed, H, W)
            else:
                batch = None
                x0 = np.repeat(seed, _cfg.BATCH_SIZE, 0)
            sub['x0_batch'] = x0

            alpha = None
            repulsion_norm = 0.0
            if gradient_policy in {'weighted_sum', 'tchebycheff'}:
                x_out, scalar_loss, grad_norm = local_train_step(
                    sub, clip_loss, text_embeddings, device,
                    step_ideal_point,
                )
            else:
                repulsion = None
                if gradient_policy == 'mgda_svgd':
                    repulsion = _repulsion_for_solution(
                        i,
                        subproblems,
                        _cfg.MOEAD_SVGD_BANDWIDTH,
                        _cfg.MOEAD_SVGD_REPULSION_COEF,
                    )
                x_out, scalar_loss, grad_norm, alpha, repulsion_norm = (
                    local_mgda_train_step(
                        sub,
                        clip_loss,
                        text_embeddings,
                        device,
                        repulsion=repulsion,
                    )
                )

            if _cfg.USE_PATTERN_POOL and batch is not None:
                batch.x[:] = x_out.cpu().numpy()
                batch.commit()
            sub['loss_log'].append(scalar_loss)
            sub['grad_norm_log'].append(float(grad_norm))
            sub['repulsion_norm_log'].append(float(repulsion_norm))
            if alpha is not None:
                sub['alpha_log'].append(np.asarray(alpha, dtype=np.float32))
            sub['update_count'] += 1

            step_records.append({
                'source_index': i,
                'x0': x0.copy(),
                'x_out': x_out.detach().cpu().numpy(),
            })

            if pool_figure_interval and step % pool_figure_interval == 0:
                _w = sub['weight_vec']
                _z = step_ideal_point.copy()
                def _visualization_rank_fn(
                        x_pool, _w=_w, _te=text_embeddings, _z=_z):
                    return _visualization_rank_indices(
                        x_pool, clip_loss, _te, _w, _z, device
                    )
                generate_pool_figures(sub['pool'], step, sub['log_dir'],
                                      rank_fn=_visualization_rank_fn, highlight_first=True)
            if (checkpoint_interval and step % checkpoint_interval == 0
                    and replacement_policy != 'stm'):
                visualize_batch(x0, x_out, step, sub['log_dir'])
                plot_loss(sub['loss_log'], save_path=os.path.join(sub['log_dir'], 'loss.png'))
                torch.save(sub['ca'].state_dict(), os.path.join(sub['log_dir'], f'{step:04d}.pt'))

        step_candidate_objectives = []
        if evaluation_step:
            # Evaluate all children under the same fixed seed/horizon before
            # any replacement, then perform one synchronized MOEA/D decision.
            evaluation_rng_seed = (
                int(getattr(_cfg, 'RUN_SEED', 0)) * 1000003 + step
            )
            for i, sub in enumerate(subproblems):
                cand_obj, _ = evaluate_solution_state(
                    sub['ca'], clip_loss, text_embeddings, seed, device,
                    rng_seed=evaluation_rng_seed,
                )
                sub['obj_vec'] = cand_obj
                step_candidate_objectives.append(cand_obj.copy())
                step_records[i]['ca_state'] = copy.deepcopy(sub['ca'].state_dict())
                step_records[i]['obj_vec'] = cand_obj.copy()

            ideal_candidates = step_candidate_objectives + [
                sub['obj_vec'] for sub in subproblems
            ]
            ideal_point = empirical_ideal_point(
                ideal_candidates,
                previous=ideal_point,
            )

            if archive_controller is not None:
                archive_controller.sync_ideal_after_evaluation(ideal_point)
                for sub, cand_obj in zip(
                        subproblems, step_candidate_objectives):
                    archive_controller.after_update(sub, cand_obj, step)

            if cooperation_step and replacement_policy == 'stm':
                assignment, _ = _stable_matching_assignment(
                    [child['obj_vec'] for child in step_records],
                    weights,
                    _cfg.MOEAD_DECOMPOSITION,
                    ideal_point,
                )
                for target_index, child_index in enumerate(assignment):
                    child = step_records[child_index]
                    target = subproblems[target_index]
                    if child['source_index'] != target_index:
                        copy_solution_state(child['ca_state'], target)
                        target['replacement_count'] += 1
                    else:
                        target['ca'].load_state_dict(copy.deepcopy(child['ca_state']))
                    target['obj_vec'] = child['obj_vec'].copy()
                    if checkpoint_interval and step % checkpoint_interval == 0:
                        visualize_batch(
                            child['x0'], child['x_out'], step, target['log_dir']
                        )
                        plot_loss(
                            target['loss_log'],
                            save_path=os.path.join(target['log_dir'], 'loss.png'),
                        )
                        torch.save(
                            target['ca'].state_dict(),
                            os.path.join(target['log_dir'], f'{step:04d}.pt'),
                        )
            elif cooperation_step and replacement_policy in {'neighborhood', 'global'}:
                matches = plan_global_replacements(
                    step_candidate_objectives,
                    subproblems,
                    neighborhoods,
                    replacement_policy,
                    _cfg.MOEAD_DECOMPOSITION,
                    ideal_point,
                    cooperation_threshold,
                )
                # The plan is computed from frozen evaluated children; only
                # after matching do we commit every selected propagation.
                for source_id, target_id, _ in matches:
                    child = step_records[source_id]
                    target = subproblems[target_id]
                    copy_solution_state(child['ca_state'], target)
                    target['obj_vec'] = child['obj_vec'].copy()
                    target['replacement_count'] += 1

        if archive_controller is not None:
            archive_controller.end_step(step)

        # Non-evaluation steps deliberately keep the last synchronized ideal.
        ideal_point_evolution.append(ideal_point.tolist())

        if step == 1 or step % 10 == 0:
            elapsed = time.perf_counter() - training_started
            eta = (elapsed / step) * (_cfg.TRAIN_STEPS - step)
            print(
                f'  completed outer step {step}/{_cfg.TRAIN_STEPS} '
                f'| elapsed={elapsed / 60.0:.1f} min '
                f'| ETA={eta / 3600.0:.1f} h',
                flush=True,
            )

        if step % 100 == 0:
            obj_mat = np.stack([s['obj_vec'] for s in subproblems])
            print(f'  step {step} | ideal={ideal_point.tolist()} '
                  f'| mean_obj={np.round(obj_mat.mean(0), 4)}')

    final_eval_steps = max(1, int(_cfg.FINAL_EVAL_STEPS))
    final_eval_repeats = max(1, int(_cfg.FINAL_EVAL_REPEATS))
    final_objectives = []
    for sub in subproblems:
        repeated = []
        for repeat in range(final_eval_repeats):
            final_rng_seed = (
                int(getattr(_cfg, 'RUN_SEED', 0)) * 1000003
                + 10000000
                + repeat
            )
            obj, _ = evaluate_solution_state(
                sub['ca'], clip_loss, text_embeddings, seed, device,
                rollout_steps=final_eval_steps,
                rng_seed=final_rng_seed,
            )
            repeated.append(obj)
        sub['obj_vec'] = np.mean(np.stack(repeated), axis=0).astype(np.float32)
        final_objectives.append(sub['obj_vec'])
    ideal_point = empirical_ideal_point(final_objectives, previous=ideal_point)

    if archive_controller is not None:
        archive_controller.finalize(ideal_point)

    if _cfg.MOEAD_SAVE_FINAL_MODELS:
        for sub in subproblems:
            torch.save(
                sub['ca'].state_dict(),
                os.path.join(sub['log_dir'], 'final.pt'),
            )

    summary_dir = os.path.join(_cfg.TRAIN_LOG_ROOT, variant_name)
    os.makedirs(summary_dir, exist_ok=True)
    summary = {
        'variant': variant_name,
        'objective_prompts': list(_cfg.OBJECTIVE_PROMPTS),
        'weights': weights.tolist(),
        'neighborhoods': neighborhoods.tolist(),
        'final_objectives': [s['obj_vec'].tolist() for s in subproblems],
        'final_scalar': [
            decomposition_value(s['obj_vec'], s['weight_vec'], ideal_point,
                                method=_cfg.MOEAD_DECOMPOSITION)
            for s in subproblems
        ],
        'ideal_point': ideal_point.tolist(),
        'reference_point': ideal_point.tolist(),
        'ideal_point_rule': 'synchronous_empirical_best_observed',
        'ideal_point_evolution': ideal_point_evolution,
        'replacement_counts': [s['replacement_count'] for s in subproblems],
        'decomposition': _cfg.MOEAD_DECOMPOSITION,
        'run_seed': int(getattr(_cfg, 'RUN_SEED', 0)),
        'n_subproblems': N,
        'neighbor_size': T,
        'max_replacements': max_rep,
        'cooperation_interval': cooperation_interval,
        'cooperation_matching': (
            'global_max_weight_one_to_one'
            if (archive_controller is not None
                or replacement_policy in {'neighborhood', 'global'})
            else 'gale_shapley'
            if replacement_policy == 'stm'
            else 'none'
        ),
        'cooperation_improvement_threshold': (
            cooperation_threshold
            if (archive_controller is not None
                or replacement_policy in {'neighborhood', 'global'})
            else None
        ),
        'replacement_policy': replacement_policy,
        'pool_policy': pool_policy,
        'gradient_policy': gradient_policy,
        'use_rta': archive_controller is not None,
        'pareto_candidate_multiplier': int(_cfg.MOEAD_PARETO_CANDIDATE_MULTIPLIER),
        'pareto_exploration': float(_cfg.MOEAD_PARETO_EXPLORATION),
        'svgd_bandwidth': float(_cfg.MOEAD_SVGD_BANDWIDTH),
        'svgd_repulsion_coef': float(_cfg.MOEAD_SVGD_REPULSION_COEF),
        'pool_figure_interval': pool_figure_interval,
        'checkpoint_interval': checkpoint_interval,
        'eval_interval': eval_interval,
        'eval_steps': int(_cfg.EVAL_STEPS),
        'final_eval_steps': final_eval_steps,
        'final_eval_repeats': final_eval_repeats,
        'pool_size': int(_cfg.POOL_SIZE),
        'save_final_models': bool(_cfg.MOEAD_SAVE_FINAL_MODELS),
        'mean_repulsion_norm': float(np.mean([
            value
            for sub in subproblems
            for value in sub['repulsion_norm_log']
        ])) if subproblems else 0.0,
    }
    if archive_controller is not None:
        summary.update(archive_controller.summary())
    with open(os.path.join(summary_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n{variant_name.upper()} done. Summary -> {summary_dir}/summary.json')

    results = {}
    for i, sub in enumerate(subproblems):
        results[f'sub_{i:02d}'] = {
            'log_dir': sub['log_dir'],
            'weights': sub['weight_vec'].tolist(),
            'final_loss': sub['loss_log'][-1] if sub['loss_log'] else float('nan'),
            'final_objectives': sub['obj_vec'].tolist() if sub['obj_vec'] is not None else None,
            'ca': sub['ca'],
        }
    return {
        'results': results,
        'active_solutions': subproblems,
        'summary': summary,
        'summary_dir': summary_dir,
        'archive_controller': archive_controller,
    }


def train_moead_ca(clip_loss: CLIPLoss, seed: np.ndarray) -> dict:
    configure_strategy()
    return train_moead_core(
        clip_loss,
        seed,
        variant_name='ca',
        archive_controller=None,
    )['results']


def configure_strategy():
    """Lock the standalone MOEA/D-CA ablation definition."""
    import config as _cfg

    _cfg.MOEAD_VARIANT_NAME = 'ca'
    _cfg.MOEAD_REPLACEMENT_POLICY = 'neighborhood'
    _cfg.MOEAD_POOL_POLICY = 'preference_reset'
    _cfg.MOEAD_USE_RTA = False
    _cfg.MOEAD_GRADIENT_POLICY = 'weighted_sum'
    _cfg.MOEAD_MAX_REPLACEMENTS = 1
    _cfg.MOEAD_COOPERATION_INTERVAL = 50
    _cfg.POOL_SIZE = 1024
    _cfg.MOEAD_EVAL_INTERVAL = 10
    _cfg.MOEAD_POOL_FIGURE_INTERVAL = 100
    _cfg.MOEAD_CHECKPOINT_INTERVAL = 100
    _cfg.EVAL_STEPS = 64
    _cfg.FINAL_EVAL_STEPS = 96
    _cfg.FINAL_EVAL_REPEATS = 4
