"""MOEA/D-RTA: the shared MOEA/D-CA backbone plus a region-wise archive.

The training loop, objective evaluation, pattern-pool maintenance, scheduler,
and neighborhood replacement all live in ``training.moead_ca``.  This module
only implements region-wise reusable-state storage, selection, and transfer.
"""

import copy
import json
import os

import numpy as np

from clip_loss import CLIPLoss
from training.moead_ca import (
    decomposition_value,
    maximum_weight_one_to_one_matching,
    relative_scalar_improvement,
    train_moead_core,
)


RTA_SCALARIZATION = 'tchebycheff'


def rta_scalar_value(obj_vec, weight_vec, ideal_point=None):
    """Scalarize an archive objective using the RTA selection policy."""
    return decomposition_value(
        obj_vec,
        weight_vec,
        ideal_point,
        method=RTA_SCALARIZATION,
    )


def assign_region(obj_vec, region_weights, ideal_point):
    """Assign a state to the subproblem with the lowest decomposition value."""
    values = [
        rta_scalar_value(obj_vec, weight, ideal_point)
        for weight in region_weights
    ]
    return int(np.argmin(values))


def make_solution_record(sol, obj_vec, assigned_region_id, solution_id, step,
                         source_pool=None, include_optimizer=True):
    record = {
        'solution_id': int(solution_id),
        'source_solution_id': int(sol['solution_id']),
        'source_region_id': int(sol['region_id']),
        'assigned_region_id': int(assigned_region_id),
        'source_weight_vec': sol['weight_vec'].copy(),
        'creation_step': int(step),
        'update_count': int(sol['update_count']),
        'obj_vec': np.asarray(obj_vec, dtype=np.float32).copy(),
        'ca_state': copy.deepcopy(sol['ca'].state_dict()),
    }
    if include_optimizer:
        record['optimizer_state'] = copy.deepcopy(sol['optimizer'].state_dict())
    if source_pool is not None:
        record['pool_x'] = source_pool.copy()
    return record


def trim_region_bucket(bucket, region_weight, max_per_region, ideal_point):
    """Keep the top-k reusable states for one decomposition region."""
    bucket = sorted(
        bucket,
        key=lambda record: (
            rta_scalar_value(record['obj_vec'], region_weight, ideal_point),
            -record['creation_step'],
        ),
    )
    return bucket[:max_per_region]


def update_region_archive(archive, record, region_weights, max_per_region,
                       ideal_point):
    rid = record['assigned_region_id']
    bucket = archive.setdefault(rid, [])
    bucket.append(record)
    archive[rid] = trim_region_bucket(
        bucket, region_weights[rid], max_per_region, ideal_point
    )


def _select_archive_parent_with_reason(archive, sol, threshold=0.10,
                                   allow_self=False, ideal_point=None,
                                   excluded_source_ids=None):
    candidates = list(archive.get(sol['region_id'], ()))
    if not candidates:
        return None, 'empty_region'
    if not allow_self:
        candidates = [
            record for record in candidates
            if record['source_solution_id'] != sol['solution_id']
        ]
        if not candidates:
            return None, 'same_source_only'
    excluded_source_ids = set(excluded_source_ids or ())
    if excluded_source_ids:
        candidates = [
            record for record in candidates
            if record['source_solution_id'] not in excluded_source_ids
        ]
        if not candidates:
            return None, 'source_already_propagated'
    if sol['obj_vec'] is None:
        return None, 'missing_current_objective'

    parent = min(
        candidates,
        key=lambda record: rta_scalar_value(
            record['obj_vec'], sol['weight_vec'], ideal_point
        ),
    )
    parent_value = rta_scalar_value(
        parent['obj_vec'], sol['weight_vec'], ideal_point
    )
    current_value = rta_scalar_value(
        sol['obj_vec'], sol['weight_vec'], ideal_point
    )
    if not np.isfinite(parent_value) or not np.isfinite(current_value):
        return None, 'non_finite_scalar_value'
    required_improvement = threshold * max(abs(current_value), 1e-12)
    if parent_value >= current_value - required_improvement:
        return None, 'below_improvement_threshold'
    return parent, 'accepted'


def select_archive_parent(archive, sol, threshold=0.10, allow_self=False,
                      ideal_point=None, excluded_source_ids=None):
    parent, _ = _select_archive_parent_with_reason(
        archive, sol, threshold=threshold, allow_self=allow_self,
        ideal_point=ideal_point, excluded_source_ids=excluded_source_ids,
    )
    return parent


def plan_archive_transfers(archive, solutions, threshold=0.10,
                               allow_self=False, ideal_point=None,
                               excluded_source_ids=None):
    """Globally match unique archive sources to active target solutions."""
    excluded_source_ids = set(excluded_source_ids or ())
    edge_benefits = {}
    edge_parents = {}
    target_reasons = {}
    target_ids = [int(sol['solution_id']) for sol in solutions]

    for sol in solutions:
        target_id = int(sol['solution_id'])
        candidates = list(archive.get(sol['region_id'], ()))
        if not candidates:
            target_reasons[target_id] = 'empty_region'
            continue
        if not allow_self:
            candidates = [
                record for record in candidates
                if int(record['source_solution_id']) != target_id
            ]
            if not candidates:
                target_reasons[target_id] = 'same_source_only'
                continue
        if excluded_source_ids:
            candidates = [
                record for record in candidates
                if int(record['source_solution_id']) not in excluded_source_ids
            ]
            if not candidates:
                target_reasons[target_id] = 'source_already_propagated'
                continue
        if sol['obj_vec'] is None:
            target_reasons[target_id] = 'missing_current_objective'
            continue

        current_value = rta_scalar_value(
            sol['obj_vec'], sol['weight_vec'], ideal_point
        )
        if not np.isfinite(current_value):
            target_reasons[target_id] = 'non_finite_scalar_value'
            continue

        best_by_source = {}
        saw_finite_parent = False
        for record in candidates:
            parent_value = rta_scalar_value(
                record['obj_vec'], sol['weight_vec'], ideal_point
            )
            if not np.isfinite(parent_value):
                continue
            saw_finite_parent = True
            source_id = int(record['source_solution_id'])
            incumbent = best_by_source.get(source_id)
            record_key = (
                parent_value,
                int(record.get('solution_id', 0)),
                int(record.get('creation_step', 0)),
            )
            if incumbent is None or record_key < incumbent[0]:
                best_by_source[source_id] = (record_key, record)

        eligible = False
        for source_id, (record_key, record) in best_by_source.items():
            benefit = relative_scalar_improvement(
                record_key[0], current_value
            )
            if benefit <= threshold:
                continue
            eligible = True
            edge_benefits[(source_id, target_id)] = benefit
            edge_parents[(source_id, target_id)] = record

        if eligible:
            target_reasons[target_id] = 'global_match_unselected'
        elif saw_finite_parent:
            target_reasons[target_id] = 'below_improvement_threshold'
        else:
            target_reasons[target_id] = 'non_finite_scalar_value'

    source_ids = {source_id for source_id, _ in edge_benefits}
    raw_matches = maximum_weight_one_to_one_matching(
        source_ids, target_ids, edge_benefits
    )
    matches = []
    for source_id, target_id, benefit in raw_matches:
        target_reasons[target_id] = 'accepted'
        matches.append({
            'source_solution_id': source_id,
            'target_solution_id': target_id,
            'benefit': benefit,
            'parent': edge_parents[(source_id, target_id)],
        })
    return matches, target_reasons


def restore_current_lrs(optimizer, saved_lrs):
    for group, lr in zip(optimizer.param_groups, saved_lrs):
        group['lr'] = lr


def apply_archive_transfer(sol, parent, transfer_pool=True):
    """Restore CA + Adam + optional pool while preserving LR/scheduler time."""
    import config as _cfg

    pre_obj = sol['obj_vec'].copy() if sol['obj_vec'] is not None else None
    current_lrs = [group['lr'] for group in sol['optimizer'].param_groups]
    sol['ca'].load_state_dict(copy.deepcopy(parent['ca_state']))
    sol['optimizer'].load_state_dict(copy.deepcopy(parent['optimizer_state']))
    restore_current_lrs(sol['optimizer'], current_lrs)

    if transfer_pool and _cfg.USE_PATTERN_POOL and 'pool_x' in parent:
        sol['pool'].x[:] = parent['pool_x'].copy()

    sol['obj_vec'] = parent['obj_vec'].copy()
    sol['archive_transfer_count'] += 1
    sol['last_parent_solution_id'] = parent['solution_id']
    return {
        'parent_solution_id': parent['solution_id'],
        'parent_source_solution_id': parent['source_solution_id'],
        'parent_source_region_id': parent['source_region_id'],
        'parent_assigned_region_id': parent['assigned_region_id'],
        'parent_step': parent['creation_step'],
        'pre_obj': pre_obj.tolist() if pre_obj is not None else None,
        'parent_obj': parent['obj_vec'].tolist(),
    }


class RegionArchive:
    """Read-snapshot/write-buffer controller for the shared MOEA/D backbone."""

    def __init__(self, enable_transfer=True):
        import config as _cfg

        self.enable_transfer = bool(enable_transfer)
        self.warmup_steps = int(getattr(_cfg, 'RTA_WARMUP_STEPS', 400))
        self.transfer_interval = int(getattr(_cfg, 'RTA_TRANSFER_INTERVAL', 50))
        self.threshold = float(getattr(_cfg, 'COOPERATION_THRESHOLD', 0.10))
        self.max_per_region = int(getattr(_cfg, 'RTA_ARCHIVE_SIZE', 3))
        self.allow_self = bool(getattr(_cfg, 'RTA_ALLOW_SELF_TRANSFER', False))
        self.transfer_pool = bool(getattr(_cfg, 'RTA_TRANSFER_POOL', True))
        if self.warmup_steps < 0:
            raise ValueError(
                f'RTA_WARMUP_STEPS must be non-negative, got {self.warmup_steps}'
            )
        if self.transfer_interval <= 0:
            raise ValueError(
                'RTA_TRANSFER_INTERVAL must be positive, '
                f'got {self.transfer_interval}'
            )
        self.archive_log_dir = os.path.join(
            _cfg.TRAIN_LOG_ROOT, 'moead_rta', 'archive'
        )
        self.archive = {}
        self.read_archive = {}
        self.pending_records = []
        self.used_parent_source_ids = set()
        self.solution_counter = 0
        self.solutions = []
        self.weights = None
        self.ideal_point = None
        self.transfer_events = []
        self.update_events = []
        self.archive_size_evolution = []
        self.region_coverage_evolution = []
        self.selection_counts = {
            'accepted': 0,
            'warmup_skipped': 0,
            'interval_skipped': 0,
            'empty_region': 0,
            'same_source_only': 0,
            'source_already_propagated': 0,
            'missing_current_objective': 0,
            'non_positive_current_value': 0,
            'non_finite_scalar_value': 0,
            'below_improvement_threshold': 0,
            'global_match_unselected': 0,
        }

    def initialize(self, solutions, weights, ideal_point):
        self.solutions = solutions
        self.weights = weights
        self.ideal_point = np.asarray(ideal_point, dtype=np.float32).copy()
        os.makedirs(self.archive_log_dir, exist_ok=True)
        # Do not seed the reusable archive with step-0 pools.  With a large
        # persistent pool, those snapshots are almost entirely single-cell
        # seeds and can otherwise be restored long after training has begun.
        self.archive = {}
        self.read_archive = {}

    def _rebuild_archive(self):
        """Reassign stored states under the current synchronized ideal."""
        records = [
            record
            for bucket in self.archive.values()
            for record in bucket
        ]
        rebuilt = {}
        for original in records:
            record = dict(original)
            record['assigned_region_id'] = assign_region(
                record['obj_vec'], self.weights, self.ideal_point
            )
            update_region_archive(
                rebuilt,
                record,
                self.weights,
                self.max_per_region,
                self.ideal_point,
            )
        self.archive = rebuilt

    def begin_step(self, step, ideal_point):
        # All subproblems in one outer step see exactly the same archive state.
        self.ideal_point = np.asarray(ideal_point, dtype=np.float32).copy()
        self._rebuild_archive()
        self.read_archive = {
            rid: tuple(bucket) for rid, bucket in self.archive.items()
        }
        self.pending_records = []
        self.used_parent_source_ids = set()

    def sync_ideal_after_evaluation(self, ideal_point):
        """Repartition the archive under the newly evaluated empirical ideal."""
        self.ideal_point = np.asarray(ideal_point, dtype=np.float32).copy()
        self._rebuild_archive()

    def before_updates(self, solutions, step):
        if not self.enable_transfer:
            return
        if step <= self.warmup_steps:
            self.selection_counts['warmup_skipped'] += len(solutions)
            return
        if step % self.transfer_interval != 0:
            self.selection_counts['interval_skipped'] += len(solutions)
            return

        matches, target_reasons = plan_archive_transfers(
            self.read_archive,
            solutions,
            threshold=self.threshold,
            allow_self=self.allow_self,
            ideal_point=self.ideal_point,
            excluded_source_ids=self.used_parent_source_ids,
        )
        for reason in target_reasons.values():
            self.selection_counts[reason] += 1

        targets_by_id = {
            int(sol['solution_id']): sol for sol in solutions
        }
        # All matches are fixed before any target is mutated.
        for match in matches:
            sol = targets_by_id[match['target_solution_id']]
            parent = match['parent']
            self.used_parent_source_ids.add(match['source_solution_id'])
            event = apply_archive_transfer(sol, parent, transfer_pool=self.transfer_pool)
            event.update({
                'step': int(step),
                'target_solution_id': int(sol['solution_id']),
                'target_region_id': int(sol['region_id']),
                'normalized_improvement': float(match['benefit']),
                'transfer_count': int(sol['archive_transfer_count']),
            })
            self.transfer_events.append(event)

    def before_update(self, sol, step):
        """Compatibility wrapper for callers that manage one target only."""
        self.before_updates([sol], step)

    def after_update(self, sol, obj_vec, step):
        import config as _cfg

        # The first W outer steps train only.  Archive records begin at W+1,
        # so the first scheduled transfer at the next K boundary cannot
        # select a seed-heavy initialization snapshot.
        if step <= self.warmup_steps:
            return

        rid = assign_region(obj_vec, self.weights, self.ideal_point)
        record = make_solution_record(
            sol,
            obj_vec,
            rid,
            self.solution_counter,
            step,
            source_pool=(
                sol['pool'].x
                if self.transfer_pool and _cfg.USE_PATTERN_POOL
                else None
            ),
        )
        self.solution_counter += 1
        self.pending_records.append(record)

    def end_step(self, step):
        for record in self.pending_records:
            rid = record['assigned_region_id']
            before = len(self.archive.get(rid, []))
            update_region_archive(
                self.archive, record, self.weights, self.max_per_region,
                self.ideal_point,
            )
            after = len(self.archive.get(rid, []))
            self.update_events.append({
                'step': int(step),
                'solution_id': record['solution_id'],
                'source_solution_id': record['source_solution_id'],
                'assigned_region_id': rid,
                'bucket_before': before,
                'bucket_after': after,
                'obj_vec': record['obj_vec'].tolist(),
            })
        self.archive_size_evolution.append(
            sum(len(bucket) for bucket in self.archive.values())
        )
        self.region_coverage_evolution.append(len(self.archive))

    def finalize(self, ideal_point):
        self.ideal_point = np.asarray(ideal_point, dtype=np.float32).copy()
        self._rebuild_archive()

    def summary(self):
        transfer_counts = [sol['archive_transfer_count'] for sol in self.solutions]
        return {
            'archive_rule': 'region_training_state',
            'archive_scalarization': RTA_SCALARIZATION,
            'archive_transfer_enabled': self.enable_transfer,
            'final_archive_size': sum(
                len(bucket) for bucket in self.archive.values()
            ),
            'archive_region_counts': {
                int(rid): len(bucket) for rid, bucket in self.archive.items()
            },
            'archive_size_evolution': self.archive_size_evolution,
            'region_coverage_evolution': self.region_coverage_evolution,
            'archive_transfer_counts': transfer_counts,
            'total_archive_transfers': sum(transfer_counts),
            'selection_counts': self.selection_counts,
            'archive_records_created': self.solution_counter,
            'archive_size_per_region': self.max_per_region,
            'archive_warmup_steps': self.warmup_steps,
            'transfer_interval': self.transfer_interval,
            'transfer_threshold': self.threshold,
            'cooperation_matching': 'global_max_weight_one_to_one',
            'allow_self_transfer': self.allow_self,
            'transfer_pool': self.transfer_pool,
            'archive_ideal_point': (
                self.ideal_point.tolist()
                if self.ideal_point is not None else None
            ),
        }

    def write_logs(self):
        os.makedirs(self.archive_log_dir, exist_ok=True)
        with open(
            os.path.join(self.archive_log_dir, 'archive_transfer_log.jsonl'),
            'w',
            encoding='utf-8',
        ) as handle:
            for event in self.transfer_events:
                handle.write(json.dumps(event) + '\n')
        with open(
            os.path.join(self.archive_log_dir, 'archive_update_log.jsonl'),
            'w',
            encoding='utf-8',
        ) as handle:
            for event in self.update_events:
                handle.write(json.dumps(event) + '\n')


def train_moead_rta(clip_loss: CLIPLoss, seed: np.ndarray,
                    enable_transfer=True) -> dict:
    """Run the shared MOEA/D-CA backbone with a region-wise region-wise archive."""
    import config as _cfg

    configure_strategy()
    controller = RegionArchive(enable_transfer=enable_transfer)
    controller.archive_log_dir = os.path.join(
        _cfg.TRAIN_LOG_ROOT, 'rta', 'archive'
    )
    core_result = train_moead_core(
        clip_loss,
        seed,
        variant_name='rta',
        archive_controller=controller,
    )
    controller.write_logs()
    archive_summary_path = os.path.join(controller.archive_log_dir, 'summary.json')
    with open(archive_summary_path, 'w', encoding='utf-8') as handle:
        json.dump(core_result['summary'], handle, indent=2)
    return {
        **core_result['summary'],
        'results': core_result['results'],
        'active_solutions': core_result['active_solutions'],
        'archive': controller.archive,
        'archive_controller': controller,
        'archive_log_dir': controller.archive_log_dir,
    }


def configure_strategy():
    """Lock the standalone MOEA/D-RTA ablation definition."""
    import config as _cfg

    _cfg.MOEAD_VARIANT_NAME = 'rta'
    _cfg.MOEAD_REPLACEMENT_POLICY = 'none'
    _cfg.MOEAD_POOL_POLICY = 'preference_reset'
    _cfg.MOEAD_USE_RTA = True
    _cfg.MOEAD_GRADIENT_POLICY = 'weighted_sum'
    _cfg.MOEAD_MAX_REPLACEMENTS = 1
    _cfg.MOEAD_COOPERATION_INTERVAL = 50
    _cfg.RTA_WARMUP_STEPS = 0
    _cfg.RTA_TRANSFER_INTERVAL = 50
    _cfg.RTA_TRANSFER_POOL = False
    _cfg.POOL_SIZE = 1024
    _cfg.MOEAD_EVAL_INTERVAL = 10
    _cfg.MOEAD_POOL_FIGURE_INTERVAL = 100
    _cfg.MOEAD_CHECKPOINT_INTERVAL = 100
    _cfg.EVAL_STEPS = 64
    _cfg.FINAL_EVAL_STEPS = 96
    _cfg.FINAL_EVAL_REPEATS = 4
