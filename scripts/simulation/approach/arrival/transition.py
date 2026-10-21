"""Explicit source seam for adding the bounded arrival policy to a saved mission.

Ordinary checkpoint validation is not relaxed. Preparation validates the old
scenario, compares all non-arrival destination declarations, and restores the
same physical guard once. It grants planning eligibility, not motion or backup.
"""
from __future__ import annotations

import copy
from pathlib import Path
import json

from orbexa.simulation.approach.arrival.policy import ArrivalPolicyConfig
from orbexa.simulation.approach.records.checkpoint import MissionCheckpoint, digest, plain, scenario_checkpoint_context


class MissionPolicyTransition:
    """A prepared, single-use in-process transition; serialization is evidence only."""

    def __init__(self, checkpoint, record, seed, target_context, guard_context):
        self.source_checkpoint = checkpoint
        self.record = copy.deepcopy(record)
        self.identity = record['identity']
        self._seed = seed
        self._seed_digest = digest(seed)
        self._target_context = digest(target_context)
        self._guard_context = digest(guard_context)
        self._used = False

    @classmethod
    def prepare(cls, *, source_checkpoint, source_scenario, target_scenario):
        if not isinstance(source_checkpoint, MissionCheckpoint):
            raise TypeError('An explicit transition requires its original typed MissionCheckpoint')
        source_checkpoint.validate_scenario(source_scenario, source_checkpoint.mission)
        if getattr(source_scenario, 'arrival_policy', None) is not None or source_checkpoint.arrival_runtime() is not None:
            raise ValueError('An existing arrival commitment must use ordinary strict resume; it cannot be replaced by a transition')
        if not isinstance(getattr(target_scenario, 'arrival_policy', None), ArrivalPolicyConfig):
            raise ValueError('The destination must explicitly enable a typed arrival policy')
        old, new = scenario_checkpoint_context(source_scenario), scenario_checkpoint_context(target_scenario)
        metadata = source_checkpoint.source_record['metadata']
        source_context = metadata.get('checkpoint_context') or source_checkpoint.source_context.get('checkpoint_context')
        if plain(source_context['configuration']) != plain(old['configuration']):
            raise ValueError('Arrival transition source must match the complete saved configuration, including solver settings')
        if target_scenario.execution_guard is None:
            raise ValueError('An arrival transition requires the same guarded physical execution interface')
        # No controller costs, geometry modes, routes, hardware, uncertainty,
        # cadence or tolerances may hide inside this policy-only exception.
        for name in set(old) | set(new):
            if name in ('policy', 'source_fingerprints'):
                continue
            if name not in old or name not in new or plain(old[name]) != plain(new[name]):
                raise ValueError(f'Arrival transition changed non-allowlisted scenario {name}')
        old_policy, new_policy = copy.deepcopy(old['policy']), copy.deepcopy(new['policy'])
        if old_policy.pop('arrival_policy', None) is not None:
            raise ValueError('Only an explicitly absent source arrival policy may be introduced')
        added = new_policy.pop('arrival_policy', None)
        if added != target_scenario.arrival_policy.as_dict() or plain(old_policy) != plain(new_policy):
            raise ValueError('Arrival transition changed an existing reference/deployment/physical policy')
        if source_scenario.reference.configured_horizon_metadata() != target_scenario.reference.configured_horizon_metadata():
            raise ValueError('Arrival transition changed its original planning horizon')
        guarded = metadata.get('continuous_execution_certificate') or {}
        exact_start = guarded.get('exact_end_anomaly')
        if exact_start is None and source_checkpoint.committed_updates == 0 and source_checkpoint.anomaly == 0.:
            exact_start = '0'
        if (exact_start is None or guarded.get('admitted_holds') !=
                source_checkpoint.committed_updates * target_scenario.config.tube.feedback_substeps):
            raise ValueError('Arrival transition requires a complete outer-grid prefix with its exact continuous clock')
        # Full original guard.restore checks remain authoritative, including
        # its prior, sensor, model/input bounds, bodies, target context and clock.
        seed = source_checkpoint._restore_validated(target_scenario)
        old_files, new_files = source_context['source_fingerprints'], new['source_fingerprints']
        changed_files = {name: dict(source=old_files.get(name), destination=new_files.get(name))
                         for name in sorted(set(old_files) | set(new_files)) if old_files.get(name) != new_files.get(name)}
        payload = dict(schema='orbexa.mission_policy_transition.v1',
            source_checkpoint_identity=source_checkpoint.identity,
            source_record_sha256=source_checkpoint.source_record_sha256,
            source_canonical_sha256=source_checkpoint.source_canonical_sha256,
            source_reference=source_checkpoint.source_context.get('source_reference'),
            committed_updates=source_checkpoint.committed_updates,
            exact_start_anomaly=exact_start,
            source_checkpoint_context=copy.deepcopy(source_context), destination_checkpoint_context=new,
            changed_source_files=changed_files,
            allowed_policy_change=dict(path='scenario.arrival_policy', before=None, after=added),
            status='PREPARED_FOR_PLANNING',
            scope='New prospective arrival-policy cohort with literal retained prefix; not candidate admission, motion or a backup theorem')
        record = dict(identity=digest(payload), **payload)
        return cls(source_checkpoint, record, seed, new, target_scenario.execution_guard.as_metadata())

    def consume(self, scenario):
        if self._used:
            raise ValueError('A prepared policy transition is single-use; prepare a new explicit object for another run')
        payload = {key: value for key, value in self.record.items() if key != 'identity'}
        if self.identity != self.record.get('identity') or digest(payload) != self.identity:
            raise ValueError('Prepared policy transition evidence changed')
        self.source_checkpoint._validate_record()
        if (digest(scenario_checkpoint_context(scenario)) != self._target_context
                or digest(self._seed) != self._seed_digest
                or scenario.execution_guard is None
                or digest(scenario.execution_guard.as_metadata()) != self._guard_context):
            raise ValueError('Prepared transition changed its scenario, literal numerical seed or inherited guard')
        self._used = True
        return copy.deepcopy(self._seed)

    def as_dict(self):
        return copy.deepcopy(self.record)

    def save(self, path):
        path = Path(path)
        if path.exists():
            raise FileExistsError('Preserve the existing policy-transition record')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.tmp')
        if temporary.exists():
            raise FileExistsError('Preserve the existing temporary policy-transition record')
        temporary.write_text(json.dumps(self.as_dict(), allow_nan=False), encoding='utf-8')
        temporary.replace(path)
        return path

    def validate_result_prefix(self, result):
        """Publication check: raw historical statistics and numeric lists stay facts."""
        old, new = self.source_checkpoint.source_record, plain(result.to_dict())
        history = ('anom_history', 'sample_phase_history', 'actual_trajectories', 'nominal_trajectories', 'controls',
            'solve_stats', 'solver_costs', 'target_attitude_history', 'target_angular_velocity_history',
            'feasible_set_history', 'parameter_estimate_history', 'smid_records', 'tube_profiles',
            'tube_radius_history', 'tube_diagnostics')
        def prefix(left, right):
            if isinstance(left, dict):
                return isinstance(right, dict) and all(key in right and prefix(value, right[key]) for key, value in left.items())
            if isinstance(left, list):
                return isinstance(right, list) and len(right) >= len(left) and right[:len(left)] == left
            return left == right
        count = self.source_checkpoint.committed_updates
        for name in history:
            left = old[name]
            if name in ('nominal_trajectories', 'tube_profiles'):
                left = {key: values[:count] for key, values in left.items()}
            if not prefix(left, new[name]):
                raise ValueError(f'Policy transition modified immutable historical prefix {name}')
        archived = new['metadata']['source_uncommitted_planning'][self.source_checkpoint.identity]
        for key in old['actual_trajectories']:
            if (archived[key]['nominal_trajectories'] != old['nominal_trajectories'][key][count:]
                    or archived[key]['tube_profiles'] != old['tube_profiles'][key][count:]):
                raise ValueError('Policy transition lost original unexecuted numerical proposals')
        if old['metadata'].get('target_kinematics') is not None and not prefix(
                old['metadata']['target_kinematics'], new['metadata']['target_kinematics']):
            raise ValueError('Policy transition changed original target kinematics')
        return True
