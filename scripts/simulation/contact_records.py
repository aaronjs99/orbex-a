"""Contact history and conservation audit serialization.

These records describe the physical plant independently of estimated feedback.
Their keys and field meanings are the finite-contact v1 persisted contract.
"""
from __future__ import annotations

from dataclasses import asdict
import numpy as np

from orbexa.simulation.contact_contracts import FiniteContactResult


class ContactRecorder:
    """Collect history and assemble the original finite-contact result/audit."""

    def __init__(self, *, fleet, options, geometry, bonds, actuation, extrema,
                 minimum_clearances, original, feedback=None, environment=None):
        self.fleet = fleet
        self.options = options
        self.geometry = geometry
        self.bonds = bonds
        self.actuation = actuation
        self.maxima = extrema
        self.minima = minimum_clearances
        self.original = original
        self.feedback = feedback
        self.environment = environment
        self.history = []
        self.contact_records = {}

    def record(self, t, values, diag, force=False):
        fleet, o = self.fleet, self.options
        started, members, ledger_start = fleet.started, fleet.members, fleet.ledger_start
        snapshot, unpack = fleet.snapshot, fleet.unpack
        history, controls = self.history, self.actuation.controls
        if history and not force and started+t-history[-1]['time_s'] < o.history_step_s-1e-10:
            return
        view, tips = snapshot(t, values), unpack(values)[1]
        history.append(dict(time_s=started+t, target=view['target'].to_dict(),
            chasers={key: state.to_dict() for key, state in view['chasers'].items()},
            pads={member.id: dict(extension_body_m=tips[k, :3].tolist(), extension_velocity_body_m_s=tips[k, 3:].tolist(),
                gap_m=row['gap'], force_eci_n=row['force'].tolist(), moment_eci_nm=row['couple'].tolist(),
                activation=row['activation'], relative_speed_m_s=float(np.linalg.norm(row['u'])),
                relative_spin_rad_s=float(np.linalg.norm(row['spin'])),
                tip_position_eci_m=(values[:3]+row['tip']).tolist())
                for k, (member, row) in enumerate(zip(members, diag['rows']))},
            controls_body={member.id: controls[k+1].tolist() for k, member in enumerate(members)},
            ledger=values[ledger_start:].tolist(), kinetic_energy_j=diag['kinetic'], contact_potential_j=diag['potential']))

    def finish(self, success, message, *, t, y, diag, dwell, lock_energy, saturation_count):
        fleet, o = self.fleet, self.options
        n, count, members = fleet.body_count, fleet.chaser_count, fleet.members
        started, ledger_start = fleet.started, fleet.ledger_start
        anchors, normals = fleet.anchors, fleet.normals
        snapshot, record = fleet.snapshot, self.record
        engagement = self.bonds.engagement
        first_touch, first_touch_speed = self.bonds.first_touch, self.bonds.first_touch_speed
        history, contact_records = self.history, self.contact_records
        maxima, minima, original = self.maxima, self.minima, self.original
        geometry, feedback, environment = self.geometry, self.feedback, self.environment
        record(t, y, diag, force=not history or history[-1]['time_s'] != started+t)
        view, ledger = snapshot(t, y), y[ledger_start:]
        p_residual = diag['linear']-original['linear']-ledger[:3]-ledger[3:6]
        h_residual = diag['angular']-original['angular']-ledger[6:9]-ledger[9:12]
        expected_energy = ledger[12]+ledger[13]+ledger[14]+ledger[15]-sum(ledger[16:19])
        energy_residual = diag['kinetic']+diag['potential']-original['kinetic']-original['potential']-expected_energy
        scale = max(1e-12, abs(diag['kinetic']-original['kinetic']), abs(expected_energy), sum(abs(ledger[12:])))
        momentum_ok = (np.linalg.norm(p_residual) <= o.momentum_absolute_tolerance+o.momentum_relative_tolerance*max(np.linalg.norm(original['linear']), np.linalg.norm(ledger[:3])) and
            np.linalg.norm(h_residual) <= o.momentum_absolute_tolerance+o.momentum_relative_tolerance*max(np.linalg.norm(original['angular']), np.linalg.norm(ledger[6:9])))
        energy_ok = abs(energy_residual) <= max(o.energy_absolute_tolerance_j, o.energy_relative_tolerance*scale)
        if success and not (momentum_ok and energy_ok):
            success, message = False, 'finite_contact_balance_tolerance_failed'
        for k, member in enumerate(members):
            row = diag['rows'][k]
            contact_records[member.id] = dict(event='finite_compliant_adhesive_capture',
                time_s=started+t, first_contact_time_s=None if np.isnan(first_touch[k]) else started+first_touch[k],
                first_contact_relative_speed_m_s=None if np.isnan(first_touch_speed[k]) else float(first_touch_speed[k]),
                contact_event_time_resolution_s=o.max_step_s,
                engagement_start_time_s=None if not np.isfinite(engagement[k]) else started+engagement[k],
                latch_time_s=started+t if success else None, target_docking_fixture_required=False,
                surface_point_body_m=(anchors[k]+row['e']-(row['e']@normals[k])*normals[k]).tolist(),
                tip_at_contact_body_m=(anchors[k]+row['e']).tolist(),
                contact_residual_m=abs(row['gap']), available_tip_reach_m=member.arm_reach_m,
                target_interface='representative adhesive pad on unprepared declared structural patch',
                tensile_limit_n=o.tensile_limit_n, shear_limit_n=o.shear_limit_n, moment_limit_nm=o.moment_limit_nm)
        finite_minima = {key: None if not np.isfinite(value) else float(value) for key, value in minima.items()}
        audit = dict(model='finite_coupled_rigid_body_compliant_adhesive.v1', options=asdict(o),
            elapsed_s=t, integrated_body_count=n, chaser_count=count, peer_pair_count=count*(count-1)//2,
            occupied_component_count=len(geometry.components) if hasattr(geometry, 'components') else 1,
            geometry_scope='All occupied components, finite buses, pads/positioners and peer buses checked at internal steps with velocity/acceleration travel reserves; no interval continuous-contact enclosure',
            observed_settling_dwell_s=dwell, final_rigidization_energy_j=float(lock_energy) if np.isfinite(lock_energy) else None,
            extrema=maxima, minimum_clearances=finite_minima,
            linear_momentum_balance_residual_kg_m_s=float(np.linalg.norm(p_residual)),
            angular_momentum_balance_residual_kg_m2_s=float(np.linalg.norm(h_residual)),
            energy_balance_residual_j=float(energy_residual), energy_balance_scale_j=float(scale),
            momentum_balance_passed=bool(momentum_ok), energy_balance_passed=bool(energy_ok),
            external_linear_impulse_kg_m_s=ledger[:3].tolist(), fuel_linear_flux_kg_m_s=ledger[3:6].tolist(),
            external_angular_impulse_kg_m2_s=ledger[6:9].tolist(), fuel_angular_flux_kg_m2_s=ledger[9:12].tolist(),
            external_work_j=float(ledger[12]), fuel_energy_flux_j=float(ledger[13]),
            positioner_work_j=float(ledger[14]), engagement_work_j=float(ledger[15]),
            compression_dissipation_j=float(ledger[16]), friction_dissipation_j=float(ledger[17]),
            bond_dissipation_j=float(ledger[18]), initial_kinetic_energy_j=original['kinetic'],
            final_kinetic_energy_j=diag['kinetic'], final_contact_potential_j=diag['potential'],
            fuel_used_kg={member.id: float(member.initial_state.fuel_kg-view['chasers'][member.id].fuel_kg) for member in members},
            saturated_control_steps=saturation_count,
            balance_frame='translating nonrotating central-gravity reference; differential gravity includes the frame acceleration; fuel carries local velocity and specific inertia',
            environment_scope='injected extra accelerations beyond central gravity and extra body torques beyond gravity gradient; included in external wrench/work ledgers' if environment else 'central gravity and gravity gradient only',
            environment_contract=environment.as_dict() if callable(getattr(environment, 'as_dict', None)) else None,
            feedback_scope='injected estimated relative navigation; physical plant retained' if feedback else 'exact physical feedback assumption',
            scope='Representative finite compliant pads and massless finite-bandwidth positioners, rigid hull and bus inertia; no mechanism material, flexible appendage, plume or flight qualification',
            reference='https://ntrs.nasa.gov/citations/19870016368')
        return FiniteContactResult(success, message, view['target'], view['chasers'], tuple(history), contact_records, audit)
