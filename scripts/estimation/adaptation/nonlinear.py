"""Observation-only initial-orbit contraction and nonlinear drag membership.

The initial osculating eccentricity is not treated as constant throughout a
J2-perturbed orbit. Drag membership acts on the actual Cartesian force law.
Unknown candidates are retained. No numerical plant truth enters this class.
"""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import numpy as np

from orbexa.control.certification.reachability import IntervalBox, _ARB_LOCK
from orbexa.control.certification.nonlinear import _Arithmetic
from orbexa.core.dynamics.relative import dot
from orbexa.core.dynamics.reference_frame import exact
from orbexa.estimation.navigation.service import infer_target_ephemeris, intersect
from orbexa.estimation.membership.nonlinear import NonlinearMembershipContractor


def initial_eccentricity_box(inertial_box,mu,*,precision_bits=128):
    """Arb image of e-vector norm for a six-state initial SI observation box."""
    import flint
    if inertial_box.dimension != 6 or exact(mu)<=0:
        raise ValueError('An initial six-state ephemeris and positive mu are required')
    with _ARB_LOCK,flint.ctx.workprec(precision_bits):
        ar=_Arithmetic(flint)
        r,v=ar.vector(inertial_box)[:3],ar.vector(inertial_box)[3:]
        radius=ar.sqrt(sum(ar.square(x) for x in r))
        speed2=sum(ar.square(x) for x in v)
        energy_term=speed2-ar.scalar(mu)/radius
        evec=tuple((energy_term*x-dot(r,v)*y)/ar.scalar(mu) for x,y in zip(r,v))
        result=ar.box((ar.sqrt(sum(ar.square(x) for x in evec)),))
    return result


class NonlinearMissionAdaptation:
    """Independent initial orbit conditioning and online physical membership.

    ``legacy`` preserves the original mode-coupled initial inference. Explicit
    ``enabled`` lets frozen/adaptive comparisons share the same initial orbit
    observations while ``mode`` alone controls subsequent drag elimination.
    """
    def __init__(self,prior_model,controller_belief,clock,initial_relative_boxes,*,mode='adaptive',
                 encloser=None,target_partitions=2,chaser_partitions=2,max_candidates=16,
                 initial_orbit_conditioning='legacy',membership_policy='independent-holds-v1',
                 membership_window_holds=8,membership_max_nuisance_cells=2,
                 membership_max_flow_evaluations_per_candidate=256):
        if mode not in ('adaptive','frozen'):
            raise ValueError('Select adaptive or frozen belief')
        if (not isinstance(initial_orbit_conditioning,str) or
                initial_orbit_conditioning not in ('legacy','enabled','disabled')):
            raise ValueError('Initial orbit conditioning must be legacy, enabled or disabled')
        self.initial_orbit_conditioning=initial_orbit_conditioning
        self.initial_orbit_conditioning_enabled=(initial_orbit_conditioning=='enabled' or
            initial_orbit_conditioning=='legacy' and mode=='adaptive')
        self.prior_model,self.belief,self.clock=prior_model,controller_belief,clock
        self.initial_relative_boxes=dict(initial_relative_boxes)
        self.mode,self.initial_target_box=mode,None
        self.smid_records,self.membership_records,self.initial_packet_ids=[],[],set()
        if membership_policy not in ('independent-holds-v1','linked-affine-window-v1'):
            raise ValueError('Select an explicit nonlinear membership policy')
        self.membership_policy=membership_policy
        self.pending_transitions=[]
        self.observation_window=None
        if membership_policy=='linked-affine-window-v1':
            from orbexa.estimation.membership.linked import LinkedObservationWindow, LinkedNonlinearMembershipContractor
            self.observation_window=LinkedObservationWindow(initial_relative_boxes,membership_window_holds)
            self.contractor=LinkedNonlinearMembershipContractor(encloser,max_boxes=max_candidates,
                max_nuisance_cells=membership_max_nuisance_cells,
                max_flow_evaluations_per_candidate=membership_max_flow_evaluations_per_candidate,
                max_agents=len(initial_relative_boxes))
        else:
            self.contractor=NonlinearMembershipContractor(encloser,max_boxes=max_candidates)
        def partitions(pair,count):
            if isinstance(count,bool) or not isinstance(count,int) or count<1:
                raise ValueError('Physical candidate partitions must be positive integers')
            return [(pair[0]+(pair[1]-pair[0])*i/count,
                     pair[0]+(pair[1]-pair[0])*(i+1)/count) for i in range(count)]
        self.candidates=tuple(replace(prior_model,sigma_target=t,sigma_chaser=c)
            for t in partitions(prior_model.sigma_target,target_partitions)
            for c in partitions(prior_model.sigma_chaser,chaser_partitions))
        self.active_model=prior_model

    def consume_delivered(self,packets,*,accepted_packet_ids,sensor_fingerprints):
        """Only navigation-accepted, delivered q=0 packets qualify the orbit."""
        if not self.initial_orbit_conditioning_enabled:
            return
        accepted=frozenset(accepted_packet_ids);sensors=frozenset(sensor_fingerprints)
        for packet in packets:
            if (packet.measurement_kind!='chaser_gnss' or packet.acquired_q!=(exact(0),exact(0))
                    or packet.packet_id in self.initial_packet_ids or packet.packet_id not in accepted
                    or packet.quality_status!='VALID' or packet.sensor_model_fingerprint not in sensors):
                continue
            self.initial_packet_ids.add(packet.packet_id)
            if packet.agent_id not in self.initial_relative_boxes:
                raise ValueError('Initial GNSS packet names an unknown release vehicle')
            target=infer_target_ephemeris(packet,self.initial_relative_boxes[packet.agent_id],self.clock)
            box=target if self.initial_target_box is None else intersect(target,self.initial_target_box)
            if box is None:
                raise ValueError('Delivered initial GNSS/relative priors are inconsistent')
            self.initial_target_box=box
            ebox=initial_eccentricity_box(box,self.clock.mu_m3_s2)
            prior=self.belief.feasible_sets['eccentricity']
            lower=max(Fraction(prior[0]),ebox.lower[0]); upper=min(Fraction(prior[1]),ebox.upper[0])
            if lower>upper:
                raise ValueError('Initial orbit observation contradicts controller eccentricity prior')
            lo,hi=IntervalBox((lower,),(upper,)).float_bounds()
            self.belief.feasible_sets['eccentricity']=(lo[0],hi[0])
            self.belief.estimates['eccentricity']=(lo[0]+hi[0])/2
            self.smid_records.append(dict(method='initial-osculating-orbit-arb-image-v1',
                packet_id=packet.packet_id,acquired_q='0',eccentricity_box=ebox.as_dict(),
                retained_eccentricity=list(self.belief.feasible_sets['eccentricity']),
                scope='Initial orbit descriptor only; not a constant eccentricity model under J2'))

    def retain_transitions(self,transitions):
        """Retain bounded observation history separately from the update cadence."""
        if self.mode=='frozen':
            return
        if self.observation_window is not None:
            self.observation_window.append(transitions)
        else:
            self.pending_transitions.extend(transitions)
            del self.pending_transitions[:-len(self.initial_relative_boxes)]

    def contract(self,transitions=None,*,update):
        if self.mode=='frozen':
            self.membership_records.append(dict(update=update,status='FROZEN',candidate_count=len(self.candidates)))
            return True
        if transitions is None:
            transitions=(self.observation_window.transitions() if self.observation_window is not None
                else tuple(self.pending_transitions))
        result=self.contractor.contract(self.candidates,transitions,clock=self.clock)
        self.membership_records.append(dict(update=update,status=result.status,
            candidate_count_before=len(self.candidates),candidate_count_after=len(result.retained_models),
            discarded_fingerprints=list(result.discarded_fingerprints),records=list(result.records)))
        if not result.retained_models:
            return False
        self.candidates=result.retained_models
        target=(min(m.sigma_target[0] for m in self.candidates),max(m.sigma_target[1] for m in self.candidates))
        chaser=(min(m.sigma_chaser[0] for m in self.candidates),max(m.sigma_chaser[1] for m in self.candidates))
        self.active_model=replace(self.prior_model,sigma_target=target,sigma_chaser=chaser)
        rho=self.prior_model.atmosphere.rho0_kg_m3
        if rho[0]!=rho[1]:
            raise ValueError('Controller drag mapping requires an explicit reference density')
        coefficient=rho[0]*self.prior_model.atmosphere.reference_radius_m/2
        for key,sigma in (('alpha',target),('beta',chaser)):
            box=IntervalBox((sigma[0]*coefficient,),(sigma[1]*coefficient,))
            lo,hi=box.float_bounds()
            # Intersect with the original prior rather than enlarge it because
            # a decimal reference mapping rounded at a binary64 boundary.
            old=self.belief.feasible_sets[key]
            lo,hi=max(old[0],lo[0]),min(old[1],hi[0])
            if lo>hi:
                raise ValueError('Physical drag candidates contradict the controller reference mapping')
            self.belief.feasible_sets[key]=(lo,hi)
            self.belief.estimates[key]=(lo+hi)/2
        return True

    def metadata(self):
        result=dict(mode=self.mode,initial_orbit_packets=sorted(self.initial_packet_ids),
            initial_orbit_conditioning=self.initial_orbit_conditioning,
            initial_orbit_conditioning_enabled=self.initial_orbit_conditioning_enabled,
            online_physical_membership_enabled=self.mode=='adaptive',
            physical_prior=self.prior_model.as_dict(),active_physical_hull=self.active_model.as_dict(),
            retained_physical_candidates=[m.as_dict() for m in self.candidates],
            semantics='Initial osculating eccentricity from own GNSS plus bounded relative observations; '
                      'Cartesian sigma=CdA/m membership uses actual held-input transitions; unresolved retained')
        if self.observation_window is not None:
            result['linked_membership']=dict(method=self.membership_policy,
                window_holds_per_agent=self.observation_window.holds_per_agent,
                max_nuisance_cells_per_agent=self.contractor.max_nuisance_cells,
                max_flow_evaluations_per_candidate=self.contractor.max_flow_evaluations_per_candidate,
                retained_observation_counts={key:len(value) for key,value in self.observation_window.history.items()},
                nuisance_persistence='Fixed affine matrix and bias per agent; independent binary64 evaluation remainder',
                scope='Finite covering branches and conditional state links; unresolved branches retained; no parameter convergence guarantee')
        return result
