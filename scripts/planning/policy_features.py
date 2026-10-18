"""Measured-state policy features and potential-based reward shaping.

Features are planning inputs, not collision geometry certificates. The original
Playground's RL environment is replaced by explicit contracts, so reset state,
moving references and physical admission remain owned by the chosen simulator.
MIT source attribution: THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RelativeObservationEncoder:
    """[state6, moving-reference error6, K*(relative pose6,radius,mask)].

    A threat provider(k) returns (states[n,6], bounding_radii[n]). Bounding
    spheres here are feature summaries only; Envisat collision admission still
    uses its actual composite geometry. Empty slots have an explicit zero mask.
    """
    threat_provider: object
    nearest: int = 5
    position_scale: float = 300.
    velocity_scale: float = 300.

    def __post_init__(self):
        if not callable(self.threat_provider) or isinstance(self.nearest,bool) or not isinstance(self.nearest,int) or self.nearest<0:
            raise ValueError('Use an explicit threat provider and nonnegative integer feature count')
        if any(not np.isfinite(v) or v<=0 for v in (self.position_scale,self.velocity_scale)):
            raise ValueError('Feature normalization scales must be positive and finite')

    @property
    def dimension(self):return 12+8*self.nearest

    def encode(self,state,reference,k):
        state=np.asarray(state,dtype=float);reference=np.asarray(reference,dtype=float)
        if state.shape!=(6,) or reference.shape!=(6,) or not np.all(np.isfinite(np.r_[state,reference])):
            raise ValueError('Policy features require finite observed six-states')
        threats,radii=self.threat_provider(k)
        threats=np.asarray(threats,dtype=float);radii=np.asarray(radii,dtype=float)
        if threats.size==0:threats=np.empty((0,6))
        if (threats.ndim!=2 or threats.shape[1]!=6 or radii.shape!=(len(threats),)
                or not np.all(np.isfinite(threats)) or not np.all(np.isfinite(radii)) or np.any(radii<0)):
            raise ValueError('Threat feature states/radii are malformed')
        scales=np.r_[np.full(3,self.position_scale),np.full(3,self.velocity_scale)]
        features=np.zeros((self.nearest,8));relative=threats-state
        order=np.argsort(np.linalg.norm(relative[:,:3],axis=1)-radii,kind='stable')[:self.nearest]
        for slot,index in enumerate(order):
            features[slot,:6]=relative[index]/scales
            features[slot,6]=radii[index]/self.position_scale;features[slot,7]=1.
        return np.r_[state/scales,(reference-state)/scales,features.reshape(-1)]


@dataclass(frozen=True)
class NavigationReward:
    """Explicit reward components, including proper discounted potential shaping.

    Success/collision/termination come from the environment's declared gates.
    A reward or intervention penalty supplies no safety guarantee.
    """
    gamma: float = .99
    progress_weight: float = 150.
    effort_weight: float = .01
    intervention_weight: float = .5
    success_bonus: float = 2000.
    collision_penalty: float = 2000.

    def __post_init__(self):
        if not 0<self.gamma<=1 or any(not np.isfinite(v) or v<0 for v in (
                self.progress_weight,self.effort_weight,self.intervention_weight,self.success_bonus,self.collision_penalty)):
            raise ValueError('Reward discount and weights are invalid')

    def evaluate(self,previous_distance,distance,proposal,applied,*,success=False,collision=False,terminated=False):
        proposal,applied=np.asarray(proposal,dtype=float),np.asarray(applied,dtype=float)
        if (not np.all(np.isfinite((previous_distance,distance))) or min(previous_distance,distance)<0
                or proposal.shape!=(3,) or applied.shape!=(3,) or not np.all(np.isfinite(np.r_[proposal,applied]))
                or any(type(v) is not bool for v in (success,collision,terminated))):
            raise ValueError('Reward inputs must be finite observations and explicit outcome events')
        potential_next=0. if terminated else -distance
        parts=dict(progress=self.progress_weight*(self.gamma*potential_next+previous_distance),
            effort=-self.effort_weight*float(applied @ applied),
            intervention=-self.intervention_weight*float((applied-proposal) @ (applied-proposal)),
            success=self.success_bonus*success,collision=-self.collision_penalty*collision)
        return float(sum(parts.values())),parts
