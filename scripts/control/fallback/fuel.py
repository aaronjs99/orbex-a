"""Exact available resources for selecting a complete finite continuation."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from orbexa.core.dynamics.reference_frame import exact, fingerprint


@dataclass(frozen=True)
class FleetFuelBudget:
    """A selection budget, never permission to spend the installed reserve.

    The admission runtime independently checks the held debit and full successor
    again. A prospective budget is the current ledger minus the proved debit of
    that same proposed hold, not a future fuel measurement.
    """
    exact_q: object
    available_kg: object
    fuel_domain_id: str

    def __post_init__(self):
        q=exact(self.exact_q)
        rows=tuple(self.available_kg.items()) if hasattr(self.available_kg,'items') else tuple(self.available_kg)
        if (not rows or any(not isinstance(key,str) or not key.strip() for key,_ in rows)
                or len({key for key,_ in rows})!=len(rows)):
            raise ValueError('Fuel budget must name each distinct fleet member')
        rows=tuple(sorted((key,exact(value)) for key,value in rows))
        if q<0 or any(value<0 for _,value in rows):
            raise ValueError('Fuel budget clock and available tanks must be nonnegative')
        if not isinstance(self.fuel_domain_id,str) or not self.fuel_domain_id.strip():
            raise ValueError('Fuel budget requires its physical resource domain')
        object.__setattr__(self,'exact_q',q)
        object.__setattr__(self,'available_kg',rows)

    @property
    def available(self):return MappingProxyType(dict(self.available_kg))

    def check_context(self,ids,q,fuel_domain_id):
        if (set(ids)!={key for key,_ in self.available_kg} or exact(q)!=self.exact_q
                or fuel_domain_id!=self.fuel_domain_id):
            raise ValueError('Fuel budget has another fleet, exact clock or physical resource domain')

    def shortfalls(self,required):
        rows=dict(required)
        if set(rows)!=set(self.available):
            raise ValueError('Complete continuation fuel must name the budget fleet')
        values={key:exact(value) for key,value in rows.items()}
        if any(value<0 for value in values.values()):
            raise ValueError('Complete continuation fuel bounds must be nonnegative')
        return {key:values[key]-available for key,available in self.available_kg if values[key]>available}

    def require(self,required):
        shortfalls=self.shortfalls(required)
        if shortfalls:
            raise ValueError('Complete finite continuation exceeds current fuel budget: '+
                '; '.join(key+' shortfall_kg='+str(value) for key,value in shortfalls.items()))

    def as_dict(self):
        return dict(method='exact-finite-fleet-fuel-budget-v1',exact_q=str(self.exact_q),
            available_kg={key:str(value) for key,value in self.available_kg},fuel_domain_id=self.fuel_domain_id)

    @property
    def fingerprint(self):return fingerprint(self.as_dict())
