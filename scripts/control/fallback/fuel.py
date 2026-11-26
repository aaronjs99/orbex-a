"""fuel.py: Exact available resources for selecting a complete finite continuation.

Copyright: Aaron John Sabu
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from orbexa.models.dynamics.reference_frame import exact, fingerprint


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
        """Validate the budget and store it in exact, sorted form.

        Outputs: ``exact_q`` becomes an exact value and ``available_kg`` a tuple of
        ``(member, exact kg)`` pairs sorted by member. Raises ValueError when no member
        is named, a member name is blank, not a string or repeated, ``exact_q`` or any
        tank is negative, or ``fuel_domain_id`` is blank."""
        q = exact(self.exact_q)
        rows = (
            tuple(self.available_kg.items())
            if hasattr(self.available_kg, "items")
            else tuple(self.available_kg)
        )
        if (
            not rows
            or any(not isinstance(key, str) or not key.strip() for key, _ in rows)
            or len({key for key, _ in rows}) != len(rows)
        ):
            raise ValueError("Fuel budget must name each distinct fleet member")
        rows = tuple(sorted((key, exact(value)) for key, value in rows))
        if q < 0 or any(value < 0 for _, value in rows):
            raise ValueError(
                "Fuel budget clock and available tanks must be nonnegative"
            )
        if not isinstance(self.fuel_domain_id, str) or not self.fuel_domain_id.strip():
            raise ValueError("Fuel budget requires its physical resource domain")
        object.__setattr__(self, "exact_q", q)
        object.__setattr__(self, "available_kg", rows)

    @property
    def available(self):
        """Return a read-only mapping from member name to available fuel in kg."""
        return MappingProxyType(dict(self.available_kg))

    def check_context(self, ids, q, fuel_domain_id):
        """Check that a fleet, exact clock and fuel domain match this budget.

        Inputs: ``ids`` are the member names, ``q`` the exact clock value and
        ``fuel_domain_id`` the resource domain.
        Outputs: None. Raises ValueError when the member set, the clock or the domain
        differs from the budget."""
        if (
            set(ids) != {key for key, _ in self.available_kg}
            or exact(q) != self.exact_q
            or fuel_domain_id != self.fuel_domain_id
        ):
            raise ValueError(
                "Fuel budget has another fleet, exact clock or physical resource domain"
            )

    def shortfalls(self, required):
        """Return the members whose required fuel exceeds their available fuel.

        Inputs: ``required`` maps each member to its required fuel in kg.
        Outputs: dict from member to required minus available kg, only for members
        with a positive shortfall. Raises ValueError when the member set differs from
        the budget fleet or a requirement is negative."""
        rows = dict(required)
        if set(rows) != set(self.available):
            raise ValueError("Complete continuation fuel must name the budget fleet")
        values = {key: exact(value) for key, value in rows.items()}
        if any(value < 0 for value in values.values()):
            raise ValueError("Complete continuation fuel bounds must be nonnegative")
        return {
            key: values[key] - available
            for key, available in self.available_kg
            if values[key] > available
        }

    def require(self, required):
        """Raise ValueError when any member's required fuel exceeds its budget.

        Inputs: ``required`` maps each member to its required fuel in kg. The message
        lists each member's shortfall in kg. Also raises as ``shortfalls`` does."""
        shortfalls = self.shortfalls(required)
        if shortfalls:
            raise ValueError(
                "Complete finite continuation exceeds current fuel budget: "
                + "; ".join(
                    key + " shortfall_kg=" + str(value)
                    for key, value in shortfalls.items()
                )
            )

    def as_dict(self):
        """Return the JSON-ready record of the budget with its method label."""
        return {
            "method": "exact-finite-fleet-fuel-budget-v1",
            "exact_q": str(self.exact_q),
            "available_kg": {key: str(value) for key, value in self.available_kg},
            "fuel_domain_id": self.fuel_domain_id,
        }

    @property
    def fingerprint(self):
        """Return the SHA-256 fingerprint of ``as_dict()``."""
        return fingerprint(self.as_dict())
