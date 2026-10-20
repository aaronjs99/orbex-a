"""Explicit contact, detumbling and disposal settings for any target design.

Vehicle mass, geometry and propulsion remain in ``SpacecraftMissionDesign``.
This profile configures stage timing, contact mechanics and disposal goals.
The existing stage constructors validate their own physical relationships.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields

from .contact_contracts import FiniteContactOptions
from .detumbling import DetumblingOptions
from .deorbit import DeorbitOptions


_CONTACT_HARDWARE = frozenset(('translation_force_n', 'attitude_nozzle_force_n',
    'attitude_moment_arm_m', 'specific_impulse_s'))


def _options(value, kind, *, forbidden=frozenset()):
    if isinstance(value, kind):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(kind.__name__+' requires named settings')
    allowed = {item.name for item in fields(kind)}
    if set(value)-allowed:
        raise ValueError('Unknown '+kind.__name__+' settings: '+', '.join(sorted(set(value)-allowed)))
    if set(value)&forbidden:
        raise ValueError('Declare propulsion hardware in vehicle_design, not completion.contact')
    if any(isinstance(item, bool) for item in value.values()):
        raise ValueError(kind.__name__+' physical settings require numbers, not booleans')
    return kind(**value)


@dataclass(frozen=True)
class MissionCompletionProfile:
    contact: FiniteContactOptions = field(default_factory=FiniteContactOptions)
    detumbling: DetumblingOptions = field(default_factory=DetumblingOptions)
    deorbit: DeorbitOptions = field(default_factory=DeorbitOptions)

    def __post_init__(self):
        object.__setattr__(self, 'contact', _options(self.contact, FiniteContactOptions,
            forbidden=_CONTACT_HARDWARE))
        object.__setattr__(self, 'detumbling', _options(self.detumbling, DetumblingOptions))
        object.__setattr__(self, 'deorbit', _options(self.deorbit, DeorbitOptions))

    @classmethod
    def from_declaration(cls, value=None):
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping) or set(value)-{'contact', 'detumbling', 'deorbit'}:
            raise ValueError('Completion profile accepts contact, detumbling and deorbit settings')
        return cls(**value)

    def as_dict(self):
        return dict(contact={key: value for key, value in asdict(self.contact).items()
                             if key not in _CONTACT_HARDWARE},
            detumbling=asdict(self.detumbling), deorbit=asdict(self.deorbit))
