"""Serializable fixed proof sources for detached planning contexts."""
from dataclasses import dataclass


@dataclass(frozen=True)
class FixedProofProvider:
    """Return an existing complete proof; consumers still verify its scope.

    This carries the same fixed certificate as a closed-over local callable,
    while allowing a source-bound preparation process to transport the policy.
    It does not make mutable certificate payloads valid or extend their domain.
    """
    certificate: object

    def __call__(self, start, end):
        return self.certificate
