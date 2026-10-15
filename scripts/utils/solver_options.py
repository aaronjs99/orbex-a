"""Backend option contracts that do not import an optimization runtime."""

from numbers import Integral


DEFAULT_GEKKO_MAX_MEMORY = 4


def gekko_memory_exponent(value=DEFAULT_GEKKO_MAX_MEMORY):
    """Validate APMonitor's base-ten sparsity-storage exponent, not megabytes."""
    if isinstance(value, bool) or not isinstance(value, Integral) or not 2 <= value <= 10:
        raise ValueError("max_memory must be an integer exponent between 2 and 10; 4 means 10**4 sparsity elements per equation")
    return int(value)
