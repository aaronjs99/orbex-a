"""Explicit process-comparable wall-clock domains for online receipts.

Windows Python 3.11's monotonic clock uses GetTickCount64 and can have a
15.625 ms resolution.  Its performance counter uses the monotonic, system-wide
QueryPerformanceCounter instead.  Selecting that domain does not establish
timely execution; all requests, deadlines and acknowledgments must use it.
"""
import time


MONOTONIC_CLOCK='monotonic-ns-v1'
PERFORMANCE_CLOCK='perf-counter-ns-v1'


def wall_time_ns(clock_id=MONOTONIC_CLOCK):
    if clock_id == MONOTONIC_CLOCK:
        return time.monotonic_ns()
    if clock_id == PERFORMANCE_CLOCK:
        return time.perf_counter_ns()
    raise ValueError('Use an explicit supported monotonic wall-clock domain')
