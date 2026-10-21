"""Explicit dispatch between current nonlinear and historical analytical jobs."""
from __future__ import annotations

from orbexa.simulation.service.jobs.historical import _execute_historical_analytical_job
from orbexa.simulation.service.jobs.nonlinear import _execute_nonlinear_job


def _execute_job(settings, parameters, directory, messages, run_event):
    if settings.resume_path is not None:
        if not settings.historical_analytical_resume:
            raise ValueError('Analytical resume must use the explicit historical route')
        return _execute_historical_analytical_job(settings, parameters, directory, messages, run_event)
    return _execute_nonlinear_job(settings, parameters, directory, messages, run_event)
