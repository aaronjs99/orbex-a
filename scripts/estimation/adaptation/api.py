# /***********************************************************
# *                                                         *
# * Copyright (c) 2026                                      *
# *                                                         *
# * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
# * University of California, Los Angeles                   *
# *                                                         *
# * Authors: Aaron John Sabu, Brett T. Lopez                *
# * Contact: {aaronjs, btlopez}@ucla.edu                    *
# *                                                         *
# ***********************************************************/

"""Stable lazy adaptation API across focused estimator modules."""

from importlib import import_module

_EXPORTS = {
    "SMIDAdaptor": 'orbexa.estimation.adaptation.smid',
    "SMIDRecord": 'orbexa.estimation.adaptation.records',
    "gen_adaptor_data": 'orbexa.estimation.adaptation.batch',
    "run_adaptor_op": 'orbexa.estimation.adaptation.batch',
    "run_adaptation": 'orbexa.estimation.adaptation.batch',
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
