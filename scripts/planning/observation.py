# /***********************************************************
# * Copyright (c) 2026                                      *
# * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
# * University of California, Los Angeles                   *
# * Authors: Aaron John Sabu                                *
# * Contact: aaronjs@ucla.edu                               *
# ***********************************************************/

"""Explicit observation-shell optimization with a caller-selected GEKKO boundary.

The objective is the existing geometric separation/angular-coverage surrogate,
not a sensor visibility or occlusion model. This module launches no mission and
requires no Hyperopt installation. The historical result dictionary is retained.
"""

import logging
import math
from collections.abc import Mapping

import numpy as np

logger = logging.getLogger(__name__)
STATUS_OK = "ok"
STATUS_FAIL = "fail"


def calc_shape(radius, target_shape, sphere_shape):
    kappa = math.exp(0.01 * radius) - 1
    return kappa * target_shape + (1 - kappa) * sphere_shape


def calc_single_observation(r1, r2, R, shape):
    difference = np.subtract(r1, r2)
    return sum(value**2 for value in difference)


def calc_mutual_observation(theta_1, theta_2, phi_1, phi_2, *args, **kwargs):
    backend = kwargs.get("m")
    if backend is None:
        return (1 + np.cos(theta_1 - theta_2)) * (1 - np.sin(np.abs(phi_1 - phi_2) / 4)) / 2
    return (1 + backend.cos(theta_1 - theta_2)) * (1 - backend.sin(backend.abs2(phi_1 - phi_2) / 4)) / 2


def calc_total_local_observation(num_chasers, r, R, shape):
    observation = 0.0
    dimensions = len(r) // num_chasers
    for first in range(num_chasers):
        for second in range(first + 1, num_chasers):
            observation += calc_single_observation(r[dimensions * first:dimensions * (first + 1)],
                                                  r[dimensions * second:dimensions * (second + 1)], R, shape) / sum(R)**2
    return observation


def calc_total_cross_observation(num_chasers, r1, r2, R1, R2, shape):
    dimensions = len(r1) // num_chasers
    return -sum(calc_single_observation(r1[dimensions * agent:dimensions * (agent + 1)],
                                        r2[dimensions * agent:dimensions * (agent + 1)], R1, shape)
                for agent in range(num_chasers))


def calc_total_mutual_observation(num_chasers, theta, phi, R, shape, *args, **kwargs):
    return -sum(calc_mutual_observation(theta[first], theta[second], phi[first], phi[second], **kwargs) / sum(R)**2
                for first in range(num_chasers) for second in range(first + 1, num_chasers))


def _positive_integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")


def observe_optimizer(hyperopt_params=None, numDims=3, Q=None, r_0=None, numLevels=3,
                      num_chasers=1, startRadii=None, finalRadii=None, target_shape=None,
                      *args, backend="gekko", remote=False, disp=False, solver_type=3,
                      max_iter=500, **kwargs):
    """Optimize nested ellipsoid shells from explicit geometry and objective weights.

    ``Q`` contains six nonnegative weights for local separation, final separation,
    motion between shells, mutual angular coverage, radial progress and isotropy.
    Initial positions must lie on the starting shell. ``hyperopt_params`` is an
    optional legacy mapping carrying ``Q`` and ``Rho_0`` initial radii; Hyperopt
    itself is not used. Local GEKKO is the default. Success returns the historical
    ``x=(radii, flattened_positions, weights)`` tuple; failure has infinite loss,
    ``status='fail'`` and no manufactured position trajectory.
    """
    if args or kwargs:
        raise TypeError("Unsupported observation optimizer arguments")
    _positive_integer(numDims, "numDims", 2)
    if numDims not in (2, 3):
        raise ValueError("numDims must be 2 or 3")
    _positive_integer(numLevels, "numLevels", 2)
    _positive_integer(num_chasers, "num_chasers")
    _positive_integer(max_iter, "max_iter")
    if backend != "gekko":
        raise ValueError("Observation optimization supports backend='gekko'")
    if not isinstance(remote, bool) or not isinstance(disp, bool):
        raise ValueError("remote and disp must be booleans")
    if isinstance(solver_type, bool) or solver_type not in (1, 2, 3):
        raise ValueError("GEKKO solver_type must be 1, 2 or 3")
    if hyperopt_params is None:
        hyperopt_params = {}
    if not isinstance(hyperopt_params, Mapping):
        raise ValueError("hyperopt_params must be a mapping")

    start = np.full(numDims, 100.0) if startRadii is None else np.asarray(startRadii, dtype=float)
    final = np.ones(numDims) if finalRadii is None else np.asarray(finalRadii, dtype=float)
    if (start.shape != (numDims,) or final.shape != (numDims,)
            or not np.all(np.isfinite(start)) or not np.all(np.isfinite(final))
            or np.any(final <= 0) or np.any(start <= final)):
        raise ValueError("Radii must be finite positive vectors with startRadii > finalRadii")
    weights = np.asarray(hyperopt_params.get("Q", np.ones(6) if Q is None else Q), dtype=float)
    if weights.shape != (6,) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Q must contain six finite nonnegative weights")
    seed_radii = np.asarray(hyperopt_params.get("Rho_0", np.linspace(start, final, numLevels)), dtype=float)
    if (seed_radii.shape != (numLevels, numDims) or not np.all(np.isfinite(seed_radii))
            or np.any(seed_radii <= 0) or np.any(seed_radii < final) or np.any(seed_radii > start)):
        raise ValueError("Rho_0 must contain numLevels finite radius vectors within the endpoint shells")
    seed_radii = seed_radii.copy()
    seed_radii[0], seed_radii[-1] = start, final
    if r_0 is None:
        positions = np.zeros((num_chasers, numDims))
        positions[:, 0] = start[0]
    else:
        positions = np.asarray(r_0, dtype=float)
        if positions.shape not in ((num_chasers * numDims,), (num_chasers, numDims)):
            raise ValueError("r_0 must contain one numDims position per chaser")
        positions = positions.reshape(num_chasers, numDims)
    if (not np.all(np.isfinite(positions))
            or not np.allclose(np.sum((positions / start)**2, axis=1), 1.0, rtol=0, atol=1e-8)):
        raise ValueError("Initial positions must be finite and lie on the starting ellipsoid")
    normalized = positions / start
    theta_initial = np.arctan2(normalized[:, 1], normalized[:, 0])
    phi_initial = (np.arctan2(np.linalg.norm(normalized[:, :2], axis=1), normalized[:, 2])
                   if numDims == 3 else np.zeros(num_chasers))

    from gekko import GEKKO

    solver = GEKKO(remote=remote)
    try:
        radii = [[(solver.Param(value=float(start[dimension])) if level == 0 else
                   solver.Param(value=float(final[dimension])) if level == numLevels - 1 else
                   solver.Var(value=float(seed_radii[level, dimension]), lb=float(final[dimension]), ub=float(start[dimension])))
                  for dimension in range(numDims)] for level in range(numLevels)]
        theta = [[solver.Var(value=float(theta_initial[agent] if level == 0 else
                                         theta_initial[agent] + 2 * np.pi * agent / num_chasers), lb=-3 * np.pi, ub=3 * np.pi)
                  for agent in range(num_chasers)] for level in range(numLevels)]
        phi = ([[solver.Var(value=float(phi_initial[agent]), lb=0, ub=np.pi)
                 for agent in range(num_chasers)] for level in range(numLevels)]
               if numDims == 3 else [[0.0] * num_chasers for _ in range(numLevels)])
        for agent in range(num_chasers):
            solver.Equation(theta[0][agent] == float(theta_initial[agent]))
            if numDims == 3:
                solver.Equation(phi[0][agent] == float(phi_initial[agent]))
        shell_positions = []
        for level in range(numLevels):
            shell = []
            for agent in range(num_chasers):
                angle = theta[level][agent]
                if numDims == 2:
                    shell.extend((radii[level][0] * solver.cos(angle), radii[level][1] * solver.sin(angle)))
                else:
                    polar = phi[level][agent]
                    shell.extend((radii[level][0] * solver.sin(polar) * solver.cos(angle),
                                  radii[level][1] * solver.sin(polar) * solver.sin(angle),
                                  radii[level][2] * solver.cos(polar)))
            shell_positions.append(shell)
        gap = (start - final) / (numLevels - 1) * 1e-6
        for level in range(numLevels - 1):
            for dimension in range(numDims):
                solver.Equation(radii[level][dimension] - radii[level + 1][dimension] >= float(gap[dimension]))
        local = [calc_total_local_observation(num_chasers, shell_positions[level], radii[level], target_shape)
                 for level in range(numLevels)]
        cross = [calc_total_cross_observation(num_chasers, shell_positions[level], shell_positions[level + 1],
                                             radii[level], radii[level + 1], target_shape) for level in range(numLevels - 1)]
        mutual = [calc_total_mutual_observation(num_chasers, theta[level], phi[level], radii[level], target_shape, m=solver)
                  for level in range(numLevels)]
        radial = [sum((radii[level][dimension] - radii[level + 1][dimension])**2 for dimension in range(numDims))
                  for level in range(numLevels - 1)]
        isotropy = [sum((radii[level][dimension] - radii[level][(dimension + 1) % numDims])**2 for dimension in range(numDims))
                    for level in range(numLevels - 1)]
        objective = (weights[0] * sum(local) + weights[1] * local[-1] + weights[2] * sum(cross)
                     + weights[3] * sum(mutual) + weights[4] * sum(radial) + weights[5] * sum(isotropy))
        solver.Maximize(objective)
        solver.options.SOLVER = int(solver_type)
        solver.options.IMODE = 3
        solver.options.MAX_ITER = int(max_iter)
        try:
            solver.solve(disp=disp)
        except Exception as error:
            logger.warning("Observation optimization failed: %s", error)
            return {"loss": float("inf"), "status": STATUS_FAIL,
                    "x": (seed_radii.tolist(), None, weights.tolist()), "error": str(error)}
        solved_radii = [[float(value.value[0]) for value in shell] for shell in radii]
        solved_positions = []
        for level in range(numLevels):
            for agent in range(num_chasers):
                angle = float(theta[level][agent].value[0])
                if numDims == 2:
                    solved_positions.extend((solved_radii[level][0] * math.cos(angle), solved_radii[level][1] * math.sin(angle)))
                else:
                    polar = float(phi[level][agent].value[0])
                    solved_positions.extend((solved_radii[level][0] * math.sin(polar) * math.cos(angle),
                                             solved_radii[level][1] * math.sin(polar) * math.sin(angle),
                                             solved_radii[level][2] * math.cos(polar)))
        loss = float(solver.options.OBJFCNVAL)
        if not np.isfinite(loss) or not np.all(np.isfinite(solved_positions)):
            raise RuntimeError("Observation solver returned nonfinite values")
        # GEKKO negates a maximization objective; its reported value already is loss.
        return {"loss": loss, "status": STATUS_OK,
                "x": (solved_radii, solved_positions, weights.tolist()), "observation": -loss}
    finally:
        solver.cleanup()
