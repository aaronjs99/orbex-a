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

"""Serializable point-fit and candidate-set update records."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class SMIDRecord:
    """Audit record for one Algorithm 1 set-membership update."""

    start_anom: float
    window_size: int
    accepted: bool
    reason: str
    fss_before: Dict[str, List[float]]
    fss_candidate: Dict[str, List[float]]
    fss_after: Dict[str, List[float]]
    estimates_before: Dict[str, float]
    estimates_after: Dict[str, float]
    prediction_error: float
    # Terminal error of the candidate-set witness, not the returned point belief.
    verification_error: Optional[float]
    solve_status: Dict[str, str] = field(default_factory=dict)
    point_prediction_error: Optional[float] = None
    point_fit_error: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_anom": float(self.start_anom),
            "window_size": int(self.window_size),
            "accepted": bool(self.accepted),
            "reason": self.reason,
            "fss_before": self.fss_before,
            "fss_candidate": self.fss_candidate,
            "fss_after": self.fss_after,
            "estimates_before": self.estimates_before,
            "estimates_after": self.estimates_after,
            "prediction_error": float(self.prediction_error),
            "verification_error": (
                None
                if self.verification_error is None
                else float(self.verification_error)
            ),
            "point_prediction_error": (
                None
                if self.point_prediction_error is None
                else float(self.point_prediction_error)
            ),
            "point_fit_error": (
                None if self.point_fit_error is None else float(self.point_fit_error)
            ),
            "solve_status": dict(self.solve_status),
        }
