"""Reusable qualified-approach → attachment → detumbling → deorbit composition."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Callable, Optional
import copy

from orbexa.simulation.contact.capture import CaptureModel, CaptureResult
from orbexa.simulation.attached.detumbling import DetumblingResult, DetumblingSimulator
from orbexa.simulation.attached.disposal.simulator import DeorbitResult, DeorbitSimulator
from orbexa.simulation.attached.propulsion import ThrusterSystem
from orbexa.utils.io_utils import load_data, save_data

LIFECYCLE_SCHEMA = "orbexa.physical-mission-lifecycle.v1"


@dataclass(frozen=True)
class MissionLifecycleResult:
    success: bool
    message: str
    approach: dict
    capture: CaptureResult
    detumbling: Optional[DetumblingResult] = None
    deorbit: Optional[DeorbitResult] = None
    propulsion: Optional[ThrusterSystem] = None
    configuration: dict = field(default_factory=dict)
    failure: dict = field(default_factory=dict)

    def to_dict(self):
        history = []
        if self.capture.state is not None:
            history.append(dict(self.capture.assembly.state_record(self.capture.state), phase="attachment"))
        if self.detumbling is not None:
            history.extend(dict(record, phase="detumbling") for record in self.detumbling.history[1:])
        if self.deorbit is not None:
            history.extend(copy.deepcopy(self.deorbit.history[1:]))
        return dict(success=self.success, message=self.message, approach=copy.deepcopy(self.approach),
                    configuration=copy.deepcopy(self.configuration),
                    terminal_handoff=copy.deepcopy(self.configuration.get("terminal_handoff")),
                    failure=copy.deepcopy(self.failure),
                    attachment=self.capture.to_dict(), detumbling=None if self.detumbling is None else self.detumbling.to_dict(),
                    deorbit=None if self.deorbit is None else self.deorbit.to_dict(),
                    propulsion=None if self.propulsion is None else self.propulsion.to_dict(), assembly_history=history,
                    units=dict(time="s", position="m ECI", velocity="m/s ECI", angular_velocity="rad/s body",
                               force="N body", torque="N m body", inertia="kg m^2 body", mass="kg", quaternion="xyzw body-to-ECI"),
                    scope="modeled physical capture and controlled orbital delivery to a descending entry interface; no flight qualification or atmospheric demise")


class MissionLifecycle:
    """Inject stage policies independently; stages retain explicit failure results."""
    def __init__(self, *, capture: Optional[CaptureModel] = None,
                 detumbling: Optional[DetumblingSimulator] = None,
                 deorbit: Optional[DeorbitSimulator] = None,
                 propulsion_factory: Optional[Callable] = None):
        self.capture = capture or CaptureModel()
        self.detumbling = detumbling or DetumblingSimulator()
        self.deorbit = deorbit or DeorbitSimulator()
        self.propulsion_factory = propulsion_factory or ThrusterSystem.for_assembly

    def configuration(self):
        def policy(value):
            return dict(implementation=f"{type(value).__module__}.{type(value).__qualname__}",
                        parameters=asdict(value) if is_dataclass(value) else "injected policy; parameters supplied by caller")
        configuration = dict(capture=dict(options=asdict(self.capture.options), target_mass_kg=self.capture.target.mass_kg,
                                 finite_contact_options=asdict(self.capture.contact_options),
                                 target_inertia_kg_m2=self.capture.target.inertia_kg_m2.tolist(),
                                 default_chaser=asdict(self.capture.default_chaser),
                                 chasers={key: asdict(value) for key, value in self.capture.chasers.items()}),
                    detumbling=dict(options=asdict(self.detumbling.options), policy=policy(self.detumbling.policy)),
                    deorbit=dict(options=asdict(self.deorbit.options), pointing_policy=policy(self.deorbit.pointing_policy)))
        if self.deorbit.burn_axis_planner is not None:
            configuration['deorbit']['burn_axis_planner'] = policy(self.deorbit.burn_axis_planner)
        if self.deorbit.burn_throttle_planner is not None:
            configuration['deorbit']['burn_throttle_planner'] = policy(self.deorbit.burn_throttle_planner)
        return configuration

    def run(self, approach, *, capture_truth=None, chaser_attitudes=None,
            terminal_handoff=None, contact_feedback=None, contact_environment=None,
            contact_initial_enclosure=None) -> MissionLifecycleResult:
        configuration = self.configuration()
        expected = (ValueError, RuntimeError, ArithmeticError, ImportError)
        def failure(stage, error):
            return dict(stage=stage, exception_type=type(error).__name__, message=str(error),
                        progress_scope="Completed preceding stages are retained; the throwing stage supplied no result or committed prefix")
        try:
            if terminal_handoff is not None:
                from orbexa.simulation.approach.terminal.handoff import PhysicalTerminalHandoff
                if not isinstance(terminal_handoff, PhysicalTerminalHandoff):
                    raise ValueError("Physical capture requires a verified PhysicalTerminalHandoff")
                terminal_handoff.verify_approach(approach)
                configuration["terminal_handoff"] = terminal_handoff.as_dict()
            arguments = dict(truth=capture_truth, chaser_attitudes=chaser_attitudes)
            arguments['contact_feedback'] = contact_feedback
            arguments['contact_environment'] = contact_environment
            arguments['contact_initial_enclosure'] = contact_initial_enclosure
            if terminal_handoff is not None:
                arguments["terminal_handoff"] = terminal_handoff
            attachment = self.capture.capture(approach, **arguments)
        except expected as error:
            message = f"attachment_exception: {type(error).__name__}: {error}"
            attachment = CaptureResult(False, message, {}, failure("attachment", error),
                                       ("No attachment state was produced by the failed capture stage",))
            return MissionLifecycleResult(False, message, approach.to_dict(), attachment,
                                          configuration=configuration, failure=failure("attachment", error))
        if not attachment.success:
            return MissionLifecycleResult(False, attachment.message, approach.to_dict(), attachment, configuration=configuration)
        if not self.capture.options.idealized_instantaneous and not attachment.audit.get('finite_contact', {}).get('success'):
            return MissionLifecycleResult(False, 'finite_contact_report_missing', approach.to_dict(), attachment, configuration=configuration)
        try:
            propulsion = self.propulsion_factory(attachment.assembly)
        except expected as error:
            return MissionLifecycleResult(False, f"propulsion_setup_rejected: {type(error).__name__}: {error}",
                approach.to_dict(), attachment, configuration=configuration, failure=failure("propulsion_setup", error))
        try:
            detumbling = self.detumbling.run(attachment.assembly, attachment.state, propulsion)
            loads = attachment.assembly.attachment_history_audit(propulsion, detumbling.controls, detumbling.history)
            detumbling.audit['attachment_loads'] = loads
            attachment.audit.setdefault('downstream_attachment_loads', {})['detumbling'] = loads
            if not loads['passed']:
                return MissionLifecycleResult(False, 'detumbling_attachment_capacity_exceeded', approach.to_dict(),
                    attachment, detumbling, propulsion=propulsion, configuration=configuration,
                    failure=dict(stage='detumbling', message='Actual allocated nozzle loads exceed a declared pad capacity'))
        except expected as error:
            return MissionLifecycleResult(False, f"detumbling_exception: {type(error).__name__}: {error}",
                approach.to_dict(), attachment, propulsion=propulsion, configuration=configuration,
                failure=failure("detumbling", error))
        if not detumbling.success:
            return MissionLifecycleResult(False, detumbling.message, approach.to_dict(), attachment, detumbling,
                                          propulsion=propulsion, configuration=configuration)
        try:
            deorbit = self.deorbit.run(attachment.assembly, detumbling.final_state, propulsion)
            loads = attachment.assembly.attachment_history_audit(propulsion, deorbit.controls, deorbit.history)
            deorbit.audit['attachment_loads'] = loads
            attachment.audit.setdefault('downstream_attachment_loads', {})['deorbit'] = loads
            if not loads['passed']:
                return MissionLifecycleResult(False, 'deorbit_attachment_capacity_exceeded', approach.to_dict(),
                    attachment, detumbling, deorbit, propulsion, configuration,
                    failure=dict(stage='deorbit', message='Actual allocated nozzle loads exceed a declared pad capacity'))
        except expected as error:
            return MissionLifecycleResult(False, f"deorbit_exception: {type(error).__name__}: {error}",
                approach.to_dict(), attachment, detumbling, propulsion=propulsion,
                configuration=configuration, failure=failure("deorbit", error))
        return MissionLifecycleResult(deorbit.success, deorbit.message, approach.to_dict(), attachment,
                                      detumbling, deorbit, propulsion, configuration)


class MissionLifecycleStore:
    filename = "lifecycle_data.json"

    @staticmethod
    def save(result: MissionLifecycleResult, output_dir: Path) -> Path:
        path = Path(output_dir) / MissionLifecycleStore.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if not save_data(path, dict(schema=LIFECYCLE_SCHEMA, result=result.to_dict())):
            raise OSError(f"Could not save mission lifecycle to {path}")
        return path

    @staticmethod
    def read(path: Path) -> dict:
        payload = load_data(Path(path))
        if payload.get("schema") != LIFECYCLE_SCHEMA:
            raise ValueError("Unsupported mission lifecycle record schema")
        return payload["result"]
