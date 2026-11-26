"""journal.py: Durable publication and acknowledged completed-hold retention.

Publication must return before the coordinator acknowledges a completed hold.
Recorder callbacks happen afterward, preserving evidence if a callback fails.

Copyright: Aaron John Sabu
"""

from __future__ import annotations


class ExecutionJournal:
    """Build archive events for completed holds and retain completed records.

    Each event goes to the optional sink before the outcome is updated, so a
    failing sink leaves the outcome without the event.
    """

    def __init__(self, sink=None):
        """Store the optional event sink and start with no events or completed records.

        Inputs:
            sink: callable taking one event dictionary, or None to keep full events in
                memory on the outcome.
        """
        self.sink = sink
        self.last_event = None
        self.completed = []

    def publish(
        self,
        kind,
        record,
        outcome,
        *,
        target=None,
        flows=None,
        nominal=None,
        navigation_flows=None,
        successor_witnesses=None,
    ):
        """Build one archive event, pass it to the sink, and note it on the outcome.

        Inputs:
            kind: value stored as ``archive_event_kind``.
            record: hold record; its ``staged_identity`` is copied into the event.
            outcome: object whose ``archive_events`` list is appended to.
            target, flows, nominal, navigation_flows, successor_witnesses: stored in
                the event; the last four default to empty dictionaries.
        Outputs:
            Sets ``last_event``. With a sink, calls it first and appends only the kind
            and staged identity to ``outcome.archive_events``; without one, appends the
            full event. Sink exceptions propagate before any append. Returns the event.
        """
        event = {
            "archive_event_kind": kind,
            "staged_identity": record.get("staged_identity"),
            "record": record,
            "target": target,
            "flows": {} if flows is None else flows,
            "nominal": {} if nominal is None else nominal,
            "navigation_flows": {} if navigation_flows is None else navigation_flows,
            "successor_witnesses": {}
            if successor_witnesses is None
            else successor_witnesses,
        }
        self.last_event = event
        # A supplied sink must finish its durable write before returning. With
        # no sink, the bounded current outcome owns the full in-memory events.
        if self.sink is not None:
            self.sink(event)
            outcome.archive_events.append(
                {
                    "archive_event_kind": kind,
                    "staged_identity": record.get("staged_identity"),
                }
            )
        else:
            outcome.archive_events.append(event)
        return event

    def record_completed(self, outcome, record, *, retain):
        """Append a completed hold record to the outcome journal.

        Inputs:
            outcome: object whose ``journal`` list is appended to.
            record: completed hold record.
            retain: when true, also keep the record in ``self.completed``.
        """
        outcome.journal.append(record)
        if retain:
            self.completed.append(record)
