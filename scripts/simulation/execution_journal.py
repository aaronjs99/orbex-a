"""Durable publication and acknowledged completed-hold retention.

Publication must return before the coordinator acknowledges a completed hold.
Recorder callbacks happen afterward, preserving evidence if a callback fails.
"""
from __future__ import annotations


class ExecutionJournal:
    def __init__(self,sink=None):
        self.sink=sink
        self.last_event=None
        self.completed=[]

    def publish(self,kind,record,outcome,*,target=None,flows=None,nominal=None,navigation_flows=None,
            successor_witnesses=None):
        event=dict(archive_event_kind=kind,staged_identity=record.get('staged_identity'),
            record=record,target=target,flows={} if flows is None else flows,
            nominal={} if nominal is None else nominal,
            navigation_flows={} if navigation_flows is None else navigation_flows,
            successor_witnesses={} if successor_witnesses is None else successor_witnesses)
        self.last_event=event
        # A supplied sink must finish its durable write before returning. With
        # no sink, the bounded current outcome owns the full in-memory events.
        if self.sink is not None:
            self.sink(event)
            outcome.archive_events.append(dict(archive_event_kind=kind,staged_identity=record.get('staged_identity')))
        else:outcome.archive_events.append(event)
        return event

    def record_completed(self,outcome,record,*,retain):
        outcome.journal.append(record)
        if retain:self.completed.append(record)
