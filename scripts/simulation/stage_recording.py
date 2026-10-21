"""Optional append-only storage of complete accepted stage evidence."""
from orbexa.core.reference_frame import fingerprint


class StageRecordRetainer:
    """Keep full records by default, or immutable references to a supplied sink.

    A sink receives every complete accepted record, including any endpoint
    diagnostics. Storage failures propagate. An archived result
    requires the separately closed full archive to resolve and verify every
    reference; references alone are not physical evidence.
    """
    def __init__(self, sink=None):
        if sink is not None and not callable(sink):
            raise TypeError('Stage record sink must be callable or None')
        self.sink = sink
        self.count = 0
        self.chain = None
        self.channels = {}

    def keep(self, channel, record, summary=None):
        if self.sink is None:
            return record
        if channel not in ('controls', 'coast_history'):
            raise ValueError('Unknown stage record channel')
        identity = fingerprint(record)
        payload = dict(channel=channel, sequence=self.count,
            record_identity=identity, previous_chain_identity=self.chain)
        chain = fingerprint(payload)
        reference = dict(schema='orbexa.stage-record-reference.v1', **payload,
            chain_identity=chain)
        reference_identity = fingerprint(reference)
        envelope = dict(schema='orbexa.stage-record-envelope.v1',
            reference=reference, record=record)
        self.sink(envelope)
        if (fingerprint(record) != identity or fingerprint(envelope['record']) != identity
                or fingerprint(envelope['reference']) != reference_identity):
            raise ValueError('Stage record sink mutated an accepted record or reference')
        self.count += 1
        self.chain = chain
        self.channels[channel] = self.channels.get(channel, 0)+1
        retained = {} if summary is None else dict(summary)
        retained['whole_record_reference'] = reference
        return retained

    def as_dict(self):
        return dict(mode='full_in_memory' if self.sink is None else 'external_full_record_archive',
            archived_records=self.count, final_chain_identity=self.chain,
            channels=dict(self.channels),
            scope='Archived mode requires every complete, ordered source-bound record and a closed archive; compact references alone establish no physical claim')
