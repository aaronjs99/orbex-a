"""Durable streaming nonlinear witnesses and a reusable source-bound hold view."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from fractions import Fraction
import gzip
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np


SCHEMA = 'orbexa.nonlinear-proof-archive.v1'
HOLD_INDEX_SCHEMA = 'orbexa.nonlinear-held-index.v1'
MAXIMUM_ENTRY_BYTES = 256*1024*1024
ZSTD_WINDOW_BYTES = 128*1024*1024
_ZSTD_MAGIC = b'\x28\xb5\x2f\xfd'


def _zstandard():
    try:
        import zstandard
    except ImportError as error:
        raise ImportError('Zstd nonlinear archives require the certification extra (zstandard==0.25.0)') from error
    return zstandard


class _ClosedZstdReader(io.RawIOBase):
    """One completed checksummed frame, with a bounded 128 MiB decoder window.

    stream_reader alone accepts a missing final checksum byte.  The explicit
    decompressobj EOF and unused/trailing-byte checks below are intentional.
    Small compressed input chunks also bound each returned decompressed block;
    the caller separately enforces the existing finite JSON-entry budget.
    """
    def __init__(self, path):
        super().__init__()
        self._raw = Path(path).open('rb')
        self._pending, self._offset, self._finished = b'', 0, False
        try:
            zstd = _zstandard()
            self._header = self._raw.read(18)
            if not self._header.startswith(_ZSTD_MAGIC):
                raise ValueError('Closed nonlinear Zstd archive needs one standard frame')
            try:
                parameters = zstd.get_frame_parameters(self._header)
            except zstd.ZstdError as error:
                raise ValueError('Closed nonlinear Zstd archive header is truncated or invalid') from error
            if (not parameters.has_checksum or parameters.window_size > ZSTD_WINDOW_BYTES or
                    parameters.dict_id != 0):
                raise ValueError('Closed nonlinear Zstd archive checksum, window or dictionary declaration is invalid')
            # python-zstandard 0.25.0 passes this value directly to libzstd in
            # bytes; passing a KiB count incorrectly rejects the declared frame.
            self._context = zstd.ZstdDecompressor(max_window_size=ZSTD_WINDOW_BYTES)
            self._decoder = self._context.decompressobj()
            self._zstd_error = zstd.ZstdError
        except BaseException:
            self._raw.close()
            raise

    def readable(self):
        return True

    def _advance(self):
        block, self._header = self._header, b''
        if not block:
            block = self._raw.read(128)
        if not block:
            raise ValueError('Closed nonlinear Zstd archive is missing its completed frame/checksum')
        try:
            self._pending = self._decoder.decompress(block)
        except self._zstd_error as error:
            raise ValueError('Closed nonlinear Zstd archive frame or checksum is corrupt') from error
        self._offset = 0
        if self._decoder.unconsumed_tail:
            raise ValueError('Closed nonlinear Zstd archive has unconsumed compressed input')
        if self._decoder.eof:
            if self._decoder.unused_data or self._raw.read(1):
                raise ValueError('Closed nonlinear Zstd archive has trailing bytes or another frame')
            self._finished = True

    def readinto(self, target):
        if self.closed:
            raise ValueError('Read from closed nonlinear Zstd archive')
        if not len(target):
            return 0
        while self._offset == len(self._pending):
            if self._finished:
                return 0
            self._advance()
        count = min(len(target), len(self._pending)-self._offset)
        target[:count] = memoryview(self._pending)[self._offset:self._offset+count]
        self._offset += count
        return count

    def close(self):
        if not self.closed:
            self._raw.close()
            self._pending = b''
            self._decoder = self._context = None
        super().close()


def _open_archive_binary(path):
    # Magic, rather than the name, preserves reads of renamed legacy gzip files.
    with Path(path).open('rb') as source:
        magic = source.read(4)
    if magic == _ZSTD_MAGIC:
        return io.BufferedReader(_ClosedZstdReader(path), buffer_size=64*1024)
    if magic.startswith(b'\x1f\x8b'):
        return gzip.open(path, 'rb')
    raise ValueError('Closed nonlinear archive is neither a Zstd frame nor legacy gzip')


class _Encoder(json.JSONEncoder):
    def default(self, value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Fraction):
            return str(value)
        if is_dataclass(value):
            return asdict(value)
        return super().default(value)


def semantic_sha256(value, *, fast=False):
    """Canonical digest, streaming by default; optional finite-entry C encoder.

    ``fast`` materializes one bounded JSON entry, never a mission journal.
    ASCII escaping makes the character and UTF-8 byte counts identical.
    """
    result = hashlib.sha256()
    encoder = _Encoder(sort_keys=True, separators=(',', ':'), allow_nan=False)
    chunks = tuple(encoder.iterencode(value, _one_shot=True)) if fast else encoder.iterencode(value)
    if fast and sum(map(len, chunks)) > MAXIMUM_ENTRY_BYTES:
        raise ValueError('Nonlinear finite hash input exceeds the archive entry budget')
    for chunk in chunks:
        result.update(chunk.encode('utf-8'))
    return result.hexdigest()


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def hold_index(record):
    """Small exact linkage to the separately retained full admitted record."""
    from orbexa.control.fallback.refuge import fingerprint
    decision = record['decision']
    return dict(schema=HOLD_INDEX_SCHEMA, exact_start=record['exact_start'], exact_end=record['exact_end'],
        hold_width=decision['hold_width'], decision_identity=decision['identity'],
        full_record_sha256=semantic_sha256(record, fast=True), command_source=decision['source'],
        actuator_realization_id=record['actuator_realization_id'], sensor_profile_id=record['sensor_profile_id'],
        geometry_id=fingerprint(decision['geometry']), runtime_commit_status=record['runtime_commit']['status'],
        navigation_status=record['navigation_status'])


class NonlinearProofArchive:
    """Flush every exact producer event before acknowledging its sink call.

    The writer retains counters and rolling identities, never the witnesses.
    Closing publishes the compressed raw identity and the reusable lazy view.
    """
    def __init__(self, path, *, maximum_entry_bytes=MAXIMUM_ENTRY_BYTES):
        self.path = Path(path).resolve()
        if (isinstance(maximum_entry_bytes, bool) or not isinstance(maximum_entry_bytes, int) or
                not 1 <= maximum_entry_bytes <= MAXIMUM_ENTRY_BYTES):
            raise ValueError('Proof archive requires a positive finite entry byte budget')
        self.maximum_entry_bytes = maximum_entry_bytes
        self._raw = None
        self._compression = None
        if self.path.suffix.lower() == '.zst':
            zstd = _zstandard()
            parameters = zstd.ZstdCompressionParameters.from_level(3, window_log=27,
                enable_ldm=True, threads=0, write_checksum=True, write_content_size=False)
            self._raw = self.path.open('xb')
            try:
                compressor = zstd.ZstdCompressor(compression_params=parameters)
                binary = compressor.stream_writer(self._raw, closefd=False)
                # Match gzip.open(..., 'xt', encoding='utf-8'): on Windows this
                # preserves the existing CRLF bytes as well as the exact JSON.
                self.stream = io.TextIOWrapper(binary, encoding='utf-8')
            except BaseException:
                self._raw.close()
                raise
            self._compression = dict(codec='zstd', python_zstandard=zstd.__version__,
                libzstd=list(zstd.ZSTD_VERSION), backend=zstd.backend, level=3,
                window_log=27, window_bytes=ZSTD_WINDOW_BYTES, long_distance_matching=True,
                checksum=True, content_size=False, threads=0, event_flush='FLUSH_BLOCK')
        else:
            self.stream = gzip.open(self.path, 'xt', encoding='utf-8', compresslevel=3)
        self.entries = self.holds = 0
        self.kinds, self.closed = {}, False
        self._rolling = hashlib.sha256()
        self._source = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def append(self, payload):
        if self.closed or not isinstance(payload, dict):
            raise ValueError('Proof archive needs an open writer and a complete producer dictionary')
        identity = semantic_sha256(payload, fast=True)
        kind = payload.get('archive_event_kind', payload.get('kind', 'APPLIED_HOLD_COMPLETE' if 'record' in payload else 'WITNESS'))
        if kind == 'APPLIED_HOLD_COMPLETE' and not isinstance(payload.get('record'), dict):
            raise ValueError('Completed hold archive event lacks its full admitted record')
        entry = dict(schema=SCHEMA, sequence=self.entries, payload_sha256=identity, payload=payload)
        encoder = _Encoder(separators=(',', ':'), allow_nan=False)
        chunks = tuple(encoder.iterencode(entry, _one_shot=True))
        # ensure_ascii=True (JSONEncoder's default) makes chunks ASCII bytes.
        # TextIOWrapper's default newline translates the final '\n' by os.linesep.
        if sum(map(len, chunks))+len(os.linesep.encode('utf-8')) > self.maximum_entry_bytes:
            raise ValueError('Nonlinear archive entry exceeds the finite reader byte budget before acknowledgement')
        for chunk in chunks:
            self.stream.write(chunk)
        self.stream.write('\n')
        self.flush()
        self.entries += 1
        self.kinds[kind] = self.kinds.get(kind, 0)+1
        if kind == 'APPLIED_HOLD_COMPLETE':
            self.holds += 1
        self._rolling.update(identity.encode('ascii'))
        return identity

    def flush(self):
        if not self.closed:
            # Text flush calls the Zstd writer's FLUSH_BLOCK (not FLUSH_FRAME),
            # retaining prior history while making this event decodable.
            self.stream.flush()
            if self._raw is not None:
                self._raw.flush()
                os.fsync(self._raw.fileno())
            else:
                os.fsync(self.stream.buffer.fileobj.fileno())

    def close(self):
        if self.closed:
            return self._source
        self.stream.close()
        if self._raw is not None:
            # Closing text/binary writes the final checksummed frame.  Finish
            # durability before releasing the large compressor context.
            self._raw.flush()
            os.fsync(self._raw.fileno())
            self._raw.close()
            self._raw = self.stream = None
        else:
            with self.path.open('ab') as durable:
                durable.flush()
                os.fsync(durable.fileno())
        self.closed = True
        record = dict(schema=SCHEMA, raw_sha256=file_sha256(self.path), entries=self.entries,
            completed_holds=self.holds, event_kinds=self.kinds, payload_identities_sha256=self._rolling.hexdigest(),
            scope='Exact streamed producer witnesses and actual hold/failure events; not a new proof or fabricated execution')
        if self._compression is not None:
            record['compression'] = self._compression
        manifest = self.path.with_name(self.path.name+'.manifest.json')
        temporary = manifest.with_name(manifest.name+'.tmp')
        temporary.write_text(json.dumps(record, allow_nan=False, indent=2)+'\n', encoding='utf-8')
        temporary.replace(manifest)
        self._source = NonlinearHoldArchive(self.path, expected_sha256=record['raw_sha256'],
            expected_holds=self.holds, expected_entries=self.entries, expected_payload_ids_sha256=self._rolling.hexdigest(),
            maximum_entry_bytes=self.maximum_entry_bytes)
        return self._source


class NonlinearHoldArchive:
    """Reopen and verify a closed archive on every complete lazy iteration."""
    def __init__(self, path, *, expected_sha256, expected_holds, expected_entries=None,
                 expected_payload_ids_sha256=None, maximum_entry_bytes=MAXIMUM_ENTRY_BYTES):
        self.path, self.raw_sha256 = Path(path).resolve(), str(expected_sha256)
        self.count, self.entries = int(expected_holds), expected_entries
        self.payload_ids_sha256 = expected_payload_ids_sha256
        self.maximum_entry_bytes = int(maximum_entry_bytes)
        if (len(self.raw_sha256) != 64 or self.count < 0 or self.maximum_entry_bytes < 1 or
                file_sha256(self.path) != self.raw_sha256):
            raise ValueError('Closed nonlinear archive source identity or count is invalid')
        self.identity = semantic_sha256(dict(schema=SCHEMA, raw_sha256=self.raw_sha256,
            completed_holds=self.count, entries=self.entries, payload_ids_sha256=self.payload_ids_sha256))

    def __len__(self):
        return self.count

    def as_dict(self):
        return dict(path=str(self.path), raw_sha256=self.raw_sha256, completed_holds=self.count,
            entries=self.entries, identity=self.identity, method='source-bound-reiterable-streaming-full-hold-view-v1')

    def __iter__(self):
        if file_sha256(self.path) != self.raw_sha256:
            raise ValueError('Nonlinear archive changed before accounting')
        entries = holds = 0
        pending = None
        rolling = hashlib.sha256()
        with _open_archive_binary(self.path) as stream:
            while True:
                line = stream.readline(self.maximum_entry_bytes+1)
                if not line:
                    break
                if len(line) > self.maximum_entry_bytes or not line.endswith(b'\n'):
                    raise ValueError('Nonlinear archive entry is truncated or exceeds the finite reader budget')
                entry = json.loads(line)
                payload = entry['payload']
                if (entry.get('schema') != SCHEMA or entry.get('sequence') != entries or
                        semantic_sha256(payload, fast=True) != entry['payload_sha256']):
                    raise ValueError('Nonlinear archive entry identity or exact ordering changed')
                rolling.update(entry['payload_sha256'].encode('ascii'))
                entries += 1
                kind = payload.get('archive_event_kind', payload.get('kind', 'APPLIED_HOLD_COMPLETE' if 'record' in payload else 'WITNESS'))
                if kind == 'ADMITTED_BEFORE_MOTION':
                    from orbexa.core.dynamics.reference_frame import fingerprint as staged_fingerprint
                    staged = payload['record']
                    identity = payload['staged_identity']
                    if (pending is not None or identity != staged['staged_identity'] or
                            staged_fingerprint({key: value for key, value in staged.items() if key != 'staged_identity'}) != identity):
                        raise ValueError('Nonlinear staged witness identity or ordering changed')
                    pending = (identity, staged['decision']['identity'], staged['exact_start'], staged['exact_end'])
                if kind == 'ASSUMPTION_VIOLATION':
                    raise ValueError('Nonlinear archive retains an assumption violation; successful continuation is unauthorized')
                if kind == 'APPLIED_HOLD_COMPLETE':
                    record = payload['record']
                    identity = payload.get('staged_identity')
                    if identity is not None:
                        expected = (identity, record['decision']['identity'], record['exact_start'], record['exact_end'])
                        if pending != expected or record.get('staged_identity') != identity:
                            raise ValueError('Completed nonlinear hold lacks its matching preceding staged witness')
                    elif pending is not None:
                        raise ValueError('Completed nonlinear hold lost its staged witness identity')
                    pending = None
                    holds += 1
                    yield record
        if (pending is not None or holds != self.count or (self.entries is not None and entries != self.entries) or
                (self.payload_ids_sha256 is not None and rolling.hexdigest() != self.payload_ids_sha256) or
                file_sha256(self.path) != self.raw_sha256):
            raise ValueError('Nonlinear archive complete count, ordering or raw source identity changed')
