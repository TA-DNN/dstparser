"""Streaming DST95 framing/CRC reader and lossless bank writer.

Only the RUSDRAW/RUSDMC/MUONEXTR layouts are interpreted. Other banks are
opaque payloads and survive writing unchanged. Layer arrays use lower, upper.
"""

import binascii
import bz2
from dataclasses import dataclass
import gzip
import hashlib
from pathlib import Path
import struct

import numpy as np

START, STOP = 1400000023, 1400000101
RAW, MC, MUON = 13101, 13105, 13114
RECON = frozenset((13103, 13104, 13107))
BSD = 13112
SIZE, END = 32000, 31994
_REVERSE = bytes(int(f"{i:08b}"[::-1], 2) for i in range(256))


class DSTError(ValueError):
    """Invalid framing, unsupported schema, or ambiguous event association."""


def crc(data):
    value = binascii.crc_hqx(data.translate(_REVERSE), 0)
    return _REVERSE[value >> 8] | (_REVERSE[value & 255] << 8)


def i4(data, offset):
    return struct.unpack_from("<i", data, offset)[0]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def _open(path):
    with open(path, "rb") as f:
        magic = f.read(3)
    return (
        gzip.open(path, "rb")
        if magic[:2] == b"\x1f\x8b"
        else bz2.open(path, "rb")
        if magic == b"BZh"
        else open(path, "rb")
    )


@dataclass(frozen=True)
class Bank:
    id: int
    version: int
    payload: bytes


def iter_banks(path):
    active = None
    checks = []
    with _open(path) as f:
        number = 0
        while True:
            block = f.read(SIZE)
            if not block:
                break
            if len(block) != SIZE:
                raise DSTError("Truncated DST physical block")
            if (
                block[:2] != b"`a"
                or i4(block, 2) != number
                or block[END : END + 2] != b"`c"
            ):
                raise DSTError("Invalid DST block framing or sequence")
            if i4(block, SIZE - 4) != crc(block[: SIZE - 4]):
                raise DSTError("DST block CRC mismatch")
            number += 1
            pos = 6
            if active is not None and block[pos : pos + 2] != b"`\x08":
                raise DSTError("Missing bank continuation")
            while True:
                marker = block[pos : pos + 2]
                if marker == b"`b":
                    if block[pos + 2 : END] != b"d" * (END - pos - 2):
                        raise DSTError("Invalid DST filler")
                    break
                if marker not in (b"`\x07", b"`\x08") or pos + 6 > END:
                    raise DSTError("Invalid bank opening")
                if marker == b"`\x07":
                    if active is not None:
                        raise DSTError("Nested bank")
                    active = bytearray()
                    checks = []
                elif active is None:
                    raise DSTError("Unexpected continuation")
                n = i4(block, pos + 2)
                stop = pos + 6 + n
                if n <= 0 or stop + 6 > END:
                    raise DSTError("Invalid segment length")
                active.extend(block[pos + 6 : stop])
                if len(active) > 128 * 1024**2:
                    raise DSTError("Bank exceeds 128 MiB safety bound")
                close = block[stop : stop + 2]
                checks.append(i4(block, stop + 2))
                pos = stop + 6
                if close == b"`\x0e":
                    payload = bytes(active)
                    checksum = crc(payload)
                    if len(payload) < 8 or any(c != checksum for c in checks):
                        raise DSTError("Bank CRC mismatch")
                    yield Bank(i4(payload, 0), i4(payload, 4), payload)
                    active = None
                elif close != b"`\x0f" or block[pos : pos + 2] != b"`b":
                    raise DSTError("Invalid bank closing")
    if active is not None:
        raise DSTError("EOF in bank continuation")


@dataclass
class Event:
    ordinal: int
    banks: dict

    @property
    def event_id(self):
        return i4(self.banks[RAW].payload, 8) if RAW in self.banks else None

    @property
    def raw(self):
        return decode_raw(self.banks[RAW]) if RAW in self.banks else None

    @property
    def muon(self):
        return decode_muon(self.banks[MUON]) if MUON in self.banks else None

    @property
    def fingerprint(self):
        h = hashlib.sha256()
        for bid in (MC, RAW):
            b = self.banks.get(bid)
            if b:
                # Known RUSDRAW v000/v001 have identical stored field layouts;
                # v001 only increases capacity. Reconstruction may update that
                # version word. Every actual truth/waveform byte still matches.
                if bid == RAW:
                    h.update(b.payload[:4] + b.payload[8:])
                else:
                    h.update(b.payload)
        return h.hexdigest()


def iter_events(path):
    banks = None
    ordinal = 0
    for b in iter_banks(path):
        if b.id == START:
            if banks is not None or b.version != 0 or len(b.payload) != 8:
                raise DSTError("Invalid START")
            banks = {}
        elif b.id == STOP:
            if banks is None or b.version != 0 or len(b.payload) != 8:
                raise DSTError("Invalid STOP")
            e = Event(ordinal, banks)
            if MC in banks:
                if banks[MC].version != 0 or len(banks[MC].payload) != 52:
                    raise DSTError("Unsupported RUSDMC layout")
                if RAW in banks and i4(banks[MC].payload, 8) != e.event_id:
                    raise DSTError("RUSDMC/RUSDRAW event mismatch")
            if RAW in banks:
                e.raw
            if MUON in banks:
                mu = e.muon
                raw = e.raw
                if raw is None or mu["event_id"] != e.event_id:
                    raise DSTError("MUONEXTR event mismatch")
                if not np.array_equal(mu["keys"], raw["keys"]):
                    raise DSTError("MUONEXTR waveform identity/order mismatch")
            yield e
            ordinal += 1
            banks = None
        else:
            if banks is None or b.id in banks:
                raise DSTError("Bank outside event or duplicate bank")
            banks[b.id] = b
    if banks is not None:
        raise DSTError("EOF before STOP")


def _keys(ids, wf):
    keys = np.column_stack((ids, wf))
    if len(np.unique(keys, axis=0)) != len(keys):
        raise DSTError("Duplicate station/waveform identity")
    return keys


def decode_raw(bank):
    p = bank.payload
    if bank.version not in (0, 1) or len(p) < 72:
        raise DSTError("Unsupported RUSDRAW version/layout")
    n = i4(p, 68)
    if not 0 <= n <= (256 if bank.version == 0 else 1024) or len(p) != 72 + 1280 * n:
        raise DSTError("Invalid RUSDRAW count/length")
    arrays = np.frombuffer(p, dtype="<i4", count=6 * n, offset=72).reshape(6, n)
    traces = np.empty((n, 2, 128), dtype=np.int32)
    for j in range(n):
        traces[j] = np.frombuffer(
            p, dtype="<i4", count=256, offset=72 + 24 * n + 1256 * j + 16
        ).reshape(2, 128)
    return {
        "keys": _keys(arrays[3], arrays[1]),
        "clocks": arrays[4:6].T,
        "traces": traces,
        "event_id": i4(p, 8),
        "event_code": i4(p, 12),
    }


def decode_muon(bank):
    p = bank.payload
    if bank.version != 0 or len(p) < 16:
        raise DSTError("Unsupported MUONEXTR version/layout")
    n = i4(p, 8)
    if not 0 <= n <= 1024 or len(p) != 16 + 32 * n:
        raise DSTError("Invalid MUONEXTR count/length")
    keys = np.frombuffer(p, dtype="<i4", count=2 * n, offset=16).reshape(2, n)
    signals = np.frombuffer(p, dtype="<f4", count=4 * n, offset=16 + 8 * n).reshape(
        4, n
    )
    sat = np.frombuffer(p, dtype="<i4", count=2 * n, offset=16 + 24 * n).reshape(2, n)
    if not np.all(np.isfinite(signals)) or np.any(sat < 0) or np.any(sat > 128):
        raise DSTError("Invalid muon values")
    return {
        "event_id": i4(p, 12),
        "keys": _keys(keys[0], keys[1]),
        "muon_signal": signals[[1, 0]].T.copy(),
        "muon_total_signal": signals[[3, 2]].T.copy(),
        "muon_nsat": sat[[1, 0]].T.copy(),
        "muon_available": np.ones(n, dtype=bool),
    }


def write_events(path, events):
    """Write complete events losslessly; output must not exist."""
    path = Path(path)
    with path.open("xb") as output:
        stream = (
            gzip.GzipFile(fileobj=output, mode="wb", mtime=0)
            if path.suffix == ".gz"
            else output
        )
        number = 0
        block = bytearray(b"`a" + struct.pack("<i", number))

        def flush():
            nonlocal block, number
            block.extend(b"`b")
            block.extend(b"d" * (END - len(block)))
            block.extend(b"`c")
            block.extend(struct.pack("<i", crc(bytes(block))))
            stream.write(block)
            number += 1
            block = bytearray(b"`a" + struct.pack("<i", number))

        try:
            for event in events:
                for payload in [
                    struct.pack("<ii", START, 0),
                    *[b.payload for b in event.banks.values()],
                    struct.pack("<ii", STOP, 0),
                ]:
                    checksum = crc(payload)
                    pos = 0
                    while pos < len(payload):
                        room = END - len(block) - 14
                        if room <= 0:
                            flush()
                            continue
                        n = min(room, len(payload) - pos)
                        block.extend(
                            bytes((96, 7 if pos == 0 else 8)) + struct.pack("<i", n)
                        )
                        block.extend(payload[pos : pos + n])
                        pos += n
                        block.extend(
                            bytes((96, 14 if pos == len(payload) else 15))
                            + struct.pack("<i", checksum)
                        )
                        if pos < len(payload):
                            flush()
            if len(block) > 6:
                flush()
        finally:
            if stream is not output:
                stream.close()
