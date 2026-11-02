"""Non-destructive, content-addressed TA reconstruction preparation."""

from dataclasses import dataclass
import fcntl
from itertools import zip_longest
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from .banks import (
    DSTError,
    RAW,
    MUON,
    RECON,
    iter_events,
    sha256_file,
    write_events,
)


@dataclass(frozen=True)
class Inspection:
    events: int
    triggered: int
    muon_events: int
    needs_reconstruction: bool


def inspect_dst(path):
    count = triggered = muon = 0
    needs = False
    for e in iter_events(path):
        count += 1
        muon += MUON in e.banks
        if RAW in e.banks and len(e.raw["keys"]):
            triggered += 1
            needs |= not RECON.issubset(e.banks)
    if count == 0:
        raise DSTError("DST contains no events")
    return Inspection(count, triggered, muon, needs)


def _libraries(executable):
    """Fingerprint linked code as well as the small reconstruction executable."""
    ldd = shutil.which("ldd")
    if not ldd:
        return {}
    result = subprocess.run(
        [ldd, executable], capture_output=True, text=True, timeout=30
    )
    libraries = {}
    for line in result.stdout.splitlines():
        if "not found" in line:
            raise DSTError(f"Reconstruction dependency unavailable: {line.strip()}")
        for word in line.split():
            if word.startswith("/") and Path(word).is_file():
                libraries[word] = sha256_file(word)
    return libraries


def _restore(source, reconstructed):
    """Reject reorder/drop/mutation; preserve native muon payloads exactly."""
    for original, new in zip_longest(iter_events(source), iter_events(reconstructed)):
        if original is None or new is None:
            raise DSTError("Reconstruction changed event count")
        if original.fingerprint != new.fingerprint:
            raise DSTError(
                f"Reconstruction changed truth/waveforms at event {original.ordinal}"
            )
        if MUON in original.banks:
            if (
                MUON in new.banks
                and new.banks[MUON].payload != original.banks[MUON].payload
            ):
                raise DSTError("Reconstruction modified MUONEXTR")
            new.banks[MUON] = original.banks[MUON]
        yield new


def prepare_dst(
    path,
    reconstruction="auto",
    cache_dir=None,
    *,
    commands=None,
    dependencies=(),
    timeout=3600,
    inspection=None,
):
    """Return reconstructed input or a verified cached copy.

    commands optionally supplies two argument lists, with {input}/{output}
    placeholders, for site-specific TA reconstruction. dependencies lists any
    external calibration/tables those commands use; their hashes key the cache.
    Missing banks never become fake zero-valued reconstruction results.
    """
    if reconstruction not in ("auto", "require"):
        raise ValueError("reconstruction must be 'auto' or 'require'")
    if (
        not isinstance(timeout, (int, float))
        or isinstance(timeout, bool)
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("reconstruction_timeout must be finite and positive")
    path = Path(path).resolve(strict=True)
    info = inspection or inspect_dst(path)
    if not info.needs_reconstruction:
        return path
    if reconstruction == "require":
        raise DSTError(
            'Missing reconstruction banks. Use reconstruction="auto" or prepare_dst().'
        )
    if commands is None:
        commands = [
            ["rufptn.run", "{input}", "-o1f", "{output}"],
            ["rufldf.run", "{input}", "-o1f", "{output}"],
        ]
    if len(commands) != 2:
        raise ValueError("Expected pattern-recognition and fitting commands")
    resolved = []
    for args in commands:
        if not args or not all(isinstance(a, str) for a in args):
            raise ValueError("Commands must be argument lists")
        binary = shutil.which(args[0])
        if not binary:
            raise DSTError(f"Reconstruction executable unavailable: {args[0]}")
        if not any("{input}" in a for a in args) or not any(
            "{output}" in a for a in args
        ):
            raise ValueError("Each reconstruction command needs {input} and {output}")
        # Preserve the invoked path: resolving a venv Python symlink changes
        # its environment, even though the executable bytes are identical.
        resolved.append([str(Path(binary).absolute()), *args[1:]])
    identity = {
        "schema": 1,
        "input_sha256": sha256_file(path),
        "commands": resolved,
        "binaries": {c[0]: sha256_file(c[0]) for c in resolved},
        "libraries": {c[0]: _libraries(c[0]) for c in resolved},
        "dependencies": {
            str(Path(p).resolve(strict=True)): sha256_file(p) for p in dependencies
        },
        "environment": {
            k: os.environ.get(k, "") for k in ("SDDIR", "ROOTSYS", "LD_LIBRARY_PATH")
        },
    }
    import hashlib

    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root = Path(
        cache_dir or os.environ.get("DSTPARSER_CACHE", Path.home() / ".cache/dstparser")
    ).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    target = root / key
    target.mkdir(exist_ok=True)
    final = target / "reconstructed.dst.gz"
    receipt = target / "complete.json"
    with (target / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            if (
                saved["identity"] != identity
                or sha256_file(final) != saved["output_sha256"]
            ):
                raise DSTError("Reconstruction cache integrity check failed")
            return final
        if final.exists():
            raise DSTError("Incomplete reconstruction cache: output without receipt")
        with tempfile.TemporaryDirectory(prefix="prepare-", dir=target) as tmp:
            tmp = Path(tmp)
            current = path
            for i, args in enumerate(resolved):
                output = tmp / f"stage{i}.dst.gz"
                command = [
                    s.format(input=str(current), output=str(output)) for s in args
                ]
                with (target / f"stage{i}.log").open("w") as log:
                    log.write(json.dumps(command) + "\n")
                    log.flush()
                    run = subprocess.run(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        cwd=tmp,
                        timeout=timeout,
                    )
                if run.returncode or not output.is_file():
                    raise DSTError(
                        f"Reconstruction stage {i} failed; see {target}/stage{i}.log"
                    )
                current = output
            restored = tmp / "restored.dst.gz"
            write_events(restored, _restore(path, current))
            result = inspect_dst(restored)
            if (
                result.needs_reconstruction
                or result.events != info.events
                or result.muon_events != info.muon_events
            ):
                raise DSTError("Reconstruction output is incomplete")
            if sha256_file(path) != identity["input_sha256"]:
                raise DSTError("Input changed during reconstruction")
            saved = {
                "identity": identity,
                "output_sha256": sha256_file(restored),
                "events": result.events,
                "muon_events": result.muon_events,
            }
            os.replace(restored, final)
            part = target / "complete.json.part"
            with part.open("w") as f:
                json.dump(saved, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(part, receipt)
    return final
