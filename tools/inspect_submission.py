"""Inspect the two-file GALEX submission format without importing its Python code.

This is a bounded format/integrity check, not a malware scanner, provenance
certificate, or proof of leaderboard performance. Python standard library only.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import math
from pathlib import Path
import stat
import struct
import sys
import zipfile


MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
MAX_UNPACKED_BYTES = 16 * 1024 * 1024
MEMBERS = {"main.py", "deck.csv"}
FIELDS = {
    "_PARAMETER_LAYOUT", "_PARAMETER_FLOAT_COUNT", "_PARAMETER_SHA256",
    "_PARAMETER_B64_CHUNKS",
}
ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_archive(path: Path) -> dict[str, bytes]:
    """Validate members before reading them; never extract an archive to disk."""
    require(path.is_file(), f"Archive not found: {path.name}")
    require(path.stat().st_size <= MAX_ARCHIVE_BYTES, "Archive exceeds size limit")
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        require(len(entries) == 2, "Expected exactly main.py and deck.csv at the ZIP root")
        require({entry.filename for entry in entries} == MEMBERS,
                "Unexpected, duplicate, or non-root archive members")
        require(sum(entry.file_size for entry in entries) <= MAX_UNPACKED_BYTES,
                "Uncompressed payload exceeds size limit")
        for entry in entries:
            mode = (entry.external_attr >> 16) & 0o170000
            require(mode in (0, stat.S_IFREG), "Links and special files are not allowed")
            require(not entry.flag_bits & 1, "Encrypted members are not supported")
        return {entry.filename: archive.read(entry) for entry in entries}


def extract_constants(source: bytes) -> dict:
    """Accept only literal assignments for the four embedded-parameter fields."""
    tree = ast.parse(source.decode("utf-8"), filename="main.py")
    constants = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in FIELDS:
                require(target.id not in constants, "Duplicate parameter declaration")
                constants[target.id] = ast.literal_eval(node.value)
    require(constants.keys() == FIELDS, "Missing embedded-parameter declarations")
    return constants


def inspect_archive(path: Path) -> dict:
    files = read_archive(path)
    constants = extract_constants(files["main.py"])
    count = constants["_PARAMETER_FLOAT_COUNT"]
    layout = constants["_PARAMETER_LAYOUT"]
    chunks = constants["_PARAMETER_B64_CHUNKS"]
    require(type(count) is int and 0 < count <= MAX_UNPACKED_BYTES // 4,
            "Invalid float count")
    require(isinstance(chunks, (tuple, list)) and all(isinstance(x, bytes) for x in chunks),
            "Parameter chunks must be literal bytes")
    payload = base64.b64decode(b"".join(chunks), validate=True)
    require(len(payload) == count * 4, "Parameter byte length mismatch")
    parameter_hash = hashlib.sha256(payload).hexdigest()
    require(parameter_hash == constants["_PARAMETER_SHA256"], "Parameter checksum mismatch")
    require(all(math.isfinite(x[0]) for x in struct.iter_unpack("<f", payload)),
            "Non-finite float32 parameter")
    require(isinstance(layout, (tuple, list)), "Invalid tensor layout")
    offset = 0
    shapes = {}
    for entry in layout:
        require(isinstance(entry, (tuple, list)) and len(entry) == 4, "Invalid tensor entry")
        name, shape, start, end = entry
        require(isinstance(name, str) and name not in shapes, "Duplicate/invalid tensor name")
        require(isinstance(shape, (tuple, list)) and len(shape) > 0,
                "Invalid tensor shape")
        require(all(type(n) is int and n > 0 for n in shape), "Invalid tensor dimension")
        require(type(start) is int and type(end) is int and start == offset,
                "Non-contiguous tensor offsets")
        require(end == start + math.prod(shape) and end <= count,
                "Tensor shape/offset mismatch")
        shapes[name] = list(shape)
        offset = end
    require(offset == count, "Tensor layout does not cover the parameter buffer")
    deck = [int(line) for line in files["deck.csv"].decode("utf-8").splitlines()]
    require(len(deck) == 60 and all(card > 0 for card in deck),
            "Expected 60 positive integer card IDs; full deck legality requires the SDK")
    return {
        "format": "galex-two-file-embedded-float32",
        "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "files": {
            name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in sorted(files.items())
        },
        "parameters": {
            "stored_float32_values": count,
            "bytes": len(payload),
            "sha256": parameter_hash,
            "tensor_count": len(shapes),
            "tensor_shapes": shapes,
        },
        "deck": {"card_count": len(deck), "unique_card_ids": len(set(deck))},
        "code_executed": False,
    }


def check_expected(report: dict, expected: dict) -> None:
    """Check member/payload identity; recompression may change the ZIP hash."""
    for name in sorted(MEMBERS):
        require(report["files"][name]["sha256"] == expected["files"][name]["sha256"],
                f"{name} does not match the recorded GALEX artifact")
    for field in ("stored_float32_values", "sha256"):
        require(report["parameters"][field] == expected["parameters"][field],
                f"Parameter {field} does not match the recorded artifact")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Path to your locally held submission2.zip")
    parser.add_argument("--expected", type=Path,
                        help="Compare with a recorded manifest, e.g. evidence/submission_manifest.json")
    args = parser.parse_args()
    try:
        report = inspect_archive(args.archive)
        if args.expected:
            check_expected(report, json.loads(args.expected.read_text(encoding="utf-8")))
            report["matches_expected_members"] = True
    except (OSError, ValueError, TypeError, KeyError, SyntaxError, zipfile.BadZipFile,
            NotImplementedError, RuntimeError) as error:
        print(f"Inspection failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
