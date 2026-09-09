"""Load the preserved float32 weights without pickle or network downloads."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np


ASSETS = Path(__file__).resolve().parent / "assets"
EXPECTED_FLOATS = 1_402_487
EXPECTED_SHA256 = "70a283984fb2f21e93d9dcb67ca7fc228622523d80124140e206560e003a144e"


def load_parameters(directory: Path | None = None) -> dict[str, np.ndarray]:
    directory = ASSETS if directory is None else Path(directory)
    metadata = json.loads((directory / "layout.json").read_text(encoding="utf-8"))
    path = directory / "parameters.f32"
    if path.stat().st_size != EXPECTED_FLOATS * 4:
        raise ValueError("Unexpected parameter-buffer size")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != EXPECTED_SHA256:
        raise ValueError("Parameter checksum does not match the preserved submission")
    if metadata["stored_float32_values"] != EXPECTED_FLOATS or metadata["sha256"] != EXPECTED_SHA256:
        raise ValueError("Unexpected parameter metadata")
    flat = np.frombuffer(payload, dtype="<f4")
    if not np.isfinite(flat).all():
        raise ValueError("Non-finite parameter value")
    weights = {}
    cursor = 0
    for name, shape, start, end in metadata["tensor_layout"]:
        if not isinstance(name, str) or name in weights:
            raise ValueError("Invalid or duplicate tensor name")
        if not shape or any(type(n) is not int or n <= 0 for n in shape):
            raise ValueError("Invalid tensor shape")
        if type(start) is not int or type(end) is not int or start != cursor:
            raise ValueError("Non-contiguous tensor offsets")
        if end != start + math.prod(shape) or end > EXPECTED_FLOATS:
            raise ValueError("Tensor shape does not match its offsets")
        weights[name] = flat[start:end].reshape(shape)
        cursor = end
    if cursor != EXPECTED_FLOATS:
        raise ValueError("Incomplete tensor layout")
    return weights


def load_deck() -> list[int]:
    cards = [int(line) for line in (ASSETS / "deck.csv").read_text().splitlines()]
    if len(cards) != 60 or any(card <= 0 for card in cards):
        raise ValueError("Expected 60 positive card IDs")
    return cards
