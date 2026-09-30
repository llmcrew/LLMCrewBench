"""Answer normalisation and extraction helpers shared by all evaluators.

The rules are deliberately narrow and frozen before any model runs: a format
point is awarded for a strictly valid answer, a content point for the right
value. Everything an evaluator accepts must be expressible here, so the same
function is used to normalise the reference answer and the model's answer.
"""

from __future__ import annotations

import json
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from fractions import Fraction

_FENCE_RE = re.compile(r"^\s*```[^\n`]*\n(.*?)\n?\s*```\s*$", re.S)
_ANY_FENCE_RE = re.compile(r"```[^\n`]*\n(.*?)```", re.S)
_EMPHASIS_RE = re.compile(r"[*_`]+")
_WS_RE = re.compile(r"[ \t\u00a0]+")
_UNIT_NOISE = re.compile(
    r"(?<=\d)\s*(?:ms|s|сек(?:унд\w*)?|мин(?:ут\w*)?|ч(?:ас\w*)?|gi?b|mib|kib|шт\.?|tok(?:ens?)?)\s*$",
    re.I,
)


# --------------------------------------------------------------------------- #
# Fences and JSON
# --------------------------------------------------------------------------- #


def strip_fence(text: str) -> tuple[str, bool]:
    """Remove one wrapping code fence. Returns ``(inner, had_fence)``."""
    m = _FENCE_RE.match(text or "")
    if m:
        return m.group(1).strip(), True
    return (text or "").strip(), False


def has_extra_text(text: str) -> bool:
    """True when the reply is not *only* a fenced block or a bare value."""
    inner, fenced = strip_fence(text)
    if fenced:
        return bool(inner) and _ANY_FENCE_RE.search(text) is None
    return False


def first_fenced_block(text: str) -> str | None:
    m = _ANY_FENCE_RE.search(text or "")
    return m.group(1).strip() if m else None


def strict_json(text: str) -> tuple[object | None, bool]:
    """Parse the whole reply as JSON.

    Returns ``(value, strict_ok)``. ``strict_ok`` is False when the reply has
    any wrapping fence or text around the JSON — that costs the format point,
    while the value may still earn content points.
    """
    raw = (text or "").strip()
    if not raw:
        return None, False
    try:
        return json.loads(raw), True
    except json.JSONDecodeError:
        pass
    inner, fenced = strip_fence(raw)
    if fenced:
        try:
            return json.loads(inner), False
        except json.JSONDecodeError:
            pass
    return None, False


def loose_json(text: str) -> object | None:
    """Best-effort extraction of the first JSON value from free text."""
    value, _ = strict_json(text)
    if value is not None:
        return value
    raw = (text or "").strip()
    for pattern in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pattern, raw, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
    return None


# --------------------------------------------------------------------------- #
# Single-value answers
# --------------------------------------------------------------------------- #


def clean_line(line: str) -> str:
    """Strip markdown emphasis, quotes and surrounding whitespace."""
    s = unicodedata.normalize("NFKC", line or "")
    s = _EMPHASIS_RE.sub("", s)
    s = s.strip().strip('"').strip("'").strip()
    s = _WS_RE.sub(" ", s).strip()
    return s


def answer_lines(text: str) -> list[str]:
    """Non-empty, cleaned lines of the reply (fence markers dropped)."""
    inner, _ = strip_fence(text)
    out = []
    for line in inner.splitlines():
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        out.append(clean_line(line))
    return [line for line in out if line]


def _strip_units(value: str, units: tuple[str, ...]) -> str:
    """Drop a trailing unit word only when a number precedes it.

    ``s`` is a unit (seconds) and also the last letter of words like ``helios``:
    stripping on a bare suffix match turned a correct answer into a wrong one.
    A unit is therefore only removed when the remaining head ends with a digit.
    """
    if not units:
        return value
    for unit in units:
        if unit and value.casefold().endswith(unit.casefold()):
            head = value[: -len(unit)].rstrip(" .,:;").strip()
            if head and head[-1].isdigit():
                return head
            break
    match = _UNIT_NOISE.search(value)
    if match:
        head = value[: match.start()].rstrip(" .,:;").strip()
        if head:
            return head
    return value


def answer_token(text: str, unit_whitelist: tuple[str, ...] = ()) -> str | None:
    """The single answer value, or ``None`` when the reply is not one value.

    A reply must consist of exactly one non-empty line. ``unit_whitelist``
    carries accepted trailing unit words ("минут", "seconds") that are removed
    before comparison — but only after a number, never from a bare word.
    Trailing periods and a leading label such as ``Ответ:`` are tolerated.
    """
    lines = answer_lines(text)
    if len(lines) != 1:
        return None
    value = lines[0]
    for prefix in ("Ответ:", "Ответ -", "answer:", "Ответ —"):
        if value.lower().startswith(prefix.lower()):
            value = value[len(prefix):].strip()
    value = value.rstrip(".").strip()
    value = _strip_units(value, unit_whitelist)
    return value or None


def parse_decimal(text: str) -> Decimal | None:
    """First decimal number in the reply, accepting a comma decimal mark."""
    raw = (text or "").replace(",", ".")
    m = re.search(r"[-+]?\d+(?:\.\d+)?", raw)
    if not m:
        return None
    try:
        return Decimal(m.group(0))
    except InvalidOperation:
        return None


def parse_integer(text: str) -> int | None:
    m = re.search(r"[-+]?\d+", (text or "").replace(" ", ""))
    return int(m.group(0)) if m else None


def parse_fraction(text: str) -> Fraction | None:
    """Parse ``a/b`` or a bare integer as an exact rational."""
    if text is None:
        return None
    raw = clean_line(text)
    m = re.fullmatch(r"([-+]?\d+)\s*/\s*([-+]?\d+)", raw)
    if m:
        den = int(m.group(2))
        return Fraction(int(m.group(1)), den) if den else None
    m = re.fullmatch(r"[-+]?\d+", raw)
    return Fraction(int(raw)) if m else None


def decimals_equal(a: Decimal | None, b: Decimal, places: int | None = None) -> bool:
    if a is None:
        return False
    if places is None:
        return a == b
    quantum = Decimal(1).scaleb(-places)
    return a.quantize(quantum) == b.quantize(quantum)


def values_equal(model: str | None, reference: str, *, casefold: bool = False) -> bool:
    """Compare two already-normalised answer tokens."""
    if model is None:
        return False
    if casefold:
        return model.casefold() == reference.casefold()
    return model == reference


# --------------------------------------------------------------------------- #
# Text checks
# --------------------------------------------------------------------------- #


def contains_all(text: str, needles: list[str]) -> bool:
    low = (text or "").casefold()
    return all(n.casefold() in low for n in needles)


def contains_any(text: str, needles: list[str]) -> bool:
    low = (text or "").casefold()
    return any(n.casefold() in low for n in needles)


def count_nonempty_lines(text: str) -> int:
    return len(answer_lines(text))


def numbered_lines(text: str, expected: int) -> list[str] | None:
    """Split a ``1. ...`` style answer into exactly ``expected`` values.

    Returns ``None`` when the numbering is missing or the count is wrong, so a
    caller can award 0 per line instead of guessing an alignment.
    """
    inner, _ = strip_fence(text)
    found: dict[int, str] = {}
    for line in inner.splitlines():
        m = re.match(r"\s*(\d{1,2})\s*[.)\]:-]\s*(.+?)\s*$", line)
        if m:
            idx = int(m.group(1))
            if idx not in found:
                found[idx] = clean_line(m.group(2))
    if sorted(found) != list(range(1, expected + 1)):
        return None
    return [found[i] for i in range(1, expected + 1)]


def known_unknown(value: str) -> tuple[str, str] | None:
    """Parse ``KNOWN: <answer>`` / ``UNKNOWN``. Returns ``(kind, payload)``."""
    raw = clean_line(value)
    m = re.match(r"(?i)^(KNOWN|ИЗВЕСТНО)\s*[:\-]\s*(.+)$", raw)
    if m:
        return "known", clean_line(m.group(2))
    if re.fullmatch(r"(?i)UNKNOWN|НЕИЗВЕСТНО", raw):
        return "unknown", ""
    return None
