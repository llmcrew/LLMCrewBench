"""Category E: generated long-context retrieval and state tracking tasks.

The fixtures deliberately use character sizing rather than a tokenizer.  The
conservative ``CHARS_PER_TOKEN`` estimate is also recorded in each task so a
run can be audited against the runtime's measured prompt length.
"""

from __future__ import annotations

import hashlib
import random
import re
from typing import Any

from .. import core, text
from ..config import RunConfig


CHARS_PER_TOKEN = 3.2
_CATEGORY = "E"
_POINTS = 2.0


def _hex(rng: random.Random, n: int = 8) -> str:
    return hashlib.sha256(str(rng.getrandbits(64)).encode("ascii")).hexdigest()[:n]


def _suffix(rng: random.Random) -> str:
    return f"{rng.randrange(100, 1000)}"


def _filler_records(rng: random.Random, count: int, kind: str) -> list[str]:
    """Make records without any target identifiers or target values."""
    records: list[str] = []
    for index in range(count):
        stamp = _suffix(rng)
        if kind == "deployment":
            records.append(
                f"deployment worker-{index:04d}-{stamp} uses artifact sha256:{_hex(rng)} "
                f"in region test-{index % 7}; replicas {2 + index % 5}."
            )
        elif kind == "service":
            records.append(
                f"service service-{index:04d}-{stamp} owns shard K{index + 300}; "
                f"shard K{index + 300} is currently assigned to node worker-{index % 19}."
            )
        elif kind == "project":
            records.append(
                f"project project-{index:04d}-{stamp} uses queue q-{index + 300}; "
                f"queue q-{index + 300} is served by broker broker-{index % 23}; "
                f"broker broker-{index % 23} runs in region lab-{index % 9}."
            )
        elif kind == "event":
            records.append(
                f"service service-{index:04d}-{stamp} owner = team-{index % 17}; "
                f"event 20{index % 10:02d}-01-01T00:{index % 60:02d}Z records routine maintenance."
            )
        else:
            records.append(
                f"incident INC-{300 + index:03d} affects node graphite-{index % 11}; "
                f"node graphite-{index % 11} hosts shard T{index % 80:02d}; "
                f"shard T{index % 80:02d} belongs to service service-{index:04d}."
            )
    return records


def _insert_at_quarter(records: list[str], line: str, fraction: float) -> None:
    position = max(0, min(len(records), int(len(records) * fraction)))
    records[position:position] = [line]


def _unique_assertions(document: str, facts: list[str]) -> None:
    assert len(facts) == len(set(facts)), "chain links must be unique"
    for fact in facts:
        assert document.count(fact) == 1, f"target fact is not unique: {fact!r}"
        assert document.find(fact) >= 0, f"target fact is absent: {fact!r}"

def _make_prompt(records: list[str], question: str) -> tuple[str, str]:
    document = "\n".join(records)
    prompt = (
        "Используй ТОЛЬКО сведения из документа ниже. Внешние знания и "
        "предположения запрещены.\n\nDOCUMENT:\n"
        + document
        + "\n\nQUESTION:\n"
        + question
        + "\n"
    )
    return prompt, document


def _ratio(cfg: RunConfig, task_id: str = "") -> float:
    """Calibrated characters-per-token for this run (see ``RunConfig``).

    The module constant is only a starting guess: a real tokenizer can pack
    repetitive filler two to three times denser than prose, and a document sized
    by the wrong constant tests a smaller context than it claims.
    """
    per_task = getattr(cfg, "task_ratios", None) or {}
    if task_id and task_id in per_task:
        value = float(per_task[task_id])
    else:
        value = float(getattr(cfg, "chars_per_token", CHARS_PER_TOKEN) or CHARS_PER_TOKEN)
    return value if value > 0 else CHARS_PER_TOKEN


def _targeted_records(
    rng: random.Random,
    target_tokens: int,
    kind: str,
    special: list[str],
    fractions: list[float],
    question: str,
    target_facts: list[str],
    ratio: float = CHARS_PER_TOKEN,
) -> tuple[str, dict[str, Any]]:
    """Size by characters using a list/join build (never repeated concatenation)."""
    target_chars = int(target_tokens * ratio)
    # This representative line keeps the estimate stable while every record
    # remains unique.  The final prompt is measured again below.
    representative = _filler_records(random.Random(0), 1, kind)[0]
    fixed_chars = len(
        "Используй ТОЛЬКО сведения из документа ниже. Внешние знания и "
        "предположения запрещены.\n\nDOCUMENT:\n\nQUESTION:\n"
        + question
        + "\n"
    ) + sum(len(line) + 1 for line in special)
    filler_count = max(100, int(max(0, target_chars - fixed_chars) / max(1, len(representative) + 1)))
    fillers = _filler_records(rng, filler_count, kind)
    records = fillers[:]
    for line, fraction in zip(special, fractions):
        _insert_at_quarter(records, line, fraction)
    prompt, document = _make_prompt(records, question)

    # Account for actual random record lengths.  A small bounded top-up is
    # enough to keep all targets in their requested band without quadratic work.
    while len(prompt) < int(target_chars * 0.94):
        top_up = _filler_records(rng, 64, kind)
        records.extend(top_up)
        prompt, document = _make_prompt(records, question)
    _unique_assertions(document, target_facts)
    estimated = round(len(prompt) / ratio)
    return prompt, {
        "document_chars": len(document),
        "prompt_chars": len(prompt),
        "estimated_tokens": estimated,
        "target_tokens": target_tokens,
        "chars_per_token": ratio,
        "target_facts": list(target_facts),
        "fact_positions": {fact: document.find(fact) for fact in target_facts},
    }
def _target(cfg: RunConfig, legacy: int, share: float) -> int:
    """Document size as a share of this run's window, in legacy tokens.

    With the 32K window the shares reproduce the original 8K/16K/24K/27K ladder
    exactly, so a run against a 32K server keeps its published meaning; a run
    against a 120K server (or a model that admits 1M) scales the whole ladder up
    and the scenarios keep their bites of the context instead of shrinking to a
    tenth of it.
    """
    window = int(getattr(cfg, "context_window", 32768) or 32768)
    if window <= 32768:
        return legacy
    return int(round(window * share))


def _base_meta(cfg: RunConfig, target_tokens: int, estimated_tokens: int, prompt_chars: int) -> dict[str, Any]:
    runtime_window = int(getattr(cfg, "n_ctx", getattr(cfg, "context_window", 32768)))
    configured_window = int(getattr(cfg, "context_window", runtime_window))
    window = min(runtime_window, configured_window)
    max_new = int(getattr(cfg, "max_new_tokens", 4096))
    # The 512-token reserve is part of the documented admission budget.  Use
    # the stricter configured/runtime window when they differ.
    budget = max(0, window - max_new - 512)
    meta: dict[str, Any] = {
        "target_tokens": target_tokens,
        "estimated_tokens": estimated_tokens,
        "chars_per_token": float(getattr(cfg, "chars_per_token", CHARS_PER_TOKEN) or CHARS_PER_TOKEN),
        "prompt_chars": prompt_chars,
        "runtime_context_window": runtime_window,
        "context_window": window,
        "max_new_tokens": max_new,
        "admission_budget_tokens": budget,
    }
    if target_tokens > budget or estimated_tokens > budget:
        # Do not resize a fixture to hide an admission failure.
        meta["admission_required"] = target_tokens
    return meta


def _task(
    task_id: str,
    prompt: str,
    oracle: dict[str, Any],
    meta: dict[str, Any],
) -> core.TaskInstance:
    oracle = dict(oracle)
    oracle.setdefault("reference", str(oracle["answer"]))
    oracle.setdefault("unit_whitelist", ())
    return core.TaskInstance(
        id=task_id,
        category=_CATEGORY,
        title="Long context",
        prompt=prompt,
        max_points=_POINTS,
        oracle=oracle,
        meta=meta,
    )


def _lc1(rng: random.Random, cfg: RunConfig) -> core.TaskInstance:
    deployment = f"violet-{rng.randrange(100, 1000)}"
    artifact = _hex(rng)
    fact = f"deployment {deployment} uses artifact sha256:{artifact}"
    question = f"Какой artifact использует deployment {deployment}? Ответь только значением sha256."
    prompt, sizing = _targeted_records(rng, _target(cfg, 8192, 0.25), "deployment", [fact], [rng.choice((0.17, 0.36, 0.58, 0.77, 0.91))], question, [fact], ratio=_ratio(cfg, "LC-1"))
    meta = _base_meta(cfg, _target(cfg, 8192, 0.25), sizing["estimated_tokens"], sizing["prompt_chars"])
    meta.update({k: sizing[k] for k in ("document_chars", "target_facts", "fact_positions")})
    return _task("LC-1", prompt, {"answer": artifact, "reference": artifact,
                          "accept_any": [f"sha256:{artifact}", f"SHA256:{artifact}", f"sha256: {artifact}"], "target_fact": fact, "target_deployment": deployment, "positions": sizing["fact_positions"]}, meta)


def _lc2(rng: random.Random, cfg: RunConfig) -> core.TaskInstance:
    service = f"meridian-{_suffix(rng)}"
    shard = f"K{rng.randrange(10, 99)}"
    node = f"beta-{rng.randrange(2, 9)}"
    link1 = f"service {service} owns shard {shard}"
    link2 = f"shard {shard} is currently assigned to node {node}"
    question = f"Какой node обслуживает shard service {service}? Только именем node."
    prompt, sizing = _targeted_records(rng, _target(cfg, 16384, 0.50), "service", [link1, link2], [0.18, 0.52], question, [link1, link2], ratio=_ratio(cfg, "LC-2"))
    meta = _base_meta(cfg, _target(cfg, 16384, 0.50), sizing["estimated_tokens"], sizing["prompt_chars"])
    meta.update({k: sizing[k] for k in ("document_chars", "target_facts", "fact_positions")})
    return _task("LC-2", prompt, {"answer": node, "reference": node, "accept_any": [f"node {node}"], "target_service": service, "target_shard": shard, "target_node": node, "target_facts": [link1, link2], "positions": sizing["fact_positions"]}, meta)


def _lc3(rng: random.Random, cfg: RunConfig) -> core.TaskInstance:
    project = f"atlas-{_suffix(rng)}"
    queue = f"q-{rng.randrange(10, 99)}"
    broker = f"nats-{rng.randrange(2, 9)}"
    region = f"eu-west-{rng.randrange(2, 9)}"
    link1 = f"project {project} uses queue {queue}"
    link2 = f"queue {queue} is served by broker {broker}"
    link3 = f"broker {broker} runs in region {region}"
    question = f"Region брокера очереди project {project}? Только region."
    prompt, sizing = _targeted_records(rng, _target(cfg, 24576, 0.75), "project", [link1, link2, link3], [0.16, 0.45, 0.74], question, [link1, link2, link3], ratio=_ratio(cfg, "LC-3"))
    meta = _base_meta(cfg, _target(cfg, 24576, 0.75), sizing["estimated_tokens"], sizing["prompt_chars"])
    meta.update({k: sizing[k] for k in ("document_chars", "target_facts", "fact_positions")})
    return _task(
        "LC-3",
        prompt,
        {
            "answer": region,
            "reference": region,
            "accept_any": [f"region {region}"],
            "target_project": project,
            "target_queue": queue,
            "target_broker": broker,
            "target_facts": [link1, link2, link3],
            "positions": sizing["fact_positions"],
            "unit_whitelist": ("region",),
            "casefold": True,
            "comparison_note": "Region values use a case-insensitive region whitelist; IDs do not.",
        },
        meta,
    )


def _lc4(rng: random.Random, cfg: RunConfig) -> core.TaskInstance:
    service = f"aurora-{_suffix(rng)}"
    # Owners are drawn without replacement: a repeated team could make the
    # naive last-line answer equal the chronological one and abort the build
    # on an admissible draw (plan T4).
    while True:
        pool = [f"team-{rng.randrange(100, 999)}" for _ in range(4)]
        if len(set(pool)) == 4:
            teams = pool
            break
    # The final timestamp is intentionally not the final line in the document.
    events = [
        ("2026-04-11T09:15:00Z", teams[0]),
        ("2026-04-11T14:31:00Z", teams[1]),
        ("2026-04-11T16:08:00Z", teams[2]),
        ("2026-04-11T18:44:00Z", teams[3]),
    ]
    initial = f"service {service} owner = team-initial"
    event_lines = [f"event {stamp}: service {service} owner changes to {team}" for stamp, team in events]
    shuffled = event_lines[:]
    rng.shuffle(shuffled)
    if shuffled[-1] == event_lines[-1]:
        # Any non-final slot works: owners are distinct, so the swap always
        # separates the last line from the latest timestamp.
        swapped = next(index for index in range(len(shuffled) - 1) if shuffled[index] != event_lines[-1])
        shuffled[-1], shuffled[swapped] = shuffled[swapped], shuffled[-1]
    question = f"Текущий owner service {service} после изменений по времени? Только командой."
    # Separate event lines are chain links; the canonical final event is a
    # unique target fact, while all four events are needed to compute state.
    prompt, sizing = _targeted_records(
        rng,
        _target(cfg, 27648, 0.84),
        "event",
        [initial, *shuffled],
        [0.10, 0.24, 0.43, 0.63, 0.82],
        question,
        [initial, *event_lines],
        ratio=_ratio(cfg, "LC-4"),
    )
    owner_by_line = {line: team for line, (_, team) in zip(event_lines, events)}
    naive_last = owner_by_line[shuffled[-1]]
    # The final timestamp is not the final document line, so line order is
    # deliberately a different answer from chronological state.
    final_answer = events[-1][1]
    assert naive_last != final_answer
    assert len(event_lines) >= 3 and len(set(event_lines)) == len(event_lines)
    meta = _base_meta(cfg, _target(cfg, 27648, 0.84), sizing["estimated_tokens"], sizing["prompt_chars"])
    meta.update({k: sizing[k] for k in ("document_chars", "target_facts", "fact_positions")})
    return _task(
        "LC-4",
        prompt,
        {
            "answer": final_answer,
            "reference": final_answer,
            "target_entity": service,
            "events": [{"timestamp": stamp, "owner": team} for stamp, team in events],
            "target_facts": [initial, *event_lines],
            "positions": sizing["fact_positions"],
            "naive_last_line_answer": naive_last,
            "unit_whitelist": ("team",),
            "casefold": False,
            "comparison_note": "Ownership is case-sensitive and events must be ordered by timestamp.",
        },
        meta,
    )


def _lc5(rng: random.Random, cfg: RunConfig) -> core.TaskInstance:
    # Build siblings by small, programmatic edits rather than unrelated names.
    digits = list(f"{rng.randrange(100, 999):03d}")
    if digits[1] == digits[2]:
        digits[2] = str((int(digits[2]) + 1) % 10)
    target_incident = "I-" + "".join(digits)
    distractor_incident = "I-" + digits[0] + digits[2] + digits[1]
    node_digit = rng.randrange(2, 8)
    target_node = f"cobalt-{node_digit}"
    distractor_node = f"cobalt-{node_digit + 1}"
    shard_digits = f"{rng.randrange(10, 99):02d}"
    if shard_digits[0] == shard_digits[1]:
        shard_digits = shard_digits[0] + str((int(shard_digits[1]) + 1) % 10)
    target_shard = "S" + shard_digits
    distractor_shard = "S" + shard_digits[1] + shard_digits[0]
    target_service = rng.choice(("helios", "selene", "orion"))
    distractor_service = target_service[:-1]

    target_links = [
        f"incident {target_incident} affects node {target_node}",
        f"node {target_node} hosts shard {target_shard}",
        f"shard {target_shard} belongs to service {target_service}",
    ]
    distractor_links = [
        f"incident {distractor_incident} affects node {distractor_node}",
        f"node {distractor_node} hosts shard {distractor_shard}",
        f"shard {distractor_shard} belongs to service {distractor_service}",
    ]
    # Both chains are complete and plausible, but only the target IDs occur in
    # the question.  Keep their links in distinct quarters of the document.
    question = f"Какой service затронут {target_incident}? Только именем service."
    all_special = [target_links[0], distractor_links[0], target_links[1], distractor_links[1], target_links[2], distractor_links[2]]
    prompt, sizing = _targeted_records(rng, _target(cfg, 27648, 0.84), "incident", all_special, [0.10, 0.24, 0.40, 0.55, 0.70, 0.84], question, all_special, ratio=_ratio(cfg, "LC-5"))
    meta = _base_meta(cfg, _target(cfg, 27648, 0.84), sizing["estimated_tokens"], sizing["prompt_chars"])
    meta.update({k: sizing[k] for k in ("document_chars", "target_facts", "fact_positions")})
    return _task(
        "LC-5",
        prompt,
        {
            "answer": target_service,
            "reference": target_service,
            "accept_any": [f"service {target_service}"],
            "target_ids": [target_incident, target_node, target_shard, target_service],
            "distractors": [distractor_incident, distractor_node, distractor_shard, distractor_service],
            "distractor_answer": distractor_service,
            "target_facts": target_links,
            "distractor_facts": distractor_links,
            "positions": sizing["fact_positions"],
            "distractor_positions": {fact: sizing["fact_positions"][fact] for fact in distractor_links},
            "unit_whitelist": ("service",),
            "casefold": True,
            "comparison_note": "Service values use a documented case-insensitive whitelist; IDs remain exact.",
        },
        meta,
    )


# --------------------------------------------------------------------------- #
# State-machine scenarios (LC-6 pilot, LC-7 stretch)
# --------------------------------------------------------------------------- #

ASSIGNED = "ASSIGNED"
UNRESOLVED = "UNRESOLVED"

_STATUS_RE = re.compile(r"СТАТУС\s*[:=\-]?\s*([A-Za-zА-Яа-я_]+)", re.IGNORECASE)
_OWNER_RE = re.compile(r"OWNER\s*[:=\-]?\s*([A-Za-z][\w\-]*)", re.IGNORECASE)
_EVIDENCE_RE = re.compile(r"СОБЫТИЯ\s*[:=\-]?\s*(.+)$", re.IGNORECASE)
_EVENT_ID_RE = re.compile(r"EV-\d{3}")


def _single_line(reply: str) -> str | None:
    lines = text.answer_lines(reply)
    return lines[0] if len(lines) == 1 else None


def _parse_state_answer(reply: str) -> tuple[str, tuple[str, ...]] | None:
    """Parse ``СТАТУС <...>; СОБЫТИЯ <ids>``; the ID list keeps its order."""
    line = _single_line(reply)
    if line is None:
        return None
    status = _STATUS_RE.search(line)
    evidence = _EVIDENCE_RE.search(line)
    if status is None or evidence is None:
        return None
    ids = tuple(_EVENT_ID_RE.findall(evidence.group(1)))
    if not ids:
        return None
    return status.group(1).upper(), ids


def _parse_chain_answer(reply: str) -> tuple[str, tuple[str, ...]] | None:
    """Parse ``OWNER <entity>; СОБЫТИЯ <ids>``."""
    line = _single_line(reply)
    if line is None:
        return None
    owner = _OWNER_RE.search(line)
    evidence = _EVIDENCE_RE.search(line)
    if owner is None or evidence is None:
        return None
    ids = tuple(_EVENT_ID_RE.findall(evidence.group(1)))
    return (owner.group(1), ids) if ids else None


def _state_answer(status: str, evidence: list[str] | tuple[str, ...]) -> str:
    return f"СТАТУС {status}; СОБЫТИЯ {','.join(evidence)}"


def _chain_answer(entity: str, evidence: list[str] | tuple[str, ...]) -> str:
    return f"OWNER {entity}; СОБЫТИЯ {','.join(evidence)}"


def _stamp(day: str, hhmm: str) -> str:
    return f"{day}T{hhmm}:00Z"


def _active_by_version(events: list[dict[str, Any]], cutoff: str) -> list[dict[str, Any]]:
    """Events in force at ``cutoff``: apply-time order with the version rule.

    Only events up to the cutoff take part, and of several versions of one key
    only the highest one survives — a later line with a lower version is a
    correction, not a newer state, and a higher version after the cutoff has not
    happened yet at the snapshot. A retraction is not a version of the
    assignment: it stays in force and clears the key it names.
    """
    active = [event for event in events if event["ts"] <= cutoff]
    best: dict[str, int] = {}
    for event in active:
        if event["op"] == "assign":
            best[event["key"]] = max(best.get(event["key"], 0), int(event["version"]))
    kept = [
        event
        for event in active
        if event["op"] != "assign" or int(event["version"]) == best[event["key"]]
    ]
    return sorted(kept, key=lambda event: (event["ts"], event["id"]))


def replay_owner(events: list[dict[str, Any]], cutoff: str) -> tuple[str, tuple[str, ...]]:
    """Reference state machine: owner status plus the deciding event IDs."""
    owner: str | None = None
    owner_key: str | None = None
    evidence: list[str] = []
    for event in _active_by_version(events, cutoff):
        if event["op"] == "assign":
            owner = str(event["team"])
            owner_key = str(event["key"])
            evidence = [str(event["id"])]
        elif owner_key == event["key"]:
            # A retraction only clears the key it names; its evidence is the
            # assignment it revokes plus itself, in application order.
            owner = None
            owner_key = None
            evidence.append(str(event["id"]))
    return (ASSIGNED if owner else UNRESOLVED), tuple(evidence)


def _fold_owner(events: list[dict[str, Any]], cutoff: str) -> tuple[str, tuple[str, ...]]:
    """Independent recomputation of :func:`replay_owner` (seen-version fold)."""
    active = [event for event in events if event["ts"] <= cutoff]
    seen: dict[str, int] = {}
    order = sorted(active, key=lambda event: (event["ts"], event["id"]))
    owner: str | None = None
    owner_key: str | None = None
    evidence: list[str] = []
    for event in order:
        if event["op"] == "assign":
            if int(event["version"]) < seen.get(event["key"], 0):
                continue
            seen[event["key"]] = int(event["version"])
            owner, owner_key, evidence = str(event["team"]), str(event["key"]), [str(event["id"])]
        elif owner_key == event["key"]:
            owner, owner_key, evidence = None, None, [*evidence, str(event["id"])]
    return (ASSIGNED if owner else UNRESOLVED), tuple(evidence)


def _event_lines(service: str, events: list[dict[str, Any]]) -> list[str]:
    lines = []
    for event in events:
        if event["op"] == "retract":
            lines.append(f"event {event['id']} at {event['ts']}: service {service} retraction of key {event['key']}")
        else:
            lines.append(
                f"event {event['id']} at {event['ts']} v{event['version']}: "
                f"service {service} owner = {event['team']} key {event['key']}"
            )
    return lines


def _lc6(rng: random.Random, cfg: RunConfig) -> core.TaskInstance:
    """Owner on a cutoff date from assignments, corrections, a tombstone and versions."""
    service = f"kanopus-{_suffix(rng)}"
    teams: list[str] = []
    while len(teams) < 8:
        candidate = f"team-{rng.randrange(100, 1000)}"
        if candidate not in teams:
            teams.append(candidate)
    base = rng.randrange(101, 780)
    ids = [f"EV-{base + index:03d}" for index in range(10)]
    day = f"2026-05-{rng.randrange(10, 28):02d}"
    times = ("09:05", "09:40", "10:20", "11:00", "11:35", "12:10", "12:40", "13:15", "13:15", "14:00")
    events = [
        {"id": ids[0], "ts": _stamp(day, times[0]), "version": 1, "key": "K1", "op": "assign", "team": teams[0]},
        {"id": ids[1], "ts": _stamp(day, times[1]), "version": 1, "key": "K2", "op": "assign", "team": teams[1]},
        # A correction re-issues K1 with a higher version: the first line is void.
        {"id": ids[2], "ts": _stamp(day, times[2]), "version": 2, "key": "K1", "op": "assign", "team": teams[2]},
        {"id": ids[3], "ts": _stamp(day, times[3]), "version": 1, "key": "K1", "op": "retract"},
        # The surviving K3 version is the earlier line; the later one is lower.
        {"id": ids[4], "ts": _stamp(day, times[4]), "version": 2, "key": "K3", "op": "assign", "team": teams[3]},
        {"id": ids[5], "ts": _stamp(day, times[5]), "version": 1, "key": "K3", "op": "assign", "team": teams[4]},
        {"id": ids[6], "ts": _stamp(day, times[6]), "version": 1, "key": "K3", "op": "retract"},
        # Equal timestamps: the larger ID is applied later and wins.
        {"id": ids[7], "ts": _stamp(day, times[7]), "version": 1, "key": "K4", "op": "assign", "team": teams[5]},
        {"id": ids[8], "ts": _stamp(day, times[8]), "version": 1, "key": "K5", "op": "assign", "team": teams[6]},
        # Higher version after the cutoff: must not be used at the snapshot.
        {"id": ids[9], "ts": _stamp(day, times[9]), "version": 2, "key": "K4", "op": "assign", "team": teams[7]},
    ]
    cutoff = _stamp(day, rng.choice(("13:00", "13:30")))
    status, evidence = replay_owner(events, cutoff)
    assert _fold_owner(events, cutoff) == (status, evidence), "state machine disagrees with itself"
    answer = _state_answer(status, evidence)

    naive_no_tombstone = replay_owner([event for event in events if event["op"] != "retract"], cutoff)
    naive_no_cutoff = replay_owner(events, "9999")

    question = (
        f"Восстанови состояние owner службы {service} на момент CUTOFF.\n"
        "Правила: 1) действуют только события с ts <= CUTOFF; 2) из событий одного key "
        "действует версия с наибольшим v, версии с меньшим v исправлены и не применяются; "
        "3) применённые события упорядочиваются по (ts, id), при равных ts позже идёт событие "
        "с большим id; 4) assign задаёт owner, retraction очищает owner своего key.\n"
        "Ответь одной строкой: СТАТУС <ASSIGNED|UNRESOLVED>; СОБЫТИЯ <ID через запятую>. "
        "Для ASSIGNED перечисли применённое назначение, для UNRESOLVED — применённое "
        "назначение и его отзыв в порядке применения."
    )
    header = f"service {service} holds an event journal for {day}."
    cutoff_line = f"cutoff = {cutoff}"
    last_line = _event_lines(service, [events[-1]])[0]
    shuffled = events[:-1]
    rng.shuffle(shuffled)
    lines = _event_lines(service, shuffled) + [last_line]
    special = [header, cutoff_line, *lines]
    fractions = [0.02, 0.05, *sorted(rng.uniform(0.08, 0.90) for _ in range(9)), 0.94]
    target = _target(cfg, 20480, 0.625)
    prompt, sizing = _targeted_records(
        rng, target, "event", special, fractions, question, special, ratio=_ratio(cfg, "LC-6")
    )
    positions = sizing["fact_positions"]
    last_event = max(events, key=lambda event: positions[_event_lines(service, [event])[0]])
    # A naive reader takes the last event line as the current state.
    naive_last = (ASSIGNED, (last_event["id"],)) if last_event["op"] == "assign" else (UNRESOLVED, (last_event["id"],))
    meta = _base_meta(cfg, target, sizing["estimated_tokens"], sizing["prompt_chars"])
    meta.update({key: sizing[key] for key in ("document_chars", "target_facts", "fact_positions")})
    meta["cutoff"] = cutoff
    return _task(
        "LC-6",
        prompt,
        {
            "answer": answer,
            "reference": answer,
            "format": "state",
            "status": status,
            "evidence": list(evidence),
            "cutoff": cutoff,
            "target_entity": service,
            "events": [dict(event) for event in events],
            "naive_last_position_answer": _state_answer(*naive_last),
            "naive_ignore_tombstone_answer": _state_answer(*naive_no_tombstone),
            "naive_ignore_cutoff_answer": _state_answer(*naive_no_cutoff),
            "answer_without_ids": f"СТАТУС {status}",
            "target_facts": special,
            "positions": positions,
            "comparison_note": "Статус и ID событий сравниваются точно, порядок ID значим.",
        },
        meta,
    )


def _edges_in_force(edges: list[dict[str, Any]], cutoff: str) -> list[dict[str, Any]]:
    active = [edge for edge in edges if edge["ts"] <= cutoff]
    best: dict[str, int] = {}
    for edge in active:
        best[edge["key"]] = max(best.get(edge["key"], 0), int(edge["version"]))
    return [edge for edge in active if int(edge["version"]) == best[edge["key"]]]


def walk_chain(
    edges: list[dict[str, Any]], start: tuple[str, str], cutoff: str, hops: int = 4
) -> tuple[str, tuple[str, ...]]:
    """Follow exactly one edge per hop from ``start``; returns the last object and IDs."""
    active = _edges_in_force(edges, cutoff)
    current = start
    evidence: list[str] = []
    for _ in range(hops):
        matches = [edge for edge in active if edge["subject"] == current]
        if len(matches) != 1:
            raise AssertionError(f"hop from {current} matched {len(matches)} edges")
        evidence.append(str(matches[0]["id"]))
        current = matches[0]["object"]
    return current[1], tuple(evidence)


def _edge_line(edge: dict[str, Any]) -> str:
    _, value = edge["subject"]
    object_kind, object_value = edge["object"]
    body = {
        "node": f"incident {value} affects node {object_value}",
        "shard": f"node {value} hosts shard {object_value}",
        "service": f"shard {value} belongs to service {object_value}",
        "owner": f"service {value} owner = {object_value}",
    }[object_kind]
    return f"edge {edge['id']} at {edge['ts']} v{edge['version']} {edge['key']}: {body}"


def _lc7(rng: random.Random, cfg: RunConfig) -> core.TaskInstance:
    """Follow incident -> node -> shard -> service -> owner through three chains."""
    day = f"2026-06-{rng.randrange(10, 28):02d}"
    cutoff = _stamp(day, "14:00")

    def distinct(count: int, low: int, high: int) -> list[int]:
        values: list[int] = []
        while len(values) < count:
            value = rng.randrange(low, high)
            if value not in values:
                values.append(value)
        return values

    incident_ids = [f"IN-{value}" for value in distinct(3, 1000, 9000)]
    node_ids = [f"zircon-{value}" for value in distinct(2, 2, 40)]
    shard_ids = [f"S{value:02d}" for value in distinct(3, 10, 99)]
    services = ["hestia", "nereid", "iris"]
    teams = [f"team-{value}" for value in distinct(4, 100, 1000)]
    base = rng.randrange(101, 700)
    ids = [f"EV-{base + index:03d}" for index in range(12)]

    def edge(index: int, hhmm: str, version: int, key: str, subject: tuple[str, str], obj: tuple[str, str]) -> dict[str, Any]:
        return {"id": ids[index], "ts": _stamp(day, hhmm), "version": version, "key": key,
                "subject": subject, "object": obj}

    target_node, distractor_node = node_ids
    target_shard, distractor_shard, post_shard = shard_ids
    target_team = teams[0]
    distractor_team = teams[1]
    post_team = teams[2]
    edges = [
        # Target chain; the node -> shard link is versioned.
        edge(0, "10:05", 1, "IN1", ("incident", incident_ids[0]), ("node", target_node)),
        edge(1, "10:40", 1, "NODE1", ("node", target_node), ("shard", target_shard)),
        edge(2, "11:20", 2, "SHARD1", ("shard", target_shard), ("service", services[0])),
        edge(3, "12:05", 1, "SVC1", ("service", services[0]), ("owner", target_team)),
        # Conflicting lower version of the same shard link, printed later.
        edge(4, "12:50", 1, "SHARD1", ("shard", target_shard), ("service", services[1])),
        # A complete, plausible second chain that shares the shard wording.
        edge(5, "09:10", 1, "IN2", ("incident", incident_ids[1]), ("node", distractor_node)),
        edge(6, "09:45", 1, "NODE2", ("node", distractor_node), ("shard", distractor_shard)),
        edge(7, "10:30", 1, "SHARD2", ("shard", distractor_shard), ("service", services[1])),
        edge(8, "11:05", 1, "SVC2", ("service", services[1]), ("owner", distractor_team)),
        # Repointed node -> shard edge after the cutoff, with its own tail.
        edge(9, "15:10", 2, "NODE1", ("node", target_node), ("shard", post_shard)),
        edge(10, "15:40", 1, "SHARD3", ("shard", post_shard), ("service", services[2])),
        edge(11, "16:05", 1, "SVC3", ("service", services[2]), ("owner", post_team)),
    ]
    answer_owner, evidence = walk_chain(edges, ("incident", incident_ids[0]), cutoff)
    answer = _chain_answer(answer_owner, evidence)

    distract_owner, distract_evidence = walk_chain(edges, ("incident", incident_ids[1]), cutoff)
    post_owner, post_evidence = walk_chain(edges, ("incident", incident_ids[0]), "9999")
    # The last document line is the superseded lower version of the shard edge.
    tail_owner, tail_ids = walk_chain(edges, edges[4]["object"], cutoff, hops=1)
    naive_last = _chain_answer(tail_owner, (edges[4]["id"], *tail_ids))

    question = (
        f"По журналу связей построй путь incident -> node -> shard -> service -> owner "
        f"для incident {incident_ids[0]} на момент CUTOFF.\n"
        "Правила: 1) действуют только рёбра с ts <= CUTOFF; 2) для одного key действует "
        "только версия с наибольшим v; 3) каждый следующий шаг берётся по ребру, субъект "
        "которого равен текущей сущности.\n"
        "Ответь одной строкой: OWNER <team>; СОБЫТИЯ <ID рёбер в порядке прохода>."
    )
    header = f"graph snapshot for {day}."
    cutoff_line = f"cutoff = {cutoff}"
    ordered = [_edge_line(edge_item) for edge_item in edges]
    rng.shuffle(ordered)
    # Deliberately positional: a superseded lower version closes the document.
    closing = _edge_line(edges[4])
    ordered.remove(closing)
    ordered.append(closing)
    special = [header, cutoff_line, *ordered]
    fractions = [0.01, 0.03, *sorted(rng.uniform(0.06, 0.88) for _ in range(11)), 0.93]
    target = _target(cfg, rng.randrange(24_000, 28_001), rng.uniform(0.73, 0.86))
    prompt, sizing = _targeted_records(
        rng, target, "incident", special, fractions, question, special, ratio=_ratio(cfg, "LC-7")
    )
    meta = _base_meta(cfg, target, sizing["estimated_tokens"], sizing["prompt_chars"])
    meta.update({key: sizing[key] for key in ("document_chars", "target_facts", "fact_positions")})
    meta["cutoff"] = cutoff
    mixed = _chain_answer(distract_owner, distract_evidence)
    return _task(
        "LC-7",
        prompt,
        {
            "answer": answer,
            "reference": answer,
            "format": "chain",
            "target_incident": incident_ids[0],
            "target_node": target_node,
            "target_shard": target_shard,
            "target_service": services[0],
            "answer_entity": answer_owner,
            "evidence": list(evidence),
            "cutoff": cutoff,
            "edges": [dict(item) for item in edges],
            "distractor_incident": incident_ids[1],
            "distractor_entities": [incident_ids[1], distractor_node, distractor_shard, services[1], distractor_team],
            "distractor_answer": mixed,
            "naive_mixed_chain_answer": mixed,
            "naive_last_position_answer": naive_last,
            "naive_last_version_answer": _chain_answer(post_owner, post_evidence),
            "answer_without_ids": f"OWNER {answer_owner}",
            "target_facts": special,
            "positions": sizing["fact_positions"],
            "comparison_note": "Сущность и ID рёбер сравниваются точно, порядок ID значим.",
        },
        meta,
    )


def generate(rng: random.Random, cfg: RunConfig) -> list[core.TaskInstance]:
    """Generate the scenarios of the run's profile, deterministically."""
    profile = core.profile_of(cfg)
    if profile == "pilot":
        return [_lc6(rng, cfg)]
    tasks = [
        _lc1(rng, cfg),
        _lc2(rng, cfg),
        _lc3(rng, cfg),
        _lc4(rng, cfg),
        _lc5(rng, cfg),
    ]
    if profile == "stretch-v1":
        tasks.append(_lc6(rng, cfg))
        tasks.append(_lc7(rng, cfg))
    return tasks


def _context_overflow(error: str | None) -> bool:
    if not error:
        return False
    lowered = error.casefold()
    return any(
        marker in lowered
        for marker in (
            "context length",
            "context_length",
            "maximum context",
            "context window",
            "prompt is too long",
            "too many tokens",
            "token limit",
            "input is too long",
            "input too long",
        )
    )


def evaluate(task: core.TaskInstance, tr: core.Transcript) -> core.Score:
    """Score one all-or-nothing answer and report context admission separately."""
    score = core.Score(task_id=task.id, category=_CATEGORY)
    prompt_tokens = tr.prompt_tokens()
    required = task.meta.get("admission_required")
    if _context_overflow(tr.error):
        score.admission = False
        score.notes.append("Runtime rejected the input for exceeding its context limit.")
    elif required:
        score.admission = False
        score.notes.append(f"Long-context admission failed: target requires {required} estimated tokens.")
    elif prompt_tokens is None:
        score.admission = None
        score.notes.append("Actual input size was unverified because the runtime reported no prompt tokens.")
    else:
        window = int(task.meta.get("context_window", 0))
        max_new = int(task.meta.get("max_new_tokens", 0))
        target = int(task.meta.get("target_tokens", 0))
        if prompt_tokens + max_new > window:
            score.admission = False
            score.notes.append(
                f"Long-context admission failed: {prompt_tokens} prompt + {max_new} output tokens exceeds {window}."
            )
        elif target and prompt_tokens < 0.8 * target:
            # The document was sized by a characters-per-token estimate. When the
            # real tokenizer packs the filler denser, the scenario silently tests a
            # smaller context than it claims — that is not the same task, so it is
            # reported as an admission failure instead of a score.
            score.admission = False
            score.notes.append(
                f"Сценарий не дотянул до целевого размера: фактический вход {prompt_tokens} токенов "
                f"при цели {target}; калибровать chars_per_token по реальному токенизатору."
            )
        else:
            score.admission = True
            if target:
                score.notes.append(f"Фактический вход {prompt_tokens} при цели {target} токенов.")

    whitelist = tuple(task.oracle.get("unit_whitelist", ()))
    answer_format = task.oracle.get("format")
    if answer_format in ("state", "chain"):
        if answer_format == "state":
            parsed_state = _parse_state_answer(tr.final_text)
            expected = (str(task.oracle["status"]), tuple(str(value) for value in task.oracle["evidence"]))
            name = "точный статус и event IDs"
            shown = None if parsed_state is None else _state_answer(*parsed_state)
        else:
            parsed_state = _parse_chain_answer(tr.final_text)
            expected = (str(task.oracle["answer_entity"]),
                        tuple(str(value) for value in task.oracle["evidence"]))
            name = "точная сущность и event IDs"
            shown = None if parsed_state is None else _chain_answer(*parsed_state)
        passed = parsed_state is not None and parsed_state == expected
        reference_text = task.oracle["reference"]
        score.add(
            name,
            passed,
            _POINTS,
            f"ответ {shown!r} совпал с {reference_text!r}" if passed else f"получено {shown!r}, ожидалось {reference_text!r}",
        )
        return score

    parsed = text.answer_token(tr.final_text, whitelist)
    reference = str(task.oracle["answer"])
    # Accepted spellings are fixed in the oracle before the run: the question asks
    # for an entity, and whether the answer repeats the document's field label
    # ("sha256:", "node ") carries no information about retrieval. The distractor
    # answer and the naive last-line answer are deliberately NOT accepted.
    accepted = [reference, *(str(value) for value in task.oracle.get("accept_any", ()))]
    if task.oracle.get("casefold"):
        accepted = [value.casefold() for value in accepted]
        parsed = parsed.casefold() if parsed else None
    passed = parsed is not None and parsed in accepted
    score.add(
        "exact single-value answer",
        passed,
        _POINTS,
        f"ответ {parsed!r} совпал с {reference!r}" if passed else f"получено {parsed!r}, ожидалось {reference!r}",
    )
    return score


BASELINE_IDS = ("LC-1", "LC-2", "LC-3", "LC-4", "LC-5")
PILOT_IDS = ("LC-6",)
ALL_IDS = BASELINE_IDS + ("LC-6", "LC-7")


core.register(
    key="E",
    title="Long context",
    points=10.0,
    generate=generate,
    evaluate=evaluate,
    expected_ids=BASELINE_IDS,
    expected_ids_by_profile={"pilot": PILOT_IDS, "stretch-v1": ALL_IDS},
)
