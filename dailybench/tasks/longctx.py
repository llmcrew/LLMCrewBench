"""Category E: generated long-context retrieval and state tracking tasks.

The fixtures deliberately use character sizing rather than a tokenizer.  The
conservative ``CHARS_PER_TOKEN`` estimate is also recorded in each task so a
run can be audited against the runtime's measured prompt length.
"""

from __future__ import annotations

import hashlib
import random
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
    teams = [f"team-{rng.randrange(100, 999)}" for _ in range(4)]
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
        shuffled[-1], shuffled[-2] = shuffled[-2], shuffled[-1]
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


def generate(rng: random.Random, cfg: RunConfig) -> list[core.TaskInstance]:
    """Generate all five deterministic scenarios from the supplied RNG."""
    return [
        _lc1(rng, cfg),
        _lc2(rng, cfg),
        _lc3(rng, cfg),
        _lc4(rng, cfg),
        _lc5(rng, cfg),
    ]


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


core.register(
    key="E",
    title="Long context",
    points=10.0,
    generate=generate,
    evaluate=evaluate,
    expected_ids=("LC-1", "LC-2", "LC-3", "LC-4", "LC-5"),
)
