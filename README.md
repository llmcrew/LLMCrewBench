# LLMCrew DailyBench

A deterministic, programmatically scored benchmark for **local** language models
run under a 16 GiB VRAM budget (internally: *DailyBench-16*).

Ten scored categories: the frozen baseline has 47 A–J tasks and 100 points,
plus two **visual blocks** — 20 simulation scenarios and 29 free-form art works
(36 and 39 in the `stretch-v1` profile) — measured in a headless browser and
shown outside the score. Every point comes from an observable result — a parsed answer, a file on
disk, a hidden test run, a tool-call log, a rendered frame — and never from a
judge model. The whole suite is regenerated from a single seed, so two runs of
two different quantizations face byte-identical tasks.

```
make setup                       # .venv + pytest (uv)
make agent                       # pi — the agent runtime for C/K/L
make selfcheck                   # no GPU: generators and evaluators must agree
python -m dailybench bench --config qwen3.8-27b-ud-q3-k-xl --speed
python -m dailybench report runs/qwen3.8-27b-ud-q3-k-xl runs/qwen3.8-27b-ud-iq3-s
```

## What the leaderboard compares

One row per *configuration* (model + quantization + runtime + KV type + context
window), not per model name: the same weights at `Q3_K_XL` and `IQ2_S` are
different results, and the report says which one was measured, on which GPU,
with which launch line, and with the SHA-256 of the weights.

Quality and speed never mix: quality is the point total, speed is a separate
profile (TTFT, prefill and generation throughput, peak VRAM at a filled
context). A configuration that does not fit or fails the long-context admission
is reported as such — an unverified result is never folded into the score as a
zero.

## Categories

| Key | Category | Points | Tasks | What is actually checked |
|---|---|---:|---:|---|
| A | Instruction following | 10 | 5 | Output format, exact content and ordering, parsed from the reply |
| B | Reasoning | 15 | 15 | One exact value per task (integers, fractions, orders, makespan) |
| C | Coding | 20 | 3 | Built by the agent in a working copy; hidden tests decide: Python repo patch, Go concurrency, SQL result sets |
| D | Tool calling | 14 | 7 | Tool-call log, file diffs after the run, no destructive action |
| E | Long context | 10 | 5 | Retrieval and state tracking over documents sized to 25–84 % of the configured window, admission measured |
| F | Hallucination resistance | 8 | 1 | 10 questions: 5 answerable, 5 not — each invention loses that question's points |
| G | Structured output | 8 | 4 | JSON-Schema conformance (types, enums, bounds, no extra keys) |
| H | Multi-turn dialogue | 6 | 3 | A correction applied mid-conversation, earlier constraints kept |
| I | Retrieval over files | 4 | 2 | Aggregation and conflict resolution over a multi-file workspace |
| J | Code review | 5 | 2 | The planted defect localised, and the fix passes hidden tests |

Every category is scored 0–100 % inside the run report, and the total is the sum
of the earned points (100 possible in the baseline).

The counts above are the baseline. `stretch-v1` adds the ids listed in
*Profiles*, in `build_suite` order, without changing the 100 A–J points: the
category total is divided evenly over the category's tasks
(`suite.apply_profile_weights`, see *Profiles*).

## Profiles

`core.PROFILES` accepts four profiles. The profile is written to `config.json`
and `suite.json`, and `resume`/`rescore` refuse to mix them.

| Profile | What it contains |
|---|---|
| `baseline-20261020` (default) | Frozen ranked suite: 47 A–J tasks, 100 points, K-1…K-20, L-1…L-29 |
| `pilot` | Only the scenarios not admitted to the ranking: IF-6, R-16, C-4, T-8, LC-6, F-2, G-5, H-4, I-3, J-3, visual hosts K-21, K-23, K-25 and L-1…L-29 — 42 tasks |
| `stretch-v1` | Baseline plus every id `build_suite` adds, in suite order: IF-6, IF-7, IF-7-KK, IF-7-EN, IF-8, IF-9, IF-10, R-16, R-17, R-17-KK, R-18, C-4, C-5, C-6, N-2, C-7, N-9, T-8, T-9, T-10, T-11, LC-6, LC-7, LC-8, F-2, F-3, G-5, H-4, H-5, I-3, I-4, I-5, J-3, J-4, J-5, K-21…K-36, L-30…L-39 — 82 A–J, 36 K and 39 L tasks; at least a fifth of every A–J category is private |
| `research-v1` | Ten text research scenarios (IF-9, IF-10, N-2, T-10, T-11, T-12, LC-8, F-3, I-5, J-5) plus the VQA track V-1…V-6 (6 informational points, labelled separately from K, outside the 71). Seven scored categories (A, C, D, E, F, I, J), 71 points; B, G, H, K and L are absent, so the profile is not ranked against a baseline run |

`pilot` is not collapsed into a new `/100`: A–J tasks keep the per-task weights
their modules declare, and each task is read on its own. It is not ranked against a baseline run.
`stretch-v1` keeps the
A–J total at 100 and only enlarges the visual showcase (K 144 points, L 39,
informational). `research-v1` declares only the categories it carries, so its
denominator is 71 and V's 6 informational points stay outside the 71. The report
marks `pilot` and `research-v1` as not rankable with the baseline.
`apply_profile_weights` spreads the declared category total over
that category's tasks, and the last task takes the rounding tail, so the sum is
exactly the declared one. Weights are assigned before the `--only` filter, so a slice keeps
the weight it has in the complete profile.

Each profile is checked against its exact id list
(`CategoryModule.expected_ids_by_profile`, `tasks/visual.PILOT_IDS`): a module
that generates another list fails the run instead of shrinking a category.

## Result states

Every score row has `status` = `ok`, `skipped` or `error`. Skip reasons stay
apart: `not_run` (the task is not in the run), `model_failed` (the task ran, but
no page was delivered or it died before a frame), `evaluator_unavailable` (no
browser, Go toolchain or similar on this machine), `capture_failed` (the page is
alive, the recording broke), `media_missing` (the clip is not on disk). An `ok`
row can still be partial.

Admission is a separate axis: `unverified` means the runtime never reported the
input size — the full numerator and denominator are kept and the verdict says so
instead of counting a guessed zero. Only `skipped` leaves both numerator and
denominator; admission `failed` reports the run out of the short total.

A declared ceiling — `--request-wall-clock` (default 1800 seconds) or the task
wall clock, including a socket timeout that reaches that ceiling — is
`resource_exhausted`: 0 points, the task stays in the denominator, and the run
stays rankable. `resume` inherits the saved `request_wall_clock` unless the flag
is passed again. An outage that is not that ceiling (connection refused, server
reset) stays `infra_failure` and is not ranked. `--retry-errors` re-runs only
`infra_failure` and `evaluator_failure`. `--allow-suite-drift` keeps every row
whose task bytes did not change and appends the decision to `resume_log.jsonl`.

Windows, Linux and macOS are supported (CI includes `windows-latest`; the
Makefile selects `.venv/Scripts/python.exe` on Windows). Agent categories run
in a sandbox when Docker is available: a per-run internal network, egress only
through a forwarder to the configured endpoint, images pinned by id. The host
gateway is not given to the agent. A comparison ranks a run only when its
contract key matches (`suite_sha256:generation_mode`); a run with no recorded
mode is unranked («режим генерации не записан»).

## Evidence layers

Three artefacts are never mixed:

* **original** — the answer page exactly as scored (the agent's `index.html`, or
  the fenced block in a chat reply) and the frame the checks scored. The frame
  hash is SHA-256 over the bytes on disk and is recomputed on every cache read;
* **display** — the report's `pages/<id>.html`: the original plus the
  `__dbscene` step hook, rewritten `./vendor/three.module.js` imports against a
  standalone `three.global.js`, a caption and, for K, zoom-to-fit. For L a
  fresh capture is *a* frame of the same work, not necessarily the byte-identical
  scored frame;
* **capture** — video clips: JPEG q90 frames encoded as H.264 CRF 21 /
  yuv420p. Lossy by construction, so a clip never stands for the scored frame;
  it sits next to its status marker and its reason.

Only the scored frame carries a content hash (`shot_sha`, SHA-256 over the file
bytes); display pages and clips are derived and identified by path. A display
page or a preview never substitutes the original evidence.

## The answer limit is the window, not a laboratory ceiling

New runs against an endpoint default to **262,144 tokens (256K)**, including the
Pi context for C/K/L. `--context-window` selects another size. The endpoint must
actually support and be configured for that window: this flag does not resize
an externally managed server.

An explicit `--config` keeps the window declared by that reference entry;
existing 32K/128K hardware profiles are not silently upgraded. The `generic-gguf`
template defaults to 256K. For a new reference-based comparison, declare the new
window in its entry and verify the loaded server with `preflight --full`.
Saved results are unchanged; use the original `--context-window` when resuming
an older endpoint run rather than mixing 32K and 256K measurements.

The answer limit is the model's window minus the prompt (`unbounded_budget`);
`--max-new-tokens` is an upper bound on top of it, the smaller of the two goes
out, and the budget actually sent — with its reason — is written to the
transcript. The 256-token floor is reported instead of passing for measured
room. A model that spent 100 KB on thinking and never wrote a line is a
*result*, not our ceiling — and the spend is documented in mandatory fields
(`spend` in every transcript: answer tokens, reasoning tokens, seconds, TTFT,
throughput). `reasoning_effort` is not forced: the stand keeps its own default,
and that is part of the provenance; the thinking cap is pinned by
`--reasoning-budget` (2048), and `--generation-mode native` hands sampling and
thinking back to the server. Both the mode and the whole contract that went out
are recorded. The old table-driven mode stays available with
`unbounded_budget = False`.

## Artefacts are built by an agent, not dictated into a chat

The three categories that produce a *file* (C coding, K/L visual pages) are not
solved by sending one prompt and parsing the reply. They are solved by
[`pi`](https://pi.dev), a coding agent, working in a copy of the task's
repository: it reads the files, edits them, runs the project's own tests, and the
**artefact is graded from disk** — a model that "answered correctly" without
writing the file has not finished the job.

The chat path stays for the narrative categories (A, B, D, E, F, G, H, I, J),
where one prompt and one parsed answer is the honest shape of the task, and is
still available for C/K/L with `--no-agent` — so the two harnesses can be
compared on the same suite.

Harness invariants (identical for every model, asserted by tests):

- **pinned agent**: exactly `pi 1.0.0` (`npm install -g @earendil-works/pi-coding-agent@1.0.0`),
  tools `read,bash,edit,write`; every start runs `pi --version` and a different release is
  refused with that install line instead of being driven with an argv it may not parse
  (0.80.10 has no `--` separator and died before the first request), and the arrived
  version plus the binary's sha256 are written to the transcript and the run identity;
- **isolated from the machine**: `--no-session` (no state between tasks),
  `--no-context-files` (this repo's `AGENTS.md` must not write itself into the
  prompt), `--no-extensions --no-skills --no-prompt-templates` (a `.pi/` a model
  creates cannot change the harness), `--offline` (no catalog refresh,
  no telemetry), `--no-approve`;
- **one endpoint, one key**: a throwaway `PI_CODING_AGENT_DIR` with a generated
  `models.json` holding just this run's endpoint, the benchmark's `temperature=0`
  and `top_p=1`, and the category's answer ceiling;
- **no hidden give-away**: the working copy holds exactly the fixture (plus the
  pinned `vendor/three.js` for K/L); the agent's own `AGENTS.md` only states where
  the library is and which file is graded;
- **a toolchain the agent can reach**: `PATH` is extended with the Go SDK the
  benchmark itself locates, so an agent cannot be measured on worse information
  than the evaluator uses;
- **the agent's own report is the provenance**: JSON event stream → tool calls,
  turns, token spend; a crashed agent keeps its events and its stderr tail
  instead of being scored as an empty directory.
- **empty stdin**: `pi` reads stdin when it is a pipe and waits for EOF, so the
  run hands it `DEVNULL` — otherwise a run started from a shell pipeline hangs
  for the whole wall clock without one event;
- **the answer ceiling is the window-aware budget**, not the chat table: an agent
  spends it on thinking *and* a tool call in the same response, and a chat-sized
  8192 was reached (8012 tokens) with nothing written yet;
- **no survivor outlives its task**: on a timeout the whole process tree dies
  (group + parent-chain walk, because `pi`'s tool shells detach), and after
  every run a sweep kills whatever still has the task's working copy as its cwd
  — a model's own `python -m http.server 8770 &` self-check outlives a *clean*
  exit, keeps the hard-coded port, and the next task's identical script then
  hangs on the occupied port for the full wall clock. The same sweep sits on
  `atexit`: an exit that skips `finally` (a debugger, `Py_Exit`) never unwinds
  the stack, and without the hook the survivor outlives the whole run.

A missing `pi` is a configuration error, not an unverified result: the run
installs the pinned release on first use, `make agent` does it up front, and
`dailybench doctor` reports where it is (or the command that would install it).
A wrong version is the same kind of error — it names the expected release and the
install command rather than scoring an empty working copy. `DAILYBENCH_AGENT_BIN`
points the run at a specific binary; it is probed the same way.

## Visual blocks (measured, outside the score)

| Key | Block | Points (baseline / stretch) | Tasks (baseline / stretch) | What is actually checked |
|---|---|---:|---:|---|
| K | Visual simulations (Three.js r186) | 80 / 144 | 20 / 36 | The agent writes `index.html`; the page runs in headless Chromium, reports its scene through `window.__probe()`, advances exactly by `window.__step(dt)`, and renders a reproducible 800×600 frame |
| L | Free-form art | 29 / 39 (informational) | 29 / 39 | The agent writes `index.html`; `L-1` is fully free, the other 28 (38 in stretch, up to `L-39`) name only a subject, and the model picks composition, style, palette and means (HTML/CSS, SVG, Canvas, WebGL; the vendored Three.js is available but optional). The single check of 1 point is technical delivery: the page loads without external requests or errors and a 1920×1080 frame is captured |

Each scenario asks one question: does the model produce a *working* page —
ballistics, pendulums, collision boxes, particle fountains, boids, heat
diffusion, cellular automata, terrain, a 3D bar chart, an orrery, a two-link
robot arm? Four checks of one point (html, scene, simulation, image) are taken
from the browser: the probe payload, the state after a stated number of steps
recomputed independently in Python, and the frame itself.

K is 80 points on K-1…K-20 in the baseline, 12 points on the three hosts
K-21/K-23/K-25 in `pilot`, and 144 points on K-1…K-36 in `stretch-v1`. L earns
1 point per work for delivery only, so its total is 29 or 39 and stays
informational: artistic quality is judged by people from the gallery frames.

Both blocks are deliberately kept out of the verdict: a browser build on the
measurement machine must not move a quality score. The frames are copied into
the run directory and collected into a `visual.html` gallery, so the leaderboard
has a screen that shows what a model actually drew.

The art block asks for something different from the simulation block: one work
is fully free, the other 28 name only a subject, and the model chooses the
idea, style, composition and means. The prompt prescribes no palette, no
formulas, no frame reproducibility and no mandatory hooks (`__probe`,
`__step`, `__ready`); a static page is acceptable and animation is optional.
The single automatic check is technical delivery (the page loads, the frame is
captured); taste is never scored by heuristics. Authoring rules:
`docs/art-contract.md`.

The library is pinned and vendored (`vendor/three/`, r186, checksums in
`vendor/three/PROVENANCE.md`); answers may import only
`./vendor/three.module.js` and any request that leaves loopback is refused and
reported. The block needs `playwright` + Chromium (`make browser` installs both);
without it the block is reported as unverified rather than failing. Authoring rules for new scenarios:
`docs/visual-contract.md`.

## Fairness rules (identical for every model)

- **Decoding**: `temperature=0`, `top_p=1`, per-task seed derived from the master
  seed, identical answer budgets (`4096` tokens; `8192` for the category that
  must write out a whole artefact), identical total-generation and tool-call
  ceilings.
- **Fixtures**: generated from `(master_seed, task_id)`, never shared between
  tasks; `suite.json` publishes a SHA-256 of every prompt so a published run can
  be tied to the exact suite, plus the per-task weights and the evaluator
  contract (a bumped `core.EVALUATOR_CONTRACT` or a changed evaluator module is
  a different measurement).
- **Long context is measured, not estimated**: the documents are sized against
  the real tokenizer at run start (`calibrate_chars_per_token`), and a document
  that came out smaller than its target, or an input that does not fit the
  window, is reported as an *admission failure* — never scored as a quality
  result.
- **Not verified is not a zero**: a category the configuration cannot do (a
  template without tool support), or a scenario that cannot be verified on this
  machine (no Go toolchain), leaves both the numerator and the denominator.
- **The probe is not the oracle**: `scene` and `simulation` compare
  `window.__probe()` — numbers the page itself reports — against an independent
  Python recomputation of the prompt's scheme. Values are rounded to 4 decimals;
  `visual.near` defaults to `1e-3`, integer counters and steps use `0`, and a
  few quantities are stricter (`1e-4`/`1e-6`), so a tolerance follows from
  rounding and cross-language accumulation rather than a copied constant. Image
  checks sample the largest visible canvas sparsely (`PIXEL_PROBE`), which does
  not prove that canvas dominates the screenshot. A page that echoes expected
  probe numbers without simulating still passes; the audit records it as a model
  error because the contract asks for actual values.
- **Provenance**: weights path and SHA-256, runtime build, quant, KV type,
  context window, offload mode and the exact server launch line are recorded in
  the run directory.

## Commands

| Command | Purpose |
|---|---|
| `dailybench setup` (Make target) | create `.venv`, install pytest |
| `dailybench selfcheck` | replay every task with a perfect answer and with garbage; both must land where expected. No GPU needed |
| `dailybench list` | show the generated suite (ids, points, sizes) |
| `dailybench generate --out suite/<seed>` | write `prompts.json` (publishable) and `oracles.json` (answers); `--profile` picks the suite version |
| `dailybench doctor` | GPU, toolchain, agent runtime, endpoint and tokenizer reachability |
| `dailybench preflight --config <id> --full` | verify a running runtime against a reference configuration |
| `dailybench visual <page.html> --steps N` | render one page exactly as the visual checks do (probe, frame, frame statistics) |
| `dailybench bench --config <id> --speed` | bring the runtime up, preflight, run the suite, write `report.md` + `report.html` (`--no-html` skips the page) |
| `dailybench run --base-url … --model …` | the same suite against an endpoint you started yourself (`--no-agent` keeps C/K/L on the chat path); results hit the disk after every task, so a crash at task 90 keeps the first 89. `--resume` finishes an interrupted run in place: saved tasks are not re-asked, truncated ones (timeout, reboot) are re-run, the calibration is taken from the saved `config.json`, the suite weights, the total denominator and the full run identity (config, budgets, agent mode, harness version) are verified, and only a checkpoint generation whose every file still hashes to its manifest is read |
| `dailybench report <dirs…>` | re-render `report.md`, one `<run>/report.html` per run, `comparison.md` and `comparison.html` (`--no-html` skips the pages, `--compare` picks the models) |
| `dailybench rescore <dir>` | re-score saved transcripts with the current evaluators — no model call; a fixture that drifted keeps its old score and is flagged (a legacy `suite.json` without per-task rows is compared by whole-suite hash) |
| `scripts/sweep.py` | load every reference configuration in turn and run the whole suite |

`--max-total-tokens` defaults to the larger of `--max-new-tokens` and 16384;
set it explicitly for thinking models with a big answer budget.

`--profile baseline-20261020` is the frozen ranked suite (47 A–J, 100 points).
`pilot` shows only the scenarios not admitted to the ranking (see *Profiles*) and
is not collapsed into a new `/100`; its A–J tasks keep the module weights.
`stretch-v1` adds scenarios on top of the baseline while keeping 100 A–J points:
a category's weight is split evenly across its tasks, the showcase grows to
K 144 / L 39. A runtime that never reports the input size leaves the admission
`unverified` instead of failing it. The profile is written to `config.json` and
`suite.json`; `resume` and `rescore` refuse to mix profiles. Every checkpoint also carries the full run identity (config,
budgets, agent mode, suite digest, harness version) and `resume` rejects a
directory written by another run; mirror records are only trusted when they
belong to that same identity and a master checkpoint exists.

## The report

`report` writes two flavours of the same page, from one renderer:

* **`<run>/report.html`** — one model's full report, next to its own `report.md`
  (a run writes it automatically; `--no-html` skips it). Its own `_vendor` and
  `_showcase` sit beside it, so the run directory is self-contained and can be
  served, zipped or published as is. Extra tab: **Провенанс** — weights hash, window,
  budgets, agent harness, suite hash.
* **`comparison.html`** — the field of models, with `--compare label,label` to
  pick exactly which runs to compare (label, model id or directory name; order is
  kept).

Both are offline (plotly and three.js vendored into `_vendor` and `_showcase`, so
nothing is fetched from the network). Tabs of the comparison page:

* **Решения моделей** — who scored what, model cards with per-category losses,
  and the documented spend (answer tokens, reasoning tokens, seconds);
* **Решения моделей по сьюту** — a task × model matrix: one row per suite task,
  one column per model, colour = how fully it was solved, tooltip = the failing
  check. Filters: scored only, showcase only, losses only;
* **Графики** — plotly: quality per category, K/L side by side, token spend split
  into answer and reasoning, cost of the result (score per minute), reasoning
  against quality;
* **Витрина** — every K/L frame; a click opens the **live page the model wrote**
  in the same pinned library, next to the frame. «Режим съёмки» opens one scene
  full screen, ←/→ walk the tasks, space advances the simulation, Esc closes;
  «сравнить выбранные» puts several models' pages side by side and steps them
  together.

ES modules do not load over `file://`, so for the live scenes serve the folder:

```bash
make serve DIR=runs        # http://127.0.0.1:8765/comparison.html
```

## From a run to a leaderboard

1. `scripts/sweep.py` walks `configs/reference.json`, loads each entry through the
   runtime's own API, runs `preflight --full`, then the suite, and keeps
   `runs/sweep-summary.json`.
2. `dailybench report runs/*` renders one `report.md` per run and one
   `comparison.md` with the quality verdict, the per-category percentages, the
   long-prompt TTFT, generation throughput and the peak VRAM side by side.
3. Ranking uses the verdict (`earned/possible`), which already excludes skipped
   and non-admitted work; the per-category columns say *where* a configuration
   wins rather than only how much, and the last column plus the `visual/` gallery
   say what it drew.

Each run directory stays self-describing: `config.json`, `suite.json`,
`provider.json`, `hardware.json`, `scores.json`, `tasks.json` and
`transcripts/<task>.json`, so a published number can be reproduced or
re-scored without touching a GPU.

## Published artefacts and the recording framework

`docs/ARTIFACTS.md` lists what is intentionally tracked (one published run and
its comparison) and what stays local — recordings and the video archive
included — with the commands to verify a restored archive. `docs/recording/` is a standalone kit for
capturing live web scenes: a step-driven recorder, a batch driver, the page
driver, the practices and the pitfalls that cost real time here.

## Adding a category

A category is one module in `dailybench/tasks/` that registers itself:

```python
IDS = ("K-1", "K-2")

def generate(rng, cfg):        # deterministic in rng only; no clock, no network
    ...

def evaluate(task, tr):        # scores observable results, never the transcript shape
    ...

core.register(key="K", title="…", points=…, generate=generate, evaluate=evaluate, expected_ids=IDS)
```

Rules the suite enforces (see `dailybench/core.py` for the full contract):

- `generate` is deterministic in `rng` and must assert every claim its fixtures
  make (the designed distractor really differs from the answer, the answer is
  unique, the document really contains the facts).
- The sum of the checks of a task equals its `max_points`.
- The reference answer scores full points, and the garbage answer scores zero —
  `dailybench selfcheck` fails the build otherwise.
- A scenario reaches the baseline only through `expected_ids`; new ones are
  declared in `IDS_BY_PROFILE` / `expected_ids_by_profile`, so `pilot`,
  `stretch-v1` and `research-v1` are checked against their exact lists — an
  empty list for a category the profile drops.
- New modules are added to `dailybench/tasks/__init__.py`; the category total
  goes into `CATEGORY_POINTS`, or into `SHOWCASE_POINTS` for a showcase block
  (the registry refuses a mismatch).

## Development

```
make test     # pytest
make cov      # pytest --cov=dailybench
make selfcheck
```

Requires Python ≥ 3.10 and no third-party runtime dependencies. The tests need
`pytest` only (`make setup`). Categories C, K and L need `pi` (`make agent`) and
the visual ones additionally need `playwright` + Chromium (`make browser`);
a chat-only run works without either, with `--no-agent`. `nvidia-smi`, `go` and
Docker are optional: the harness detects what the machine has and reports what it
cannot verify instead of guessing — and extends the agent's `PATH` with the
toolchain it did find, so the agent is not measured on worse information than the
evaluator.

The BigCodeBench adapter's `--evaluate-cmd` is split POSIX-style (`shlex.split`).
On Windows, quote paths that contain spaces.


## What stays local (endpoints, keys, machine paths)

The repository is public; the measurement machine is not. Nothing machine-specific
is required to be committed, and nothing private has to be edited out before a
push:

| Thing | Where it lives | Override |
|---|---|---|
| API key | environment only, never a tracked value | `export DAILYBENCH_API_KEY=sk-…` (default `not-needed`) |
| Endpoint and model | tracked default is `127.0.0.1:1234/v1` + `local-model` | `export DAILYBENCH_BASE_URL=…`, `export DAILYBENCH_MODEL=…` |
| Reference configs (absolute weight paths, ports, tags) | `configs/reference.json` (tracked, 13 entries) | `export DAILYBENCH_REFERENCE=configs/reference.local.json` — that file is git-ignored |
| Run output, server logs, sweep summary | `runs/` | git-ignored |
| Rendered frame cache (versioned key) | `~/.cache/dailybench/visual` | `DAILYBENCH_VISUAL_CACHE` |
| Agent binary | resolved from `PATH` and the usual prefixes | `export DAILYBENCH_AGENT_BIN=/path/to/pi` |

`configs/reference.json` ships as a working example *and* as the fallback table
inside `dailybench/reference.py`; a test fails if the two ever drift. Absolute
paths (`C:\\Users\\…`, `/Users/…`) are allowed in exactly those two places — a
test enumerates every other file and fails on a third copy.

A clone therefore runs end to end after two exports:

```bash
export DAILYBENCH_BASE_URL=http://127.0.0.1:8888/v1
export DAILYBENCH_API_KEY=sk-local
python -m dailybench bench --config qwen3.8-27b-ud-q3-k-xl
```

`dailybench doctor` prints the resolved endpoint, so what the run will talk to is
visible before it starts. The key itself is neither printed nor written to
`runs/<name>/config.json`: the config keeps only the `api_key_set` marker, and a
re-score or resume resolves the value from the environment again.

## Where the numbers come from

- `configs/reference.json` holds the policy (decoding, budgets, window, KV type,
  prompt-cache and RoPE rules) and one entry per measured configuration; the
  exact launch line for each runtime template is part of the report.
- `suite/<seed>/` holds the generated suite and its digest; `oracles.json`
  contains the answers and is deliberately kept out of the publishable
  `prompts.json`.
