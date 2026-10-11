# LLMCrew DailyBench

A deterministic, programmatically scored benchmark for **local** language models
run under a 16 GiB VRAM budget (internally: *DailyBench-16*).

Ten scored categories, 47 tasks, 100 points, plus two **visual blocks** — 20
simulation scenarios and 24 art scenarios on Three.js — measured in a headless
browser and shown outside the score. Every point comes from an observable result — a parsed answer, a file on
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
| E | Long context | 10 | 5 | One value retrieved from a 8K–28K-token document, admission measured |
| F | Hallucination resistance | 8 | 1 | 10 questions: 5 answerable, 5 not — inventing one scores zero |
| G | Structured output | 8 | 4 | JSON-Schema conformance (types, enums, bounds, no extra keys) |
| H | Multi-turn dialogue | 6 | 3 | A correction applied mid-conversation, earlier constraints kept |
| I | Retrieval over files | 4 | 2 | Aggregation and conflict resolution over a multi-file workspace |
| J | Code review | 5 | 2 | The planted defect localised, and the fix passes hidden tests |

Every category is scored 0–100 % inside the run report, and the total is the sum
of the earned points (100 possible).

## The answer limit is the window, not a laboratory ceiling

A run does not cap generation with a separate budget: the limit is the model's
window minus the prompt (`unbounded_budget`). A model that spent 100 KB on
thinking and never wrote a line is a *result*, not our ceiling — and the spend is
documented in mandatory fields (`spend` in every transcript: answer tokens,
reasoning tokens, seconds, TTFT, throughput). `reasoning_effort` is not forced:
the stand keeps its own default, and that is part of the provenance. The old
table-driven mode stays available with `unbounded_budget = False`.

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

- **pinned agent**: `pi 1.0.0`, tools `read,bash,edit,write`;
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
installs it on first use, `make agent` does it up front, and `dailybench doctor`
reports where it is (or the command that would install it). `DAILYBENCH_AGENT_BIN`
points the run at a specific binary.

## Visual blocks (measured, outside the score)

| Key | Block | Tasks | What is actually checked |
|---|---|---:|---|
| K | Visual simulations (Three.js r186) | 20 | The agent writes `index.html`; the page runs in headless Chromium, reports its scene through `window.__probe()`, advances exactly by `window.__step(dt)`, and renders a reproducible 800×600 frame |
| L | Art visualisations + LoFi Rain Sci-Fi (Three.js r186) | 24 | Same render loop (agent writes the file), scored on the *picture*: declared palette coverage (hue-first), composition (horizon, gradient, symmetry, radial falloff), detail (edges, colours — not a flat fill), motion (frames at two moments differ, each load reproduces its own) |

Each scenario asks one question: does the model produce a *working* page —
ballistics, pendulums, collision boxes, particle fountains, boids, heat
diffusion, cellular automata, terrain, a 3D bar chart, an orrery, a two-link
robot arm? Four checks of one point (html, scene, simulation, image) are taken
from the browser: the probe payload, the state after a stated number of steps
recomputed independently in Python, and the frame itself.

Both blocks are deliberately kept out of the verdict: a browser build on the
measurement machine must not move a quality score. The frames are copied into
the run directory and collected into a `visual.html` gallery, so the leaderboard
has a screen that shows what a model actually drew.

The art block asks for something different from the simulation block: the prompt
declares a palette, a composition and a motion, the checks measure them in the
frame, and the reference pages are the author's own renders. Palette is checked
strictly (a wrong hue loses the point), composition loosely (taste is not
punished). Authoring rules: `docs/art-contract.md`.

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
  be tied to the exact suite.
- **Long context is measured, not estimated**: the documents are sized against
  the real tokenizer at run start (`calibrate_chars_per_token`), and a document
  that came out smaller than its target, or an input that does not fit the
  window, is reported as an *admission failure* — never scored as a quality
  result.
- **Not verified is not a zero**: a category the configuration cannot do (a
  template without tool support), or a scenario that cannot be verified on this
  machine (no Go toolchain), leaves both the numerator and the denominator.
- **Provenance**: weights path and SHA-256, runtime build, quant, KV type,
  context window, offload mode and the exact server launch line are recorded in
  the run directory.

## Commands

| Command | Purpose |
|---|---|
| `dailybench setup` (Make target) | create `.venv`, install pytest |
| `dailybench selfcheck` | replay every task with a perfect answer and with garbage; both must land where expected. No GPU needed |
| `dailybench list` | show the generated suite (ids, points, sizes) |
| `dailybench generate --out suite/<seed>` | write `prompts.json` (publishable) and `oracles.json` (answers) |
| `dailybench doctor` | GPU, toolchain, agent runtime, endpoint and tokenizer reachability |
| `dailybench preflight --config <id> --full` | verify a running runtime against a reference configuration |
| `dailybench visual <page.html> --steps N` | render one page exactly as the visual checks do (probe, frame, frame statistics) |
| `dailybench bench --config <id> --speed` | bring the runtime up, preflight, run the suite, write `report.md` + `report.html` (`--no-html` skips the page) |
| `dailybench run --base-url … --model …` | the same suite against an endpoint you started yourself (`--no-agent` keeps C/K/L on the chat path); results hit the disk after every task, so a crash at task 90 keeps the first 89. `--resume` finishes an interrupted run in place: saved tasks are not re-asked, truncated ones (timeout, reboot) are re-run, the calibration is taken from the saved `config.json`, the suite is verified by SHA-256, and a torn `tasks.json` is rebuilt from `transcripts/` |
| `dailybench report <dirs…>` | re-render `report.md`, one `<run>/report.html` per run, `comparison.md` and `comparison.html` (`--no-html` skips the pages, `--compare` picks the models) |
| `dailybench rescore <dir>` | re-score saved transcripts with the current evaluators — no model call |
| `scripts/sweep.py` | load every reference configuration in turn and run the whole suite |

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
- New modules are added to `dailybench/tasks/__init__.py`; the category total goes
  into `CATEGORY_POINTS` (the registry refuses a mismatch).

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
visible before it starts; the key itself is never printed (only its absence or
presence matters) and is recorded in `runs/<name>/config.json`, because a run has
to be reproducible.

## Where the numbers come from

- `configs/reference.json` holds the policy (decoding, budgets, window, KV type,
  prompt-cache and RoPE rules) and one entry per measured configuration; the
  exact launch line for each runtime template is part of the report.
- `suite/<seed>/` holds the generated suite and its digest; `oracles.json`
  contains the answers and is deliberately kept out of the publishable
  `prompts.json`.
