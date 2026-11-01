# Review 2026-11-01 Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close every finding L1–L20, S1–S18, P1–P9 of the 2026-11-01 review.

**Architecture:** Bug fixes inside the existing modules; no new subsystems except a stdlib egress forwarder for the sandbox (S13). Every behavioral fix gets a failing-before regression test in the module's existing test file.

**Tech Stack:** Python ≥3.10 (reference 3.12), pytest, uv, Playwright/Chromium (optional), Go toolchain (optional), Docker (optional).

**Spec:** `docs/specs/2026-11-01-review-findings.md` — finding IDs and the owner's decisions table are binding.

## Global Constraints

- Python ≥3.10 syntax; `[project] dependencies = []` stays empty (core is stdlib-only); new third-party deps only in optional extras.
- Must pass on Windows with a non-UTF-8 console code page and **without** `PYTHONUTF8`/`PYTHONIOENCODING` (run tests with `env -u PYTHONUTF8 -u PYTHONIOENCODING`), and on Linux/macOS.
- Test command in a worktree: Windows `.venv/Scripts/python.exe -m pytest <files> -q`; POSIX `.venv/bin/python -m pytest <files> -q`.
- A test that needs an optional tool (playwright+Chromium, numpy, Pillow, yaml, Go, postgres, docker, ffmpeg) skips with a reason when the tool is absent; it never fails for that reason.
- Code comments and user-facing messages follow the surrounding style (Russian).
- `core.EVALUATOR_CONTRACT` is bumped exactly once, 3 → 4, in Task 6 (covers every evaluator change in this plan).
- Published artefacts under `runs/`, `reports/`, `suite/` are never regenerated or edited.
- Identity hashes of existing runs must not change when new options are at their defaults (old run dirs stay resumable).
- No push, no MR.

## Review Focus

- A run directory created **before** this plan (identity without the new exclusions, `config.json` without `generation_mode`) is resumed/reported → resume still accepts it (Task 1 test `test_legacy_identity_is_unchanged_by_resume_flags`); report shows it unranked with reason «режим генерации не записан» (Task 6 test).
- `--allow-suite-drift` where only one task's bytes changed → only that task re-runs, every other row is kept (Task 1 test `test_drift_keeps_rows_whose_task_digest_is_unchanged`).
- A socket timeout that is **not** the declared ceiling (connection refused / server reset) stays `infra_failure`; only an elapsed declared ceiling becomes `resource_exhausted` (Task 2 test `test_connection_reset_stays_infrastructure`).
- Sandbox endpoint on host loopback vs remote https endpoint with CA → both reachable through the forwarder, everything else unreachable (Task 11 tests).
- Chromium crash mid-probe followed by the next K task → next task gets a fresh browser and a real verdict (Task 10 test `test_probe_crash_relaunches_the_worker`).

---

### Task 1: Resume identity, drift record, retry policy, POSIX manifest keys

**Review:** per-task — persisted format (run identity, checkpoint manifest)

Findings: L1, S4, S9; `--retry-errors` part of L2.

**Files:**
- Modify: `dailybench/runner.py:45-48` (`_IDENTITY_EXCLUDED`, `config_identity`/`run_identity`)
- Modify: `dailybench/cli.py:1068-1256` (`_resume_tasks`, `_truncated`, change-list at 1162)
- Modify: `dailybench/generation.py:729-746` (manifest keys)
- Test: `tests/test_harness.py`, `tests/test_cli.py`

**Interfaces:**
- Produces: `runner.run_identity(cfg, suite_sha) -> str` — value identical for any `retry_errors`/`allow_suite_drift` setting and identical to the pre-change value when both are False.
- Produces: `<out_dir>/resume_log.jsonl`, one JSON object per accepted drift: `{"accepted_at": <ISO-8601 UTC>, "old_suite_sha256": str, "new_suite_sha256": str, "old_identity": str, "new_identity": str, "rerun": [task ids]}`, written with the same atomic/generation writer the run uses for its other files.
- Produces: `cli._truncated(transcript) -> bool` no longer returns True merely because `transcript.error` is set.

- [ ] **Step 1: Write failing tests**
  - `test_resume_flags_do_not_change_the_identity`: `run_identity(cfg,'x') == run_identity(replace(cfg, retry_errors=True),'x') == run_identity(replace(cfg, allow_suite_drift=True),'x')`.
  - `test_legacy_identity_is_unchanged_by_resume_flags`: identity of a default `RunConfig()` equals a hard-coded hex computed on the parent commit `ccb0863` (compute it once with the old code before editing and paste it).
  - `test_drift_keeps_rows_whose_task_digest_is_unchanged`: saved run of 3 tasks; mutate one task's prompt so `suite_digest(...)["per_task"]` differs for it; resume with `allow_suite_drift=True` → `missing == [changed id]`, the other two kept; `resume_log.jsonl` has one line with `rerun == [changed id]`.
  - `test_drift_without_the_flag_still_refuses` (SystemExit).
  - `test_completed_error_rows_are_kept_without_retry_errors`: a transcript with `error` set, `exit_ok` True and a non-empty last finish_reason → kept; with `retry_errors=True` and `score.failure == "infra_failure"` → re-run; with `retry_errors=True` and `score.failure == "resource_exhausted"` → kept.
  - `test_manifest_keys_are_posix` (generation): committing a stage containing `transcripts/a.json` stores key `"transcripts/a.json"`; a second commit with the file unchanged hard-links it (same `st_ino` where the FS supports links; skip otherwise).
- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement.**
  - Identity: hash the config with `retry_errors` and `allow_suite_drift` replaced by their dataclass defaults (do not drop the keys — that would change every legacy identity). Same exclusion in the "changed" list at `cli.py:1162`.
  - Drift: when the suite sha differs and the flag is set, load saved rows by the **saved** identity; keep a row only when `saved_digest["per_task"][id] == current["per_task"][id]`; write the `resume_log.jsonl` line; later checkpoints are written under the new identity.
  - `retry_errors` re-runs rows whose `core.canonical_failure(score.failure)` is in `core.INCOMPLETE_FAILURES`, nothing else.
  - `_truncated`: True only for `exit_ok is False`, a finish reason in `("error","aborted")`, or an empty last finish reason. Message «повтор из-за обрыва» lists only those.
  - Manifest keys: `.as_posix()` at every `relative_to(stage)` used as a key and every lookup.
- [ ] **Step 4: Run** `tests/test_harness.py tests/test_cli.py tests/test_generation*.py` (whichever exist) — PASS.
- [ ] **Step 5: Commit** `Resume: флаги вне identity, запись дрейфа, повтор только инфраструктурных ошибок, POSIX-ключи manifest`.

### Task 2: One taxonomy for declared ceilings; wall-clock flag

**Review:** per-task — scoring taxonomy (decides rankability of every run)

Findings: L2 (code part), L17; the "missing usage → infra_failure" part of S4.

**Files:**
- Modify: `dailybench/providers.py:395-426`
- Modify: `dailybench/runner.py:680-700, 905-960, 1110-1120`
- Modify: `dailybench/config.py:150-215` (defaults/docstrings: `request_wall_clock` default 1800 to match policy; ceiling docstrings say `resource_exhausted`, not "unverified")
- Modify: `dailybench/cli.py:70-110` (flags)
- Test: `tests/test_runner.py` or the file where runner deadline tests live (find with grep `предел времени`), `tests/test_cli.py`

**Interfaces:**
- Produces: `providers.ProviderError` raised for a socket timeout that equals the request's declared wall-clock carries kind `"resource_exhausted"` (use the existing kind field/mechanism of `ProviderError`; add one if absent: `ProviderError(msg, kind="infra_failure")`).
- Produces: CLI `--request-wall-clock SECONDS` → `cfg.request_wall_clock`; `--timeout` keeps setting `request_timeout` and its help says it is the socket timeout.

- [ ] **Step 1: Failing tests**
  - `test_socket_timeout_at_the_declared_ceiling_is_resource_exhausted`: stub endpoint that never answers, `request_wall_clock=1` → score `failure == "resource_exhausted"`, `report.summarise(...)["eligible"] is True`.
  - `test_task_deadline_between_requests_is_resource_exhausted` (existing behavior pinned).
  - `test_connection_reset_stays_infrastructure`: stub closes the socket → `infra_failure`, `eligible is False`.
  - `test_missing_usage_is_not_infrastructure`: stub answers without `usage` → the row is not `infra_failure` (failure `None` or the existing model outcome).
  - `test_request_wall_clock_flag_reaches_the_config`; `test_default_request_wall_clock_matches_the_policy` (`RunConfig().request_wall_clock == 1800`).
- [ ] **Step 2: Run, confirm fail.**
- [ ] **Step 3: Implement** the mapping; fix the misleading comment at `runner.py:684-686`.
- [ ] **Step 4: Run** the covering test files — PASS.
- [ ] **Step 5: Commit** `Таксономия: исчерпанный предел — resource_exhausted на обоих путях; --request-wall-clock`.

### Task 3: Review fixtures: UTF-8 subprocesses and fast J-5 (+ BigCodeBench decode)

Findings: S1 (review.py, bigcodebench), P1.

**Files:**
- Modify: `dailybench/tasks/review.py:1076-1088, 1860-1874, 2320-2345` and the public-test runner that uses `text=True`
- Modify: `dailybench/adapters/bigcodebench.py:510-530`
- Test: `tests/test_review.py`, `tests/test_adapter_bcb.py`

**Interfaces:**
- Produces: `review._env()` always contains `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8`; every `subprocess.run` in review.py/bigcodebench.py passes `encoding="utf-8"` (diagnostic text: `errors="replace"`).

- [ ] **Step 1: Failing tests**
  - `test_j5_fixture_builds_without_utf8_mode`: run `python -m dailybench generate --profile research-v1 --out <tmp>` in a subprocess with `PYTHONUTF8`/`PYTHONIOENCODING` removed from env and `PYTHONLEGACYWINDOWSSTDIO` unset → exit 0. (On POSIX also run with `LC_ALL=C`.)
  - `test_bcb_score_survives_non_utf8_output`: scorer stub prints bytes `b"\xef\xf0"` → `score()` returns an `unverified`/recorded result, no exception.
  - `test_review_fixture_proof_runs_once_per_process`: building the stretch suite twice in one process invokes `_verify_run` for J-5 at most once (monkeypatch counter).
  - Fix `test_score_passes_pinned_argv_and_stores_pass1_as_ratio` if it fails for an encoding/path reason on Windows.
- [ ] **Step 2: Run, confirm fail.**
- [ ] **Step 3: Implement.** Cache fixture proofs per process keyed by `(task.id, sha256 of the fixture files)`; selfcheck keeps proving them. Keep proofs in `generate` (they guard the fixture) but never repeat for identical bytes.
- [ ] **Step 4: Run** `tests/test_review.py tests/test_adapter_bcb.py` and time `python -m dailybench generate --profile stretch-v1 --out <tmp>`: must be < 4 s on second in-process build (report the measured number).
- [ ] **Step 5: Commit** `J-5/BCB: UTF-8 подпроцессы, однократная проверка фикстур`.

### Task 4: Packaging, dependencies, lock, CI matrix

Findings: S2, S3, S15, S18, L20; CI part of S16.

**Files:**
- Modify: `pyproject.toml` (extras: `visual = ["playwright>=1.40", "numpy>=1.24", "Pillow>=10"]`, `vision = ["Pillow>=10"]`, `dev` adds `pyyaml>=6`; delete the `shot.sh` package-data block and its comment)
- Modify: `setup.py` (packages incl. `dailybench.adapters`; data_files incl. `schema/*.json`; derive both lists from one place so they cannot drift from pyproject — the existing test compares them)
- Modify: `dailybench/tasks/visual_input.py:33` (import PIL lazily inside the functions that draw; a missing Pillow raises `RuntimeError("для профиля research-v1 нужен Pillow: uv pip install -e .[vision]")` only when V tasks are generated)
- Create: `uv.lock` (`uv lock`)
- Modify: `.github/workflows/ci.yml` (pin every `uses:` to a full commit SHA with the tag in a comment; install with `uv sync --locked --extra dev`; test matrix `os: [ubuntu-latest, macos-latest, windows-latest]` × `python: ["3.10", "3.12"]` for the pytest job; Windows job runs with `PYTHONUTF8` unset)
- Test: `tests/test_ci_contracts.py`, new `tests/test_packaging.py` (do not edit `tests/test_htmlreport.py` — Task 6 owns it)

- [ ] **Step 1: Failing tests**
  - `test_list_works_without_pillow`: subprocess with `-c "import sys; sys.modules['PIL']=None; from dailybench.cli import main; raise SystemExit(main(['list']))"` → exit 0.
  - `test_visual_extra_carries_frame_stats_dependencies`: parsed pyproject `visual` extra contains numpy and Pillow.
  - `test_legacy_setup_packages_match_pyproject`: `setup.py` packages == pyproject `tool.setuptools.packages`; data_files cover pyproject data-files incl. schema.
  - `test_no_package_data_points_to_a_missing_file`.
  - `test_ci_actions_are_pinned_by_sha` and `test_ci_tests_windows_and_python_310` in `tests/test_ci_contracts.py`.
- [ ] **Step 2: Run, confirm fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the covering tests; `uv lock --check` exits 0.
- [ ] **Step 5: Commit** `Упаковка: extras visual/vision, setup.py синхронизирован, uv.lock, CI Windows и 3.10`.

### Task 5: Windows process tree and Ctrl-C for the agent; portable agent tests

**Review:** per-task — process lifecycle / concurrency

Findings: S5; `tests/test_agent.py` part of S16.

**Files:**
- Modify: `dailybench/agent.py:890-990, 1230-1250, 1590-1645`
- Test: `tests/test_agent.py`

**Interfaces:**
- Produces: on any `BaseException` during `communicate`, `agent.run` calls `_kill_tree(proc)` and re-raises; a failed `AssignProcessToJobObject` is recorded and the timeout/interrupt path falls back to `taskkill /T /F /PID`.

- [ ] **Step 1: Failing tests**
  - Replace every POSIX shell fake binary in `tests/test_agent.py` with a Python fake: a `.py` script launched through a generated launcher (`<name>.cmd` on Windows invoking `sys.executable`, executable shebang script on POSIX). One helper `_fake_agent(tmp_path, body: str) -> Path`.
  - `test_keyboard_interrupt_kills_the_agent_tree`: fake agent spawns a grandchild that sleeps; monkeypatch `proc.communicate` to raise `KeyboardInterrupt` → after the raise, neither pid is alive.
  - `test_job_attach_failure_still_kills_by_taskkill` (Windows-only, `skipif` elsewhere).
  - All currently failing `test_agent.py` tests on Windows pass (list in the spec run: 23 tests).
- [ ] **Step 2: Run, confirm fail.**
- [ ] **Step 3: Implement.** Fix product bugs the portable tests expose (e.g. `WinError 6` handle use in the job path); do not skip a test on Windows unless the behavior is genuinely POSIX-only (pgrep-based stray sweep), and say why in the skip reason.
- [ ] **Step 4: Run** `tests/test_agent.py` on this Windows host — 0 failures.
- [ ] **Step 5: Commit** `Агент: дерево процессов на Windows и Ctrl-C, переносимые тесты`.

### Task 6: Profiles, ranking key, report labels, deterministic CI

**Review:** per-task — persisted format (contract key / rankability)

Findings: L3, L4, L14, L18 (code), S12; `EVALUATOR_CONTRACT` bump.

**Files:**
- Modify: `dailybench/core.py:38-52, 159-182` (`PROFILE_POINTS` without `pilot`; `BASELINE_INCOMPARABLE_PROFILES = {"research-v1", "pilot"}`; `EVALUATOR_CONTRACT = 4`; docstrings)
- Modify: `dailybench/suite.py:54-78` only if pilot weighting needs it (pilot tasks keep module `max_points`)
- Modify: `dailybench/report.py:240-255, 365-410, 514-534, 900-1000`
- Modify: `dailybench/htmlreport.py` (~611 and the K/V showcase lines)
- Modify: `dailybench/stats.py:225-245`
- Test: `tests/test_profiles.py`, `tests/test_report.py`, `tests/test_htmlreport.py`, `tests/test_stats.py`

**Interfaces:**
- Produces: `report.contract_key(run) -> str` = `f"{suite_sha}:{generation_mode}"`, `generation_mode` read from the run's `config.json` dict (`run["config"]["generation_mode"]`, or wherever the loaded run keeps config); `rankable(summary)` is False when the mode is missing, and the verdict text names «режим генерации не записан».
- Produces: `summary["showcase"]` rows per category so K and V are labelled separately («Симуляции K», «VQA V»).

- [ ] **Step 1: Failing tests**
  - `test_pilot_keeps_module_weights`: pilot IF-6 `max_points == 2.0`; `scored_total("pilot")` equals the sum of module weights, not 100.
  - `test_pilot_report_says_not_rankable_with_baseline`.
  - `test_fixed_and_native_runs_are_not_ranked_together`; `test_run_without_generation_mode_is_unranked`.
  - `test_research_sensitivity_uses_profile_categories`: research schemes contain only A,C,D,E,F,I,J; B/G/H are absent, not "unknown".
  - `test_research_showcase_is_labelled_v_not_k` (md and html).
  - `test_paired_ci_is_independent_of_score_order`: reversed scores → identical interval (seed 7, 50 resamples).
  - Fix `test_rebuild_without_tasks_keeps_published_pages_and_frames` (fails on Windows: path separators in `_showcase/run-a/pages/K-1.html`).
- [ ] **Step 2: Run, confirm fail.**
- [ ] **Step 3: Implement.** Sort strata keys and units by task id before resampling.
- [ ] **Step 4: Run** the covering files — PASS; `python -m dailybench selfcheck` — PASS.
- [ ] **Step 5: Commit** `Профили и рейтинг: pilot по весам модулей, режим в contract key, V≠K, детерминированный CI`.

### Task 7: Portable test suite, Makefile, fast suite fixtures

Findings: rest of S16, S17, P8.

**Files:**
- Modify: `tests/test_harness.py:12` (fcntl only where used, `pytest.importorskip`/`skipif(os.name == "nt")` on the lock test with reason) and its port-9 uses
- Modify: `tests/test_cli.py:54-57`, `tests/test_reference.py:230-233` (port 9)
- Modify: `tests/test_coding.py` postgres cases (skip when no postgres), `tests/test_schema.py` (importorskip yaml), `tests/test_local_config.py`, `tests/test_video_manifests.py`
- Create: `tests/conftest.py` (session fixtures)
- Modify: `tests/test_profiles.py`, `tests/test_cli.py` to use them
- Modify: `Makefile`
- Do NOT touch: `tests/test_agent.py` (Task 5), `tests/test_visual*.py` (Task 10), `tests/test_measure.py` (Task 13), `tests/test_htmlreport.py`/`test_report.py`/`test_profiles.py` assertions owned by Task 6 (only swap their suite construction to the fixture — if that collides, leave test_profiles.py to Task 6 and note it)

**Interfaces:**
- Produces: `tests/conftest.py`: `refusing_endpoint() -> str` fixture (binds `127.0.0.1:0`, closes the socket, returns `http://127.0.0.1:<port>/v1`); `suite_for(profile: str) -> list[TaskInstance]` session-cached factory returning `copy.deepcopy` of a once-built suite.

- [ ] **Step 1:** Make `tests/test_harness.py` collect on Windows; run the full suite on this host with `PYTHONUTF8` unset and record failures.
- [ ] **Step 2:** For each failure in the files above: if the tool is optional → skip with reason; if it is a product bug on Windows (paths, CRLF, encoding) → write/keep the failing test and fix the product code in the file owned by this task or report it in the task report (do not edit files owned by other tasks).
- [ ] **Step 3:** Replace port 9 with `refusing_endpoint`; introduce `suite_for` in the slow tests (`test_profiles.py`, `test_cli.py` resume tests).
- [ ] **Step 4:** Makefile: `PY ?=` resolves to `.venv/Scripts/python.exe` on Windows (`ifeq ($(OS),Windows_NT)`), `.venv/bin/python` otherwise; `setup` uses `$(UV) venv` without `test -x`; `clean` uses `$(PY) -c "import shutil; ..."`.
- [ ] **Step 5: Run** the touched test files; report full-suite wall time before/after (baseline 14 min 20 s on this host).
- [ ] **Step 6: Commit** `Тесты: переносимость на Windows, отказной endpoint вместо порта 9, session-фикстуры сьютов, Makefile`.

### Task 8: Text evaluators match their prompts

Findings: L5, L6, L7, L8, L9.

**Files:**
- Modify: `dailybench/tasks/instruction.py:1140-1160` (IF-8 compares raw emitted keys with the NFC oracle)
- Modify: `dailybench/tasks/tools.py:2700-2712` (T-11: version as its own token anywhere in the final text, plus reservation id required), `tools.py:2805-2816` (T-12: revision, build id and effect count 1; no phrase list)
- Modify: `dailybench/tasks/retrieval.py:1430-1445` (I-5 ordered sources; content points require `set(payload) == {"value","sources"}`)
- Modify: `dailybench/tasks/hallucination.py:895-912` (F-3: only a bare object is closed)
- Test: `tests/test_instruction.py`, `tests/test_tools.py`, `tests/test_retrieval.py`, `tests/test_hallucination.py` (existing names; grep)

- [ ] **Step 1: Failing tests** — one per finding: NFD keys lose `keys`/`order`; «ревизия 12, сборка b-1, эффектов 1» earns T-12 full; «версия слота теперь 2» + id earns T-11, missing id loses; swapped I-5 sources lose provenance, extra key loses content points; fenced F-3 loses `json_shape` and `no_extra_text`. Each oracle answer still earns full points.
- [ ] **Step 2: Run, confirm fail.** **Step 3: Implement.** **Step 4: Run** covering files + `python -m dailybench selfcheck --only A,D,F,I` — PASS.
- [ ] **Step 5: Commit** `Оценщики A/D/F/I проверяют то, что требует промпт`.

### Task 9: Deterministic hidden concurrency tests; persistent Go cache

**Review:** per-task — concurrency

Findings: S11, P4.

**Files:**
- Modify: `dailybench/tasks/coding_go.py:480-495, 1370-1395, 755-775`
- Modify: `dailybench/tasks/coding_python.py:1300-1310`
- Test: `tests/test_coding.py`

**Interfaces:**
- Produces: Go grading uses `GOCACHE = <user cache dir>/dailybench/go-build/<sha256(go version)[:16]>` (`os.environ["DAILYBENCH_GOCACHE"]` overrides), still `GOPROXY=off`, `GOTOOLCHAIN=local`, `GOFLAGS=-mod=mod` unchanged.

- [ ] **Step 1: Failing tests:** the existing oracle/mutant matrix for C-2/C-5/C-4 stays exact; add a stress variant that runs the hidden Go tests with `GOMAXPROCS=1` and `-count=20` against the oracle (must pass) and against a per-job-goroutine mutant and a leaking mutant (must fail). Skip without Go.
- [ ] **Step 2: Implement:** worker bound via a blocking gate the test controls (jobs block until the test counts started workers, then release); leak check waits on pool-owned completion signals, not on `NumGoroutine` after a sleep. C-4: await the candidate task, then spin `await asyncio.sleep(0)` up to 100 times until only the test's own tasks remain.
- [ ] **Step 3: Run** `tests/test_coding.py` twice; second run measurably faster for Go cases (report times).
- [ ] **Step 4: Commit** `C-2/C-4/C-5: детерминированные проверки конкурентности, постоянный GOCACHE`.

### Task 10: Visual capture and scoring

Findings: L10, L11, L12, L13, S7, S8, S10, P2, P3; visual-test part of S16.

**Files:**
- Modify: `dailybench/browser.py` (lines ~93-120, 280-290, 870-900, 995-1010, 1160-1240, 1305-1315)
- Modify: `dailybench/visual.py` (~730-740, 880-890, 920-940, 1030-1060)
- Modify: `dailybench/art.py:100-110`
- Test: `tests/test_visual.py`, `tests/test_visual_particles.py`, `tests/test_art*.py`

**Interfaces:**
- Produces: `RenderResult.failure == "evaluator_failure"` and a relaunched worker when a harness-side page evaluation throws; the PNG is captured before `PIXEL_PROBE`.
- Produces: scoring renders install a deterministic frame clock before any page script (same mechanism as `video/tools/record_clip.py:816`, extracted to a shared constant in `browser.py` and imported by `record_clip.py`).

- [ ] **Step 1: Failing tests** (pure-Python where possible; Chromium tests skip without playwright):
  - `test_shot_stage_gets_its_own_budget`: drive the parent loop with a fake worker reporting `STAGE_SHOT` for 60 s of fake monotonic time → not killed before `SCREENSHOT_TIMEOUT_MS`.
  - `test_probe_crash_is_evaluator_failure` and `test_probe_crash_relaunches_the_worker`.
  - `test_every_varied_sample_must_move` (L10), `test_motion_needs_a_visible_object` (L11), `test_inline_image_is_not_art` (`data:image/` and `blob:`).
  - `test_unstable_frame_is_not_cached`.
  - `test_hidden_points_render_once` (P2: `again=False` for hidden samples; `PIXEL_PROBE` not run in scoring).
  - `test_vendor_tree_is_prepared_once_per_process` (P3).
  - Make `test_frame_statistics_*`, `test_every_check_is_one_fifth_of_the_task`, `test_every_task_carries_four_checks_summing_to_max_points` and the live-render tests skip when numpy/Pillow/playwright are absent.
- [ ] **Step 2: Implement:** shot stage compares to `stage_deadline` (absolute cap still applies to every other stage); two rAF + `document.fonts.ready` before `screenshot`.
- [ ] **Step 3: Run** the covering files; with Chromium available (`python -m playwright install chromium` in the worktree venv is allowed) also run the live-render tests.
- [ ] **Step 4: Commit** `K/L: бюджет скриншота, сбой пробы — оценщик, AND скрытых проб, кадровые часы, без лишних навигаций`.

### Task 11: Sandbox: enforced endpoint-only egress, pinned images, robust supervisor

**Review:** per-task — security

Findings: S13, S6, S14, P5, L19; sandbox part of S1.

**Files:**
- Modify: `dailybench/isolation.py`, `scripts/sandbox_run.py`
- Create: `docker/egress_forward.py` (stdlib TCP forwarder: listens on one port, connects only to the configured `host:port`; nothing else)
- Modify: `docker/Dockerfile.agent` (base `node:22-bookworm-slim@sha256:<digest>`; apt from `snapshot.debian.org/archive/debian/<timestamp>` with pinned versions), `docker/Dockerfile.python`, `docker/Dockerfile.judge`, `docker/Dockerfile.capture` (same pinning), `docker/compose.yaml` (judge on an internal network via the forwarder; fix the comment at 11-13, 90-91), `docker/README.md`
- Test: `tests/test_sandbox.py`, `tests/test_isolation.py`

**Interfaces:**
- Produces: `network="endpoint"` = a per-run `docker network create --internal dailybench-<run>`; a forwarder container on that network **and** the default bridge runs `egress_forward.py --listen 0.0.0.0:8080 --target <endpoint host>:<port>`; the agent container joins only the internal network and gets `base_url` rewritten to `http://dailybench-egress:8080/...` (https endpoints: forward TCP to 443, keep SNI by connecting with the original hostname mapped via `--add-host <hostname>:<forwarder ip>`). Both containers and the network are removed in `finally`.
- Produces: images are pinned by immutable id: a `name@sha256:<64 hex>` reference is used as is; a tag (e.g. the default `dailybench/runner-agent:local`) is resolved once per run to its local image id (`docker image inspect --format {{.Id}}`), the container is started **by that id** (a rebuild mid-run cannot change it), and the id is recorded in the supervisor report; an unresolvable reference is refused. `latest` stays refused.
- Produces: supervisor streams stdout/stderr to the given files (bounded 64 KiB tails in memory for the report), decodes UTF-8 with `errors="replace"`, kills via `_kill_tree` (taskkill on Windows), always attempts `docker rm` even if `docker kill` fails, preserves the primary exception; SBOM cached per `image_id` per process.
- Produces: docstring at `sandbox_run.py:10-13` maps timeout/OOM to `resource_exhausted`, same as `agent.py:1543-1563`.

- [ ] **Step 1: Failing tests** (argv/unit tests always run; docker tests `@pytest.mark.docker` skip when the daemon is unavailable):
  - `test_endpoint_network_is_internal_with_a_forwarder` (argv inspection).
  - `test_forwarder_refuses_other_targets` (pure Python: start `egress_forward` locally against a stub, verify only the target is reachable).
  - docker: `test_container_cannot_reach_the_internet` (`curl https://example.com` fails), `test_container_reaches_the_endpoint` (stub on host).
  - `test_tag_without_digest_is_refused`; `test_cleanup_runs_rm_after_kill_failure`; `test_timeout_recovery_on_windows_uses_kill_tree` (monkeypatch `os.name`/platform); `test_output_is_streamed_not_buffered` (10 MB output, peak tail ≤ 64 KiB); `test_sbom_is_computed_once_per_image`.
- [ ] **Step 2: Implement.** Resolve the Node base digest with `docker buildx imagetools inspect node:22-bookworm-slim` or the registry API (`https://registry-1.docker.io` manifest HEAD) and the snapshot timestamp of the build date; write both into the Dockerfiles.
- [ ] **Step 3: Run** unit tests; if Docker is available run `tests/test_sandbox.py -m docker`; report which ran.
- [ ] **Step 4: Commit** `Песочница: egress только к endpoint, образы по digest, потоковый вывод и надёжная очистка`.

### Task 12: Scripts: coder_loop tree kill and incremental repeats; sweep without rebuild

Findings: P6, P7 (and the coder_loop tree-kill part).

**Files:**
- Modify: `scripts/coder_loop.py:70-120, 200-215`, `scripts/sweep.py:25-40, 100-110`
- Test: `tests/test_scripts.py` (create if no test covers these scripts; grep first)

- [ ] **Step 1: Failing tests:** `_repeat_counts` replaced by an incremental counter fed by the pump (unit test feeds lines, asserts counts, and asserts the log file is not reopened); a round timeout kills a grandchild (pid not alive afterwards); `sweep._suite_size(run_dir)` reads `suite.json` `per_task` length and never calls `build_suite` (monkeypatch to raise).
- [ ] **Step 2: Implement** (reuse `dailybench.agent._kill_tree` or the same Job Object/process-group approach; flush the pump writes).
- [ ] **Step 3: Run** the covering tests. **Step 4: Commit** `Скрипты: убийство дерева и инкрементальные повторы в coder_loop, sweep без пересборки`.

### Task 13: Speed profile measures what it names

Findings: P9; `tests/test_measure.py` Windows failure.

**Files:**
- Modify: `dailybench/measure.py:60-80, 255-315`
- Test: `tests/test_measure.py`

- [ ] **Step 1: Failing tests:** `long_<N>` prompt is fitted with the provider's tokenizer when available (stub `/tokenize` returns counts; prompt token count within ±1 % of N), falls back to the estimate and labels the row `sized: "estimate"` otherwise; prefill rate subtracts a measured empty-prompt TTFT (`overhead_s` recorded in the row); generation rate is `None` when `completion_tokens` is missing; VRAM row carries `measured_on: "harness_host"` unless the endpoint is loopback; `test_gpu_inventory_and_used_mib_parse_the_smi_output` passes on Windows (CRLF).
- [ ] **Step 2: Implement.** **Step 3: Run** `tests/test_measure.py`. **Step 4: Commit** `Speed: промпт по токенизатору, накладные TTFT, без выдуманной скорости генерации`.

### Task 14: Documentation and reference policy

Findings: L15, L16, L18 (docs), L19 (docs), L2/L3/L4 wording; preflight warning for L16.

**Files:**
- Modify: `README.md`, `README.ru.md` (stretch counts 36/39 and the exact new-id list generated from `build_suite`; pilot per-task and unranked; V carries 6 informational points outside the 71; ceiling = `resource_exhausted` in the denominator; Windows supported; sandbox egress enforced; `--request-wall-clock`)
- Modify: `configs/reference.json` (remove `--seed -1`; `quantization`/`kv_cache` notes say «measured at 32768; not verified at 262144»; add `"quantization_verified_window": 32768`; `request_wall_clock_note` → resource_exhausted)
- Modify: `dailybench/reference.py` (preflight warning when `context_window > quantization_verified_window`)
- Modify: `docs/PROTOCOL.md` (mode in contract key, unranked without mode), `docs/TECHNICAL_REVIEW.md` (M08 status note: reopened and closed 2026-11-01), `docker/README.md` only if Task 11 left it inconsistent
- Test: `tests/test_reference.py`, `tests/test_docs*.py` if present

- [ ] **Step 1: Failing tests:** `test_preflight_warns_when_window_exceeds_the_verified_quant_window`; `test_launch_line_has_no_random_seed`; a doc-consistency test that the README stretch K/L counts equal `build_suite` counts (if such doc tests exist, extend them; otherwise add `tests/test_docs_counts.py`).
- [ ] **Step 2: Implement.** **Step 3: Run** covering tests + `python -m dailybench selfcheck`. **Step 4: Commit** `Документация и эталонная политика приведены к коду`.
