"""Scenario C-2 — Go worker pool (10 hidden checks × 1 point).

The model returns the complete corrected file for the public ``Process``
signature. Ten hidden Go tests cover the specification's requirements; they are
compiled and executed with the real toolchain, and a missing toolchain is
reported as a failure with an explicit reason rather than a silent pass.
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator

from .. import core


#: Points per hidden check; three scenarios share the constant so a check
#: weighs the same wherever it is earned.
_POINTS_PER_CHECK = 0.8
from .coding_python import code_of

CHECK_ORDER = (
    ("concurrency_limit", "TestConcurrencyLimit"),
    ("at_most_once", "TestAtMostOnce"),
    ("output_closes", "TestOutputCloses"),
    ("no_new_jobs_after_cancel", "TestNoNewJobsAfterCancel"),
    ("inflight_context_cancelled", "TestInflightContextCancelled"),
    ("no_hang_after_consumer_stops", "TestNoHangWhenConsumerStops"),
    ("non_positive_workers", "TestNonPositiveWorkers"),
    ("empty_input", "TestEmptyInputCloses"),
    ("no_goroutine_per_job", "TestNoGoroutinePerJob"),
    ("no_leak", "TestNoLeakAfterCompletion"),
)

TARGET_FILE = "pool/process.go"

_GO_MOD = "module dailybench/pool\n\ngo 1.21\n"

_STUB = '''package pool

import "context"

// Job is one unit of work. Run performs the work and must honour ctx.
type Job struct {
\tID  int
\tRun func(ctx context.Context) error
}

// Result carries the outcome of a Job.
type Result struct {
\tID  int
\tErr error
}

// Process starts at most `workers` concurrent handlers reading from jobs and
// returns a channel that is closed once every accepted Job has been handled.
//
// Contract (see the task description): never more than `workers` handlers run
// at once, each Job is handled at most once, the output channel is closed when
// the input is drained, no new Job is started after ctx is cancelled, in-flight
// Run calls receive the cancelled context, the function must not deadlock when
// the consumer stops reading after cancellation, workers <= 0 must return a
// closed channel without panicking, an empty input must close the output, and
// no goroutines may leak.
func Process(ctx context.Context, jobs <-chan Job, workers int) <-chan Result {
\tpanic("not implemented")
}
'''

_REFERENCE = '''package pool

import (
\t"context"
\t"sync"
)

// Job is one unit of work. Run performs the work and must honour ctx.
type Job struct {
\tID  int
\tRun func(ctx context.Context) error
}

// Result carries the outcome of a Job.
type Result struct {
\tID  int
\tErr error
}

// Process runs at most `workers` handlers over jobs.
func Process(ctx context.Context, jobs <-chan Job, workers int) <-chan Result {
\tout := make(chan Result)
\tif workers <= 0 {
\t\tclose(out)
\t\treturn out
\t}

\tvar wg sync.WaitGroup
\twg.Add(workers)
\tfor i := 0; i < workers; i++ {
\t\tgo func() {
\t\t\tdefer wg.Done()
\t\t\tfor {
\t\t\t\tif ctx.Err() != nil {
\t\t\t\t\treturn
\t\t\t\t}
\t\t\t\tselect {
\t\t\t\tcase <-ctx.Done():
\t\t\t\t\treturn
\t\t\t\tcase job, ok := <-jobs:
\t\t\t\t\tif !ok {
\t\t\t\t\t\treturn
\t\t\t\t\t}
\t\t\t\t\terr := job.Run(ctx)
\t\t\t\t\tselect {
\t\t\t\t\tcase out <- Result{ID: job.ID, Err: err}:
\t\t\t\t\tcase <-ctx.Done():
\t\t\t\t\t\treturn
\t\t\t\t\t}
\t\t\t\t}
\t\t\t}
\t\t}()
\t}

\tgo func() {
\t\twg.Wait()
\t\tclose(out)
\t}()

\treturn out
}
'''

_PUBLIC_TEST = '''package pool

import (
\t"context"
\t"testing"
)

func TestProcessHandlesEveryJob(t *testing.T) {
\tjobs := make(chan Job, 4)
\tseen := make([]int, 0, 4)
\tfor i := 0; i < 4; i++ {
\t\tjobs <- Job{ID: i, Run: func(ctx context.Context) error { return nil }}
\t}
\tclose(jobs)

\tfor result := range Process(context.Background(), jobs, 2) {
\t\tseen = append(seen, result.ID)
\t}
\tif len(seen) != 4 {
\t\tt.Fatalf("expected 4 results, got %d", len(seen))
\t}
}
'''

_HIDDEN_TESTS = '''package pool

import (
\t"context"
\t"runtime"
\t"sync"
\t"sync/atomic"
\t"testing"
\t"time"
)

func jobsOf(count int, fn func(ctx context.Context) error) chan Job {
\tjobs := make(chan Job, count)
\tfor i := 0; i < count; i++ {
\t\tjobs <- Job{ID: i, Run: fn}
\t}
\tclose(jobs)
\treturn jobs
}

func drain(ch <-chan Result) int {
\tcount := 0
\tfor range ch {
\t\tcount++
\t}
\treturn count
}

func TestConcurrencyLimit(t *testing.T) {
\tvar running, peak int32
\tconst workers = 3
\tjobs := jobsOf(60, func(ctx context.Context) error {
\t\tcurrent := atomic.AddInt32(&running, 1)
\t\tfor {
\t\t\told := atomic.LoadInt32(&peak)
\t\t\tif current <= old || atomic.CompareAndSwapInt32(&peak, old, current) {
\t\t\t\tbreak
\t\t\t}
\t\t}
\t\ttime.Sleep(time.Millisecond)
\t\tatomic.AddInt32(&running, -1)
\t\treturn nil
\t})
\tif got := drain(Process(context.Background(), jobs, workers)); got != 60 {
\t\tt.Fatalf("expected 60 results, got %d", got)
\t}
\tif peak > workers {
\t\tt.Fatalf("peak concurrency %d exceeds workers %d", peak, workers)
\t}
}

func TestAtMostOnce(t *testing.T) {
\tvar calls int32
\tvar mu sync.Mutex
\tseen := map[int]int{}
\tjobs := jobsOf(100, func(ctx context.Context) error {
\t\tatomic.AddInt32(&calls, 1)
\t\treturn nil
\t})
\tfor result := range Process(context.Background(), jobs, 5) {
\t\tmu.Lock()
\t\tseen[result.ID]++
\t\tmu.Unlock()
\t}
\tif atomic.LoadInt32(&calls) != 100 {
\t\tt.Fatalf("expected 100 calls, got %d", calls)
\t}
\tfor id, count := range seen {
\t\tif count != 1 {
\t\t\tt.Fatalf("job %d handled %d times", id, count)
\t\t}
\t}
}

func TestOutputCloses(t *testing.T) {
\tjobs := jobsOf(10, func(ctx context.Context) error { return nil })
\tdone := make(chan int, 1)
\tgo func() { done <- drain(Process(context.Background(), jobs, 2)) }()
\tselect {
\tcase count := <-done:
\t\tif count != 10 {
\t\t\tt.Fatalf("expected 10 results, got %d", count)
\t\t}
\tcase <-time.After(5 * time.Second):
\t\tt.Fatal("output channel never closed")
\t}
}

func TestNoNewJobsAfterCancel(t *testing.T) {
\tctx, cancel := context.WithCancel(context.Background())
\tvar startedAfterCancel int32
\tvar cancelled int32
\tvar started int32
\tjobs := make(chan Job, 200)
\tfor i := 0; i < 200; i++ {
\t\tjobs <- Job{ID: i, Run: func(ctx context.Context) error {
\t\t\tif atomic.LoadInt32(&cancelled) == 1 {
\t\t\t\tatomic.AddInt32(&startedAfterCancel, 1)
\t\t\t}
\t\t\tatomic.AddInt32(&started, 1)
\t\t\ttime.Sleep(time.Millisecond)
\t\t\treturn nil
\t\t}}
\t}
\tclose(jobs)

\tout := Process(ctx, jobs, 4)
\tfor i := 0; i < 4; i++ {
\t\tif _, ok := <-out; !ok {
\t\t\tbreak
\t\t}
\t}
\tatomic.StoreInt32(&cancelled, 1)
\tcancel()
\tdeadline := time.After(10 * time.Second)
\tfor {
\t\tselect {
\t\tcase _, ok := <-out:
\t\t\tif !ok {
\t\t\t\tif startedAfterCancel > 4 {
\t\t\t\t\tt.Fatalf("jobs started after cancellation: %d", startedAfterCancel)
\t\t\t\t}
\t\t\t\treturn
\t\t\t}
\t\tcase <-deadline:
\t\t\tt.Fatal("output channel did not close after cancellation")
\t\t}
\t}
}

func TestInflightContextCancelled(t *testing.T) {
\tctx, cancel := context.WithCancel(context.Background())
\tinside := make(chan struct{}, 1)
\tsawCancel := make(chan struct{}, 1)
\tjobs := make(chan Job, 1)
\tjobs <- Job{ID: 1, Run: func(ctx context.Context) error {
\t\tinside <- struct{}{}
\t\t<-ctx.Done()
\t\tsawCancel <- struct{}{}
\t\treturn ctx.Err()
\t}}
\tclose(jobs)

\tout := Process(ctx, jobs, 1)
\t<-inside
\tcancel()
\tselect {
\tcase <-sawCancel:
\tcase <-time.After(5 * time.Second):
\t\tt.Fatal("in-flight Run did not receive the cancelled context")
\t}
\tselect {
\tcase <-out:
\tcase <-time.After(5 * time.Second):
\t\tt.Fatal("Process did not return a result after the in-flight job finished")
\t}
}

func TestNoHangWhenConsumerStops(t *testing.T) {
\tctx, cancel := context.WithCancel(context.Background())
\tjobs := make(chan Job, 50)
\tfor i := 0; i < 50; i++ {
\t\tjobs <- Job{ID: i, Run: func(ctx context.Context) error { time.Sleep(time.Millisecond); return nil }}
\t}
\tclose(jobs)

\t_ = Process(ctx, jobs, 3)
\tcancel()
\ttime.Sleep(200 * time.Millisecond)
\t// The test passes when the process is still able to run: no producer
\t// goroutine may be blocked forever on an unread output channel.
\tdone := make(chan struct{})
\tgo func() { time.Sleep(50 * time.Millisecond); close(done) }()
\tselect {
\tcase <-done:
\tcase <-time.After(5 * time.Second):
\t\tt.Fatal("test process is stuck")
\t}
}

func TestNonPositiveWorkers(t *testing.T) {
\tfor _, workers := range []int{0, -1} {
\t\tjobs := jobsOf(3, func(ctx context.Context) error { return nil })
\t\tout := Process(context.Background(), jobs, workers)
\t\tcount := 0
\t\tdeadline := time.After(5 * time.Second)
\t\tfor {
\t\t\tselect {
\t\t\tcase _, ok := <-out:
\t\t\t\tif !ok {
\t\t\t\t\tcount = -1
\t\t\t\t\tgoto finished
\t\t\t\t}
\t\t\t\tcount++
\t\t\tcase <-deadline:
\t\t\t\tt.Fatalf("workers=%d: output channel never closed", workers)
\t\t\t}
\t\t}
\tfinished:
\t\tif count > 0 {
\t\t\tt.Fatalf("workers=%d: expected no results, got %d", workers, count)
\t\t}
\t}
}

func TestEmptyInputCloses(t *testing.T) {
\tjobs := make(chan Job)
\tclose(jobs)
\tout := Process(context.Background(), jobs, 3)
\tselect {
\tcase _, ok := <-out:
\t\tif ok {
\t\t\tt.Fatal("expected no results for empty input")
\t\t}
\tcase <-time.After(5 * time.Second):
\t\tt.Fatal("empty input did not close the output channel")
\t}
}

func TestNoGoroutinePerJob(t *testing.T) {
\tbaseline := runtime.NumGoroutine()
\trelease := make(chan struct{})
\tjobs := make(chan Job, 500)
\tfor i := 0; i < 500; i++ {
\t\tjobs <- Job{ID: i, Run: func(ctx context.Context) error {
\t\t\t<-release
\t\t\treturn nil
\t\t}}
\t}
\tclose(jobs)

\tout := Process(context.Background(), jobs, 2)
\ttime.Sleep(100 * time.Millisecond)
\tpeak := runtime.NumGoroutine()
\tclose(release)
\tdrain(out)
\tif peak-baseline > 20 {
\t\tt.Fatalf("goroutine count grew by %d with 2 workers and 500 jobs", peak-baseline)
\t}
}

func TestNoLeakAfterCompletion(t *testing.T) {
\tbaseline := runtime.NumGoroutine()
\tfor iteration := 0; iteration < 5; iteration++ {
\t\tjobs := jobsOf(40, func(ctx context.Context) error { return nil })
\t\tdrain(Process(context.Background(), jobs, 3))
\t}
\tdeadline := time.Now().Add(5 * time.Second)
\tfor time.Now().Before(deadline) {
\t\truntime.GC()
\t\tif runtime.NumGoroutine() <= baseline+2 {
\t\t\treturn
\t\t}
\t\ttime.Sleep(50 * time.Millisecond)
\t}
\tt.Fatalf("goroutine leak: baseline %d, now %d", baseline, runtime.NumGoroutine())
}
'''

_PROMPT = """В пакете pool нужно реализовать функцию Process.

```go
func Process(
    ctx context.Context,
    jobs <-chan Job,
    workers int,
) <-chan Result
```

Ограничения:

1. Одновременно должны работать не более workers обработчиков.
2. Каждый полученный Job должен обрабатываться максимум один раз.
3. После обработки всех jobs выходной channel должен закрыться.
4. После ctx cancellation новые jobs начинать нельзя.
5. Уже выполняющиеся операции должны получить отменённый context.
6. Функция не должна зависать, если consumer перестал читать Result
   после cancellation.
7. workers <= 0 должен корректно завершать работу без panic.
8. Пустой входной channel должен корректно закрыть output.
9. Нельзя запускать goroutine на каждый Job.
10. После завершения не должно оставаться goroutine leak.

Публичные типы и сигнатуры менять запрещено.
Тесты менять запрещено.

Обработка одного Job — это вызов job.Run(ctx), результат отправляется
в выходной channel как Result с ID задания и Err.

Верни ТОЛЬКО полный текст файла __TARGET__ целиком, в одном блоке кода.

Файл pool/process.go в рабочей копии:

```go
__STUB__```"""


def _workspace() -> dict[str, str]:
    return {
        "go.mod": _GO_MOD,
        TARGET_FILE: _STUB,
        "pool/process_test.go": _PUBLIC_TEST,
        "README.md": "# pool\n\nWorker pool used by the ingestion service.\n",
    }


def generate(rng: random.Random, cfg: Any) -> core.TaskInstance:
    return core.TaskInstance(
        id="C-2",
        category="C",
        title="Go Worker Pool",
        prompt=_PROMPT.replace("__TARGET__", TARGET_FILE).replace("__STUB__", _STUB),
        max_points=8.0,
        workspace=_workspace(),
        oracle={
            "reference": _REFERENCE,
            "solution": _REFERENCE,
            "hidden_tests": _HIDDEN_TESTS,
            "target_file": TARGET_FILE,
            "answer_file": TARGET_FILE,
            "go_mod": _GO_MOD,
            "checks": [name for name, _ in CHECK_ORDER],
        },
        meta={"scenario": "C-2"},
    )


#: Where a Go toolchain may live when it is not on PATH (a user-local SDK, the
#: usual install prefixes). A benchmark box may have it without exporting PATH.
GO_CANDIDATES = (
    Path.home() / "go-sdk" / "go" / "bin" / "go",
    Path("/usr/local/go/bin/go"),
    Path("/opt/homebrew/bin/go"),
    Path("C:/Go/bin/go.exe"),
    Path.home() / "sdk",  # найденные версии: ~/sdk/go*/bin/go
)


def _go_binary() -> str | None:
    found = shutil.which("go") or shutil.which("go.exe")
    if found:
        return found
    for candidate in GO_CANDIDATES:
        if candidate.is_file():
            return str(candidate)
    for candidate in sorted(Path.home().glob("sdk/go*/bin/go"), reverse=True):
        if candidate.is_file():
            return str(candidate)
    return None


def _json_events(output: str) -> "Iterator[dict]":
    """Only well-formed ``go test -json`` events; raw text is a journal.

    A package may print anything around the stream (build diagnostics), and the
    candidate may print ``--- PASS: Test...`` lines from ``init``. Neither is
    interpreted: the verdict comes from the JSON actions alone.
    """
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            yield event


_ACTIONS = ("run", "pass", "fail", "skip")


def _run_go_tests(
    repo: Path, timeout: int = 300, order: tuple[tuple[str, str], ...] = CHECK_ORDER
) -> tuple[dict[str, tuple[bool, str]], str]:
    go = _go_binary()
    if not go:
        return {name: (False, "go toolchain unavailable") for name, _ in order}, "go toolchain unavailable"

    try:
        proc = subprocess.run(
            [go, "test", "-json", "-count=1", "-timeout", "120s", "./..."],
            cwd=str(repo),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=_go_env(),
        )
    except subprocess.TimeoutExpired:
        return {name: (False, "go test превысил 300 с") for name, _ in order}, "timeout"
    except OSError as exc:
        return {name: (False, f"не удалось запустить go test: {exc}") for name, _ in order}, str(exc)

    output = (proc.stdout or "") + (proc.stderr or "")
    names = {test: name for name, test in order}
    results: dict[str, tuple[bool, str]] = {}
    logs: dict[str, list[str]] = {}
    for event in _json_events(output):
        test = str(event.get("Test", ""))
        action = event.get("Action")
        if not test:
            continue
        if action == "output":
            logs.setdefault(test, []).append(str(event.get("Output", "")))
            continue
        if action not in _ACTIONS or test not in names:
            continue
        name = names[test]
        if action == "run":
            results.pop(name, None)  # a rerun invalidates an earlier verdict
        elif action == "pass":
            results[name] = (True, "ok")
        elif action == "skip":
            # A skip is not a pass: the check was never exercised.
            results[name] = (False, "skipped")
        else:
            results[name] = (False, _failure_detail(logs.get(test, [])) or "тест упал")

    # A non-zero status means the package did not finish cleanly — a panic in
    # ``init`` may come after perfectly plausible output. Nothing unconfirmed by
    # a terminal test event survives it.
    if proc.returncode != 0:
        cause = f"go test завершился с кодом {proc.returncode}"
        tail = _last_meaningful_line(output)
        for name, test in order:
            if name not in results:
                results[name] = (False, f"{cause}: {tail}" if tail else f"{cause} ({test})")

    fallback = "сборка пакета не удалась" if _looks_like_build_failure(output) else "тест не выполнился"
    tail = _last_meaningful_line(output)
    for name, test in order:
        results.setdefault(name, (False, f"{fallback}: {tail}" if tail else f"{fallback} ({test})"))
    return results, output


def _looks_like_build_failure(output: str) -> bool:
    markers = ("cannot use", "undefined:", "syntax error", "build failed", "undeclared name", "# dailybench/pool")
    return any(marker in output for marker in markers)


def _last_meaningful_line(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[-1][:200] if lines else ""


def _failure_detail(lines: list[str]) -> str:
    tail = [line.strip() for line in lines if line.strip()]
    return " | ".join(tail[:12])[:400]


def _go_env() -> dict[str, str]:
    import os

    env = dict(os.environ)
    env.setdefault("GOFLAGS", "-mod=mod")
    env.setdefault("GOTOOLCHAIN", "local")
    env["CGO_ENABLED"] = "0"
    return env


def evaluate(task: core.TaskInstance, tr: core.Transcript) -> core.Score:
    score = core.Score(task_id=task.id, category="C")
    if not _go_binary():
        # Unverifiable, never a zero: without the toolchain the hidden tests
        # cannot run at all, so the scenario is reported as not verified and
        # leaves both the numerator and the denominator untouched.
        score.skipped = "go toolchain недоступен — сценарий C-2 не мог быть проверен на этой машине"
        score.notes.append(score.skipped)
        for name, _ in CHECK_ORDER:
            score.add(name, False, _POINTS_PER_CHECK, "go toolchain unavailable")
        return score
    code = code_of(task, tr)
    if not code:
        for name, _ in CHECK_ORDER:
            score.add(name, False, _POINTS_PER_CHECK, "ответ не содержит кода файла")
        score.notes.append("не удалось извлечь файл из ответа")
        return score

    sandbox = Path(tempfile.mkdtemp(prefix="dailybench-c2-"))
    repo = sandbox / "poolrepo"
    repo.mkdir(parents=True)
    for rel, content in task.workspace.items():
        target = repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    (repo / TARGET_FILE).write_text(code, encoding="utf-8")
    hidden = repo / "pool" / "process_hidden_test.go"
    hidden.write_text(task.oracle["hidden_tests"], encoding="utf-8")

    results, output = _run_go_tests(repo)
    for name, test in CHECK_ORDER:
        passed, detail = results[name]
        score.add(name, passed, _POINTS_PER_CHECK, detail)

    if "not implemented" in output:
        score.notes.append("реализация осталась заглушкой (panic \"not implemented\")")
    if "class AsyncTTLCache" in code or "def get_or_load" in code:
        score.notes.append("в ответе оказался Python-код, а не Go")
    return score


# --------------------------------------------------------------------------- #
# Scenario C-5 (stretch) — pool with backpressure and acknowledged stop
# --------------------------------------------------------------------------- #

C5_TARGET_FILE = "pool/pool.go"

CHECK_ORDER_C5 = (
    ("backpressure", "TestBackpressure"),
    ("concurrency_limit", "TestConcurrencyLimit"),
    ("accepted_all_complete", "TestAcceptedAllComplete"),
    ("no_start_after_ack", "TestNoStartAfterAck"),
    ("stop_rejects_submit", "TestStopRejectsSubmit"),
    ("results_close", "TestResultsClose"),
    ("cancel_ack", "TestCancelAck"),
    ("inflight_context_cancelled", "TestInflightContextCancelled"),
    ("errors_reported", "TestErrorsReported"),
    ("no_goroutine_leak", "TestNoGoroutineLeak"),
)

_STUB_C5 = '''package pool

import (
\t"context"
\t"sync"
)

// Job is one unit of work. Run performs the work and must honour ctx.
type Job struct {
\tID  int
\tRun func(ctx context.Context) error
}

// Result carries the outcome of a Job.
type Result struct {
\tID  int
\tErr error
}

// Pool runs accepted Jobs on a fixed number of workers.
//
// New starts the workers. Submit accepts a Job and reports whether it was
// accepted; it applies backpressure while the queue is full. Stop acknowledges
// the end of acceptance: its channel closes once no further Job can be
// accepted, and every accepted Job still runs to completion. Results is the
// output stream; it closes after all accepted Jobs were handled and every
// worker exited. Cancelling ctx behaves like Stop.
type Pool struct {
\t_ sync.Mutex
}

// New starts a pool with `workers` workers and `capacity` queued Jobs.
func New(ctx context.Context, workers, capacity int) *Pool {
\tpanic("not implemented")
}

// Submit accepts a Job or reports false once the pool stopped accepting.
func (p *Pool) Submit(job Job) bool {
\tpanic("not implemented")
}

// Stop requests the end of acceptance and returns the acknowledgement channel.
func (p *Pool) Stop() <-chan struct{} {
\tpanic("not implemented")
}

// Results returns the output channel.
func (p *Pool) Results() <-chan Result {
\tpanic("not implemented")
}
'''

_REFERENCE_C5 = '''package pool

import (
\t"context"
\t"sync"
)

// Job is one unit of work. Run performs the work and must honour ctx.
type Job struct {
\tID  int
\tRun func(ctx context.Context) error
}

// Result carries the outcome of a Job.
type Result struct {
\tID  int
\tErr error
}

// Pool runs accepted Jobs on a fixed number of workers.
type Pool struct {
\tctx     context.Context
\tjobs    chan Job
\tresults chan Result
\tstop    chan struct{}
\tack     chan struct{}
\tonce    sync.Once
\tmu      sync.RWMutex
\tclosed  bool
\twg      sync.WaitGroup
}

// New starts a pool with `workers` workers and `capacity` queued Jobs.
func New(ctx context.Context, workers, capacity int) *Pool {
\tif workers < 1 {
\t\tworkers = 1
\t}
\tif capacity < 0 {
\t\tcapacity = 0
\t}
\tp := &Pool{
\t\tctx:     ctx,
\t\tjobs:    make(chan Job, capacity),
\t\tresults: make(chan Result),
\t\tstop:    make(chan struct{}),
\t\tack:     make(chan struct{}),
\t}
\tp.wg.Add(workers)
\tfor i := 0; i < workers; i++ {
\t\tgo p.worker()
\t}
\tgo p.supervise()
\treturn p
}

// Submit accepts a Job or reports false once the pool stopped accepting.
func (p *Pool) Submit(job Job) bool {
\tp.mu.RLock()
\tdefer p.mu.RUnlock()
\tif p.closed {
\t\treturn false
\t}
\tp.jobs <- job
\treturn true
}

// Stop requests the end of acceptance and returns the acknowledgement channel.
func (p *Pool) Stop() <-chan struct{} {
\tp.once.Do(func() { close(p.stop) })
\treturn p.ack
}

// Results returns the output channel.
func (p *Pool) Results() <-chan Result {
\treturn p.results
}

func (p *Pool) supervise() {
\tselect {
\tcase <-p.stop:
\tcase <-p.ctx.Done():
\t}
\t// The write lock waits for a Submit that is already inside the channel
\t// send, so closing the queue cannot race with an accepted Job.
\tp.mu.Lock()
\tp.closed = true
\tclose(p.jobs)
\tp.mu.Unlock()
\tclose(p.ack)
\tp.wg.Wait()
\tclose(p.results)
}

func (p *Pool) worker() {
\tdefer p.wg.Done()
\tfor {
\t\tselect {
\t\tcase job, ok := <-p.jobs:
\t\t\tif !ok {
\t\t\t\treturn
\t\t\t}
\t\t\tp.handle(job)
\t\tcase <-p.ctx.Done():
\t\t\t// Accepted Jobs are already in the queue (or inside a blocked
\t\t\t// Submit): drain them before leaving.
\t\t\tfor job := range p.jobs {
\t\t\t\tp.handle(job)
\t\t\t}
\t\t\treturn
\t\t}
\t}
}

func (p *Pool) handle(job Job) {
\tvar err error
\tif job.Run != nil {
\t\terr = job.Run(p.ctx)
\t}
\tp.results <- Result{ID: job.ID, Err: err}
}
'''

_PUBLIC_TEST_C5 = '''package pool

import (
\t"context"
\t"testing"
)

func TestPoolHandlesEveryJob(t *testing.T) {
\tp := New(context.Background(), 2, 8)
\tfor i := 0; i < 4; i++ {
\t\tif !p.Submit(Job{ID: i}) {
\t\t\tt.Fatalf("job %d was rejected", i)
\t\t}
\t}
\t<-p.Stop()
\tdone := make(chan int, 1)
\tgo func() {
\t\tcount := 0
\t\tfor range p.Results() {
\t\t\tcount++
\t\t}
\t\tdone <- count
\t}()
\tif got := <-done; got != 4 {
\t\tt.Fatalf("expected 4 results, got %d", got)
\t}
}
'''

_HIDDEN_TESTS_C5 = '''package pool

import (
	"context"
	"errors"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func waitUntil(t *testing.T, cond func() bool, message string) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatal(message)
}

// collect drains Results from the start: workers deliver unbuffered results, so
// a test that only reads after submitting would deadlock on backpressure.
func collect(p *Pool) <-chan []Result {
	out := make(chan []Result, 1)
	go func() {
		rows := make([]Result, 0, 16)
		for result := range p.Results() {
			rows = append(rows, result)
		}
		out <- rows
	}()
	return out
}

func waitRows(t *testing.T, rows <-chan []Result) []Result {
	t.Helper()
	select {
	case collected := <-rows:
		return collected
	case <-time.After(10 * time.Second):
		t.Fatal("results channel did not close")
		return nil
	}
}

func TestBackpressure(t *testing.T) {
	p := New(context.Background(), 1, 1)
	rows := collect(p)
	release := make(chan struct{})
	var started int32
	job := func(id int) Job {
		return Job{ID: id, Run: func(context.Context) error {
			atomic.AddInt32(&started, 1)
			<-release
			return nil
		}}
	}
	if !p.Submit(job(1)) {
		t.Fatal("first job was rejected")
	}
	waitUntil(t, func() bool { return atomic.LoadInt32(&started) == 1 }, "worker never started the first job")
	if !p.Submit(job(2)) {
		t.Fatal("buffered job was rejected")
	}
	blocked := make(chan bool, 1)
	go func() { blocked <- p.Submit(job(3)) }()
	select {
	case <-blocked:
		t.Fatal("Submit did not block while the queue was full")
	case <-time.After(200 * time.Millisecond):
	}
	close(release)
	select {
	case ok := <-blocked:
		if !ok {
			t.Fatal("blocked Submit reported false before the stop")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("blocked Submit never returned after the queue moved")
	}
	<-p.Stop()
	if got := len(waitRows(t, rows)); got != 3 {
		t.Fatalf("expected 3 results, got %d", got)
	}
}

func TestConcurrencyLimit(t *testing.T) {
	const workers = 4
	p := New(context.Background(), workers, 128)
	rows := collect(p)
	var running, peak int32
	for i := 0; i < 64; i++ {
		p.Submit(Job{ID: i, Run: func(context.Context) error {
			current := atomic.AddInt32(&running, 1)
			for {
				old := atomic.LoadInt32(&peak)
				if current <= old || atomic.CompareAndSwapInt32(&peak, old, current) {
					break
				}
			}
			time.Sleep(time.Millisecond)
			atomic.AddInt32(&running, -1)
			return nil
		}})
	}
	<-p.Stop()
	if got := len(waitRows(t, rows)); got != 64 {
		t.Fatalf("expected 64 results, got %d", got)
	}
	if peak > workers {
		t.Fatalf("peak concurrency %d exceeds workers %d", peak, workers)
	}
}

func TestAcceptedAllComplete(t *testing.T) {
	p := New(context.Background(), 3, 16)
	rows := collect(p)
	boom := errors.New("boom")
	var mu sync.Mutex
	accepted := map[int]bool{}
	for i := 0; i < 64; i++ {
		id := i
		var want error
		if id%2 == 0 {
			want = boom
		}
		if p.Submit(Job{ID: id, Run: func(context.Context) error { return want }}) {
			mu.Lock()
			accepted[id] = true
			mu.Unlock()
		}
	}
	<-p.Stop()
	collected := waitRows(t, rows)
	if len(collected) != len(accepted) {
		t.Fatalf("expected %d results, got %d", len(accepted), len(collected))
	}
	seen := map[int]bool{}
	for _, row := range collected {
		if seen[row.ID] {
			t.Fatalf("job %d was reported twice", row.ID)
		}
		seen[row.ID] = true
		if !accepted[row.ID] {
			t.Fatalf("job %d ran but was never accepted", row.ID)
		}
		if row.ID%2 == 0 && !errors.Is(row.Err, boom) {
			t.Fatalf("job %d: expected the Run error, got %v", row.ID, row.Err)
		}
		if row.ID%2 == 1 && row.Err != nil {
			t.Fatalf("job %d: expected nil error, got %v", row.ID, row.Err)
		}
	}
	for id := range accepted {
		if !seen[id] {
			t.Fatalf("accepted job %d never completed", id)
		}
	}
}

func TestNoStartAfterAck(t *testing.T) {
	p := New(context.Background(), 2, 4)
	rows := collect(p)
	gate := make(chan struct{})
	var mu sync.Mutex
	accepted := map[int]bool{}
	started := map[int]bool{}
	var startedCount int32
	mk := func(id int) Job {
		return Job{ID: id, Run: func(context.Context) error {
			mu.Lock()
			started[id] = true
			mu.Unlock()
			atomic.AddInt32(&startedCount, 1)
			<-gate
			return nil
		}}
	}
	submitted := make(chan struct{})
	go func() {
		defer close(submitted)
		for i := 0; i < 24; i++ {
			if !p.Submit(mk(i)) {
				return
			}
			mu.Lock()
			accepted[i] = true
			mu.Unlock()
		}
	}()
	waitUntil(t, func() bool { return atomic.LoadInt32(&startedCount) >= 2 }, "workers never started")
	ack := p.Stop()
	// The gate is released before the acknowledgement is awaited: a Submit can
	// be parked on the full queue while the supervisor needs the write lock.
	close(gate)
	<-ack
	<-submitted
	waitRows(t, rows)

	mu.Lock()
	defer mu.Unlock()
	for id := range started {
		if !accepted[id] {
			t.Fatalf("job %d started without being accepted", id)
		}
	}
	if len(started) != len(accepted) {
		t.Fatalf("expected exactly the %d accepted jobs to run, got %d", len(accepted), len(started))
	}
}

func TestStopRejectsSubmit(t *testing.T) {
	p := New(context.Background(), 1, 1)
	rows := collect(p)
	var calls int32
	ack := p.Stop()
	<-ack
	if p.Stop() != ack {
		t.Fatal("second Stop returned a different channel")
	}
	if p.Submit(Job{ID: 7, Run: func(context.Context) error {
		atomic.AddInt32(&calls, 1)
		return nil
	}}) {
		t.Fatal("Submit accepted a job after the acknowledgement")
	}
	waitRows(t, rows)
	if atomic.LoadInt32(&calls) != 0 {
		t.Fatal("rejected job still ran")
	}
}

func TestResultsClose(t *testing.T) {
	p := New(context.Background(), 2, 4)
	rows := collect(p)
	<-p.Stop()
	if got := len(waitRows(t, rows)); got != 0 {
		t.Fatalf("expected no results, got %d", got)
	}
}

func TestCancelAck(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	p := New(ctx, 1, 4)
	rows := collect(p)
	gate := make(chan struct{})
	for i := 1; i <= 3; i++ {
		p.Submit(Job{ID: i, Run: func(context.Context) error {
			<-gate
			return nil
		}})
	}
	waitUntil(t, func() bool { return true }, "noop")
	cancel()
	select {
	case <-p.Stop():
	case <-time.After(5 * time.Second):
		t.Fatal("cancelling ctx did not produce an acknowledgement")
	}
	if p.Submit(Job{ID: 9}) {
		t.Fatal("Submit accepted a job after ctx cancellation")
	}
	close(gate)
	collected := waitRows(t, rows)
	if len(collected) != 3 {
		t.Fatalf("accepted jobs were dropped on cancel: got %d results", len(collected))
	}
	seen := map[int]bool{}
	for _, row := range collected {
		seen[row.ID] = true
	}
	for id := 1; id <= 3; id++ {
		if !seen[id] {
			t.Fatalf("accepted job %d was lost on cancel", id)
		}
	}
}

func TestInflightContextCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	p := New(ctx, 1, 4)
	rows := collect(p)
	inside := make(chan struct{}, 1)
	p.Submit(Job{ID: 1, Run: func(ctx context.Context) error {
		close(inside)
		<-ctx.Done()
		return ctx.Err()
	}})
	<-inside
	cancel()
	collected := waitRows(t, rows)
	if len(collected) != 1 {
		t.Fatalf("expected the in-flight result, got %d results", len(collected))
	}
	if !errors.Is(collected[0].Err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", collected[0].Err)
	}
}

func TestErrorsReported(t *testing.T) {
	p := New(context.Background(), 1, 4)
	rows := collect(p)
	boom := errors.New("boom")
	p.Submit(Job{ID: 1, Run: func(context.Context) error { return boom }})
	p.Submit(Job{ID: 2})
	<-p.Stop()
	collected := waitRows(t, rows)
	if len(collected) != 2 {
		t.Fatalf("expected 2 results, got %d", len(collected))
	}
	if !errors.Is(collected[0].Err, boom) && !errors.Is(collected[1].Err, boom) {
		t.Fatal("the Run error was lost")
	}
}

func TestNoGoroutineLeak(t *testing.T) {
	baseline := runtime.NumGoroutine()
	for iteration := 0; iteration < 3; iteration++ {
		p := New(context.Background(), 4, 16)
		rows := collect(p)
		for i := 0; i < 32; i++ {
			p.Submit(Job{ID: i})
		}
		<-p.Stop()
		waitRows(t, rows)
	}
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		runtime.GC()
		if runtime.NumGoroutine() <= baseline+2 {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("goroutine leak: baseline %d, now %d", baseline, runtime.NumGoroutine())
}
'''

_PROMPT_C5 = """В пакете pool нужно реализовать пул с backpressure и подтверждённой
остановкой.

```go
func New(ctx context.Context, workers, capacity int) *Pool
func (p *Pool) Submit(job Job) bool
func (p *Pool) Stop() <-chan struct{}
func (p *Pool) Results() <-chan Result
```

Job — задание с ID и Run(ctx) error.

Требования:

1. Одновременно работают не более workers обработчиков; workers < 1
   считается за 1.
2. Submit принимает Job; когда очередь заполнена, Submit ждёт
   (backpressure), а не плодит goroutine.
3. Каждый принятый Job (Submit вернул true) выполняется ровно один раз и
   попадает в Results с ID и Err его Run.
4. Stop возвращает канал ack: он закрывается, когда новые Job больше не
   принимаются.
5. После закрытия ack Submit возвращает false, а Run отклонённых Job не
   вызывается. Повторный Stop возвращает тот же канал.
6. Все принятые Job всё равно выполняются до конца, включая уже стоящие в
   очереди.
7. Results закрывается после обработки всех принятых Job и выхода всех
   worker-ов.
8. Отмена ctx равнозначна Stop: ack закрывается, новые Job не принимаются.
9. Уже выполняющийся Run получает отменённый ctx и его результат
   доставляется.
10. После завершения не остаётся goroutine leak.

Порядок берётся из каналов и ack, а не из wall-clock; проверки не
предполагают конкретный порядок независимых результатов. До 8 workers и
128 jobs.

Публичные типы и сигнатуры менять запрещено.
Тесты изменять запрещено.

Верни ТОЛЬКО полный текст файла __TARGET__ целиком, в одном блоке кода.

Файл pool/pool.go в рабочей копии:

```go
__STUB__```"""


def _workspace_c5() -> dict[str, str]:
    return {
        "go.mod": _GO_MOD,
        C5_TARGET_FILE: _STUB_C5,
        "pool/pool_test.go": _PUBLIC_TEST_C5,
        "README.md": "# pool\n\nBounded worker pool used by the ingestion service.\n",
    }


def generate_c5(rng: random.Random, cfg: Any) -> core.TaskInstance:
    return core.TaskInstance(
        id="C-5",
        category="C",
        title="Go bounded pool",
        prompt=_PROMPT_C5.replace("__TARGET__", C5_TARGET_FILE).replace("__STUB__", _STUB_C5),
        max_points=8.0,
        workspace=_workspace_c5(),
        oracle={
            "reference": _REFERENCE_C5,
            "solution": _REFERENCE_C5,
            "hidden_tests": _HIDDEN_TESTS_C5,
            "target_file": C5_TARGET_FILE,
            "answer_file": C5_TARGET_FILE,
            "go_mod": _GO_MOD,
            "checks": [name for name, _ in CHECK_ORDER_C5],
        },
        meta={"scenario": "C-5"},
    )


def evaluate_c5(task: core.TaskInstance, tr: core.Transcript) -> core.Score:
    score = core.Score(task_id=task.id, category="C")
    if not _go_binary():
        score.skipped = "go toolchain недоступен — сценарий C-5 не мог быть проверен на этой машине"
        score.notes.append(score.skipped)
        for name, _ in CHECK_ORDER_C5:
            score.add(name, False, _POINTS_PER_CHECK, "go toolchain unavailable")
        return score
    code = code_of(task, tr)
    if not code:
        for name, _ in CHECK_ORDER_C5:
            score.add(name, False, _POINTS_PER_CHECK, "ответ не содержит кода файла")
        score.notes.append("не удалось извлечь файл из ответа")
        return score

    sandbox = Path(tempfile.mkdtemp(prefix="dailybench-c5-"))
    repo = sandbox / "poolrepo"
    repo.mkdir(parents=True)
    for rel, content in task.workspace.items():
        target = repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    (repo / C5_TARGET_FILE).write_text(code, encoding="utf-8")
    hidden = repo / "pool" / "pool_hidden_test.go"
    hidden.write_text(task.oracle["hidden_tests"], encoding="utf-8")

    results, output = _run_go_tests(repo, order=CHECK_ORDER_C5)
    for name, test in CHECK_ORDER_C5:
        passed, detail = results[name]
        score.add(name, passed, _POINTS_PER_CHECK, detail)

    if "not implemented" in output:
        score.notes.append("реализация осталась заглушкой (panic \"not implemented\")")
    return score
