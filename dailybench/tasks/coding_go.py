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
from typing import Any

from .. import core


#: Points per hidden check; three scenarios share the constant so a check
#: weighs the same wherever it is earned.
_POINTS_PER_CHECK = 0.8
from .coding_python import extract_code

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


def _run_go_tests(repo: Path, timeout: int = 300) -> tuple[dict[str, tuple[bool, str]], str]:
    go = _go_binary()
    if not go:
        return {name: (False, "go toolchain unavailable") for name, _ in CHECK_ORDER}, "go toolchain unavailable"

    try:
        proc = subprocess.run(
            [go, "test", "-count=1", "-timeout", "120s", "-v", "./..."],
            cwd=str(repo),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=_go_env(),
        )
    except subprocess.TimeoutExpired:
        return {name: (False, "go test превысил 300 с") for name, _ in CHECK_ORDER}, "timeout"
    except OSError as exc:
        return {name: (False, f"не удалось запустить go test: {exc}") for name, _ in CHECK_ORDER}, str(exc)

    output = (proc.stdout or "") + (proc.stderr or "")
    results: dict[str, tuple[bool, str]] = {}
    for line in output.splitlines():
        stripped = line.strip()
        for name, test in CHECK_ORDER:
            if stripped.startswith("--- PASS: " + test):
                results[name] = (True, "ok")
            elif stripped.startswith("--- FAIL: " + test):
                results[name] = (False, _failure_detail(output, test))

    fallback = "сборка пакета не удалась" if _looks_like_build_failure(output) else "тест не выполнился"
    tail = _last_meaningful_line(output)
    for name, test in CHECK_ORDER:
        results.setdefault(name, (False, f"{fallback}: {tail}" if tail else f"{fallback} ({test})"))
    return results, output


def _looks_like_build_failure(output: str) -> bool:
    markers = ("cannot use", "undefined:", "syntax error", "build failed", "undeclared name", "# dailybench/pool")
    return any(marker in output for marker in markers)


def _last_meaningful_line(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[-1][:200] if lines else ""


def _failure_detail(output: str, test: str) -> str:
    lines = output.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if f"--- FAIL: {test}" in line)
    except StopIteration:
        return "тест упал"
    tail = [line.strip() for line in lines[start : start + 12] if line.strip()]
    return " | ".join(tail)[:400]


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
    code = extract_code(tr.final_text)
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
