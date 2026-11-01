# Sandboxed execution

Hardened runtime for untrusted candidate code, implementing the isolation
contract in [`docs/FRAMEWORK_SPEC.md`](../docs/FRAMEWORK_SPEC.md) §6:

| Control | Value |
|---|---|
| rootfs | `read_only`; writable surface is exactly: tmpfs `/tmp` (64 MiB), tmpfs `/dev/shm` (64 MiB, `noexec`, explicit — not a daemon default), the explicit `--write-dir` mount (`/out`), plus Docker's `/dev`, `/dev/pts` and the `/etc/{hosts,hostname,resolv.conf}` bind files |
| identity | UID/GID 1000, no login shell, no `HOME`-driven writes |
| capabilities | `cap_drop: ALL`, `no-new-privileges:true` |
| syscalls | `seccomp.json`, `defaultAction: SCMP_ACT_ERRNO` + `defaultErrnoRet: 38` (ENOSYS) with an allow-list |
| processes | `pids.max = 256`, `--init` reaper |
| CPU / memory / IO | cgroups v2 `cpu.max`, `memory.max`, `memory.swap.max=0`, `io.max` |
| network | `none` for the runner and capture; an agent run uses a per-run `--internal` network plus `egress_forward.py`, which dials only the configured endpoint. The agent joins that network alone (`http://dailybench-egress:8080`, or the original hostname pinned to the forwarder for HTTPS/SNI). A loopback HTTPS endpoint is reached with SNI `host.docker.internal`, so its certificate must carry that name. The host gateway is not given to the agent |
| time | absolute deadline, the whole process tree gets `SIGKILL` (`taskkill /T /F` on Windows) |

`timeout` and `OOM` exhaust a declared ceiling and map to `resource_exhausted`
(same as the agent), not a zero that looks like the model. `infra_error`
(daemon/CLI failure, unreadable container state) is `evaluator_failure`.

## Images

Both bases are pinned by digest (`latest` is rejected at run time):

| Image | Base |
|---|---|
| `docker/Dockerfile.agent` (default) | `python:3.12-slim@sha256:2259debd…` + pinned `pi` from `node:22-bookworm-slim@sha256:4b150b48…`; apt from `snapshot.debian.org/archive/debian/20261101T082805Z` (trixie) with pinned versions |
| `docker/Dockerfile.python` | `python:3.12-slim@sha256:2259debd…`; apt sources frozen to the same snapshot |
| `docker/Dockerfile.capture` | `debian:bookworm-slim@sha256:7ff6ba31…`; apt from the same snapshot timestamp (bookworm), versions pinned |
| `docker/Dockerfile.judge` | `python:3.12-slim@sha256:2259debd…`; `ca-certificates` pinned from the trixie snapshot |

```bash
# local build, pinned default base
.venv/bin/python scripts/sandbox_run.py build

# rebuild against an explicit digest
.venv/bin/python scripts/sandbox_run.py build --digest sha256:2259debdc6af85626058a15a7bf44ad6a68b8c98c9ae82a02cf5cdb92f096d47
```

For a published run, build/tag and record the immutable reference:

```bash
docker buildx build --load -f docker/Dockerfile.python \
  --build-arg PYTHON_IMAGE=python:3.12-slim@sha256:2259debd… \
  -t registry.example/dailybench/runner-python .
docker push registry.example/dailybench/runner-python
docker image inspect registry.example/dailybench/runner-python \
  --format '{{index .RepoDigests 0}}'   # -> record image_digest
```

## Provenance: digest, runtime, SBOM

Every run records the executor in the generation manifest (`environment`):
`image_digest` (from `docker image inspect`), the runtime (`runc`/`runsc`),
`SOURCE_DATE_EPOCH` and the image package list (`dpkg-query` inside the
image). The report prints them in the «Среда» section. A resume on another
digest is refused with an explicit error; a host without Docker records
`unverified` and continues. Point the collector at a published image and
gVisor with `DAILYBENCH_SANDBOX_IMAGE` and `DAILYBENCH_SANDBOX_RUNTIME`.

Pin the artefacts before publishing:

```bash
IMAGE=registry.example/dailybench/runner-python
DIGEST=$(docker image inspect "$IMAGE" --format '{{index .RepoDigests 0}}')

# SBOM of the exact digest (SPDX JSON), then pin its hash
syft "$DIGEST" -o spdx-json=docker/runner-python.spdx.json
sha256sum docker/runner-python.spdx.json > docker/runner-python.spdx.json.sha256

# sign the digest, not the mutable tag
cosign sign --yes "$DIGEST"
cosign verify "$DIGEST" \
  --certificate-identity-regexp 'https://github.com/.*' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
```

Commit `docker/runner-python.spdx.json` and its `.sha256` next to this README;
the digest in the report must match the published one. `SOURCE_DATE_EPOCH`
is passed to the build (`--build-arg`, default `1700000000`) and reproduced by
the manifest, so the same inputs give the same bytes.

## Running a command

```bash
.venv/bin/python scripts/sandbox_run.py run --deadline 600 -- \
    python solution.py
```

The task directory (`--task-dir`, default: the repository root) is mounted
read-only at `/workspace` and used as the working directory; `--workspace-rw`
makes that mount writable for a command that must leave an answer behind. A directory passed
with `--write-dir` is the only writable bind mount (at `/out`, exposed to the
child as `$SANDBOX_OUT`). Both are attached with `--mount type=bind`, so a `:`
in the host path is legal; a `,` is refused with an infrastructure error because
it is the field separator of the same syntax. Everything else outside the two
tmpfs mounts is read-only, and `/dev/shm` is mounted explicitly rather than
inherited from the daemon default.

Useful flags:

| Flag | Default | Meaning |
|---|---|---|
| `--runtime runsc\|runc` | `runc` | `runsc` refuses to start with a clear message when gVisor is not registered |
| `--network none\|bridge\|endpoint\|<name>` | `none` | `endpoint` is a per-run internal network plus the forwarder; pass `--endpoint URL` |
| `--endpoint URL` | – | the only host:port the forwarder dials when `--network endpoint` |
| `--deadline SECONDS` | `600` | absolute wall-clock deadline |
| `--cpus`, `--memory`, `--pids-limit` | `2`, `2g`, `256` | cgroup v2 limits |
| `--workspace-rw` | off | let the command write the task directory |
| `--env NAME=VALUE` | – | one allowlisted container variable; Docker inherits nothing, so this list *is* the container environment |
| `--host-gateway` | off | add `host.docker.internal` → host gateway on an explicit bridge. Not used for `endpoint` |
| `--report PATH` | – | also write the JSON report to `PATH` |
| `--allow-latest` | off | permit a tag without a digest |

The JSON report on stdout has the shape:

```json
{
  "status": "ok | timeout | oom | infra_error | failed",
  "error": "null | reason for infra_error",
  "runtime": "runc | runsc",
  "image": "…",
  "image_digest": "sha256:…",
  "source_date_epoch": 1700000000,
  "sbom": ["libc6\t2.36-9+deb12u4", "…"],
  "command": ["python", "solution.py"],
  "network": "none",
  "mounts": [{"src": "/tmp/task", "dst": "/workspace", "readonly": true}],
  "env": ["PATH"],
  "cgroups": {"cpus": "2", "memory": "2g", "pids_limit": 256},
  "rootfs": "read_only",
  "capabilities": "dropped",
  "container": "dailybench-sandbox-1a2b3c4d5e",
  "deadline_s": 600,
  "duration_s": 1.23,
  "exit_code": 0,
  "oom_killed": false,
  "timed_out": false,
  "runtime_verified": true
}
```

Exit codes: `0` ok, `1` failed (candidate), `2` infrastructure error
(image, runtime, daemon, malformed `SOURCE_DATE_EPOCH`, malformed mount path),
`3` timeout, `4` OOM. Docker's own failure code `125` is `infra_error`/`2`;
`126`/`127` (exec failure inside the container) stay candidate failures. The
container must be observed in a terminal state before the exit code is
classified — if the daemon cannot report one, the run is infrastructure. The
`runtime` field is always recorded, so the manifest and the published report can
state which kernel boundary was used.

## The agent runs here

`dailybench run` executes the agentic categories (C/K/L) inside this sandbox, not
on the host: `dailybench/isolation.py` mounts the task fixture as `/workspace`
(writable — the agent must leave the answer), the harness config as `/agent`, an
allowlisted environment and nothing else. The repository, `task.oracle` and the
neighbouring working copies are therefore invisible, which is the point: a model
that can read the hidden tests next door is measured on better information than
the next one.

The agent image ships the pinned `pi` and checks it (`pi --version`) inside
the container before a task starts; a foreign release is an infrastructure
error, and the run falls back to the host with the reason recorded
(`executor_status: host_run`). Build it with `make agent-image` (or
`scripts/sandbox_run.py build`), point `DAILYBENCH_SANDBOX_IMAGE` at it, and use
`--no-sandbox` only to reproduce a host run deliberately.

## gVisor

`runsc` is optional locally and mandatory for runner/capture in a published
run. Install it on the Docker host (Linux):

```bash
ARCH=$(uname -m)   # x86_64 or aarch64
BASE=https://storage.googleapis.com/gvisor/releases/release/latest/${ARCH}
wget -q "${BASE}/runsc" "${BASE}/runsc.sha512"
sha512sum -c runsc.sha512
chmod +x runsc && sudo mv runsc /usr/local/bin/
sudo runsc install
sudo systemctl restart docker
docker info --format '{{json .Runtimes}}'   # must list "runsc"
```

Then:

```bash
.venv/bin/python scripts/sandbox_run.py run --runtime runsc --deadline 600 -- \
    python solution.py
```

If `runsc` is missing, the script exits `2` and prints the install hint above
instead of silently falling back to `runc`. Rootless Docker (`dockerd-rootless.sh`)
is the expected deployment; the images never rely on root-owned paths.

## Compose baseline

```bash
docker compose -f docker/compose.yaml config -q
docker compose -f docker/compose.yaml run --rm runner python --version
```

`TASK_DIR`, `STORE_DIR`, `STORE_SUBDIR`, `RUNNER_IMAGE`, `CAPTURE_IMAGE`,
`JUDGE_IMAGE`, `SECCOMP_PROFILE` and `IO_DEVICE` (block device for `io.max`,
default `/dev/vda`) are overridable.  `STORE_SUBDIR` defaults to `generation`:
`/store` is that one subdirectory, so code under test cannot rewrite scores or
manifests of an earlier run.  On hosts without that device, drop the
`blkio_config` stanza or point `IO_DEVICE` at the root disk.  Compose is the
reference shape of the controls; the supervisor script is what enforces
per-task deadlines and `SIGKILL`.

The judge is on `dailybench-internal` and reaches the endpoint only through
the `egress` service (`http://dailybench-egress:8080`). That forwarder also
joins the non-internal `egress-out` network so it can dial
`${ENDPOINT_HOST}:${ENDPOINT_PORT}` and nothing else. The supervisor does the
same split with `docker network connect bridge` for a per-run network. Do not
put the judge or the agent on the default bridge.

## Tests

```bash
.venv/bin/python -m pytest tests/test_sandbox.py -q
```

The suite exercises a real `docker run` (build, deadline kill, read-only rootfs,
no network, pids limit, a `:` in the mount path, `sched_setaffinity` plus
threads under the seccomp profile) and a fake `docker` on `PATH` for the exit
`125` classification. The whole module skips when no Docker daemon answers.


## CA для https-endpoint

`run` автоматически монтирует host CA (из `SSL_CERT_FILE`/`CURL_CA_BUNDLE`
или `~/.config/omp/certs/sberdevices-ca-bundle.pem`) read-only и отдаёт
`SSL_CERT_FILE`/`NODE_EXTRA_CA_CERTS` внутрь; без файла запуск идёт с
`ca_mounted=false` и причиной. Образ агента собирается один раз:
`dailybench preflight sandbox` (или автосборка при первом agent-задании).
