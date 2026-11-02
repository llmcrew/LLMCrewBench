from __future__ import annotations

import copy
import socket
from collections.abc import Callable

import pytest

from dailybench import core
from dailybench.config import RunConfig
from dailybench.suite import build_suite


def pytest_configure(config):
    """CI memory cap: shrink the default long-context window everywhere.

    A suite built at the published 262144-token window peaks at ~37 GiB of
    resident memory while the LC records are joined — the hosted CI runners
    have ~7 GiB free and the step dies as a SIGTERM cancel mid-suite. The
    tests assert structure and determinism, not the absolute scale, so the
    ``DAILYBENCH_LONGCTX_WINDOW`` env caps the *default* only (explicit
    windows in a test still win).
    """
    import os

    window = int(os.environ.get("DAILYBENCH_LONGCTX_WINDOW") or 0)
    if window and window < RunConfig.context_window:
        # A dataclass default read through the class stays a plain int;
        # mutating it is visible to every RunConfig() without the field set.
        RunConfig.__dataclass_fields__["context_window"].default = window

@pytest.fixture()
def refusing_endpoint() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}/v1"


@pytest.fixture(scope="session")
def suite_for() -> Callable[[str], list[core.TaskInstance]]:
    suites: dict[str, list[core.TaskInstance]] = {}

    def cached(profile: str) -> list[core.TaskInstance]:
        if profile not in suites:
            suites[profile] = build_suite(RunConfig(profile=profile))
        return copy.deepcopy(suites[profile])

    return cached
