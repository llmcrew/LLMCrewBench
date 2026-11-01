from __future__ import annotations

import copy
import socket
from collections.abc import Callable

import pytest

from dailybench import core
from dailybench.config import RunConfig
from dailybench.suite import build_suite


@pytest.fixture()
def refusing_endpoint() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}/v1"


@pytest.fixture(scope="session")
def suite_for() -> Callable[[str], list[core.TaskInstance]]:
    suites: dict[str, list[core.TaskInstance]] = {}
    # Same OOM cap as tests/test_longctx.py: the CI runners cannot hold the
    # full-window stretch suite in memory while it is being built.
    import os

    window = int(os.environ.get("DAILYBENCH_LONGCTX_WINDOW") or 0)

    def cached(profile: str) -> list[core.TaskInstance]:
        if profile not in suites:
            cfg = RunConfig(profile=profile)
            if window and window < cfg.context_window:
                cfg.context_window = window
            suites[profile] = build_suite(cfg)
        return copy.deepcopy(suites[profile])

    return cached
