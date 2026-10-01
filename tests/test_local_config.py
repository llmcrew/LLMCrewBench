"""What must stay local: absolute model paths, endpoints and API keys.

The repository is public, the measurement machine is not. Three things therefore
have to survive a clone without leaking, and be overridable without editing
tracked files:

* model paths that are absolute on the author's machine (``C:\\Users\\...``,
  ``/Users/...``) live only in ``configs/reference.json``;
* keys and endpoints come from the environment (``DAILYBENCH_API_KEY``,
  ``DAILYBENCH_BASE_URL``), never from a committed value;
* editing the ignored local copy must not disturb the shipped defaults, and a
  run's ``config.json`` must carry no secret.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from dailybench import reference
from dailybench.config import RunConfig

ROOT = Path(__file__).resolve().parent.parent
SHIPPED = ROOT / "configs" / "reference.json"

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git недоступен")


def _tracked(relative: str) -> bool:
    import subprocess

    result = subprocess.run(["git", "check-ignore", "-q", relative], cwd=ROOT, capture_output=True)
    return result.returncode != 0


def test_the_shipped_reference_file_is_tracked_and_parses():
    assert _tracked("configs/reference.json")
    data = json.loads(SHIPPED.read_text(encoding="utf-8"))
    assert data["models"], "эталонные конфигурации обязаны ехать в git"
    for entry in data["models"]:
        assert entry.get("id"), entry
        assert "base_url" in entry or "runtime" in entry, entry


#: Machine paths may live in exactly two places: the data file a run reads, and
#: the in-code defaults used when that file is absent. A third copy is a bug.
ALLOWED_PATH_HOLDERS = {"configs/reference.json", "dailybench/reference.py", "tests/test_local_config.py"}


def test_absolute_paths_live_only_in_the_reference_table():
    offenders: list[str] = []
    for path in sorted(ROOT.rglob("*.py")) + [SHIPPED]:
        if any(part in {".venv", "__pycache__"} for part in path.parts):
            continue
        relative = str(path.relative_to(ROOT))
        if relative in ALLOWED_PATH_HOLDERS:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for needle in ("C:\\\\Users\\\\", "/Users/", "/home/"):
            if needle in text:
                offenders.append(f"{relative}: {needle}")
    assert not offenders, f"пути машины обязаны жить только в {ALLOWED_PATH_HOLDERS}: {offenders}"


def test_the_data_file_and_the_code_defaults_do_not_drift():
    """The table exists twice so a clone can edit it; divergence must fail here."""
    shipped = json.loads(SHIPPED.read_text(encoding="utf-8"))
    assert shipped["policy"] == reference.POLICY
    assert shipped["runtimes"] == reference.RUNTIMES
    by_id = {entry["id"]: entry for entry in shipped["models"]}
    code = {entry["id"]: entry for entry in reference.MODELS}
    assert set(by_id) == set(code), "списки конфигураций разошлись"
    for config_id, entry in by_id.items():
        assert entry == code[config_id], f"{config_id}: запись в файле и в коде разошлись"


def test_a_local_override_wins_over_shipped_defaults(tmp_path: Path):
    """An ignored local file may replace the shipped one without touching it."""
    local = tmp_path / "reference.json"
    shipped = json.loads(SHIPPED.read_text(encoding="utf-8"))
    shipped["models"] = [dict(shipped["models"][0], id="local-only", model="/home/person/models/x.gguf")]
    local.write_text(json.dumps(shipped, ensure_ascii=False), encoding="utf-8")

    assert reference.resolve("local-only", local).entry["model"] == "/home/person/models/x.gguf"
    with pytest.raises(KeyError):
        reference.resolve("local-only", SHIPPED)

    entries = reference.model_entries(local)
    assert len(entries) == 1 and entries[0].id == "local-only"


def test_keys_come_from_the_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DAILYBENCH_API_KEY", "sk-local-secret")
    assert RunConfig().api_key == "sk-local-secret"
    monkeypatch.delenv("DAILYBENCH_API_KEY")
    assert RunConfig().api_key == "not-needed"


def test_a_run_config_carries_the_key_but_no_path(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DAILYBENCH_API_KEY", "sk-local-secret")
    payload = json.loads(RunConfig().to_json())
    assert payload["api_key"] == "sk-local-secret"  # ключ нужен для воспроизведения
    assert payload["weights_path"] == ""  # путь появляется только из reference-записи
    assert "/Users/" not in json.dumps(payload) and "C:\\\\Users" not in json.dumps(payload)


def test_env_override_reaches_the_client(monkeypatch: pytest.MonkeyPatch):
    """A clone with one env var is enough to point the harness at its own server."""
    monkeypatch.setenv("DAILYBENCH_BASE_URL", "http://127.0.0.1:9999/v1")
    assert RunConfig().base_url == "http://127.0.0.1:9999/v1"
