"""Category modules. Each one registers itself through ``core.register``.

Imported lazily through :func:`load_all` so a partially written working tree
still imports, while a real run fails loudly on a missing category.
"""

from __future__ import annotations

import importlib

MODULES: tuple[str, ...] = (
    "instruction",
    "reasoning",
    "coding",
    "tools",
    "longctx",
    "hallucination",
    "structured",
    "dialogue",
    "retrieval",
    "review",
    "visual",
    "art",
    "visual_input",
)

_loaded = False


def load_all(force: bool = False) -> list[str]:
    """Import every category module, returning the module names."""
    global _loaded
    if _loaded and not force:
        return list(MODULES)
    for name in MODULES:
        importlib.import_module(f"{__name__}.{name}")
    _loaded = True
    return list(MODULES)
