"""``python -m dailybench`` entry point."""

import sys

from .cli import main

if __name__ == "__main__":
    # A C-locale Python (LC_ALL=C, PYTHONUTF8=0) defaults stdout to ASCII and
    # every Russian message would crash the CLI mid-command. The output
    # encoding is a terminal property, not a data format: mojibake on an
    # ASCII terminal is recoverable, a mid-run UnicodeEncodeError is not.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):  # pragma: no cover - closed stream
                pass
    raise SystemExit(main())