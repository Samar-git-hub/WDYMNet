"""WDYMNet — word-level emphasis, emotion and sarcasm tagging from speech."""
import sys

# Windows consoles default to cp1252; keep unicode log lines (→, —) from crashing scripts.
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")
