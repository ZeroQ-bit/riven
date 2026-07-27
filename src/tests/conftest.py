"""Pytest configuration.

Provides lightweight stubs for native dependencies that are unavailable in some
test environments (e.g. pyfuse3, which requires libfuse3 to build and is only
used by the optional fuse mount feature - not by the downloader logic).
"""

from __future__ import annotations

import sys
import types
from typing import Any


def _ensure_stub(name: str, attrs: dict | None = None) -> None:
    """Register a bare module stub if the real package isn't importable."""
    if name in sys.modules:
        return
    try:
        __import__(name)
    except ImportError:
        module = types.ModuleType(name)
        for attr, value in (attrs or {}).items():
            setattr(module, attr, value)
        sys.modules[name] = module


# pyfuse3 is imported eagerly by program.services.streaming.media_stream but is
# only used by the fuse mount feature. Stub it (and the type aliases media_stream
# references at class-definition time) so the import chain resolves without the
# native libfuse3 dependency. On CI the real pyfuse3 is installed and wins.
_ensure_stub(
    "pyfuse3",
    {
        "EntryAttributes": types.SimpleNamespace,
        "FUSEError": RuntimeError,
        "FileHandleT": int,
        "FileInfo": types.SimpleNamespace,
        "FileNameT": bytes,
        "InodeT": int,
        "ModeT": int,
        "Operations": object,
        "ROOT_INODE": 1,
        "ReaddirToken": Any,
        "RequestContext": Any,
        "default_options": set(),
    },
)
