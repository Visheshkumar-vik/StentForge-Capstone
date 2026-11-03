"""Append-only JSON-lines event stream for live campaign progress.

Today, progress is human-readable text only (``campaign_18.engine.progress()``)
or per-stage snapshots (``status.json``) -- there is no structured, appendable,
machine-tailable stream.  This matters more on Gadi than on a laptop: a PBS array
job running unattended for hours needs a way to answer "how far did it get?"
without ``qstat``-ing every array index by hand.

``EVENTS.jsonl`` is that stream: one JSON object per line, opened-appended-closed
per call (no held file handle, so parallel array-task workers can't corrupt each
other's writes), readable live with ``tail -f EVENTS.jsonl | jq`` or
:mod:`stentforge.hpc.metrics_tail`.

ADDITIVE ONLY.  This module never replaces ``progress()`` or ``status.json`` --
both keep writing exactly as before.  :func:`emit` mirrors the backend's ``_sync_scratch``
contract exactly: catch, warn, continue -- a metrics-write problem (disk full,
permissions, a bad campaign_root) must never fail a solve.
"""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from stentforge.config import STENTFORGE_ROOT

# The default campaign root (mirrors campaign_18.engine.CAMPAIGN_ROOT's own
# resolution).  NOT imported from campaign_18.engine: that module imports
# stentforge.hpc.metrics (to emit), so importing back would be circular.  Keeping
# the same env var name ($STENTFORGE_CAMPAIGN_ROOT) is what keeps the two in sync.
_DEFAULT_CAMPAIGN_ROOT = STENTFORGE_ROOT / "results_fea" / "campaign_18"


def events_path(campaign_root: "str | Path | None" = None) -> Path:
    """Resolve the EVENTS.jsonl path.

    Resolution order (mirrors ``config.py``'s env-first chain shape):

      1. ``$STENTFORGE_METRICS``       -- an explicit full file path override.
      2. ``<campaign_root>/EVENTS.jsonl`` -- when the caller names its own root
         (e.g. ``campaign_18.engine`` passes its ``CAMPAIGN_ROOT``, or
         ``sweep_worker`` passes its ``iter_dir``).
      3. ``$STENTFORGE_CAMPAIGN_ROOT/EVENTS.jsonl``, defaulting to
         ``<StentForge>/results_fea/campaign_18/EVENTS.jsonl`` -- same env var and
         same default ``campaign_18.engine.CAMPAIGN_ROOT`` itself resolves, so a
         caller that emits with no ``campaign_root`` still lands next to that
         campaign's own PROGRESS.log.

    A pure resolver (like ``config.scratch_root()``): no mkdir, no side effects,
    and re-reads the env var on every call rather than caching it at import time --
    this is what keeps it correct under ``tests/conftest.py``'s per-test
    ``$STENTFORGE_CAMPAIGN_ROOT`` redirect, which only sweeps ``campaign_18``-
    namespaced modules and would not otherwise touch this one.
    """
    env = os.environ.get("STENTFORGE_METRICS")
    if env:
        return Path(env)
    if campaign_root is not None:
        return Path(campaign_root) / "EVENTS.jsonl"
    root = os.environ.get("STENTFORGE_CAMPAIGN_ROOT")
    return (Path(root) if root else _DEFAULT_CAMPAIGN_ROOT) / "EVENTS.jsonl"


def _jsonable(value):
    """NaN/Inf -> ``null``, recursively.

    ``json.dumps`` writes bare ``NaN``/``Infinity`` by default -- a Python extension
    that ``json.loads`` accepts but STRICT parsers reject, ``jq`` included.  That
    would break the file's headline use (``tail -f EVENTS.jsonl | jq``) on a very
    ordinary line, because a timed-out or errored design's stub result carries NaN
    for every QoI -- and timeouts are normal here, not an edge case.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def emit(event_type: str, *, campaign_root: "str | Path | None" = None, **fields) -> None:
    """Append one JSON line: ``{"ts": ISO-8601, "event": event_type, **fields}``.

    Open-append-close per call -- matches ``progress()``'s own pattern and avoids
    corrupting concurrent writers from parallel array-task workers.  ``campaign_root``
    (if given) is only routing -- see :func:`events_path` -- it is never itself
    written as a field.

    NEVER RAISES.  Same contract as the backend's ``_sync_scratch``: a metrics-write failure
    (disk full, permissions, an unwritable path) is caught, warned once to stderr,
    and swallowed -- it must not turn a completed design/job/poll into a crash.
    """
    try:
        path = events_path(campaign_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"ts": datetime.now(timezone.utc).isoformat(), "event": event_type,
                  **_jsonable(fields)}
        line = json.dumps(record, default=str, allow_nan=False)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as exc:                             # pragma: no cover - best-effort
        try:
            print(f"[metrics] warning: failed to emit {event_type!r}: {exc}",
                  file=sys.stderr, flush=True)
        except Exception:
            pass
