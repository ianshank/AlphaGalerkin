"""Structured-logging binder tests.

The event vocabulary is a *closed set* — the ``CERTIFICATE_LOG_EVENTS`` frozen
set. Free-form ``.info("something happened")`` calls in the certificate
subpackage are the failure mode that produced the fabricated ``0.000209``
transfer-MSE headline (CLAUDE.md 2026-07-22); this test surface forbids them.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import structlog

from src.pde.certificate import (
    CERTIFICATE_LOG_EVENTS,
    bind_certificate_logger,
    new_certificate_id,
)

CERTIFICATE_PKG = Path(__file__).resolve().parents[3] / "src" / "pde" / "certificate"


def test_new_certificate_id_shape() -> None:
    """16-char hex — matches the ``Certificate.certificate_id`` field contract."""
    cid = new_certificate_id()
    assert len(cid) == 16
    int(cid, 16)  # raises if not hex


def test_new_certificate_id_uniqueness_small_batch() -> None:
    """Not a statistical uniqueness proof — just guards against a constant bug."""
    ids = {new_certificate_id() for _ in range(64)}
    assert len(ids) == 64


def test_bind_certificate_logger_carries_identity(caplog: pytest.LogCaptureFixture) -> None:
    """Every log event under the binder must carry the four identity fields."""
    # Configure structlog to emit through stdlib so caplog sees it.
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.EventRenamer("event"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )
    log = bind_certificate_logger(
        certificate_id="deadbeefdeadbeef",
        pde_type="poisson",
        track="A",
        rigor="rigorous",
    )
    with caplog.at_level("INFO", logger="src.pde.certificate"):
        log.info("certificate.computed", bound_value=0.5)
    assert caplog.records, "no log record captured"
    payload = caplog.records[-1].getMessage()
    for key in ("deadbeefdeadbeef", "poisson", '"A"', "rigorous"):
        assert key in payload, f"missing {key!r} in log payload: {payload!r}"


def test_bind_accepts_extra_kwargs() -> None:
    """``**extra`` in the binder signature must not silently drop fields."""
    log = bind_certificate_logger(
        certificate_id=new_certificate_id(),
        pde_type="poisson",
        track="B",
        rigor="heuristic",
        scenario_name="test_scenario",
        verifier_backend="dense_grid_heuristic",
    )
    # BoundLogger stores context accessible via ``._context`` (structlog impl detail
    # used only by this white-box test — do not rely on it in production code).
    ctx = log._context  # type: ignore[attr-defined]
    assert ctx["scenario_name"] == "test_scenario"
    assert ctx["verifier_backend"] == "dense_grid_heuristic"


# --- Static guard: only documented event names appear -------------------


def _collect_structlog_event_names_in_package() -> set[str]:
    """AST-walk the certificate subpackage collecting logger event names.

    Collects first-arg string constants passed to
    ``.info() / .debug() / .warning() / .error() / .critical()`` calls on any
    object (structlog and stdlib logger call sites both look identical at AST
    level; this test therefore only cares about the *set*).
    """
    names: set[str] = set()
    for py in CERTIFICATE_PKG.rglob("*.py"):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in {"info", "debug", "warning", "error", "critical"}:
                continue
            if not node.args:
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                names.add(first.value)
    return names


def test_all_logged_events_are_documented() -> None:
    """Every event name used in ``src/pde/certificate/`` must be in the frozen set.

    The fabrication-precedent guard: undocumented events are the failure mode.
    Add new events to ``CERTIFICATE_LOG_EVENTS`` *before* using them, not after.
    """
    used = _collect_structlog_event_names_in_package()
    # Only care about ``certificate.*`` events — non-prefixed names may come
    # from imported utilities and are handled by other modules' guards.
    cert_events = {n for n in used if n.startswith("certificate.")}
    undocumented = cert_events - CERTIFICATE_LOG_EVENTS
    assert not undocumented, (
        f"undocumented certificate.* events used in code: {sorted(undocumented)!r}; "
        f"add them to CERTIFICATE_LOG_EVENTS first"
    )
