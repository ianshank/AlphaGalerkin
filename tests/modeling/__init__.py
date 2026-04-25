"""Unit tests for ``src.modeling``.

Keeping this as a real package (with ``__init__.py``) matches the
convention used by ``tests/agents/``, ``tests/templates/``, and most
other ``tests/*`` subpackages. It also guarantees that
``from tests.modeling._signature_utils import ...`` resolves through
regular import machinery rather than PEP 420 namespace fallback — the
latter is fragile when a matching path lives elsewhere on ``sys.path``
(e.g. an installed site-packages copy).
"""
