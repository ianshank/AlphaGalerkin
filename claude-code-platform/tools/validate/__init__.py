"""Static validation gates for the marketplace (dev-side, CI-enforced).

Run as ``python -m tools.validate``. Covers what ``claude plugin
validate`` does not: catalog/manifest description parity, release-pin
consistency, vendored-runtime parity, path-literal and stdlib-import
gates, and frontmatter linting with the progressive-disclosure limit.
"""

from .config import ValidatorConfig
from .gates import Violation, run_all_gates

__all__ = ["ValidatorConfig", "Violation", "run_all_gates"]
