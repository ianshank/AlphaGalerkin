"""Dev-side tooling for the claude-code-platform marketplace.

Subpackages:
- ``tools.hook_runtime``: canonical stdlib-only runtime vendored into each
  plugin (ADR-0002). Must never grow third-party imports.
- ``tools.validate``: pydantic-based static validation gates run in CI.

Modules:
- ``tools.sync_runtime``: vendors ``hook_runtime`` into every plugin.
- ``tools.sync_catalog``: regenerates the marketplace catalog from plugin
  manifests plus the release pin manifest (ADR-0003).
"""
