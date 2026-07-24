# Architecture Decision Records

An **Architecture Decision Record (ADR)** captures a single significant
architectural decision, its context, and its consequences. ADRs are immutable
once accepted — to change a decision, add a new ADR that *supersedes* the old one
rather than editing history.

## Convention

- Files are named `NNNN-kebab-title.md` (zero-padded sequential number).
- Use the [template](TEMPLATE.md) for new records.
- Reference an ADR by its number and title (e.g. "ADR 0002").
- When a decision changes, add a new ADR and mark the old one *Superseded by NNNN*.

## Index

| # | Title | Status |
| --- | --- | --- |
| [0001](0001-chess-self-play.md) | Chess self-play training architecture | Accepted |
| [0002](0002-mouse-droid-fusion-integration.md) | Mouse-Droid-AGI fusion-head integration (stable `src.modeling` surface) | Accepted |

> ADR 0002 is enforced in CI: `tests/modeling/test_public_surface_contract.py`
> turns its "frozen signatures" rule into a mechanical check.
