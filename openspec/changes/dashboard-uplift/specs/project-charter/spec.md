# Delta for project-charter

This change adds one Requirement. The existing seven are untouched — nothing is MODIFIED or
REMOVED.

## ADDED Requirements

### Requirement: UI Claim Fidelity

Every numeric figure an interactive surface renders SHALL be traceable to a committed artifact,
and no interactive surface SHALL state a retracted claim as live.

*Evidence-Backed Claims* governs what the project's documents may assert. It does not reach the
Gradio dashboard, which renders figures from Pydantic defaults and hardcoded markdown — and
which reaches more people than any document. A number shown to a user is a claim regardless of
whether it appears in prose.

This Requirement also bans the self-comparison framing the transfer benchmark retracts: a ratio
against an arbitrary pass threshold is not a result. Where a committed baseline exists, the
comparison SHALL be against that baseline, reported in whichever direction the artifacts
support.

#### Scenario: A spike figure is rendered by the dashboard
- GIVEN `dashboard/config.py` declares a transfer MSE that no committed artifact contains
- WHEN the UI claim guard runs
- THEN it SHALL fail naming the metric and the committed value it disagrees with
- AND it SHALL stay failing until the figure matches `config/baselines/transfer_ci.json`

#### Scenario: A retracted figure reappears in a UI surface
- GIVEN a file under `dashboard/` contains the fabricated `0.000209` transfer figure or the
  retracted blanket novelty claim
- WHEN the UI claim guard runs
- THEN it SHALL fail naming the file

#### Scenario: A favourable framing replaces the honest comparison
- GIVEN the operator's zero-shot MSE is compared against a pass threshold rather than against
  the retrained-CNN baseline it loses to
- WHEN that framing is rendered as the headline result
- THEN this is a charter violation
- AND the surface SHALL report the baseline comparison, stating that the value is zero
  retraining rather than peak accuracy
