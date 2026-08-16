# Academic PI Cold-Outreach Email Template — AlphaGalerkin / DOE Genesis Phase I

**Purpose:** Reusable cold-email template for first-contact outreach to candidate academic co-PIs listed in `partners/academic_candidates.md`. Designed to secure a 30-minute call within 2 weeks.

**Usage rules**
- Do NOT send this template unmodified. Every `{PLACEHOLDER}` and `[HUMAN WRITES]` block must be personalized — generic cold emails to senior applied mathematicians are ignored.
- Attach the one-pager at `/home/user/AlphaGalerkin/proposal/concept_note.md` (convert to PDF first).
- The demo-link placeholder points at the headline Pareto plot artifact (`benchmarks/results/headline_2026_04/pareto_plot.png`) — do NOT send the email before that artifact exists. See Track B.6 in the execution plan.
- Keep to ≤ 250 words in the body. Senior PIs scan.
- Send from an institutional or company domain, never a generic `@gmail.com`.
- Bcc the human PI on every send for tracking.

---

## Template

**Subject line options (pick one — `[HUMAN DECISION]`):**
1. `DOE Genesis Phase I co-PI inquiry — MCTS + Galerkin methods for adaptive PDE solving`
2. `{THEIR_PUBLICATION} + MCTS look-ahead — Genesis Phase I collaboration?`
3. `Phase I co-PI invitation: learn the method, not the solution`

---

```
To: {FIRST_NAME}.{LAST_NAME}@{INSTITUTION}.edu  [VERIFY canonical format per institutional directory]
From: {PI_NAME} <{PI_EMAIL}>
Bcc: {INTERNAL_TRACKING_ADDRESS}

Subject: {SUBJECT_LINE_CHOICE}

Dear Professor {LAST_NAME},

[HUMAN WRITES — 1 sentence genuine tie-in to their recent work; cite
{THEIR_PUBLICATION} by short title. Example: "I've been studying your
{YEAR} paper on {SHORT_TITLE} and in particular the result that
{SPECIFIC_TIE_IN}."]

I'm {PI_NAME}, PI on AlphaGalerkin, a Monte Carlo Tree Search + learned
policy/value system that selects among classical Galerkin discretizations —
we learn the *method*, not the solution, so classical convergence guarantees
are preserved. This differentiates us from PINNs / FNO / DeepONet (which
learn the solution and lose those guarantees) and from policy-gradient
RL-for-AMR (which is myopic; AlphaZero-style look-ahead is the lift).

We are preparing a DOE Genesis Mission Phase I application
(DE-FOA-0003612, $500K–$750K, 9 months) and I am looking for an academic
co-PI whose work on {SPECIFIC_TIE_IN} would strengthen the convergence-theory
narrative. Concrete ask: **1–2 months of effort over 9 months** as co-PI,
with an optional postdoc subaward.

A one-page concept note is attached. Early benchmarks (L-shaped Poisson,
Burgers shock, Taylor-Green vortex) are here: {DEMO_LINK_PARETO_PLOT}.

Would you be open to a 30-minute call in the next 2 weeks? I would need a
directional yes/no by {DECISION_DEADLINE — 2 weeks from send date} so that
biosketches and letters of support can be finalized ahead of submission.

[HUMAN WRITES — 1 sentence specific follow-up question that demonstrates
you've read their recent work and are not mass-mailing. Optional but
strongly recommended.]

Best regards,
{PI_NAME}
{TITLE}, AlphaGalerkin
{PI_EMAIL} | {PI_PHONE}
Repo: {REPO_URL}
One-pager: attached
Demo: {DEMO_LINK_PARETO_PLOT}
```

---

## Placeholder reference

| Placeholder | Fill with | Source |
|---|---|---|
| `{FIRST_NAME}` | Recipient's first name | `partners/academic_candidates.md` |
| `{LAST_NAME}` | Recipient's last name | same |
| `{INSTITUTION}` | Lowercase institution domain slug | institutional directory |
| `{THEIR_PUBLICATION}` | Short title (no year) of one specific 2022–2026 paper | Google Scholar |
| `{SPECIFIC_TIE_IN}` | 1-clause description of what in their paper connects to AlphaGalerkin | researcher's recent abstract |
| `{YEAR}` | Publication year | — |
| `{PI_NAME}` | Ian Cruickshank | — |
| `{PI_EMAIL}` | `[HUMAN DECISION]` institutional/company email | — |
| `{PI_PHONE}` | `[HUMAN DECISION]` | — |
| `{TITLE}` | `[HUMAN DECISION]` e.g. "Founder & PI" | — |
| `{REPO_URL}` | `[HUMAN DECISION]` public or gated repo link | — |
| `{DEMO_LINK_PARETO_PLOT}` | URL to the headline Pareto plot | Track B.6 artifact |
| `{DECISION_DEADLINE}` | Send-date + 14 days | calendar |
| `{INTERNAL_TRACKING_ADDRESS}` | Internal bcc address for tracking | `[HUMAN DECISION]` |
| `{SUBJECT_LINE_CHOICE}` | One of the 3 above | `[HUMAN DECISION]` |

---

## Follow-up cadence

1. **T+0:** Initial send.
2. **T+7 days:** Soft bump if no response — one-sentence reply to own thread: *"Bumping this in case it got buried. Happy to pass if timing is bad — just want to know either way before {DECISION_DEADLINE}."*
3. **T+14 days:** Final email with the decision-deadline reminder. After T+14 with no response, move down the priority ranking in `academic_candidates.md`.

---

## Variants

**Variant A — Warm intro through mutual collaborator.** Replace opening with: *"{MUTUAL_COLLABORATOR} suggested I reach out about a DOE Genesis Phase I we're preparing. [HUMAN WRITES — context from mutual]."* Drop the general-purpose elevator pitch; get to the ask faster.

**Variant B — Follow a conference interaction.** Replace opening with: *"Thanks for the conversation at {VENUE_AND_DATE} — as promised, quick follow-up on the DOE Genesis Phase I."*

**Variant C — Lab POC (not academic PI).** For national-lab targets in `partners/lab_candidates.md`, reframe the ask from "co-PI at 1–2 months effort" to "letter of support + Phase II subaward discussion." Lab researchers rarely sign on as formal Phase I co-PIs without program-office involvement.

---

## Anti-patterns — do NOT do any of the following

- Do not send before the headline Pareto plot exists — senior PIs will click the link first.
- Do not send identical text to multiple candidates simultaneously — if they compare notes it destroys credibility.
- Do not overclaim preliminary results. Every numerical claim should be traceable to a committed experiment.
- Do not pre-announce the Genesis topic you are targeting unless the co-PI has said they work in that area. Keep Topic flexibility as long as possible — see `proposal/concept_note.md`.
- Do not attach the full proposal draft in the first email. The one-pager is the bait.
