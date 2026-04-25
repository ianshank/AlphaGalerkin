# DOE ASCR SBIR Release 2 — Topic Fit Analysis (Skeleton)

**Status:** Awaiting publication of DOE ASCR SBIR Release 2 topics. This file is a scoring skeleton to be populated when topics publish.

**Primary monitor URL:** `https://science.osti.gov/sbir/Funding-Opportunities`

**Secondary monitors (`[VERIFY]` each before relying on):**
- Grants.gov saved-search template — see §Saved searches below.
- DOE SBIR/STTR office announcements.
- Publication-date announcements via DOE ASCR program office email list (`[HUMAN DECISION]` subscribe via `https://science.osti.gov/ascr`).

---

## Saved-search template (Grants.gov)

Create a saved search at `https://www.grants.gov/` with the following criteria:

| Field | Value |
|-------|-------|
| Keyword | `SBIR Phase I` |
| Funding instrument type | `Grant` |
| Agency | `Department of Energy` |
| CFDA Number | `81.049` `[VERIFY current assignment]` |
| Posted date range | Rolling 30 days |
| Notification frequency | Daily email |

Duplicate the saved search with keyword swap for: `ASCR`, `Advanced Scientific Computing Research`, `scientific machine learning`, `PDE`, `adaptive mesh`. Some topics publish under different framings.

---

## Scoring rubric (to apply per topic when Release 2 publishes)

For each candidate topic, score on four axes:

| Axis | Scale | Definition |
|------|-------|------------|
| **1. Keyword match score** | 0–10 | Count of AlphaGalerkin-core keywords present in topic description: `MCTS`, `Monte Carlo Tree Search`, `Galerkin`, `adaptive mesh`, `FEM`, `PDE`, `reinforcement learning`, `operator learning`, `scientific machine learning`, `simulation surrogate`. Max 10. |
| **2. Technical fit** | 1–5 | 5 = direct match (topic names MCTS or AlphaZero-style planning); 4 = strong match (adaptive discretization or RL-for-AMR); 3 = adaptable (generic SciML topic); 2 = stretch (adjacent but narrative-intensive); 1 = poor. |
| **3. Narrative adaptation estimate (days)** | Integer | How many engineer-days to adapt the Genesis proposal narrative to this topic. Baseline: 2–5 days if only §1 (motivation) and §5 (work plan) rewrite; 10+ days if §4 (technical approach) also needs rework. |
| **4. Go / No-Go** | GO / NO-GO / DEFER | GO if KW≥6 AND fit≥4 AND days≤10. NO-GO if fit ≤ 2. DEFER otherwise; revisit when proposal bandwidth frees. |

**Composite score** (for ranking when multiple GO topics exist):

```
composite = (keyword_match * 0.3) + (fit * 2 * 0.5) + ((10 - days_capped_at_10) * 0.2)
```

Higher is better. Ties broken by budget ceiling (higher ceiling first), then by deadline (nearer first) if both are GO.

---

## Topic scoring table

`TODO: populate when Release 2 topics publish`

| # | Topic number | Topic title | KW match | Tech fit | Days | Go/No-Go | Composite | Notes |
|---|--------------|-------------|----------|----------|------|----------|-----------|-------|
| 1 | `TODO` | `TODO` | `TODO` | `TODO` | `TODO` | `TODO` | `TODO` | `TODO` |
| 2 | `TODO` | `TODO` | `TODO` | `TODO` | `TODO` | `TODO` | `TODO` | `TODO` |
| 3 | `TODO` | `TODO` | `TODO` | `TODO` | `TODO` | `TODO` | `TODO` | `TODO` |

---

## Expected ASCR SBIR topic areas (historical pattern, `[VERIFY]` against actual Release 2)

Based on ASCR historical releases (2022–2025), topics have typically covered:

- AI / ML for scientific applications
- Advanced algorithms for exascale computing
- Quantum computing + simulation
- Data management and analytics for scientific workflows
- Edge computing for experimental facilities
- Software productivity and sustainability

AlphaGalerkin is a natural fit for the AI/ML-for-science and advanced-algorithms bucket. If Release 2 continues that pattern, expect at least 1 GO-level topic.

---

## Narrative reuse plan (Genesis → ASCR SBIR)

Per the Execution Plan (Track D), approximately 70% of the Genesis technical narrative transfers directly to ASCR SBIR. Expected adaptations:

| Section | Reuse from Genesis? | Rewrite effort |
|---------|---------------------|----------------|
| §1 Motivation | Partial — swap Genesis National S&T Challenge for topic-specific framing | 1–2 days |
| §3 Background / related work | Full reuse | 0 days |
| §4 Technical approach | Full reuse | 0 days |
| §5 Preliminary results | Full reuse | 0 days |
| §6 Work plan | Rewrite — align to SBIR Phase I 12-month $250K structure (vs Genesis 9-month $500K–$750K) | 2 days |
| §7 Transition plan | Rewrite — SBIR requires commercialization plan; Genesis does not | 3 days — pull from `sbir/commercialization_plan.md` |
| §8 Team | Full reuse | 0 days |
| §10 Budget | Rewrite — $250K vs $500K–$750K ceiling | 1 day |
| §11 DMSP | Full reuse with light edits | 0.5 day |
| Commercialization plan | NEW (not required by Genesis) | 3 days — `sbir/commercialization_plan.md` |

**Estimated total adaptation effort:** 10–12 engineer-days from Genesis completion to ASCR SBIR submission, assuming the topic fit is strong.

---

## Timeline integration

```
Track D timeline (from Execution Plan):
  Day 28: baseline Pareto plot complete (from Track B.6)
  Day ~45: Genesis Phase I proposal complete (primary focus)
  Day ~56: Genesis submission
  Day 56+: begin ASCR SBIR adaptation, contingent on Release 2 publication date
```

If Release 2 publishes before day 56, do NOT split focus. Finish Genesis first. Missing a Genesis deadline to chase an ASCR SBIR deadline is not a trade we make.

---

## Open items

- [ ] `[HUMAN DECISION]` Subscribe to ASCR SBIR announcement channels.
- [ ] `[VERIFY]` CFDA number for ASCR SBIR (`81.049` listed above — confirm at submission time).
- [ ] `[HUMAN DECISION]` Who owns the Release-2-publication-watch task (weekly monitor of `https://science.osti.gov/sbir/Funding-Opportunities`).
- [ ] `[HUMAN DECISION]` Whether to file a Direct-to-Phase-II on ASCR track if topic fit permits and preliminary results are strong enough (bypasses Phase I, 6–12 months earlier to market, but requires stronger evidence).
