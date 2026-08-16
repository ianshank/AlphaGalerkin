# AlphaGalerkin — DOE Genesis Mission Phase I Budget Justification (Template)

**Solicitation:** DOE Genesis Mission DE-FOA-0003612 (Phase I, $500K–$750K, 9 months)

**Status of this document:** Template only. All dollar figures are `[HUMAN DECISION]` or `[PLACEHOLDER]`. Claude Code does not pick numbers. The human PI must fill in actual rates, confirm percentages, and total everything.

**Required inputs before finalizing (`[HUMAN DECISION]` all):**
- Entity fringe rate
- Entity indirect rate (negotiated vs de minimis)
- Actual salary basis for PI / co-PI / postdoc
- Final co-PI identity and institution (for subaward budget)
- Compute vendor and rate sheet (cloud vs on-prem vs DOE allocation)

---

## Summary table

| Category | Subtotal (`[HUMAN DECISION]` / `[PLACEHOLDER]`) |
|---|---|
| A. Personnel (salary) | `[HUMAN DECISION]` |
| B. Fringe benefits | `[HUMAN DECISION]` |
| C. Travel | `[PLACEHOLDER]` $4,500 total |
| D. Materials & supplies / compute | `[PLACEHOLDER]` $25,000 |
| E. Subaward(s) to co-PI institution(s) | `[HUMAN DECISION]` |
| F. Other direct costs | `[HUMAN DECISION]` |
| **G. Modified Total Direct Costs (MTDC)** | **Sum of A–F less excluded items** |
| H. Indirect (F&A) costs | `[HUMAN DECISION]` — see §H below |
| **I. Total proposed cost** | **Sum of G + H** — must fit in $500K–$750K |

---

## A. Personnel

| Role | Person | Effort | Basis salary | Period | Requested |
|------|--------|--------|--------------|--------|-----------|
| PI | Ian Cruickshank | `[HUMAN DECISION]` — DOE typically expects > 1 CM; SBIR PI requires > 50% effort but Genesis may differ `[VERIFY]` | `[HUMAN DECISION]` | 9 months | `[HUMAN DECISION]` |
| Co-PI (academic, subawardee) | `[HUMAN DECISION]` from `partners/academic_candidates.md` | 1–2 summer months `[HUMAN DECISION]` | `[HUMAN DECISION]` institutional rate | 9 months | `[HUMAN DECISION]` |
| Postdoc (optional) | `[HUMAN DECISION]` — yes/no drives budget by roughly $90K–$130K fully loaded `[PLACEHOLDER]` | 100% | `[HUMAN DECISION]` institutional scale | 9 months | `[HUMAN DECISION]` |
| Graduate student (optional) | `[HUMAN DECISION]` — alternative to postdoc; lower cost, lower throughput | 50% `[HUMAN DECISION]` | `[HUMAN DECISION]` | 9 months | `[HUMAN DECISION]` |

**Note on PI effort:** SBIR rules require > 50% PI employment with the small business and > 50% effort. Genesis Mission rules for PI effort `[VERIFY — may differ from SBIR]`. Calibrate before locking.

## B. Fringe benefits

`[PLACEHOLDER]` 25–35% of applicable personnel salary. Actual entity fringe rate `[HUMAN DECISION]`. Subaward fringe rate applies to co-PI and any subaward postdoc at subaward institution's negotiated rate.

## C. Travel

| Trip | Purpose | Cost estimate |
|------|---------|---------------|
| 1 DOE PI meeting (DC or site) | Required PI meeting attendance | `[PLACEHOLDER]` $2,000 |
| 1 conference | SIAM CSE / ICML / NeurIPS / SC — `[HUMAN DECISION]` | `[PLACEHOLDER]` $2,500 |
| **Travel subtotal** | | **`[PLACEHOLDER]` $4,500** |

## D. Materials & supplies / compute

| Item | Purpose | Cost |
|------|---------|------|
| Compute credits | Benchmark runs, MCTS self-play, hp-adaptive FEM sweeps | `[PLACEHOLDER]` $15,000 |
| Cloud (GPU instances) | Bursty large-scale training runs | `[PLACEHOLDER]` $10,000 |
| **M&S subtotal** | | **`[PLACEHOLDER]` $25,000** |

**Notes:**
- Compute budget may be `[HUMAN DECISION]` replaced or reduced if lab partner provides allocation, or if INCITE/ALCC allocation is secured (long lead-time — unlikely within Phase I 9-month window).
- Cloud vendor `[HUMAN DECISION]` GCP / AWS / Azure / Lambda Cloud / CoreWeave — affects rate sheet.

## E. Subaward(s)

Subaward budget to academic co-PI institution. Typical structure includes:

| Component | Note |
|-----------|------|
| Co-PI summer salary | See §A |
| Co-PI fringe | Institutional rate |
| Postdoc (if subawarded instead of primary) | See §A |
| Subaward institution indirect | Institutional negotiated rate, applied to MTDC at the subawardee level |
| Subaward travel | Bundled or separate `[HUMAN DECISION]` |

First $25K of each subaward is included in prime's MTDC base for indirect cost calculation; amounts above $25K per subaward are excluded. `[VERIFY]` against current DOE policy at submission time.

## F. Other direct costs

| Item | Note |
|------|------|
| Publication fees (open-access APCs) | `[PLACEHOLDER]` per-paper $2K–$5K `[HUMAN DECISION]` |
| Software licenses (if any non-OSS required) | `[HUMAN DECISION]` — AlphaGalerkin stack is OSS; no licenses expected |
| Equipment (>$5K per item) | None planned unless `[HUMAN DECISION]` on-prem GPU purchase |

## G. Modified Total Direct Costs (MTDC)

MTDC = (A + B + C + D + E + F) − (equipment > $5K + each subaward above $25K + tuition + certain other exclusions).

`[HUMAN DECISION]` — finalize MTDC once A–F are filled.

## H. Indirect (F&A) costs

**Default for new entities without a negotiated rate:** de minimis 10% MTDC per 2 CFR §200.414 (Uniform Guidance). `[VERIFY]` that Genesis Mission accepts de minimis — most DOE solicitations do, but some larger programs require a negotiated rate.

**If entity has a negotiated rate agreement (NICRA):** use negotiated rate. `[HUMAN DECISION]`.

**Subawardee indirect:** applies at subawardee's own rate against subawardee's MTDC.

Formula:
```
Indirect (prime) = prime_rate × MTDC_prime
```

Where `prime_rate` is either `0.10` (de minimis) or the negotiated agreement rate.

## I. Total proposed cost

```
Total = MTDC + Indirect
```

**Must fit in $500K–$750K range per solicitation.** If the build-up exceeds the ceiling, cut order (suggested, `[HUMAN DECISION]` to ratify):
1. Postdoc effort (drop from 100% to 50%, or swap for grad student)
2. Conference travel (drop from 1 trip to 0 if necessary)
3. Cloud compute (shift to lab partner allocation)
4. Subaward scope

If the build-up is well below the floor ($500K), reconsider whether postdoc or additional benchmark scope should be added.

---

## Cost-volume attachments checklist

- [ ] DOE Phase I budget template (`phase_i_template.xlsx`) `[VERIFY current amendment]`
- [ ] Narrative justification (this file, post-fill-in)
- [ ] Subaward budget from co-PI institution (separate template from subawardee)
- [ ] Subaward budget narrative from co-PI institution
- [ ] Current and pending support forms (one per key personnel)
- [ ] Negotiated indirect rate agreement (NICRA) if applicable, or de minimis declaration

---

## Key decisions the human PI must make before this template is usable

1. `[HUMAN DECISION]` Entity structure and negotiated indirect rate status (or declare de minimis 10%).
2. `[HUMAN DECISION]` PI salary basis and exact effort percentage.
3. `[HUMAN DECISION]` Co-PI identity, institution, and effort level (1 month vs 2 months).
4. `[HUMAN DECISION]` Postdoc vs graduate student vs neither.
5. `[HUMAN DECISION]` Compute strategy: cloud credits purchased, DOE lab allocation, or hybrid.
6. `[HUMAN DECISION]` Total requested amount within the $500K–$750K window.
7. `[VERIFY]` Genesis Mission PI effort rules (may differ from SBIR 50% floor).
8. `[VERIFY]` Genesis Mission indirect-cost rules (de minimis acceptable? subaward policy?).
