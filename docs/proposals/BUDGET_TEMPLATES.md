# SBIR Budget Templates

## Overview
Budget templates for each target solicitation. All amounts are pre-negotiation estimates. Adjust overhead rates and labor rates based on actual business costs. SBIR requires the small business to perform minimum 2/3 of Phase I work.

---

## Template A: DoD Phase I ($150K-$250K / 6 months)

**Solicitations**: Navy N252-088, AFRL topics

| Category | Amount | Notes |
|----------|--------|-------|
| **PI Labor** | $90,000-$150,000 | PI at >51% effort, 6 months. Rate: $75-$125/hr |
| **Other Direct Labor** | $0-$20,000 | Part-time research assistant (if applicable) |
| **Fringe Benefits** | $13,500-$25,500 | 15% of labor (self-employment tax, health) |
| **Equipment** | $0-$10,000 | GPU compute (A100 cloud instances or hardware) |
| **Travel** | $3,000-$5,000 | 1-2 program reviews at sponsor location |
| **Materials & Supplies** | $1,000-$2,000 | Software licenses, cloud storage |
| **Subcontractor/Consultant** | $0-$15,000 | University advisor (<1/3 of total) |
| **Indirect Costs** | $15,000-$30,000 | Overhead rate (15-20% of direct costs) |
| **Profit/Fee** | $7,500-$17,500 | 5-7% of total costs |
| **TOTAL** | **$150,000-$250,000** | |

**Key Constraints**:
- PI must be primarily employed (>51%) by the small business
- Small business must perform >=2/3 of work (dollar basis)
- No more than 1/3 to subcontractors/consultants
- Equipment >$5K requires prior approval in most cases

---

## Template B: NSF Phase I ($305K / 12 months)

**Solicitation**: NSF SBIR (rolling Project Pitch system)

| Category | Amount | Notes |
|----------|--------|-------|
| **PI Labor** | $130,000-$160,000 | PI at >51% effort, 12 months |
| **Other Direct Labor** | $20,000-$40,000 | Research engineer or graduate student |
| **Fringe Benefits** | $22,500-$30,000 | 15% of labor |
| **Equipment** | $10,000-$15,000 | GPU hardware or cloud allocation |
| **Travel** | $5,000-$8,000 | NSF SBIR Beat-the-Odds Boot Camp + conference |
| **Materials & Supplies** | $2,000-$5,000 | Software, cloud services |
| **Subcontractor/Consultant** | $15,000-$30,000 | Domain expert consultant |
| **Indirect Costs** | $30,000-$45,000 | Overhead (15-20%) |
| **TABA Supplement** | $0-$6,500 | Technical and Business Assistance (auto-eligible) |
| **TOTAL** | **$305,000** | |

**NSF-Specific Notes**:
- TABA funds ($6,500) are in addition to the $305K award
- I-Corps supplement ($50K) available for customer discovery
- Broader Impacts section required (education, diversity, societal benefit)
- No profit/fee on NSF awards (nonprofit model)

---

## Template C: AFWERX Open Topic Phase I ($75K / 3 months)

**Solicitation**: AFWERX 26.1 Open Topic

| Category | Amount | Notes |
|----------|--------|-------|
| **PI Labor** | $45,000-$55,000 | PI at >51% effort, 3 months intensive |
| **Fringe Benefits** | $6,750-$8,250 | 15% of labor |
| **Equipment** | $0-$3,000 | Cloud GPU (short-term) |
| **Travel** | $2,000-$3,000 | AFWERX Showcase event |
| **Materials & Supplies** | $500-$1,000 | Software, minor supplies |
| **Indirect Costs** | $7,500-$10,000 | Overhead (15-20%) |
| **TOTAL** | **$75,000** | |

**AFWERX-Specific Notes**:
- Fastest award cycle (~6 weeks from submission to award)
- Open Topic = no specific technical requirement (proposer-defined)
- Frame for "hypersonic vehicle design optimization" or "autonomous UAV CFD"
- Gateway to AFRL follow-on funding and STRATFI/TACFI matching

---

## Template D: DARPA Direct-to-Phase-II ($750K-$1.5M / 24 months)

**Solicitation**: DARPA STO/DSO BAA (next AI-simulation topic)

| Category | Amount | Notes |
|----------|--------|-------|
| **PI Labor** | $200,000-$400,000 | PI at >51% effort, 24 months |
| **Other Direct Labor** | $100,000-$200,000 | 1-2 research engineers |
| **Fringe Benefits** | $45,000-$90,000 | 15% of labor |
| **Equipment** | $30,000-$80,000 | Multi-GPU workstation or DGX allocation |
| **Travel** | $15,000-$25,000 | Quarterly program reviews at DARPA |
| **Materials & Supplies** | $5,000-$10,000 | Software, cloud compute |
| **Subcontractor/Consultant** | $75,000-$200,000 | University partner (STTR-like arrangement) |
| **Indirect Costs** | $75,000-$150,000 | Overhead (15-20%) |
| **Profit/Fee** | $50,000-$100,000 | 5-7% of total costs |
| **TOTAL** | **$750,000-$1,500,000** | |

**DARPA-Specific Notes**:
- Requires 10-page feasibility report proving Phase I-equivalent results
- 20-page technical proposal with detailed milestones
- Quarterly milestone reviews with DARPA PM
- Phase III (unlimited, sole-source) potential for transition to services
- Higher overhead rates acceptable given scope

---

## Overhead Rate Calculation

For a single-person LLC with minimal facilities:

| Cost Element | Estimated Annual | Notes |
|---|---|---|
| Office/coworking space | $6,000-$12,000 | Home office or shared space |
| Business insurance | $2,000-$4,000 | E&O + general liability |
| Accounting/legal | $3,000-$5,000 | Tax prep + contract review |
| Software subscriptions | $2,000-$4,000 | Cloud, IDE, collaboration |
| Professional development | $1,000-$3,000 | Conferences, training |
| **Total Indirect** | **$14,000-$28,000** | |
| **Direct Labor Base** | ~$120,000-$180,000 | |
| **Overhead Rate** | **8-20%** | Indirect / Direct Labor |

**Tip**: A 15% overhead rate is defensible for a lean startup. Higher rates (25-40%) require documented cost accounting system (CAS). DCAA audit risk increases above 25%.

## References
- SBIR/STTR Budget Guidelines: https://www.sbir.gov/tutorials/preparing-budget
- Cost volume guidance: See `docs/proposals/templates/sbir_phase1.md` Section C
- Agency-specific configs: `config/proposals/*.yaml`
