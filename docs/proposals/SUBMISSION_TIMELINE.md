# SBIR/STTR Submission Timeline

## Overview
Prioritized submission calendar for AlphaGalerkin SBIR targets. SBIR reauthorization (S. 3971) extends program through September 30, 2031, with compressed FY2026 cycles to obligate backlogged funds.

## Submission Calendar

| Priority | Solicitation | Agency | Phase | Funding | Duration | Window | Prep Start | Submit By | Award Est. | Config |
|----------|-------------|--------|-------|---------|----------|--------|------------|-----------|------------|--------|
| **1** | Open Topic 26.1 | AFWERX | I | $75K | 3 mo | Upon reopening (est. Q2 2026) | Immediate | 2 weeks after open | 4-6 weeks | `config/proposals/afwerx_open.yaml` |
| **2** | NSF SBIR Pitch | NSF | I | $305K | 12 mo | Rolling (upon reopening) | Immediate | Rolling | 6-9 mo | `config/proposals/nsf_sbir.yaml` |
| **3** | N252-088 | Navy (NAVAIR) | I | $150-250K | 6 mo | DoD 26.1 cycle (est. Apr-May 2026) | Mar 2026 | ~Jun 2026 | 4-6 mo | `config/proposals/navy_n252_088.yaml` |
| **4** | ASCR C59-01 | DOE | I | $200-250K | 12 mo | FY2026 Release 1 (est. Apr-May 2026) | Mar 2026 | ~Jul 2026 | 6-9 mo | `config/proposals/doe_ascr_c59.yaml` |
| **5** | Direct-to-Phase-II | DARPA (STO/DSO) | II | $750K-$1.5M | 24 mo | Next DSO/STO BAA | Jun 2026 | Per BAA | 4-6 mo | `config/proposals/darpa_d2p2.yaml` |

## Pre-Submission Dependencies

| Dependency | Required For | Status | Reference |
|------------|-------------|--------|-----------|
| SAM.gov registration (UEI + CAGE) | All submissions | Pending | `docs/proposals/SAM_REGISTRATION_GUIDE.md` |
| SBIR.gov company registration | All submissions | Pending | SAM_REGISTRATION_GUIDE.md Step 6 |
| Benchmark results generated | Navy, DOE, DARPA | Available | `config/benchmarks/sbir_suite.yaml` |
| Provisional patent filed | DARPA D2P2, strengthen all | Pending | `docs/proposals/IP_STRATEGY.md` |
| Budget prepared | All submissions | Pending | `docs/proposals/BUDGET_TEMPLATES.md` |

## Preparation Gantt (Weeks from Now)

```
Week:  1   2   3   4   5   6   7   8   9  10  11  12
       |---|---|---|---|---|---|---|---|---|---|---|---|
SAM:   [===REGISTER===]
EIN:   [=]
CAGE:               [===WAIT===]
AFWERX:    [==PREP==][SUBMIT]
NSF:       [====PREP PITCH====][SUBMIT]
NAVY:              [========PREP PROPOSAL========]
DOE:               [========PREP PROPOSAL========]
DARPA:                     [======PREP D2P2 PACKAGE======]
```

## Agency-Specific Notes

### AFWERX (Priority 1 - Lowest Barrier)
- Open Topic: no specific technical topic required, frame for "hypersonic vehicle design optimization"
- Fastest turnaround: $75K in ~6 weeks from submission
- Gateway to larger AFRL funding

### NSF (Priority 2 - Rolling Submissions)
- Project Pitch system: 2500-character description reviewed in 3 weeks
- If pitch accepted, full proposal invited (6 months to submit)
- Frame under "Mathematical and Physical Sciences" division

### Navy N252-088 (Priority 3 - Best Topic Match)
- Topic explicitly seeks "AI/ML toolkits that automate finite element meshing"
- Almost exact match for AlphaGalerkin capabilities
- Requires TPOC (Technical Point of Contact) engagement before submission

### DOE ASCR C59-01 (Priority 4 - Largest Scope)
- Frame as deploying AI-guided algorithms atop ASCR libraries (PETSc, MFEM)
- DOE values exascale readiness and reproducibility
- Longer timeline but larger long-term funding potential

### DARPA D2P2 (Priority 5 - Highest Value)
- Requires 10-page feasibility report + 20-page technical proposal
- Must demonstrate Phase I-equivalent results without Phase I funding
- AlphaGalerkin TRL 3-4 qualifies for Direct-to-Phase-II
- $750K-$1.5M award, sole-source Phase III potential
