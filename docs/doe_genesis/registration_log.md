# DOE Genesis Mission — Registration & Compliance Tracker

**Solicitation:** DE-FOA-0003612 (DOE Genesis Mission, Phase I)
**Applicant Entity:** `[HUMAN DECISION]` (LLC / C-corp / academic host TBD)
**Principal Investigator:** `[HUMAN DECISION]`
**Authorized Organizational Representative (AOR):** `[HUMAN DECISION]`

---

## 1. Purpose

This document tracks the administrative registrations required to submit a
Phase I proposal to DOE Genesis Mission under DE-FOA-0003612. Registrations
must be completed in a strict order because downstream accounts (PAMS,
Grants.gov) cannot be fully provisioned without an upstream Unique Entity
Identifier (UEI) from SAM.gov. Historically SAM.gov issuance has taken
3–10 business days after Entity Validation completes, so we treat it as
the critical-path item.

> **START DAY 1 — SAM.gov UEI request.** Do not wait on anything else.
> SAM.gov Entity Validation (IRS match, physical-address verification,
> notarized letter if required) is the long-pole. Every other registration
> blocks on it.

---

## 2. Registration Matrix

| # | Registration | Portal / URL | Status | Started | Completed | Next Action | Blocker |
|---|---|---|---|---|---|---|---|
| 1 | SAM.gov UEI (Entity Validation) | https://sam.gov | Not started | YYYY-MM-DD | YYYY-MM-DD | **START DAY 1.** Gather EIN, articles of incorporation, utility-bill proof of physical address. | `[HUMAN DECISION]` on legal entity |
| 2 | SAM.gov Full Entity Registration (CAGE code, reps & certs) | https://sam.gov | Not started | YYYY-MM-DD | YYYY-MM-DD | Follow immediately once UEI is issued. Complete Reps & Certs, banking (for grant disbursement), and NAICS selection. | UEI (row 1) |
| 3 | Login.gov account for AOR | https://secure.login.gov | Not started | YYYY-MM-DD | YYYY-MM-DD | Create AOR account with MFA. Required to act on SAM.gov. | None |
| 4 | Grants.gov Organization Registration | https://grants.gov | Not started | YYYY-MM-DD | YYYY-MM-DD | Link UEI + CAGE to Grants.gov org profile; designate AOR role. | UEI (row 1) |
| 5 | Grants.gov Workspace for DE-FOA-0003612 | https://grants.gov | Not started | YYYY-MM-DD | YYYY-MM-DD | Create workspace, assign roles, download opportunity package. | Row 4 |
| 6 | DOE PAMS (Portfolio Analysis & Management System) | https://pamspublic.science.energy.gov | Not started | YYYY-MM-DD | YYYY-MM-DD | PI account with ORCID linkage; institution profile linked to UEI. | UEI (row 1); ORCID |
| 7 | ORCID iD for PI and any Co-PIs | https://orcid.org | Not started | YYYY-MM-DD | YYYY-MM-DD | Create / verify ORCIDs. Required by PAMS and DOE biosketch. | None |
| 8 | NSF SciENcv Biosketch + Current & Pending | https://www.ncbi.nlm.nih.gov/sciencv | Not started | YYYY-MM-DD | YYYY-MM-DD | Generate DOE-format biosketch PDFs for PI + Co-PI(s). | ORCID (row 7) |
| 9 | Genesis Mission Consortium registration | `[CITATION NEEDED — confirm exact portal when FOA Q&A opens]` | Not started | YYYY-MM-DD | YYYY-MM-DD | Monitor DOE Genesis Mission site for consortium onboarding. Submit letter of intent if required. | None |
| 10 | FedConnect / DOE IDC (for award management) | https://www.fedconnect.net | Not started | YYYY-MM-DD | YYYY-MM-DD | Defer until award-stage; register to receive DOE notices. | Award |
| 11 | Cybersecurity — CUI/FISMA posture self-attestation | Internal | Not started | YYYY-MM-DD | YYYY-MM-DD | Document data-handling plan; no CUI expected at Phase I, but confirm with PO. | None |
| 12 | Cost-accounting system (DCAA-acceptable) | Internal / CPA | Not started | YYYY-MM-DD | YYYY-MM-DD | Required only if indirect-rate negotiation triggered. Consult CPA. | Entity type |
| 13 | Human subjects / animal use determinations | Internal | **N/A** | — | — | Not applicable — project is computational mathematics. | — |
| 14 | Environmental compliance (NEPA categorical exclusion) | DOE | Not started | YYYY-MM-DD | YYYY-MM-DD | Request Cat-Ex designation in proposal cover letter (computational work). | None |

---

## 3. Hard Dependencies (Critical Path)

```
Day 1:  SAM.gov UEI request (START DAY 1)  ──┐
                                              ├──► SAM Full Registration
                                              ├──► Grants.gov Org Registration
                                              ├──► DOE PAMS Institution Profile
                                              └──► Genesis Consortium
ORCID (parallel, no blocker)  ──► SciENcv biosketch ──► PAMS PI profile complete
```

Typical wall-clock: 10–20 business days from Day 1 to a submission-ready
state, assuming no SAM.gov validation hiccups (notarized letter, address
mismatch, EIN-name mismatch).

---

## 4. Document Checklist for SAM.gov Entity Validation

Gather before starting row 1:

- [ ] Legal entity name (exact, matching IRS records) — `[HUMAN DECISION]`
- [ ] Physical address (no PO boxes) with supporting utility bill or lease
- [ ] EIN / TIN confirmation letter from IRS (CP-575 or 147-C)
- [ ] Articles of incorporation / formation (state-stamped)
- [ ] Authorized signatory designation
- [ ] Bank routing + account info (for EFT of grant funds)
- [ ] NAICS code selection: proposed **541715** (Research and Development in
      the Physical, Engineering, and Life Sciences (except Nanotechnology
      and Biotechnology)). Confirm with CPA.

---

## 5. Weekly Status Log

Update the matrix above weekly. Log significant events here:

| Date | Event | Owner | Notes |
|---|---|---|---|
| YYYY-MM-DD | — | — | (Log entries added as registrations progress.) |

---

## 6. Contacts

| Role | Name | Email | Phone |
|---|---|---|---|
| PI | `[HUMAN DECISION]` | — | — |
| AOR | `[HUMAN DECISION]` | — | — |
| DOE Program Manager (Genesis Mission) | `[CITATION NEEDED — confirm from FOA]` | — | — |
| SAM.gov Federal Service Desk | https://fsd.gov | (866) 606-8220 | — |
| Grants.gov Support | support@grants.gov | (800) 518-4726 | — |

---

## 7. Risk Notes

- **SAM.gov validation rejection** is the single highest-probability delay.
  If initial entity validation fails, SAM.gov will request a notarized
  letter signed by an authorized official; this adds 5–10 business days.
  Start a notarized letter draft on Day 1 as a hedge.
- **CAGE code delay**: CAGE is issued by DLA after SAM.gov completes. It
  can lag the UEI by 2–5 business days. Grants.gov submission does not
  strictly require CAGE, but DOE PAMS may.
- **UEI-to-PAMS latency**: PAMS syncs with SAM.gov on a periodic schedule.
  Expect up to 48 hours after SAM completion before PAMS recognizes the
  institution.
