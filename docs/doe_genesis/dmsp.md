# Data Management and Sharing Plan (DMSP)

**Solicitation:** DE-FOA-0003612 (DOE Genesis Mission, Phase I)
**Compliance:** DOE Public Access Plan (Statement of the Department
of Energy Public Access Plan, 2014, as amended); DOE Order 243.1B
"Records Management Program"; OSTP 2022 memorandum on ensuring free,
immediate, and equitable access to federally funded research.
**Project:** AlphaGalerkin — MCTS-guided Galerkin discretization for
PDEs.

---

## 1. Summary

All data products generated under this award will be made publicly
accessible at the point of peer-reviewed publication (or at award
close, whichever is earlier) under an OSI-approved open-source
license. Source code lives on GitHub; long-term archival snapshots
and citable DOIs are minted via Zenodo. FAIR principles (Findable,
Accessible, Interoperable, Reusable) guide every choice below.

---

## 2. Data Types and Volumes

| Category | Description | Format | Est. Volume (Phase I) |
|---|---|---|---|
| Source code | Python/PyTorch implementation of AlphaGalerkin | `.py`, text | < 50 MB |
| Configuration | Pydantic configs, YAML Hydra schemas | `.yaml`, `.json` | < 10 MB |
| Benchmark results | Error/DOF/time metrics per run | `.csv`, `.json` | 1–5 GB |
| Trained model checkpoints | PyTorch state-dicts with metadata | `.pt`, `.safetensors` | 10–100 GB |
| Exported models | ONNX for cross-framework inference | `.onnx` | 1–10 GB |
| Figures & tables | Publication-quality plots | `.png`, `.pdf`, `.svg` | < 500 MB |
| Provenance logs | Per-experiment run logs | `.jsonl` (structlog) | 1–10 GB |
| Training datasets | Synthetic PDE solutions | `.npz`, `.h5` | 10–50 GB |
| Tests | pytest suites for verification | `.py` | < 20 MB |

No human-subjects data. No animal data. No PII. No CUI. No export-
controlled content anticipated at Phase I; classification review will
be requested if the Phase II roadmap shifts toward
defense-sensitive applications.

---

## 3. Data Formats and Standards

### 3.1 Format Choices

- **Tabular results:** CSV (human-readable) + JSON Lines (programmatic).
  Column names documented in a sibling `schema.json`.
- **Model checkpoints:** PyTorch `.pt` (canonical) plus `.safetensors`
  (security-hardened) for any release asset.
- **Exported inference models:** ONNX opset ≥ 17; validated via
  `src/deployment/validate.py` before release.
- **Training datasets:** NumPy `.npz` for arrays < 1 GB; HDF5 `.h5`
  for larger, structured datasets.
- **Configs:** YAML consumed by Hydra + Pydantic; JSON Schema emitted
  by `make schema` target for external validation.
- **Logs:** JSONL via `structlog` (CLAUDE.md — *Structured logging via
  structlog throughout*).

### 3.2 Metadata Standard

Every public release ships a top-level `metadata.json` following a
documented JSON Schema, with fields:

```json
{
  "title": "string",
  "version": "semver",
  "doi": "string",
  "description": "string",
  "creators": [{"name": "string", "orcid": "string", "affiliation": "string"}],
  "date_created": "YYYY-MM-DDTHH:MM:SSZ",
  "keywords": ["MCTS", "Galerkin", "AMR", "PDE"],
  "license": "string",
  "related_identifiers": [{"type": "doi", "value": "...", "relation": "IsDerivedFrom"}],
  "funding": [{"funder": "DOE ASCR", "award": "DE-FOA-0003612"}],
  "git_sha": "string",
  "python_version": "string",
  "torch_version": "string",
  "hardware": "string"
}
```

This schema is aligned with **DataCite 4.4** and **CodeMeta 3.0** so
that crosswalks to Zenodo, OSTI ELink2, and DOE OSTI are automatic.

### 3.3 Provenance

Every benchmark artifact is tagged with the commit SHA of
AlphaGalerkin that produced it, the exact Pydantic config hash (via
`BaseModuleConfig.compute_hash()` per CLAUDE.md template), the
random seed(s), and the hardware profile. This makes every
published number reproducible from a single `hydra.run` invocation.

---

## 4. Data Repository and Access

### 4.1 Code and Configs

**Primary:** GitHub, public repository
https://github.com/`[HUMAN DECISION — org/repo]`/AlphaGalerkin.
Tagged releases follow semver (`vX.Y.Z`). Each release is mirrored
to Zenodo via GitHub's native Zenodo integration, yielding a
citable DOI per release.

### 4.2 Large Artifacts (Checkpoints, Datasets)

- Tagged release assets: GitHub release page (up to 2 GB per asset).
- Larger artifacts: Zenodo record linked from the corresponding GitHub
  release (up to 50 GB per record, ≤ 200 MB per file with
  multi-file support). For Phase-I-scale artifacts this is
  sufficient.
- Fallback for datasets > 50 GB: DOE OSTI Data Explorer or
  a mutually-agreed DOE facility data repository (NERSC, OLCF).

### 4.3 Publications

Peer-reviewed manuscripts deposited to **DOE OSTI** (via DOE PAGES)
within the DOE-mandated embargo window (currently zero-month
embargo per the 2022 OSTP memo). Preprints also on **arXiv** at
submission time.

### 4.4 Access Model

- Open access, no registration required for read-only.
- Contribution / issue-tracking via GitHub.
- Mirror to a long-term archival host (Zenodo, CERN-backed) ensures
  survival beyond the project funding period.

---

## 5. License

**Code license:** `[HUMAN DECISION]` — choose between:

- **MIT License** — maximally permissive; preferred if broadest
  adoption is the goal.
- **Apache License 2.0** — includes an explicit patent grant,
  recommended given the project's three provisional patent claims
  (CLAUDE.md: *IP Strategy Documented — 3 provisional patent claims,
  publication plan, dual-licensing*).

Recommendation pending: **Apache-2.0** for the public release plus a
commercial-terms licensing option for derivative works requiring
patent indemnity. This is consistent with the documented
dual-licensing strategy.

**Data license:** Creative Commons Attribution 4.0 International
(CC-BY-4.0) for benchmark result tables and any documentation. This
is the DOE-preferred data license (per DOE Public Access Plan
guidance) and satisfies OSTP open-access requirements.

**Model weights license:** Same as code (Apache-2.0 or MIT once the
`[HUMAN DECISION]` is made). We will avoid non-OSI "responsible AI"
licenses (e.g., RAIL) because DOE Public Access Plan requires no
access barriers beyond attribution.

---

## 6. FAIR Compliance

| Principle | Implementation |
|---|---|
| **F1** Globally unique identifier | Zenodo DOI per release; ORCID for each author; software citation file (`CITATION.cff`) at repo root |
| **F2** Rich metadata | `metadata.json` per release; DataCite 4.4 + CodeMeta 3.0 crosswalks |
| **F3** Metadata explicitly includes identifier | `doi` field in `metadata.json`; DOI also embedded in `CITATION.cff` |
| **F4** Registered/indexed | GitHub (indexed by Google Scholar), Zenodo (indexed by DataCite and OpenAIRE), OSTI ELink2 |
| **A1** Retrievable by identifier over standardized protocol | HTTPS from GitHub/Zenodo/OSTI; no authentication required |
| **A1.1** Protocol is open and free | HTTPS; Git over HTTPS |
| **A1.2** Protocol allows authentication when needed | N/A — no access restrictions |
| **A2** Metadata persists even if data not available | Zenodo guarantees metadata persistence by design |
| **I1** Broadly applicable formalism | JSON, YAML, CSV, HDF5, ONNX, PyTorch — all widely supported |
| **I2** Uses FAIR vocabularies | CodeMeta for software; DataCite for citations |
| **I3** Qualified references | `related_identifiers` in `metadata.json` per DataCite schema |
| **R1** Rich, accurate attributes | Metadata schema above |
| **R1.1** Clear & accessible data-usage license | Apache-2.0 / CC-BY-4.0 (pending `[HUMAN DECISION]`) |
| **R1.2** Detailed provenance | Git SHA, config hash, seed, hardware in every artifact |
| **R1.3** Meets domain-relevant standards | Follows PyTorch/ONNX conventions; complies with DOE Public Access Plan |

---

## 7. Retention and Preservation

### 7.1 Retention Period

Minimum **10 years** post-project-end, per DOE Order 243.1B
"Records Management Program" and NARA General Records Schedule
6.1 (research & development records).

### 7.2 Preservation Mechanism

- **Immediate (0–3 years):** Active GitHub repository + Zenodo DOIs.
- **Medium-term (3–10 years):** Zenodo guarantees preservation for at
  least 20 years (CERN commitment). GitHub snapshots mirrored
  automatically via Zenodo integration on each tagged release.
- **Long-term (10+ years):** Software Heritage
  (https://softwareheritage.org) archives the entire git repository
  history.
- **OSTI submission:** Final technical report, publications, and a
  frozen dataset manifest deposited to DOE OSTI at award close.

### 7.3 Loss / Disaster Recovery

- GitHub: mirrored by GitLab + self-hosted Gitea.
- Zenodo: backed by CERN's tape archive.
- Project machines: off-site encrypted backup of any local-only
  artifacts (primarily logs not yet promoted to a release).

---

## 8. Cost

All primary repositories (GitHub, Zenodo, arXiv, OSTI) are free to the
project. Budget line items are limited to:

- Staff time for release curation (~0.05 FTE).
- Optional cloud cost for one-time bulk uploads of training datasets.

No per-gigabyte fees anticipated at Phase-I data volumes.

---

## 9. Responsible Party and Points of Contact

- **Data Management Lead:** `[HUMAN DECISION — typically PI or a
  designated data steward]`.
- **Technical Contact:** PI (`[HUMAN DECISION]`).
- **Records Liaison:** AOR (`[HUMAN DECISION]`).

Annual review of this DMSP is scheduled at each project anniversary
to confirm schema, license, and retention-plan continued compliance.

---

## 10. Items Pending Human Decision

Surfaced explicitly so that nothing in this plan silently drifts:

1. **Code license:** Apache-2.0 vs MIT.
2. **Legal entity name:** consistent with SAM.gov registration.
3. **PI + Co-PI identities and ORCIDs.**
4. **GitHub organization / repository path.**
5. **DOI prefix & Zenodo community name** (recommend joining the
   "DOE Office of Science" Zenodo community if available; otherwise
   create a project-specific community).
6. **DOE Program Manager contact** (to confirm submission route for
   OSTI deposits).

---

## 11. Revision History

| Version | Date | Author | Changes |
|---|---|---|---|
| 0.1 | YYYY-MM-DD | `[HUMAN DECISION]` | Initial draft for DE-FOA-0003612 submission. |
