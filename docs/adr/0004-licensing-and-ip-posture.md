# 0004. Licensing and IP posture: MIT stays, and disclosure has already happened

- **Status:** Accepted
- **Date:** 2026-08-23
- **Deciders:** @ianshank

## Context

An external review of this project's strategy raised licensing and IP as the one
decision that is **irreversible, costs no engineering time, and blocks nothing** —
and therefore should be settled before further work, rather than discovered later.
It is the only item of its kind in the current plan.

The facts that constrain the decision:

- `LICENSE` is MIT, and `pyproject.toml` declares MIT in both `license` and the
  OSI classifier. There is no internal inconsistency to resolve.
- The repository is **public**, with 663 commits of history.
- A **public HuggingFace Space mirror** ships from `hf_space/` via
  `deploy_space.py`, carrying a ~55k-LOC copy of `src/`.
- `docs/business/proposals/IP_STRATEGY.md` describes three provisional patent
  claims and a dual-licensing option, and
  `docs/business/proposals/SUBMISSION_TIMELINE.md` lists provisional filing as
  **Pending**.

The tension is between that IP strategy and the disclosure that has already
occurred. In the United States, a public disclosure starts a 12-month clock for a
US filing; most other jurisdictions apply absolute novelty, so public disclosure
before filing forfeits foreign rights outright. This project's method has been
publicly described, in a public repository with a public model mirror, for months.

A separate consideration, raised in the same review: a self-hosted neural video
codec (`src/video_compression/`, the largest package in the tree) sits close to
the maintainer's professional domain. That is an employment-IP question, not an
open-source-licensing one, and it is out of scope here.

## Decision

**We will keep the MIT licence and continue developing in the open.**

We will **not** pause public pushes pending an IP review, and we will not
retroactively restrict the licence.

Any future patent work must **precede** the disclosure it protects, not follow it.
Concretely: a provisional application covering a *new* method must be filed before
that method is described in this repository, in a paper, or in an SBIR proposal
that becomes public. Filing on what is already published here is not available for
foreign rights and is on a clock for US rights that has, in practice, already run.

The employment-IP question about `src/video_compression/` is recorded as an open
owner decision, to be resolved with written advice before that subsystem becomes
an SBIR deliverable or the basis of a commercial entity. This ADR does not decide
it. Note that the subsystem is separately **frozen** for the current cycle, so the
question does not block any planned work.

## Consequences

**Easier.** No licence audit, no history rewrite, no split of the public and
private surfaces, and no interruption to the current cycle. Contributions,
reviewer access, and the HuggingFace Space all continue to work. The project keeps
the credibility that comes from published negative results — which, given that
this repository has retracted two of its own headline claims in public, is a real
asset rather than a slogan.

**Harder.** Patent protection for what is already disclosed is largely
unavailable, and the `IP_STRATEGY.md` provisional claims should be re-read with
that in mind rather than treated as live options. Dual-licensing remains possible
for *future* code but is complicated by an MIT history that anyone may fork.

**Follow-up.** `docs/business/proposals/IP_STRATEGY.md` and
`SUBMISSION_TIMELINE.md` should be reconciled against this ADR. That is a
documentation task, not an engineering one, and it is deliberately not bundled
here — this record exists so the decision stops being implicit.

This decision is **not** enforced by a CI test. It constrains future filing
sequence, not repository state, and there is nothing mechanical to check. The
absence is deliberate rather than an oversight: the charter's rule is that a
Requirement without a guard is a wish, and this is an ADR, not a Requirement.

## Alternatives considered

**Pause public pushes pending IP review.** Rejected. It would stall the current
cycle to protect rights that public disclosure has already largely spent. The cost
is immediate and certain; the benefit is speculative and mostly foreclosed.

**Relicense to a source-available or dual licence.** Rejected for the existing
code. MIT is irrevocable for what is already published — any fork of the current
history stays MIT — so relicensing would fragment the project's identity while
protecting nothing already released. It remains available for genuinely new,
separately-developed components.

**Split now: core public, video codec private.** Deferred rather than rejected.
There is a real argument for it on employment-IP grounds, but restructuring the
repository around an unfalsified central thesis is the expensive mistake this
cycle is explicitly avoiding. The subsystem is frozen instead, which buys the same
protection at no restructuring cost, and the split can be revisited once the
thesis has been tested.
