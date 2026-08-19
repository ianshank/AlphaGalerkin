# Next Steps Review — 2026-08-18

> **What this is:** an evidence-based, peer-reviewed case for the highest-leverage
> next steps in code development, produced by scanning the codebase, documentation,
> and GitHub activity and then adversarially re-verifying the findings against the
> repo itself. It is a companion to — not a replacement for — CLAUDE.md's own "Next
> Steps" table and `docs/CODE_HYGIENE_AUDIT.md`'s backlog; where this document cites a
> number or a file/line, it was independently checked, and where it corrects an
> existing doc, the correction is called out explicitly rather than silently
> overwriting the original.
>
> **Method:** three research passes (documentation/backlog, code-level signals,
> git/GitHub activity), direct verification of the highest-stakes claims against the
> charter and hygiene audit, an adversarial peer-review pass, then a second, deeper
> round using AlphaGalerkin's own repo-specific specialist reviewers (MCTS, PDE/
> refinement, SQE, integrations) with live command execution rather than doc
> paraphrase, plus direct GitHub API reads of the two most relevant open PRs. A final
> adversarial pass stress-tested the two biggest claims from that second round —
> and caught a real conflation error in an intermediate draft, corrected below rather
> than edited away.

## Framing

AlphaGalerkin is CI-green with real recent hardening investment, excellent TODO
hygiene in production code (3 hits in `src/`, all trivial), zero unconditional test skips, and an
unusual, genuine strength: it self-reports negative results — two prior
fabricated-number incidents were caught and retracted in-repo rather than left to
stand. Current green was reached via active firefighting in the commits immediately
preceding HEAD (four straight `fix(ci)`/`fix(test)`/`fix(lint)` commits, one
resolving "10 test failures from merge mismatches") — real, but not quiet
steady-state.

The project's central scientific claim — MCTS-guided adaptive refinement/basis-
selection beats classical baselines — currently **loses** empirically, in the
project charter's own words. This review's specialist pass confirmed the MCTS
*engine* itself is not at fault (96% branch coverage; the 2026-08-16 backup-sign fix
verified genuinely applied by tracing the actual call chain, not just trusting the
changelog) — but it also surfaced something no existing doc had connected: a real,
already-built, already-tested finite-element solver capable of the local mesh
refinement this comparison actually needs has been sitting unused in the repo, and
the literal tool the spec names as the prerequisite was never wired to the code that
already implements it. That is this review's single highest-leverage new finding.

Two GitHub-side findings also upgrade materially on closer read: PR #118 already
implements most of what would otherwise be a "build the certificate work from
scratch" recommendation, and PR #57 — easy to write off as stale — contains ~95
tested, partly-reviewed changes that never landed, including a `src/pde`
coverage-gate raise the charter's own live gate register confirms never actually
shipped.

## Tier 0 — Days, do first (correctness & credibility)

1. **Fix P0-1 — but only the Burgers slice is a same-fix-covers-all situation; Heat
   and AdvectionDiffusion are separate work.** `BurgersOperator.__init__`
   (`src/pde/operators.py:628`) overwrites the class-level `is_time_dependent = True`
   with `config.is_time_dependent` (defaults `False`), so `exact_solution()` returns
   `None` and the OOD reward on Burgers is structurally flat/zero — root-caused in
   `docs/CODE_HYGIENE_AUDIT.md:576-614`. This review's re-verification found the
   audit's own "same pattern in `AdvectionDiffusionOperator`" claim is not quite
   right: `AdvectionDiffusionOperator.exact_solution` (`:901-908`) is actually gated
   on `time is None`, and `BasisSelectionGame.__init__` never passes a `time`
   argument — so it returns `None` regardless of any `is_time_dependent` fix; a
   *different* code path needs fixing there. `HeatOperator` (`:923-1027`) has **no**
   `exact_solution` override at all — it needs a new manufactured solution derived
   from scratch, not a config-flag fix. The audit's proposed remedy ("honour the
   class default") would also break an existing, intentional, tested feature
   (`tests/pde/test_operators.py::test_steady_returns_none`, which builds a config
   with `is_time_dependent=False` and asserts `None` on purpose) — the real fix needs
   a sentinel (`bool | None`) or `model_fields_set` introspection, not a blanket
   override. It also touches real training-loss composition
   (`src/training/losses/physics.py:531,540`), widening the blast radius past the OOD
   demo. **Realistic estimate: S/M, ~2-4 engineer-days for the Burgers slice plus
   threshold re-derivation; Heat and AdvectionDiffusion are additional, separately
   scoped work**, not covered by "the fix" as currently documented. After the Burgers
   fix lands, re-derive `config/scenarios/llm_prior_demo.yaml`'s `ood_*` thresholds
   from a real measurement, and re-measure `noyron_basis`'s "~2-4% best-case
   reduction" — the audit itself flags this may share the same root cause.

2. **Do not flip mypy to a hard CI gate.** Hygiene-audit item B6 recommends this; the
   charter's own Deviations register (`openspec/specs/project-charter/spec.md:288`)
   explicitly rejects it — *"Diagnostics depend on the installed torch version, so a
   hard gate is flaky across environments."* Per CLAUDE.md's own stated authority
   rule, the charter overrides the audit backlog here. Drop this item from active
   tracking rather than executing it.

3. **Document or restore the 9 undocumented `--deselect`s** in `ci.yml`'s
   `test-fast` job — every other exclusion in that file carries an inline rationale
   comment; these nine don't.

4. **Two cheap, unrelated hygiene items**: delete `onnx_err.txt` (a 77KB stray
   Windows pytest failure log committed at repo root, no purpose); and fix one stale
   prose line — CLAUDE.md's 2026-07-22 milestone entry says `video_compression` "no
   longer exists," but it was deliberately reinstated afterward (Codec Model-Zoo
   work), and the reinstatement is already correctly documented in the code itself
   (`tests/support/cut_modules.py`'s own docstring). This was independently
   rediscovered by two different reviewers in this process before either had checked
   that file — itself a small signal the stale prose line is worth fixing so it stops
   costing review time.

5. **PR triage — four items, not a uniform "staleness" bucket:**
   - **Close #47** (proposes dashboard tabs for reentry/firefighting/intercept — all
     three are in `tests/support/cut_modules.py::CUT_MODULES` and confirmed absent
     from `src/`; structurally impossible to build under the current charter).
   - **Lean toward closing #99**, not rebasing (video_compression numerical-
     instability fixes; merge-conflict-stuck since 2026-07-24 at 380 changed files;
     opened the day after the 2026-07-22 cut, before the later reinstatement via a
     different code path — a mechanical rebase across "deleted then reinstated
     differently" is unlikely to apply cleanly).
   - **PR #57 needs salvage triage, not a staleness write-off.** Its own body
     documents 5 tracks plus a gap-fix and a review-fix commit: ONNX production
     export, a `VectorFieldHead` for multi-field PDE output, adaptive time-stepping,
     per-rank distributed batching, and — notably — a `src/pde` coverage-gate raise
     from 75% to 85%, claimed measured at 85.58% locally. The charter's live gates
     register still shows `src/pde` at 75 today, confirming this never landed. The PR
     claims 2,471 passing tests and two sub-agent reviews (security-auditor,
     code-reviewer) addressed pre-push. It is merge-conflicted against a base commit
     now roughly four months stale, so a straight merge won't work — but the
     recommendation is to evaluate cherry-picking the still-needed tracks (ONNX
     export and the coverage-gate raise both still appear undone elsewhere in the
     repo), not to close it as simply stale.
   - **PR #118 is time-sensitive and closer to done than it looks — but not a
     rubber-stamp.** See Tier 2, item 9; listed here too because its mergeable state
     just turned conflicted against last week's hardening merges, and every day of
     delay makes the eventual rebase larger.

## Tier 1 — 1-2 weeks (governance & backlog decisions already half-made)

6. **CI's merge-gate scope is narrower than it looks.** The required `ci-success`
   check depends on `[lint, test-fast, coverage, transfer-baseline-regression]` —
   four jobs, but the fourth is explicitly soft/informational ("do not block" per its
   own comment), so in effect only three can fail the build. `test-integration`,
   `test-chess`, `test-jax`, and `test-extras` all run on every push but cannot block
   a merge. Fold in the ones that matter, or document explicitly why not — matching
   this project's own drift-guard-everything discipline applied everywhere else.

7. **The JAX/backend question is bigger than "some untested classes" — it's a whole
   unconsumed abstraction layer.** The seven `Jax*`-suffixed classes in
   `src/math_kernel/{basis,spectral,integral}.py` (~1900 combined LOC) have zero
   call sites in production code — confirmed by three independent checks in this
   review (CI has a dedicated `test-jax` job exercising them, but nothing in `src/` outside the backend itself calls them). Going further: `src/backend/interface.py::BackendInterface`, a 67-method
   `Protocol` apparently meant to unify the Torch/JAX backends, is referenced *only*
   within `src/backend/` itself (`jax_backend.py`, `torch_backend.py`, `__init__.py`)
   — nothing in `training/`, `modeling/`, `pde/`, or anywhere else in `src/` consumes
   it, and a live abstraction audit (run twice in this review, both clean elsewhere)
   confirms seven of its members have no reader at all. This is the single largest
   concrete "keep or cut" call in the repo: either the JAX backend is strategically
   live — write tests, pay down `pyproject.toml:136`'s own "drop these upper bounds
   after modernising the JaxBackend" debt — or it isn't, in which case `src/backend/`
   and the `Jax*` classes should be deleted together in one pass.

8. **Clear hygiene-audit item B10** — the keep-or-cut decisions for
   `prototyping`/`tournament`/`analysis`/`curriculum`/`deployment`/`demos`. The
   analysis is already written; it needs the owner's per-package sign-off.

## Tier 2 — Weeks (the two strategic bets — both materially re-scoped by this review)

9. **Don't build `verified_error_certificate` from scratch — finish and land PR
   #118, which already implements most of Track A/WS1.** An earlier draft of this
   review treated the spec as 0% built; it isn't. PR #118 (Copilot-authored,
   approved by the repo owner on 2026-08-13) ships a real `src/pde/certificate/`
   package — `interface.py`, `verifiers/heuristic_grid.py`, `config.py`, a verifier
   registry, an ADR (`docs/adr/0003-jax-track-b-verifier.md`), and a test file one
   reviewer reply cites as "17/17 passed" locally. Five Copilot review threads each
   have a same-day, specific "Fixed" reply from `copilot-swe-agent` — but none of the
   five are marked resolved, and the PR is now merge-conflicted against the base
   branch after last week's hardening merges (38 files, 4,495 additions — a real
   rebase, not a trivial one). The check runs visible on the PR show a reviewer bot
   and a docs build, but no visible run of the main CI/coverage/regression-surface
   pipelines — so "it passes CI" is not yet a confirmed fact, only "it was reviewed
   and iterated on recently." Recommended sequence: rebase against the current base,
   resolve the five threads (each already has a proposed fix in its reply — verify,
   don't re-derive), get a fresh full CI run, then merge. Only after that should
   Track B (the neural-operator path, roughly 45 minutes per certificate via
   autoLiRPA/∂-CROWN) be scoped as a small, deliberately batched follow-up — not
   "everything, immediately," given the spec's AC1 literally asking for 100%
   coverage on every pinned-scenario solution.

10. **The AMR novelty-claim fork has a concrete, effort-estimated answer now — not
    just an abstract owner decision.** The charter states plainly that on the
    current tensor-product-only substrate, no marking-policy comparison means
    anything (Dörfler marking itself loses to plain uniform refinement by 5-9x at
    matched DOF). This review's specialist tracing found three things that change
    what to actually do about it:
    - The MCTS *engine* is clean (96% branch coverage; the 2026-08-16 backup-sign fix
      verified genuinely applied by tracing the actual call chain).
    - The *way* MCTS is configured for this specific comparison has two real,
      cheap-to-fix issues confirmed by direct code inspection: `use_intermediate_rewards`
      is never plumbed as a config field at all (it stays at its `False` default, so
      the search never uses the real per-step reward
      `PDEGameAdapter.get_last_reward()` already exposes), and Dirichlet exploration
      noise (`add_noise=True`) is applied even at the `temperature=0.0` greedy
      decision that actually gets scored, with no comment anywhere suggesting this is
      deliberate. Both are worth fixing for correctness regardless of outcome — but
      this review's final verification pass traced MCTS's own `apply_action` and
      found it inserts full-axis grid lines exactly like Dörfler does, i.e. both arms
      already share the dominant tensor-product defect. These two fixes should not be
      oversold as likely to reverse the 1.0996x / 2.04x headline ratios — on their
      own, they probably won't. (A third possible lever, raising `n_simulations`, is
      explicitly not free: each simulation is a real solve, and the module's own
      three-metric design exists specifically because raising it worsens the
      already-losing matched-solves and wall-clock numbers — a documented trade-off,
      not an oversight, and not something to "fix.")
    - **The real lever: `src/research/fem_baseline.py::ScikitFEMLShapedSolver`
      already exists.** It is a genuine triangular-mesh solver (via `scikit-fem`, an
      existing optional `[fem]` extra) with real local element refinement
      (`mesh.refined(np.where(marked)[0])`) and a Zienkiewicz-Zhu error indicator,
      with its own 279-line test file. It is referenced nowhere outside its own
      module and test, confirmed by an exhaustive repo-wide search.
      `specs/lshape_amr_compare.spec.md:150` literally names "skfem" as what v2.1
      needs, but never connects that to this already-built class — the answer was
      named in the spec's own text and still never wired up. (Caveat: `scikit-fem`
      is not installed in the environment this review ran in, so the test file's
      "already tested" status is true in principle and correctly skip-gated, not
      independently re-verified passing today.) It was originally built as a
      DOE-reviewer-credible baseline for the `verified_error_certificate` spec
      (item 9's Track A reference implementation), not for this comparison — which
      means wiring it into `lshape_amr_compare` as the shared discretization for
      *both* arms would double-serve both Tier-2 bets from one investment. Estimated
      effort: M, roughly 1-3 engineer-weeks (swap the FDM solver for this one in the
      Dörfler arm, redesign `LShapeAMRGame`'s action space around mesh elements
      instead of grid edges, reconcile FDM-node vs. FEM-DOF accounting, re-derive the
      convergence gate) — not the open-ended octree-from-scratch project the Next
      Steps table implies. The real technical risk flagged is `scikit-fem` mesh-clone
      cost inside MCTS's per-simulation re-solve loop, not a research unknown.
    - **Recommended sequence**: (1) plumb `use_intermediate_rewards` and drop
      eval-time exploration noise — cheap, correct, worth doing regardless of whether
      it moves the number; (2) invest the 1-3 weeks in wiring
      `ScikitFEMLShapedSolver` in as the shared discretization — this is the actual,
      currently-missing precondition for a valid comparison, per the charter's own
      words; (3) only after both are done and the method still loses, revisit
      repositioning the pitch away from "beats classical AMR" toward what's already
      defensible today (unified basis-and-refinement search, zero-shot resolution
      transfer, framework value). This remains ultimately the owner's call, but it is
      now a scoped, sequenced, effort-estimated plan rather than a blind fork.

## Tier 3 — Backlog / opportunistic

11. **B4 god-module splits** (`operators.py` at 1,841 LOC, `trainer.py` at 1,476
    LOC) — real, but flagged XL effort / High risk in the hygiene audit; needs its
    own planning pass rather than being picked up incidentally.
12. **15 of 28 `src/` packages have no `AGENT.md`** — cheap, mechanical, low-risk;
    good async/filler work.
13. **PicoGK STL ingestion** (`src/pde/sdf.py:435,447,450`, currently
    `NotImplementedError` stubs) — replaces the analytical helix surrogate with real
    Leap 71 geometry; the actual commercial/TRL proof point the SBIR positioning
    depends on.
14. **Multigrid solver + PettingZoo swarm-loop completion** — corroborated
    incomplete independently by both PR #57's own stated follow-up list and live
    `NotImplementedError` stubs in `src/research/extra_solvers/`.
15. **Go vs. GnuGo/KataGo tournament (M7 "real-world validation")** — P3, roughly
    2 weeks; lower priority now that the charter has moved the project's primary
    identity from game-AI to PDE/MCTS research.
16. **Minor footnote on the LLM-prior integrations work**: CLAUDE.md's "LLM-prior
    alternative backends ✅ DONE (2026-06-12)" milestone describes config
    *registration* for vLLM/llama.cpp, not live GPU *validation*. Real-server proof
    is 0-for-3 backends today (LM Studio included), not just the already-flagged
    0-for-1. This is not a fresh priority — CLAUDE.md's own Next Steps table already
    lists the GPU run as open/manual — but the milestone wording is worth tightening
    so a skim-reader doesn't conflate "registered" with "validated."

## Flagged for the owner — not an engineering call

17. **SBIR administrative dependencies are all "Pending."**
    `docs/business/proposals/SUBMISSION_TIMELINE.md` lists SAM.gov registration,
    provisional patent filing, and budget preparation as all still pending, and the
    same document's own stated submission windows (AFWERX Q2 2026, Navy/DOE
    Apr-May 2026) have already elapsed relative to today. Worth a direct
    conversation about whether SBIR pursuit is still active before investing more
    engineering time chasing benchmark numbers in its service.

18. **`docs/GALERKIN_FUSION_HEAD_PLAN.md`'s decision record was never written
    back.** The cross-repo experiment with the external "Mouse-Droid-AGI" project
    shipped its Day-1 deliverable, but the Day-9/10 "continue vs. redirect" ADR
    (`docs/architecture/ADR-post-fusion-direction.md`) does not exist — an open
    external thread with no answer recorded in this repo.
