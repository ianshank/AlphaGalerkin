"""LMStudioEvaluator — an MCTS ``Evaluator`` backed by a local LLM.

Implements ``src/mcts/evaluator.py::Evaluator`` structurally: the
``evaluate``/``evaluate_batch`` signatures match byte-for-byte so MCTS can
consume an instance interchangeably with ``RandomEvaluator`` or
``FNetEvaluator``.

For each ``evaluate`` call the evaluator:
    1. Renders a deterministic prompt (``build_policy_prompt``).
    2. Asks the ``LMStudioClient`` for a validated ``LMStudioPolicyResponse``.
    3. Applies illegal-action masking + temperature softmax (mirrors
       ``FNetEvaluator._process_policy`` so MCTS sees consistent semantics).
    4. Returns an ``EvaluationResult(policy, value)``.

On exhausted retries the evaluator either:
    - Re-raises the typed exception (``config.fallback_to_uniform_on_parse_error
      is False``, default), or
    - Returns a uniform-over-legal policy with ``value=0.0`` and emits a
      ``lm_studio_fallback_uniform`` event.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
from numpy.typing import NDArray

from src.integrations.lm_studio.client import LMStudioClient
from src.integrations.lm_studio.prompt import build_policy_prompt
from src.integrations.lm_studio.schema import (
    LMStudioActionSpaceMismatchError,
    LMStudioConnectionError,
    LMStudioParseError,
)
from src.mcts.evaluator import EvaluationResult

if TYPE_CHECKING:
    from src.poc.logging import ScenarioLogger

logger = structlog.get_logger(__name__)


_SOFTMAX_NORMALIZER_FLOOR = 1e-8
"""Lower bound added to the softmax denominator to avoid divide-by-zero."""


class LMStudioEvaluator:
    """MCTS ``Evaluator`` backed by a locally-served LLM.

    Construct with an already-built ``LMStudioClient`` so the same client
    instance can be reused across MCTS rollouts (preflight runs only once).
    """

    def __init__(
        self,
        client: LMStudioClient,
        *,
        action_space_size: int,
        pde_family: str,
        basis_descriptions: list[str],
        seed: int,
        temperature: float = 1.0,
        scenario_logger: ScenarioLogger | None = None,
    ) -> None:
        """Construct the evaluator.

        Args:
            client: Pre-constructed ``LMStudioClient``.
            action_space_size: Number of candidate basis functions. Must
                equal ``BasisSelectionGame.action_space_size`` for the
                game this evaluator is paired with.
            pde_family: Human-readable PDE name embedded in the prompt
                (e.g. ``"poisson"``, ``"burgers"``).
            basis_descriptions: One short string per action describing the
                candidate basis (e.g. ``"fourier(k=2,3)"``). ``len`` must
                equal ``action_space_size``.
            seed: Sampling seed forwarded to the LLM on every call.
            temperature: Softmax temperature applied to LLM logits inside
                ``_process_policy``. Distinct from the LLM's own
                ``config.temperature``.
            scenario_logger: Optional ``ScenarioLogger`` for context-bound
                events. When ``None`` the module-level logger is used.

        """
        if action_space_size <= 0:
            raise ValueError(f"action_space_size must be > 0, got {action_space_size!r}")
        if len(basis_descriptions) != action_space_size:
            raise ValueError(
                "basis_descriptions length "
                f"({len(basis_descriptions)}) must equal action_space_size "
                f"({action_space_size})"
            )
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature!r}")
        self._client = client
        self._action_space_size = action_space_size
        self._pde_family = pde_family
        self._basis_descriptions = list(basis_descriptions)
        self._seed = seed
        self._temperature = temperature
        self._logger = scenario_logger if scenario_logger is not None else logger
        self._latencies_ms: list[float] = []

    @property
    def action_space_size(self) -> int:
        return self._action_space_size

    @property
    def latencies_ms(self) -> list[float]:
        """Per-``evaluate`` wall-clock samples in milliseconds.

        Each call to ``evaluate`` appends one sample regardless of success
        or fallback. Scenarios consume this list to compute p95 latency
        without parsing the structlog stream.
        """
        return list(self._latencies_ms)

    def reset_latencies(self) -> None:
        """Clear collected latency samples (per-cell reset hook for scenarios)."""
        self._latencies_ms = []

    def evaluate(
        self,
        state: NDArray[np.float32],
        legal_actions: list[int],
    ) -> EvaluationResult:
        """One LLM call → masked-softmax policy + scalar value."""
        if not legal_actions:
            # No legal moves: return zero policy and neutral value so MCTS
            # treats this as a terminal-equivalent leaf.
            policy = np.zeros(self._action_space_size, dtype=np.float32)
            return EvaluationResult(policy=policy, value=0.0)

        prompt = build_policy_prompt(
            state,
            legal_actions=legal_actions,
            action_space_size=self._action_space_size,
            pde_family=self._pde_family,
            basis_descriptions=self._basis_descriptions,
        )
        start = time.perf_counter()
        try:
            response = self._client.complete_policy(
                prompt,
                expected_action_size=self._action_space_size,
                seed=self._seed,
            )
        except (
            LMStudioParseError,
            LMStudioActionSpaceMismatchError,
            LMStudioConnectionError,
        ) as exc:
            self._latencies_ms.append((time.perf_counter() - start) * 1000.0)
            if self._client.config.fallback_to_uniform_on_parse_error:
                self._log_fallback(legal_actions, reason=type(exc).__name__)
                policy = self._uniform_over_legal(legal_actions)
                return EvaluationResult(policy=policy, value=0.0)
            raise

        self._latencies_ms.append((time.perf_counter() - start) * 1000.0)
        logits = np.asarray(response.logits, dtype=np.float32)
        policy = self._process_policy(logits, legal_actions)
        value = float(np.clip(response.value, -1.0, 1.0))
        return EvaluationResult(policy=policy, value=value)

    def evaluate_batch(
        self,
        states: list[NDArray[np.float32]],
        legal_actions_batch: list[list[int]],
    ) -> list[EvaluationResult]:
        """Sequential batch (MCTS's main path is single-state)."""
        if len(states) != len(legal_actions_batch):
            raise ValueError(
                f"states length ({len(states)}) != legal_actions_batch length "
                f"({len(legal_actions_batch)})"
            )
        return [
            self.evaluate(state, legal_actions)
            for state, legal_actions in zip(states, legal_actions_batch, strict=True)
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_policy(
        self,
        logits: NDArray[np.float32],
        legal_actions: list[int],
    ) -> NDArray[np.float32]:
        """Mask illegal actions, apply temperature, softmax over the legal set."""
        mask = np.full(self._action_space_size, -np.inf, dtype=np.float32)
        for action in legal_actions:
            if 0 <= action < self._action_space_size:
                mask[action] = 0.0
        scaled = logits / self._temperature
        masked = scaled + mask
        masked -= np.max(masked[np.isfinite(masked)], initial=0.0)
        exp = np.exp(masked, dtype=np.float32)
        denom = float(np.sum(exp)) + _SOFTMAX_NORMALIZER_FLOOR
        policy = (exp / denom).astype(np.float32)
        return policy

    def _uniform_over_legal(self, legal_actions: list[int]) -> NDArray[np.float32]:
        """Fallback policy when the LLM exhausted retries and fallback is on."""
        policy = np.zeros(self._action_space_size, dtype=np.float32)
        if not legal_actions:
            return policy
        weight = 1.0 / float(len(legal_actions))
        for action in legal_actions:
            if 0 <= action < self._action_space_size:
                policy[action] = weight
        return policy

    def _log_fallback(self, legal_actions: list[int], *, reason: str) -> None:
        log: Any = self._logger
        log.warning(
            "lm_studio_fallback_uniform",
            reason=reason,
            n_legal_actions=len(legal_actions),
        )
