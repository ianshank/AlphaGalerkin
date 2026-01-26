"""Neural network evaluator for MCTS.

Provides policy and value estimates using the AlphaGalerkin model.
Supports batch evaluation for efficient leaf evaluation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Protocol

import numpy as np
import torch

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.modeling.model import AlphaGalerkinModel


class EvaluationResult(NamedTuple):
    """Result from neural network evaluation."""

    policy: NDArray[np.float32]  # (n_actions,) probability distribution
    value: float  # Scalar value in [-1, 1]


class Evaluator(Protocol):
    """Protocol for MCTS evaluators."""

    def evaluate(
        self,
        state: NDArray[np.float32],
        legal_actions: list[int],
    ) -> EvaluationResult:
        """Evaluate a single state.

        Args:
            state: Game state tensor.
            legal_actions: List of legal action indices.

        Returns:
            Policy and value estimates.

        """
        ...

    def evaluate_batch(
        self,
        states: list[NDArray[np.float32]],
        legal_actions_batch: list[list[int]],
    ) -> list[EvaluationResult]:
        """Evaluate a batch of states.

        Args:
            states: List of game state tensors.
            legal_actions_batch: List of legal actions for each state.

        Returns:
            List of policy and value estimates.

        """
        ...


class FNetEvaluator:
    """Fast evaluator using FNet-accelerated model.

    Uses the fast forward pass for MCTS rollout evaluation.
    Supports batch inference for parallel leaf evaluation.
    """

    def __init__(
        self,
        model: AlphaGalerkinModel,
        device: torch.device | str = "cpu",
        use_fast_path: bool = True,
        temperature: float = 1.0,
    ) -> None:
        """Initialize evaluator.

        Args:
            model: AlphaGalerkin model.
            device: Device for inference.
            use_fast_path: Use fast FNet-only path.
            temperature: Policy temperature.

        """
        self.model = model
        self.device = torch.device(device)
        self.use_fast_path = use_fast_path
        self.temperature = temperature

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def evaluate(
        self,
        state: NDArray[np.float32],
        legal_actions: list[int],
    ) -> EvaluationResult:
        """Evaluate a single state.

        Args:
            state: Game state of shape (channels, height, width).
            legal_actions: List of legal action indices.

        Returns:
            Policy and value estimates.

        """
        # Convert to tensor and add batch dimension
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)

        # Forward pass
        if self.use_fast_path and hasattr(self.model, "forward_fast"):
            output = self.model.forward_fast(state_tensor)
        else:
            output = self.model(state_tensor)

        # Extract policy and value
        policy_logits = output.policy_logits[0].cpu().numpy()
        value = output.value[0, 0].cpu().item()

        # Apply temperature and mask illegal actions
        policy = self._process_policy(policy_logits, legal_actions)

        return EvaluationResult(policy=policy, value=value)

    @torch.no_grad()
    def evaluate_batch(
        self,
        states: list[NDArray[np.float32]],
        legal_actions_batch: list[list[int]],
    ) -> list[EvaluationResult]:
        """Evaluate a batch of states.

        Args:
            states: List of game state tensors.
            legal_actions_batch: List of legal actions for each state.

        Returns:
            List of policy and value estimates.

        """
        if not states:
            return []

        # Stack states into batch tensor
        batch_tensor = torch.stack([
            torch.from_numpy(s) for s in states
        ]).to(self.device)

        # Forward pass
        if self.use_fast_path and hasattr(self.model, "forward_fast"):
            output = self.model.forward_fast(batch_tensor)
        else:
            output = self.model(batch_tensor)

        # Extract results
        policy_logits = output.policy_logits.cpu().numpy()
        values = output.value.cpu().numpy().flatten()

        # Process each result
        results = []
        for i, legal_actions in enumerate(legal_actions_batch):
            policy = self._process_policy(policy_logits[i], legal_actions)
            results.append(EvaluationResult(
                policy=policy,
                value=float(values[i]),
            ))

        return results

    def _process_policy(
        self,
        logits: NDArray[np.float32],
        legal_actions: list[int],
    ) -> NDArray[np.float32]:
        """Process policy logits into probability distribution.

        Args:
            logits: Raw policy logits.
            legal_actions: List of legal action indices.

        Returns:
            Probability distribution over all actions.

        """
        # Mask illegal actions
        mask = np.full(len(logits), float("-inf"))
        mask[legal_actions] = 0.0

        # Apply temperature
        if self.temperature > 0:
            logits = logits / self.temperature

        # Masked logits
        masked_logits = logits + mask

        # Softmax
        exp_logits = np.exp(masked_logits - masked_logits.max())
        policy = exp_logits / (exp_logits.sum() + 1e-8)

        return policy.astype(np.float32)


class RandomEvaluator:
    """Random evaluator for testing and baseline comparison."""

    def __init__(
        self,
        n_actions: int,
    ) -> None:
        """Initialize random evaluator.

        Args:
            n_actions: Total number of possible actions.

        """
        self.n_actions = n_actions

    def evaluate(
        self,
        state: NDArray[np.float32],
        legal_actions: list[int],
    ) -> EvaluationResult:
        """Generate random policy and value.

        Args:
            state: Game state (ignored).
            legal_actions: List of legal actions.

        Returns:
            Uniform random policy and zero value.

        """
        policy = np.zeros(self.n_actions, dtype=np.float32)
        if legal_actions:
            uniform_prob = 1.0 / len(legal_actions)
            for action in legal_actions:
                policy[action] = uniform_prob

        return EvaluationResult(policy=policy, value=0.0)

    def evaluate_batch(
        self,
        states: list[NDArray[np.float32]],
        legal_actions_batch: list[list[int]],
    ) -> list[EvaluationResult]:
        """Batch evaluate with random values.

        Args:
            states: List of game states.
            legal_actions_batch: Legal actions for each state.

        Returns:
            List of random evaluations.

        """
        return [
            self.evaluate(s, la)
            for s, la in zip(states, legal_actions_batch)
        ]
