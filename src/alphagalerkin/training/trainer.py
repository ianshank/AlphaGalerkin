"""Main training loop orchestrator."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import structlog
import torch
import torch.nn as nn
import torch.optim as optim

from src.alphagalerkin.core.config import AlphaGalerkinConfig
from src.alphagalerkin.nn.model import AlphaGalerkinNetwork
from src.alphagalerkin.training.checkpointing import CheckpointManager
from src.alphagalerkin.training.curriculum import CurriculumManager
from src.alphagalerkin.training.metrics import MetricCollector
from src.alphagalerkin.training.replay_buffer import ReplayBuffer
from src.alphagalerkin.training.self_play import SelfPlayEngine
from src.alphagalerkin.utils.io import resolve_device
from src.alphagalerkin.utils.logging import log_context

logger = structlog.get_logger("training.trainer")


class Trainer:
    """Main training orchestrator.

    Coordinates the three-phase training loop:

    1. **Self-play** -- generate episodes via MCTS and feed
       the resulting experiences into the replay buffer.
    2. **Network update** -- sample mini-batches from the
       replay buffer and update the network via back-propagation.
    3. **Evaluation / curriculum** -- track metrics and
       optionally advance the curriculum stage.

    Parameters
    ----------
    config:
        Root ``AlphaGalerkinConfig`` controlling all components.

    """

    def __init__(
        self,
        config: AlphaGalerkinConfig,
    ) -> None:
        self._config = config

        # Device
        self._device = resolve_device(config.device)

        # Neural network
        self._network = AlphaGalerkinNetwork(
            config.network,
        ).to(self._device)

        # Optimizer
        self._optimizer = self._create_optimizer()

        # Components
        self._replay_buffer = ReplayBuffer(
            config.training.replay,
        )
        self._curriculum = CurriculumManager(
            config.training.curriculum,
        )
        self._checkpoint_mgr = CheckpointManager(
            config.checkpoint,
        )
        self._metrics = MetricCollector()
        self._self_play = SelfPlayEngine(config)

    # ---------------------------------------------------------------
    # Properties
    # ---------------------------------------------------------------

    @property
    def network(self) -> AlphaGalerkinNetwork:
        """The neural network being trained."""
        return self._network

    @property
    def device(self) -> str:
        """The resolved PyTorch device string."""
        return self._device

    # ---------------------------------------------------------------
    # Optimizer creation
    # ---------------------------------------------------------------

    def _create_optimizer(self) -> optim.Optimizer:
        """Instantiate the optimizer from config."""
        opt_config = self._config.training.optimizer
        params = self._network.parameters()

        if opt_config.name == "adamw":
            return optim.AdamW(
                params,
                lr=opt_config.learning_rate,
                weight_decay=opt_config.weight_decay,
                betas=opt_config.betas,
            )
        if opt_config.name == "adam":
            return optim.Adam(
                params,
                lr=opt_config.learning_rate,
                weight_decay=opt_config.weight_decay,
                betas=opt_config.betas,
            )
        if opt_config.name == "rmsprop":
            return optim.RMSprop(
                params,
                lr=opt_config.learning_rate,
                weight_decay=opt_config.weight_decay,
                momentum=opt_config.momentum,
            )
        # Default: SGD
        return optim.SGD(
            params,
            lr=opt_config.learning_rate,
            weight_decay=opt_config.weight_decay,
            momentum=opt_config.momentum,
        )

    # ---------------------------------------------------------------
    # Training iteration
    # ---------------------------------------------------------------

    def train_iteration(
        self,
        iteration: int,
    ) -> dict[str, float]:
        """Run one training iteration.

        A single iteration consists of:

        1. Generating ``self_play_games_per_step`` self-play
           episodes and adding them to the replay buffer.
        2. Training the network for one or more gradient steps
           on mini-batches from the replay buffer (if ready).
        3. Updating the curriculum based on performance.

        Parameters
        ----------
        iteration:
            Zero-based training iteration index.

        Returns
        -------
        dict[str, float]
            Summary metrics for this iteration.

        """
        with log_context(iteration=iteration):
            # --- Phase 1: Self-play ---
            episodes = []
            n_episodes = (
                self._config.training.self_play_games_per_step
            )
            for _ in range(n_episodes):
                episode = self._self_play.play_episode()
                episodes.append(episode)
                experiences = episode.to_experiences(
                    iteration,
                )
                self._replay_buffer.add_batch(experiences)

            avg_length = float(
                np.mean([e.length for e in episodes]),
            )
            avg_reward = float(
                np.mean(
                    [e.total_reward for e in episodes],
                ),
            )

            self._metrics.record(
                "self_play/avg_length", avg_length,
            )
            self._metrics.record(
                "self_play/avg_reward", avg_reward,
            )

            # --- Phase 2: Network update ---
            total_loss = 0.0
            if self._replay_buffer.is_ready:
                batch = self._replay_buffer.sample(
                    self._config.training.batch_size,
                )
                loss = self._train_step(batch)
                total_loss += loss
                self._metrics.record(
                    "training/loss", loss,
                )

            # --- Phase 3: Curriculum ---
            if episodes:
                self._curriculum.update(avg_reward)

            # --- Summary ---
            metrics = self._metrics.get_iteration_summary()
            metrics["total_loss"] = total_loss
            metrics["buffer_size"] = float(
                self._replay_buffer.size,
            )
            metrics["curriculum_stage"] = float(
                self._curriculum.current_stage_index,
            )

            logger.info(
                "training.iteration.complete",
                iteration=iteration,
                **metrics,
            )
            return metrics

    # ---------------------------------------------------------------
    # Single gradient step
    # ---------------------------------------------------------------

    def _train_step(self, batch: list) -> float:
        """Execute one gradient step on *batch*.

        Parameters
        ----------
        batch:
            List of ``Experience`` objects sampled from the
            replay buffer.

        Returns
        -------
        float
            Scalar loss value.

        """
        self._network.train()
        self._optimizer.zero_grad()

        # Stack features into a batch tensor
        features = torch.stack([
            torch.from_numpy(exp.state_features)
            for exp in batch
        ]).to(self._device)

        # Ensure 3-D input: (batch, num_elements, features)
        if features.dim() == 2:
            features = features.unsqueeze(0)

        # Forward pass
        policy_logits, values = self._network(features)

        # Value loss (MSE)
        value_targets = torch.tensor(
            [exp.value_target for exp in batch],
            dtype=torch.float32,
            device=self._device,
        )
        if values.dim() > 1:
            values = values.squeeze(-1)
        value_loss = nn.functional.mse_loss(
            values, value_targets,
        )

        # Weighted total loss
        loss = (
            self._config.training.value_loss_weight
            * value_loss
        )

        loss.backward()

        # Gradient clipping
        grad_clip = (
            self._config.training.optimizer.gradient_clip_norm
        )
        nn.utils.clip_grad_norm_(
            self._network.parameters(),
            grad_clip,
        )

        self._optimizer.step()

        return float(loss.item())

    # ---------------------------------------------------------------
    # Checkpointing
    # ---------------------------------------------------------------

    def save_checkpoint(self, iteration: int) -> Path:
        """Save a full training checkpoint.

        Parameters
        ----------
        iteration:
            Current training iteration.

        Returns
        -------
        Path
            Path to the saved checkpoint file.

        """
        return self._checkpoint_mgr.save(
            iteration=iteration,
            network_state=self._network.state_dict(),
            optimizer_state=self._optimizer.state_dict(),
            replay_buffer_state=(
                self._replay_buffer.get_state()
            ),
            training_metrics=(
                self._metrics.get_full_history()
            ),
        )

    def load_checkpoint(self, path: Path) -> None:
        """Restore training state from a checkpoint.

        Parameters
        ----------
        path:
            Path to the checkpoint file to load.

        """
        checkpoint = self._checkpoint_mgr.load(path)
        self._network.load_state_dict(
            checkpoint["network_state_dict"],
        )
        self._optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"],
        )
        if "replay_buffer_state" in checkpoint:
            self._replay_buffer.load_state(
                checkpoint["replay_buffer_state"],
            )
        logger.info(
            "trainer.checkpoint_restored",
            path=str(path),
            iteration=checkpoint.get("iteration", -1),
        )
