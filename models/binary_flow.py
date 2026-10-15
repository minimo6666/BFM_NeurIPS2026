"""Binary Flow Matching with flip prediction and plug-in reverse transitions."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1.0,
    gamma: float = 0.0,
) -> torch.Tensor:
    """Class-balanced focal loss."""
    probability = torch.sigmoid(inputs)
    cross_entropy = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )
    probability_correct = (
        probability * targets
        + (1.0 - probability) * (1.0 - targets)
    )
    error_probability = (1.0 - probability_correct).clamp(
        min=1e-6, max=1.0 - 1e-6
    )
    loss = cross_entropy * error_probability.pow(gamma)

    if alpha == -1:
        positive_fraction = targets.sum(dim=(-1, -2)) / targets[0].numel()
        positive_fraction = positive_fraction.view(-1, 1, 1)
        class_weight = (
            (1.0 - positive_fraction) * targets
            + positive_fraction * (1.0 - targets)
        )
        loss = class_weight * loss
    elif alpha > 0:
        class_weight = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = class_weight * loss
    return loss


class BinaryFlowMatching(nn.Module):
    def __init__(self, config, denoise_fn: nn.Module, codebook_size: int):
        super().__init__()
        self._denoise_fn = denoise_fn
        self.codebook_size = int(codebook_size)
        self.shape = tuple(config.latent_shape)
        self.num_timesteps = int(config.total_steps)
        self.n_samples = int(config.batch_size)
        self.focal_gamma = float(config.focal_gamma)
        if self.focal_gamma < 0.0:
            raise ValueError("focal_gamma must be non-negative")
        self.posterior_eps = 1e-30

        self.register_buffer(
            "interpolation_t",
            torch.linspace(1.0, 0.0, self.num_timesteps + 1),
            persistent=False,
        )

    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(
            1, self.num_timesteps + 1, (batch_size,), device=device
        )

    def q_sample(self, x0: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """Return q(X_t=1 | X_0) for the linear Bernoulli path."""
        retention = self.interpolation_t[time].view(
            -1, *([1] * (x0.ndim - 1))
        )
        return (1.0 + (2.0 * x0 - 1.0) * retention) / 2.0

    def _cross_step_flip_probability(
        self,
        current_time: torch.Tensor,
        target_time: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        retention_current = self.interpolation_t[current_time]
        retention_target = self.interpolation_t[target_time]
        probability = 0.5 * (retention_target - retention_current) / (
            retention_target + self.posterior_eps
        )
        probability = probability.clamp(0.0, 0.5)
        return probability.view(-1, *([1] * (reference.ndim - 1)))

    @staticmethod
    def _binary_channel(candidate: torch.Tensor, flip: torch.Tensor) -> torch.Tensor:
        return candidate * (1.0 - flip) + (1.0 - candidate) * flip

    def plugin_reverse_probability(
        self,
        clean_probability: torch.Tensor,
        xt: torch.Tensor,
        current_time: torch.Tensor,
        target_time: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the soft-X0 plug-in reverse rule."""
        xt = xt.float()
        soft_prior_probability = self.q_sample(clean_probability, target_time)
        soft_prior = torch.stack(
            (soft_prior_probability, 1.0 - soft_prior_probability), dim=-1
        )

        observed_states = torch.stack((xt, 1.0 - xt), dim=-1)
        flip = self._cross_step_flip_probability(
            current_time, target_time, observed_states
        )
        likelihood = self._binary_channel(observed_states, flip)

        posterior = soft_prior * likelihood
        posterior = posterior / posterior.sum(
            dim=-1, keepdim=True
        ).clamp_min(self.posterior_eps)
        return posterior[..., 0].clamp(0.0, 1.0)

    @staticmethod
    def _flip_logits_to_clean_logits(
        raw_logits: torch.Tensor, xt: torch.Tensor
    ) -> torch.Tensor:
        """Convert logits for P(X0 != Xt) into logits for P(X0 = 1)."""
        return (
            xt * (-raw_logits)
            + (1.0 - xt) * raw_logits
        )

    def training_loss(self, x0: torch.Tensor):
        x0 = x0.float()
        batch_size = x0.shape[0]
        time = self.sample_time(batch_size, x0.device)
        xt = torch.bernoulli(self.q_sample(x0, time))

        raw_logits = self._denoise_fn(xt, time_steps=time - 1)
        flip_target = torch.logical_xor(x0.bool(), xt.bool()).float()
        per_bit_loss = focal_loss(
            raw_logits, flip_target, gamma=self.focal_gamma
        )
        loss = per_bit_loss.mean()
        clean_logits = self._flip_logits_to_clean_logits(raw_logits, xt)
        with torch.no_grad():
            accuracy = ((clean_logits > 0.0) == x0.bool()).float().mean()
        return {"loss": loss, "bce_loss": loss.detach(), "acc": accuracy}

    def _sampling_grid(self, sample_steps: int) -> np.ndarray:
        if not 1 <= sample_steps <= self.num_timesteps:
            raise ValueError(
                f"sample_steps must be in [1, {self.num_timesteps}], got "
                f"{sample_steps}"
            )
        steps = np.arange(1, self.num_timesteps + 1)
        if sample_steps != self.num_timesteps:
            indices = np.linspace(
                0.0, self.num_timesteps - 1, sample_steps
            ).astype(np.int64)
            steps = steps[indices]
        return steps[::-1]

    @torch.no_grad()
    def sample(
        self,
        temp: float = 1.0,
        sample_steps: Optional[int] = None,
        b: int = 8,
        shape=None,
        return_all: bool = False,
        **unused,
    ) -> torch.Tensor:
        del unused
        device = next(self._denoise_fn.parameters()).device
        if shape is None:
            shape = (b, int(np.prod(self.shape)), self.codebook_size)
        else:
            b = int(shape[0])
        xt = torch.bernoulli(torch.full(shape, 0.5, device=device))

        if sample_steps is None:
            sample_steps = self.num_timesteps
        grid = self._sampling_grid(int(sample_steps))
        trajectory = [xt] if return_all else None

        for index, step in enumerate(grid):
            current_time = torch.full(
                (b,), int(step), device=device, dtype=torch.long
            )
            raw_logits = self._denoise_fn(
                xt, time_steps=current_time - 1
            ) / temp
            clean_logits = self._flip_logits_to_clean_logits(raw_logits, xt)
            clean_probability = torch.sigmoid(clean_logits)

            if int(step) == 1:
                xt = (clean_probability > 0.5).float()
            else:
                target_time = torch.full(
                    (b,), int(grid[index + 1]), device=device, dtype=torch.long
                )
                target_probability = self.plugin_reverse_probability(
                    clean_probability, xt, current_time, target_time
                )
                xt = torch.bernoulli(target_probability)
            if return_all:
                trajectory.append(xt)

        if return_all:
            return torch.cat(trajectory, dim=0)
        return xt

    def forward(self, x0: torch.Tensor):
        return self.training_loss(x0)
