"""Time-travel helpers for LookingGlass FlowMatch sampling."""

from __future__ import annotations

from typing import Literal

import torch


TimeTravelVelocityMode = Literal["sync", "noise", "unsync", "full_sync", "full_unsync", "weighted_noise_unsync"]


def flowmatch_euler_step(
    sample: torch.Tensor,
    model_output: torch.Tensor,
    current_sigma: torch.Tensor,
    next_sigma: torch.Tensor,
) -> torch.Tensor:
    """FlowMatch Euler transition for arbitrary forward or backward sigma jumps."""

    dtype = sample.dtype
    while current_sigma.ndim < sample.ndim:
        current_sigma = current_sigma.unsqueeze(-1)
    while next_sigma.ndim < sample.ndim:
        next_sigma = next_sigma.unsqueeze(-1)
    out = sample.float() + (next_sigma.float() - current_sigma.float()) * model_output.float()
    return out.to(dtype=dtype)


def time_travel_transitions(
    total_steps: int,
    *,
    sync_repeat_start: float = 0.20,
    sync_repeat_stop: float = 0.80,
    sync_repeats: int = 1,
) -> list[tuple[int, int, int]]:
    """Build one-step denoise/noise-back transitions for paper-style time travel.

    Each tuple is ``(base_step_index, current_sigma_index, next_sigma_index)``.
    For ``sync_repeats=N``, timesteps in the selected range become
    ``i -> i+1 -> i -> i+1 ...``, ending at ``i+1``.
    """

    first_repeat = int(total_steps * sync_repeat_start)
    last_repeat = int(total_steps * sync_repeat_stop)
    repeat_count = max(1, int(sync_repeats))

    transitions: list[tuple[int, int, int]] = []
    for step_index in range(total_steps):
        repeats = repeat_count if first_repeat <= step_index < last_repeat and step_index < total_steps - 1 else 1
        for repeat_index in range(repeats):
            transitions.append((step_index, step_index, step_index + 1))
            if repeat_index < repeats - 1:
                transitions.append((step_index, step_index + 1, step_index))
    return transitions


def is_backward_transition(current_sigma: torch.Tensor, next_sigma: torch.Tensor) -> bool:
    """Return true when a transition moves to a noisier sigma."""

    return bool((next_sigma > current_sigma).detach().flatten()[0].item())


def validate_time_travel_velocity_mode(mode: str) -> TimeTravelVelocityMode:
    """Validate and normalize a backward time-travel velocity mode."""

    mode = mode.lower()
    aliases = {
        "v_sync": "sync",
        "synced": "sync",
        "original_noise": "noise",
        "pure_noise": "noise",
        "raw": "unsync",
        "x0_raw": "unsync",
        "model": "unsync",
        "unsynced": "unsync",
        "full_synced": "full_sync",
        "full_raw": "full_unsync",
        "full_unsynced": "full_unsync",
    }
    normalized = aliases.get(mode, mode)
    if normalized not in {"sync", "noise", "unsync", "full_sync", "full_unsync", "weighted_noise_unsync"}:
        raise ValueError(
            "time_travel_velocity_mode must be one of: "
            "'sync', 'noise', 'unsync', 'full_sync', 'full_unsync', 'weighted_noise_unsync'"
        )
    return normalized  # type: ignore[return-value]


def needs_original_pure_noise(mode: str) -> bool:
    """Return true when a velocity mode needs the initial pure-noise latent."""

    return validate_time_travel_velocity_mode(mode) in {"noise", "full_sync", "full_unsync", "weighted_noise_unsync"}


def select_time_travel_velocity(
    *,
    mode: str,
    current_latents: torch.Tensor,
    current_sigma: torch.Tensor,
    next_sigma: torch.Tensor,
    synced_velocity: torch.Tensor,
    raw_velocity: torch.Tensor,
    original_pure_noise: torch.Tensor | None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Choose the velocity for a FlowMatch transition.

    Forward denoising always uses ``synced_velocity``. The selected mode is
    applied only when the transition goes backward to a noisier state.

    Modes:
    - ``sync``: ``(z_t - x0_sync) / sigma_t``.
    - ``noise``: ``(original_pure_noise - z_t) / (1 - sigma_t)``.
    - ``unsync``: ``(z_t - x0_raw) / sigma_t``, equivalent to the model velocity.
    - ``full_sync``: ``original_pure_noise - x0_sync``.
    - ``full_unsync``: ``original_pure_noise - x0_raw``.
    """

    if not is_backward_transition(current_sigma, next_sigma):
        return synced_velocity

    normalized = validate_time_travel_velocity_mode(mode)
    if normalized == "sync":
        return synced_velocity
    if normalized == "unsync":
        return raw_velocity

    if original_pure_noise is None:
        raise ValueError(f"original_pure_noise is required for time_travel_velocity_mode='{normalized}'")

    sigma = current_sigma
    while sigma.ndim < current_latents.ndim:
        sigma = sigma.unsqueeze(-1)
    if normalized == "full_sync":
        synced_clean = current_latents.float() - sigma.float() * synced_velocity.float()
        return original_pure_noise.float() - synced_clean
    if normalized == "full_unsync":
        raw_clean = current_latents.float() - sigma.float() * raw_velocity.float()
        return original_pure_noise.float() - raw_clean
    if normalized == "noise":
        denom = (1.0 - sigma).clamp_min(eps)
        return (original_pure_noise.float() - current_latents.float()) / denom.float()
    if normalized == "weighted_noise_unsync":
        noise_velocity = (original_pure_noise.float() - current_latents.float()) / (1.0 - sigma).clamp_min(eps)
        return (0.60) * noise_velocity + (0.4) * raw_velocity