"""NGRPO advantage calibration used by the ms-swift trainer reroute."""

from __future__ import annotations

from numbers import Real


def compute_ngrpo_advantages(grouped_rewards, virtual_max_reward=None):
    """Return flattened NGRPO advantages for a 2D group reward tensor.

    NGRPO appends a virtual maximum reward to every prompt group and computes
    population z-scores over the extended group. When no fixed virtual reward is
    configured, the virtual sample is the group maximum plus max(10%, 0.5).
    """
    import torch

    if grouped_rewards.ndim != 2:
        raise ValueError(f"grouped_rewards must be 2D, got shape {tuple(grouped_rewards.shape)}")
    if grouped_rewards.shape[1] < 1:
        raise ValueError("each reward group must contain at least one sample")

    group_count, generation_count = grouped_rewards.shape
    if virtual_max_reward is None:
        group_max = grouped_rewards.max(dim=1).values
        margin = torch.clamp(group_max.abs() * 0.1, min=0.5)
        virtual_max = group_max + margin
    elif isinstance(virtual_max_reward, Real):
        virtual_max = torch.full(
            (group_count,),
            float(virtual_max_reward),
            device=grouped_rewards.device,
            dtype=grouped_rewards.dtype,
        )
    else:
        virtual_max = torch.as_tensor(
            virtual_max_reward,
            device=grouped_rewards.device,
            dtype=grouped_rewards.dtype,
        )
        if virtual_max.shape != (group_count,):
            raise ValueError("virtual_max_reward must be scalar or have one value per reward group")

    extended_mean = (grouped_rewards.sum(dim=1) + virtual_max) / (generation_count + 1)
    reward_delta = grouped_rewards - extended_mean.unsqueeze(1)
    virtual_delta = virtual_max - extended_mean
    extended_variance = (reward_delta.square().sum(dim=1) + virtual_delta.square()) / (generation_count + 1)
    extended_std = (extended_variance + 1e-8).sqrt()
    return (reward_delta / extended_std.unsqueeze(1)).reshape(-1)
