"""Version-tolerant camera timestamp helpers for G2 RGB-D policies.

Isaac Lab resets a camera's frame counter and sensor clock at every environment
reset.  Camera age must therefore be computed on that same local clock; mixing
it with a process-global transition counter makes every later episode appear
progressively more stale.
"""

from __future__ import annotations

import torch


def _tensor_value(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    converted = getattr(value, "torch", None)
    if isinstance(converted, torch.Tensor):
        return converted
    # Isaac Lab 3 exposes several sensor clocks as Warp arrays.  Calling
    # ``warp_array.to(torch.float32)`` treats the dtype as a Warp device and
    # fails.  DLPack keeps the conversion explicit and zero-copy where the
    # producer supports it.
    try:
        return torch.from_dlpack(value)
    except (AttributeError, TypeError, RuntimeError, ValueError) as exc:
        raise TypeError(
            f"unsupported camera timestamp tensor type: {type(value)!r}"
        ) from exc


def camera_capture_time_and_age(
    camera,
    *,
    fallback_episode_time_s: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-environment capture timestamp and age in seconds.

    Isaac Lab 2.3 and 3.0 both maintain ``_timestamp`` and
    ``_timestamp_last_update`` on the sensor clock.  They are used when
    available because they identify the actual latest RGB-D update.  The
    public frame counter is retained as a compatibility fallback and is only
    combined with an episode-local time supplied by the caller.
    """

    current = getattr(camera, "_timestamp", None)
    captured = getattr(camera, "_timestamp_last_update", None)
    if current is not None and captured is not None:
        current_tensor = _tensor_value(current).to(torch.float32)
        captured_tensor = _tensor_value(captured).to(torch.float32)
        if current_tensor.shape != captured_tensor.shape:
            raise ValueError("camera timestamp buffers have different shapes")
        age = torch.clamp(current_tensor - captured_tensor, min=0.0)
        if fallback_episode_time_s is not None:
            # Isaac camera clocks can retain an offset relative to
            # ManagerBasedEnv's episode-local clock.  The sensor-clock
            # difference is still the authoritative frame age, but the
            # dataset timestamp contract is episode-local.  Translate only
            # the clock origin; never recompute or quantize the measured age.
            local_time = fallback_episode_time_s.to(
                device=age.device, dtype=torch.float32
            )
            if local_time.shape != age.shape:
                raise ValueError(
                    "camera timestamp and fallback episode time shapes differ"
                )
            return (local_time - age).clone(), age.clone()
        return captured_tensor.clone(), age.clone()

    if fallback_episode_time_s is None:
        raise RuntimeError("camera sensor clock unavailable and no local fallback supplied")
    local_time = fallback_episode_time_s.to(torch.float32)
    frame = _tensor_value(camera.frame).to(device=local_time.device, dtype=torch.float32)
    if frame.shape != local_time.shape:
        raise ValueError("camera frame and fallback episode time shapes differ")
    update_period = float(camera.cfg.update_period)
    if update_period <= 0.0:
        raise ValueError("camera update period must be positive")
    capture_time = frame * update_period
    return capture_time, torch.clamp(local_time - capture_time, min=0.0)


def dual_camera_capture_timing(
    head_camera,
    wrist_camera,
    *,
    fallback_episode_time_s: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``[N,2]`` timestamps/ages ordered as head, right wrist."""

    head_timestamp, head_age = camera_capture_time_and_age(
        head_camera, fallback_episode_time_s=fallback_episode_time_s
    )
    wrist_timestamp, wrist_age = camera_capture_time_and_age(
        wrist_camera, fallback_episode_time_s=fallback_episode_time_s
    )
    return (
        torch.stack((head_timestamp, wrist_timestamp), dim=-1),
        torch.stack((head_age, wrist_age), dim=-1),
    )


__all__ = ["camera_capture_time_and_age", "dual_camera_capture_timing"]
