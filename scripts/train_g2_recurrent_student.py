#!/usr/bin/env python3
"""Offline recurrent-student distillation from canonical G2 HDF5.

This runner is deliberately separate from Isaac Sim.  It cannot silently mix
legacy keyboard data, restore SAC replay, or reinterpret privileged state as a
deployment observation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))


def parse_args() -> argparse.Namespace:
    from geniesim.rl.isaaclab.g2_recurrent_profile import (
        default_recurrent_profile_name,
        recurrent_profile_names,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--checkpoint-interval", type=int, default=2000)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--recurrent-profile",
        choices=recurrent_profile_names(),
        default=default_recurrent_profile_name(),
    )
    parser.add_argument(
        "--camera-encoder-profile",
        choices=(
            "baseline_4layer",
            "cnn5_160",
            "cnn7_160",
            "cnn9_160",
            "cnn11_160",
            "cnn13_160",
        ),
        default="baseline_4layer",
    )
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--gru-num-layers", type=int)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--gradient-clip", type=float)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--burn-in-steps", type=int)
    parser.add_argument("--sequence-stride", type=int)
    parser.add_argument("--random-shift-pad", type=int, default=4)
    parser.add_argument("--cross-camera-contrastive-weight", type=float, default=0.02)
    parser.add_argument("--cross-camera-contrastive-temperature", type=float, default=0.1)
    parser.add_argument(
        "--cross-camera-false-negative-position-threshold-m",
        type=float,
        default=0.01,
    )
    parser.add_argument("--temporal-pose-residual-weight", type=float, default=0.05)
    parser.add_argument(
        "--temporal-pose-residual-rotation-weight", type=float, default=0.1
    )
    parser.add_argument("--expert-arm-action-scale", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--share-camera-encoder-weights", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument("--wandb-project", default="geniesim-g2-student")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    return parser.parse_args()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _combine_datasets(paths, sequence_contract, *, expert_arm_action_scale):
    from geniesim.rl.isaaclab.g2_student_training import load_canonical_sequences

    loaded = [
        load_canonical_sequences(
            path,
            sequence_contract=sequence_contract,
            expert_arm_action_scale=expert_arm_action_scale,
        )
        for path in paths
    ]
    burn_in = {int(item["burn_in_steps"]) for item in loaded}
    if burn_in != {sequence_contract.burn_in_steps}:
        raise RuntimeError("student dataset burn-in contract differs")
    names = tuple(
        name for name, value in loaded[0].items() if isinstance(value, torch.Tensor)
    )
    if any(
        tuple(name for name, value in item.items() if isinstance(value, torch.Tensor))
        != names
        for item in loaded
    ):
        raise RuntimeError("student compact sequence fields differ between files")
    return {
        **{name: torch.cat([item[name] for item in loaded], dim=0) for name in names},
        "burn_in_steps": sequence_contract.burn_in_steps,
    }


def main() -> int:
    args = parse_args()
    if args.updates <= 0 or args.batch_size <= 0 or args.checkpoint_interval <= 0:
        raise ValueError("updates, batch-size and checkpoint-interval must be positive")
    from geniesim.rl.isaaclab.g2_student_training import (
        G2RecurrentStudentTrainer,
        G2StudentTrainingConfig,
        estimate_student_rgbd_storage,
    )
    from geniesim.rl.isaaclab.g2_visual_sac import (
        G2_CAMERA_ENCODER_PROFILES,
        G2RecurrentVisualPolicyContract,
    )
    from geniesim.rl.isaaclab.g2_recurrent_profile import load_g2_recurrent_profile

    profile = load_g2_recurrent_profile(args.recurrent_profile)

    config = G2StudentTrainingConfig(
        hidden_dim=profile.hidden_dim if args.hidden_dim is None else args.hidden_dim,
        gru_num_layers=(
            profile.gru_num_layers
            if args.gru_num_layers is None
            else args.gru_num_layers
        ),
        learning_rate=args.learning_rate,
        gradient_clip=(
            profile.gradient_clip
            if args.gradient_clip is None
            else args.gradient_clip
        ),
        sequence_length=(
            profile.sequence_length
            if args.sequence_length is None
            else args.sequence_length
        ),
        burn_in_steps=(
            profile.burn_in_steps
            if args.burn_in_steps is None
            else args.burn_in_steps
        ),
        stride=(
            profile.sequence_stride
            if args.sequence_stride is None
            else args.sequence_stride
        ),
        share_camera_encoder_weights=args.share_camera_encoder_weights,
        camera_encoder_channels=G2_CAMERA_ENCODER_PROFILES[
            args.camera_encoder_profile
        ],
        random_shift_pad=args.random_shift_pad,
        cross_camera_contrastive_loss_weight=(
            args.cross_camera_contrastive_weight
        ),
        cross_camera_contrastive_temperature=(
            args.cross_camera_contrastive_temperature
        ),
        cross_camera_false_negative_position_threshold_m=(
            args.cross_camera_false_negative_position_threshold_m
        ),
        temporal_pose_residual_loss_weight=args.temporal_pose_residual_weight,
        temporal_pose_residual_rotation_weight=(
            args.temporal_pose_residual_rotation_weight
        ),
        seed=args.seed,
    ).validate()
    recurrent_policy_contract = G2RecurrentVisualPolicyContract(
        hidden_dim=config.hidden_dim,
        gru_num_layers=config.gru_num_layers,
        sequence_length=config.sequence_length,
        burn_in_steps=config.burn_in_steps,
        sequence_stride=config.stride,
        camera_encoder_channels=config.camera_encoder_channels,
    ).validated()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and args.resume is None:
        raise RuntimeError("output directory is non-empty; use --resume or a new directory")
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)

    trainer = G2RecurrentStudentTrainer(config, device=args.device)
    if args.resume is not None:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        trainer.load_state_dict(state)
    sequences = _combine_datasets(
        args.dataset,
        config.sequence_contract(),
        expert_arm_action_scale=args.expert_arm_action_scale,
    )
    window_count = int(sequences["padding_mask"].shape[0])
    if window_count <= 0:
        raise RuntimeError("student dataset contains no sequence windows")

    run = None
    if args.wandb and args.wandb_mode != "disabled":
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            dir=str(output),
            config={
                **asdict(config),
                "recurrent_profile": profile.serializable(),
                "dataset": [str(path.resolve()) for path in args.dataset],
                "sequence_window_count": window_count,
                "student_only_offline_distillation": True,
                "expert_arm_action_scale": args.expert_arm_action_scale,
                "recurrent_visual_policy_contract": (
                    recurrent_policy_contract.serializable()
                ),
            },
        )
        wandb.define_metric("student/update")
        wandb.define_metric("*", step_metric="student/update")

    started = time.monotonic()
    final_metrics: dict[str, float] = {}
    try:
        while trainer.update_count < args.updates:
            indices = torch.as_tensor(
                trainer.rng.integers(0, window_count, size=args.batch_size),
                dtype=torch.long,
            )
            batch = {
                name: value.index_select(0, indices)
                for name, value in sequences.items()
                if isinstance(value, torch.Tensor)
            }
            batch["burn_in_steps"] = int(sequences["burn_in_steps"])
            final_metrics = trainer.update(batch)
            if run is not None:
                run.log(final_metrics, step=trainer.update_count)
            if trainer.update_count % args.checkpoint_interval == 0:
                trainer.save_checkpoint(
                    checkpoints / f"student_update_{trainer.update_count:012d}.pt"
                )
        final_checkpoint = trainer.save_checkpoint(checkpoints / "student_final.pt")
        summary = {
            "status": "COMPLETED",
            "recurrent_profile": profile.serializable(),
            "student_update": trainer.update_count,
            "elapsed_seconds": time.monotonic() - started,
            "sequence_window_count": window_count,
            "50m_rgbd_storage_floor": estimate_student_rgbd_storage(50_000_000),
            "dataset": [str(path.resolve()) for path in args.dataset],
            "expert_arm_action_scale": args.expert_arm_action_scale,
            "checkpoint": str(final_checkpoint),
            "config": asdict(config),
            "recurrent_visual_policy_contract": (
                recurrent_policy_contract.serializable()
            ),
            "final_metrics": final_metrics,
        }
        _atomic_json(output / "summary.json", summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
        return 0
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    raise SystemExit(main())
