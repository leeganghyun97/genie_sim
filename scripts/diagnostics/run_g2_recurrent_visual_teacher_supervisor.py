#!/usr/bin/env python3
"""Supervisor for the recurrent two-camera G2 visual Teacher SAC."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
CHILD = ROOT / "scripts/diagnostics/g2_visual_reverse_sac_phase.py"
sys.path.insert(0, str(ROOT / "source"))

from geniesim.rl.isaaclab.g2_process_lifecycle import (  # noqa: E402
    G2_SHUTDOWN_MODES,
    G2_SHUTDOWN_OFFICIAL_FAST,
    verify_pre_close_attestation,
)
from geniesim.rl.isaaclab.g2_recurrent_profile import (  # noqa: E402
    default_recurrent_profile_name,
    recurrent_profile_names,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=3600.0,
        help="child timeout in seconds; 0 disables the wall-clock timeout for long runs",
    )
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument(
        "--recurrent-profile",
        choices=recurrent_profile_names(),
        default=default_recurrent_profile_name(),
    )
    parser.add_argument("--total-transitions", type=int, default=2000)
    parser.add_argument("--episode-horizon-steps", type=int, default=1600)
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=50000)
    parser.add_argument("--checkpoint-interval", type=int, default=2000)
    parser.add_argument("--checkpoint-retention", type=int, default=5)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--settling-steps", type=int, default=100)
    parser.add_argument("--reverse-preroll-max-steps", type=int, default=80)
    parser.add_argument(
        "--reverse-preroll-position-tolerance-m", type=float, default=0.002
    )
    parser.add_argument(
        "--reverse-preroll-maximum-normalized-translation", type=float, default=0.25
    )
    parser.add_argument("--camera-capture-interval-steps", type=int, default=10)
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument("--camera-height", type=int, default=192)
    parser.add_argument("--clone-in-fabric", action="store_true")
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
    parser.add_argument("--demonstration-dataset", action="append", type=Path)
    parser.add_argument("--demonstration-bc-updates", type=int, default=0)
    parser.add_argument("--demonstration-batch-size", type=int, default=32)
    parser.add_argument(
        "--demonstration-expert-arm-action-scale", type=float, default=0.75
    )
    parser.add_argument(
        "--demonstration-online-bc-initial-coefficient", type=float, default=1.0
    )
    parser.add_argument(
        "--demonstration-online-bc-minimum-coefficient", type=float, default=0.1
    )
    parser.add_argument(
        "--demonstration-online-bc-decay-transitions", type=int, default=50000
    )
    parser.add_argument("--demonstration-q-filter-temperature", type=float, default=0.1)
    parser.add_argument("--demonstration-success-priority-weight", type=float, default=2.0)
    parser.add_argument("--demonstration-reach-fraction", type=float, default=0.25)
    parser.add_argument("--demonstration-pregrasp-fraction", type=float, default=0.20)
    parser.add_argument("--demonstration-contact-fraction", type=float, default=0.20)
    parser.add_argument("--demonstration-stable-fraction", type=float, default=0.20)
    parser.add_argument("--demonstration-lift-fraction", type=float, default=0.15)
    parser.add_argument("--online-replay-reach-fraction", type=float, default=0.55)
    parser.add_argument("--online-replay-contact-fraction", type=float, default=0.20)
    parser.add_argument("--online-replay-stable-fraction", type=float, default=0.15)
    parser.add_argument("--online-replay-lift-fraction", type=float, default=0.10)
    parser.add_argument("--her-sequence-fraction", type=float, default=0.25)
    parser.add_argument("--her-force-temperature-j", type=float, default=0.02)
    parser.add_argument("--her-force-sampling-mixture", type=float, default=0.25)
    parser.add_argument(
        "--demonstration-force-sampling-mixture", type=float, default=0.25
    )
    parser.add_argument("--near-contact-auxiliary-distance-m", type=float, default=0.10)
    parser.add_argument("--near-contact-relative-pose-multiplier", type=float, default=4.0)
    parser.add_argument("--student-head-distillation-weight", type=float, default=0.1)
    parser.add_argument(
        "--demonstration-success-augmented-windows", type=int, default=0
    )
    parser.add_argument(
        "--demonstration-failure-augmented-windows", type=int, default=0
    )
    parser.add_argument("--curriculum-evaluation-envs", type=int, default=1)
    parser.add_argument("--replay-samples-per-transition", type=float, default=1.0)
    parser.add_argument("--maximum-updates-per-vector-step", type=int, default=32)
    parser.add_argument("--arm-joint-target-speed-rad-s", type=float, default=0.6)
    parser.add_argument(
        "--reference-arm-joint-target-speed-rad-s", type=float, default=0.4
    )
    parser.add_argument(
        "--arm-joint-target-acceleration-rad-s2", type=float, default=2.0
    )
    parser.add_argument(
        "--reverse-preroll-arm-target-acceleration-rad-s2",
        type=float,
        default=3.0,
    )
    parser.add_argument("--gripper-joint-target-speed-rad-s", type=float, default=0.8)
    parser.add_argument("--gripper-minimum-hold-steps", type=int, default=5)
    parser.add_argument(
        "--reference-gripper-close-position-tolerance-m",
        type=float,
        default=0.016,
    )
    parser.add_argument("--scripted-reverse-bootstrap", action="store_true")
    parser.add_argument("--bootstrap-lift-action", type=float, default=0.03)
    parser.add_argument("--maximum-arm-action-magnitude", type=float, default=0.10)
    parser.add_argument(
        "--reference-maximum-arm-action-magnitude", type=float, default=0.075
    )
    parser.add_argument("--arm-action-slew-per-policy-step", type=float, default=0.02)
    parser.add_argument("--reference-behavior-initial-probability", type=float, default=0.80)
    parser.add_argument("--reference-behavior-minimum-probability", type=float, default=0.20)
    parser.add_argument("--reference-behavior-probability-decrement", type=float, default=0.10)
    parser.add_argument("--contact-approach-distance-m", type=float, default=0.070)
    parser.add_argument("--contact-approach-arm-speed-rad-s", type=float, default=0.05)
    parser.add_argument("--contact-approach-gripper-speed-rad-s", type=float, default=0.40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="offline")
    parser.add_argument("--wandb-project", default="geniesim-g2-recurrent-visual-teacher")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument(
        "--shutdown-mode", choices=G2_SHUTDOWN_MODES, default="official-fast"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    child_output = args.output / "child_artifacts"
    stdout_path = args.output / "stdout.log"
    stderr_path = args.output / "stderr.log"
    command = [
        str(args.python),
        str(CHILD),
        "--output", str(child_output),
        "--num-envs", str(args.num_envs),
        "--recurrent-profile", args.recurrent_profile,
        "--total-transitions", str(args.total_transitions),
        "--episode-horizon-steps", str(args.episode_horizon_steps),
        "--learning-starts", str(args.learning_starts),
        "--batch-size", str(args.batch_size),
        "--replay-capacity", str(args.replay_capacity),
        "--checkpoint-interval", str(args.checkpoint_interval),
        "--checkpoint-retention", str(args.checkpoint_retention),
        "--log-interval", str(args.log_interval),
        "--settling-steps", str(args.settling_steps),
        "--reverse-preroll-max-steps", str(args.reverse_preroll_max_steps),
        "--reverse-preroll-position-tolerance-m",
        str(args.reverse_preroll_position_tolerance_m),
        "--reverse-preroll-maximum-normalized-translation",
        str(args.reverse_preroll_maximum_normalized_translation),
        "--camera-capture-interval-steps", str(args.camera_capture_interval_steps),
        "--camera-width", str(args.camera_width),
        "--camera-height", str(args.camera_height),
        "--random-shift-pad", str(args.random_shift_pad),
        "--cross-camera-contrastive-weight", str(args.cross_camera_contrastive_weight),
        "--cross-camera-contrastive-temperature", str(args.cross_camera_contrastive_temperature),
        "--cross-camera-false-negative-position-threshold-m",
        str(args.cross_camera_false_negative_position_threshold_m),
        "--temporal-pose-residual-weight", str(args.temporal_pose_residual_weight),
        "--temporal-pose-residual-rotation-weight",
        str(args.temporal_pose_residual_rotation_weight),
        "--demonstration-bc-updates", str(args.demonstration_bc_updates),
        "--demonstration-batch-size", str(args.demonstration_batch_size),
        "--demonstration-expert-arm-action-scale",
        str(args.demonstration_expert_arm_action_scale),
        "--demonstration-online-bc-initial-coefficient",
        str(args.demonstration_online_bc_initial_coefficient),
        "--demonstration-online-bc-minimum-coefficient",
        str(args.demonstration_online_bc_minimum_coefficient),
        "--demonstration-online-bc-decay-transitions",
        str(args.demonstration_online_bc_decay_transitions),
        "--demonstration-q-filter-temperature",
        str(args.demonstration_q_filter_temperature),
        "--demonstration-success-priority-weight",
        str(args.demonstration_success_priority_weight),
        "--demonstration-reach-fraction", str(args.demonstration_reach_fraction),
        "--demonstration-pregrasp-fraction", str(args.demonstration_pregrasp_fraction),
        "--demonstration-contact-fraction", str(args.demonstration_contact_fraction),
        "--demonstration-stable-fraction", str(args.demonstration_stable_fraction),
        "--demonstration-lift-fraction", str(args.demonstration_lift_fraction),
        "--online-replay-reach-fraction", str(args.online_replay_reach_fraction),
        "--online-replay-contact-fraction", str(args.online_replay_contact_fraction),
        "--online-replay-stable-fraction", str(args.online_replay_stable_fraction),
        "--online-replay-lift-fraction", str(args.online_replay_lift_fraction),
        "--her-sequence-fraction", str(args.her_sequence_fraction),
        "--her-force-temperature-j", str(args.her_force_temperature_j),
        "--her-force-sampling-mixture", str(args.her_force_sampling_mixture),
        "--demonstration-force-sampling-mixture",
        str(args.demonstration_force_sampling_mixture),
        "--near-contact-auxiliary-distance-m", str(args.near_contact_auxiliary_distance_m),
        "--near-contact-relative-pose-multiplier", str(args.near_contact_relative_pose_multiplier),
        "--student-head-distillation-weight", str(args.student_head_distillation_weight),
        "--demonstration-success-augmented-windows",
        str(args.demonstration_success_augmented_windows),
        "--demonstration-failure-augmented-windows",
        str(args.demonstration_failure_augmented_windows),
        "--curriculum-evaluation-envs", str(args.curriculum_evaluation_envs),
        "--replay-samples-per-transition", str(args.replay_samples_per_transition),
        "--maximum-updates-per-vector-step", str(args.maximum_updates_per_vector_step),
        "--arm-joint-target-speed-rad-s", str(args.arm_joint_target_speed_rad_s),
        "--reference-arm-joint-target-speed-rad-s",
        str(args.reference_arm_joint_target_speed_rad_s),
        "--arm-joint-target-acceleration-rad-s2",
        str(args.arm_joint_target_acceleration_rad_s2),
        "--reverse-preroll-arm-target-acceleration-rad-s2",
        str(args.reverse_preroll_arm_target_acceleration_rad_s2),
        "--gripper-joint-target-speed-rad-s", str(args.gripper_joint_target_speed_rad_s),
        "--gripper-minimum-hold-steps", str(args.gripper_minimum_hold_steps),
        "--reference-gripper-close-position-tolerance-m",
        str(args.reference_gripper_close_position_tolerance_m),
        "--bootstrap-lift-action", str(args.bootstrap_lift_action),
        "--maximum-arm-action-magnitude", str(args.maximum_arm_action_magnitude),
        "--reference-maximum-arm-action-magnitude",
        str(args.reference_maximum_arm_action_magnitude),
        "--arm-action-slew-per-policy-step", str(args.arm_action_slew_per_policy_step),
        "--reference-behavior-initial-probability", str(args.reference_behavior_initial_probability),
        "--reference-behavior-minimum-probability", str(args.reference_behavior_minimum_probability),
        "--reference-behavior-probability-decrement", str(args.reference_behavior_probability_decrement),
        "--contact-approach-distance-m", str(args.contact_approach_distance_m),
        "--contact-approach-arm-speed-rad-s", str(args.contact_approach_arm_speed_rad_s),
        "--contact-approach-gripper-speed-rad-s", str(args.contact_approach_gripper_speed_rad_s),
        "--seed", str(args.seed),
        "--shutdown-mode", args.shutdown_mode,
    ]
    if args.wandb:
        command.extend(("--wandb", "--wandb-mode", args.wandb_mode))
        command.extend(("--wandb-project", args.wandb_project))
        if args.wandb_entity:
            command.extend(("--wandb-entity", args.wandb_entity))
        if args.wandb_run_name:
            command.extend(("--wandb-run-name", args.wandb_run_name))
    for dataset in args.demonstration_dataset or ():
        command.extend(("--demonstration-dataset", str(dataset.resolve())))
    if args.clone_in_fabric:
        command.append("--clone-in-fabric")
    if args.scripted_reverse_bootstrap:
        command.append("--scripted-reverse-bootstrap")

    environment = os.environ.copy()
    for key in ("PYTHONPATH", "ISAAC_PATH", "CARB_APP_PATH"):
        environment.pop(key, None)
    environment["OMNI_KIT_ACCEPT_EULA"] = "YES"
    environment["PYTHONUNBUFFERED"] = "1"
    started = time.monotonic()
    timed_out = False
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        child = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        try:
            if args.timeout_s == 0.0:
                child.wait()
            elif args.timeout_s > 0.0:
                child.wait(timeout=args.timeout_s)
            else:
                raise ValueError("timeout-s must be non-negative")
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)

    result_path = child_output / "result_pre_close.json"
    durability_path = child_output / "pre_close_durability.json"
    result = json.loads(result_path.read_text()) if result_path.is_file() else None
    markers = stdout_path.read_text(errors="replace").splitlines()
    durability_pass = verify_pre_close_attestation(durability_path)
    functional_pass = bool(
        result
        and result.get("status") == "COMPLETE"
        and result.get("teacher_actor_uses_rgbd") is True
        and result.get("teacher_actor_uses_gru") is True
        and (
            args.demonstration_bc_updates == 0
            or (
                result.get("demonstration_bc_updates")
                == args.demonstration_bc_updates
                and bool(result.get("demonstration_datasets_sha256"))
                and result.get("demonstration_success_augmented_window_count", 0)
                == args.demonstration_success_augmented_windows
                and result.get("demonstration_failure_augmented_window_count", 0)
                == args.demonstration_failure_augmented_windows
            )
        )
    )
    if timed_out:
        shutdown = "SHUTDOWN_HANG"
    elif child.returncode in (-11, 139):
        shutdown = "SHUTDOWN_FINALIZATION_SIGSEGV"
    elif child.returncode in (-6, 134):
        shutdown = "SHUTDOWN_SIGABRT"
    elif (
        args.shutdown_mode == G2_SHUTDOWN_OFFICIAL_FAST
        and child.returncode == 0
        and durability_pass
        and "PRE_CLOSE_DURABILITY_COMMITTED" in markers
    ):
        shutdown = "SHUTDOWN_OFFICIAL_FAST_AFTER_DURABLE_COMMIT"
    elif "APP_CLOSE_RETURNED" in markers and "MAIN_RETURNED" in markers:
        shutdown = "SHUTDOWN_CLEAN"
    else:
        shutdown = "SHUTDOWN_UNCLASSIFIED"
    report = {
        "schema": "geniesim_g2_recurrent_visual_teacher_supervisor_v1",
        "functional_verdict": (
            "G2_RECURRENT_VISUAL_TEACHER_PASS"
            if functional_pass
            else "G2_RECURRENT_VISUAL_TEACHER_FAIL"
        ),
        "shutdown_verdict": shutdown,
        "rgbd_encoder": True,
        "gru": True,
        "recurrent_profile": args.recurrent_profile,
        "keyboard_shared_model_contract": True,
        "demonstration_dataset": [
            str(path.resolve()) for path in (args.demonstration_dataset or ())
        ],
        "demonstration_bc_updates": args.demonstration_bc_updates,
        "demonstration_success_augmented_windows": (
            args.demonstration_success_augmented_windows
        ),
        "demonstration_failure_augmented_windows": (
            args.demonstration_failure_augmented_windows
        ),
        "exit_code": child.returncode,
        "timeout": timed_out,
        "elapsed_seconds": time.monotonic() - started,
        "pre_close_durability_pass": durability_pass,
        "result": str(result_path) if result_path.is_file() else None,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "command": command,
    }
    (args.output / "supervisor_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0 if functional_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
