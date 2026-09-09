#!/usr/bin/env python3
"""Live two-camera reverse-curriculum asymmetric SAC for G2 Lift."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))

from geniesim.rl.isaaclab.g2_process_lifecycle import (
    G2_SHUTDOWN_MODES,
    commit_pre_close_artifacts,
    use_official_fast_shutdown,
)
from geniesim.rl.isaaclab.g2_recurrent_profile import (
    default_recurrent_profile_name,
    load_g2_recurrent_profile,
    recurrent_profile_names,
)


def run(args) -> dict:
    print("RUNTIME_IMPORTS_BEGIN", flush=True)
    import numpy as np
    import torch
    from isaaclab.envs import ManagerBasedRLEnv

    from geniesim.rl.isaaclab.g2_lift_methodology import (
        G2LiftSandboxContract, RIGHT_ARM_JOINTS, RIGHT_GRIPPER_MASTER,
        TORSO_JOINTS,
    )
    from geniesim.rl.isaaclab import g2_lift_task_mdp
    from geniesim.rl.isaaclab.g2_camera_timing import dual_camera_capture_timing
    from geniesim.rl.isaaclab.g2_camera_visibility import (
        G2DualCameraVisibilityEvaluator,
    )
    from geniesim.rl.isaaclab.g2_teacher_runtime import G2TeacherRuntimeObservationBuilder
    from geniesim.rl.isaaclab.g2_teacher_sac import (
        G2TeacherObservationContract,
        G2TeacherStatefulActionTransform,
    )
    from geniesim.rl.isaaclab.g2_training_schedule import G2OffPolicyUpdateSchedule
    from geniesim.rl.isaaclab.g2_redundancy_teleop import tensor_value
    from geniesim.rl.isaaclab.g2_visual_sac import (
        G2StudentObservationContract,
        G2EpisodeReferenceBehavior,
        G2RecurrentVisualAsymmetricSAC,
        G2RecurrentVisualPolicyContract,
        G2RecurrentVisualReplayBuffer,
        G2RecurrentVisualSACConfig,
        G2_CAMERA_ENCODER_PROFILES,
        G2_RECURRENT_VISUAL_POLICY_SCHEMA,
        G2_ONLINE_REPLAY_PHASES,
        G2_VISUAL_ARM_ACTION_DIM,
        G2_VISUAL_CAMERA_SHAPE,
        G2_VISUAL_REPLAY_SCHEMA,
        classify_demonstration_sequence_phases,
        demonstration_bc_schedule,
        demonstration_force_energy_priorities,
        demonstration_policy_mastery,
        g2_reference_controller_action,
        pack_rgbd,
        sample_phase_stratified_demonstration_indices,
    )
    from geniesim.rl.isaaclab.g2_teacher_her import (
        relabel_recurrent_visual_batch_with_future_goals,
    )
    from geniesim.rl.isaaclab.g2_teacher_sac import G2_TEACHER_OBSERVATION_SCHEMA
    from geniesim.rl.isaaclab.g2_student_training import (
        list_canonical_episodes,
        load_canonical_episode,
        load_canonical_sequences,
    )
    from geniesim.rl.isaaclab.g2_demonstration_seed import select_safe_pregrasp_seed
    from geniesim.rl.isaaclab.g2_keyboard_pose import (
        G2_KEYBOARD_CUBE_CENTER_WORLD_M,
        G2_KEYBOARD_PHOTO_EE_CUBE_DISTANCE_M,
        G2_KEYBOARD_PHOTO_POSE_PROFILE,
        G2_KEYBOARD_PHOTO_RIGHT_ARM_Q,
        G2_KEYBOARD_SIDE_GRASP_CENTER_HEIGHT_OFFSET_M,
        G2_KEYBOARD_SIDE_PREGRASP_AXIAL_OFFSET_M,
        G2_KEYBOARD_TABLE_CENTER_WORLD_M,
    )
    from geniesim.rl.isaaclab.g2_keyboard_teacher_dataset import (
        G2StudentSequenceContract,
    )
    from geniesim.rl.isaaclab.g2_visual_sac_env_cfg import G2LiftVisualSACEnvCfg
    from geniesim.rl.sac.stage2_logging import Stage2RunLogger
    print("RUNTIME_IMPORTS_DONE", flush=True)

    if args.num_envs <= 0 or args.total_transitions <= 0:
        raise ValueError("num_envs and total_transitions must be positive")
    if args.episode_horizon_steps <= 0:
        raise ValueError("episode_horizon_steps must be positive")
    if args.camera_capture_interval_steps <= 0:
        raise ValueError("camera capture interval must be positive")
    if args.camera_width <= 0 or args.camera_height <= 0:
        raise ValueError("camera width and height must be positive")
    if args.arm_joint_target_acceleration_rad_s2 <= 0.0:
        raise ValueError("arm joint target acceleration must be positive")
    if not (
        0.0
        < args.reference_arm_joint_target_speed_rad_s
        <= args.arm_joint_target_speed_rad_s
    ):
        raise ValueError(
            "reference arm joint target speed must be positive and no greater "
            "than the policy arm speed"
        )
    if not (
        0.0
        < args.reference_maximum_arm_action_magnitude
        <= args.maximum_arm_action_magnitude
    ):
        raise ValueError(
            "reference arm action magnitude must be positive and no greater "
            "than the policy action magnitude"
        )
    if args.reverse_preroll_arm_target_acceleration_rad_s2 <= 0.0:
        raise ValueError("reverse preroll arm target acceleration must be positive")
    if args.gripper_minimum_hold_steps < 0:
        raise ValueError(
            "gripper_minimum_hold_steps must be non-negative"
        )
    if args.reference_gripper_close_position_tolerance_m <= 0.0:
        raise ValueError("reference gripper close tolerance must be positive")
    if not 0.0 < args.bootstrap_lift_action <= 1.0:
        raise ValueError("bootstrap_lift_action must be in (0,1]")
    if not (
        0.0
        < args.reference_behavior_minimum_probability
        <= args.reference_behavior_initial_probability
        <= 1.0
    ):
        raise ValueError("invalid reference behavior probability range")
    if not 0.0 < args.reference_behavior_probability_decrement <= 1.0:
        raise ValueError("reference behavior probability decrement must be in (0,1]")
    if args.contact_approach_distance_m <= 0.0:
        raise ValueError("contact approach distance must be positive")
    if not (
        0.0
        < args.contact_approach_arm_speed_rad_s
        <= args.arm_joint_target_speed_rad_s
    ):
        raise ValueError("invalid contact approach arm speed")
    if not (
        0.0
        < args.contact_approach_gripper_speed_rad_s
        <= args.gripper_joint_target_speed_rad_s
    ):
        raise ValueError("invalid contact approach gripper speed")
    if args.checkpoint_interval <= 0 or args.checkpoint_retention <= 0:
        raise ValueError("checkpoint interval and retention must be positive")
    if args.demonstration_bc_updates < 0 or args.demonstration_batch_size <= 0:
        raise ValueError("demonstration BC updates must be non-negative and batch positive")
    if not 0.0 < args.demonstration_expert_arm_action_scale <= 1.0:
        raise ValueError("demonstration expert arm action scale must be in (0,1]")
    if args.demonstration_bc_updates and not args.demonstration_dataset:
        raise ValueError("demonstration BC updates require --demonstration-dataset")
    if args.demonstration_success_augmented_windows < 0:
        raise ValueError("successful demonstration augmentation cannot be negative")
    if args.demonstration_failure_augmented_windows < 0:
        raise ValueError("failed demonstration augmentation cannot be negative")
    if args.demonstration_success_augmented_windows and not args.demonstration_bc_updates:
        raise ValueError("successful demonstration augmentation requires BC updates")
    if args.demonstration_failure_augmented_windows and not args.demonstration_bc_updates:
        raise ValueError("failed demonstration augmentation requires BC updates")
    if not (
        0.0
        <= args.demonstration_online_bc_minimum_coefficient
        <= args.demonstration_online_bc_initial_coefficient
    ):
        raise ValueError("invalid online demonstration BC coefficient range")
    if args.demonstration_online_bc_decay_transitions <= 0:
        raise ValueError("online demonstration BC decay transitions must be positive")
    if args.demonstration_success_priority_weight < 1.0:
        raise ValueError("demonstration success priority weight must be at least one")
    if args.student_head_distillation_weight < 0.0:
        raise ValueError("student head distillation weight cannot be negative")
    if args.near_contact_auxiliary_distance_m <= 0.0:
        raise ValueError("near-contact auxiliary distance must be positive")
    if args.near_contact_relative_pose_multiplier < 1.0:
        raise ValueError("near-contact relative-pose multiplier must be at least one")
    if args.cross_camera_contrastive_weight < 0.0:
        raise ValueError("cross-camera contrastive weight cannot be negative")
    if args.cross_camera_contrastive_temperature <= 0.0:
        raise ValueError("cross-camera contrastive temperature must be positive")
    if args.cross_camera_false_negative_position_threshold_m < 0.0:
        raise ValueError("cross-camera false-negative threshold cannot be negative")
    if args.temporal_pose_residual_weight < 0.0:
        raise ValueError("temporal pose-residual weight cannot be negative")
    if args.temporal_pose_residual_rotation_weight < 0.0:
        raise ValueError("temporal pose-residual rotation weight cannot be negative")
    demonstration_phase_probabilities = (
        args.demonstration_reach_fraction,
        args.demonstration_pregrasp_fraction,
        args.demonstration_contact_fraction,
        args.demonstration_stable_fraction,
        args.demonstration_lift_fraction,
    )
    if not np.isclose(sum(demonstration_phase_probabilities), 1.0):
        raise ValueError("demonstration phase fractions must sum to one")
    online_replay_phase_probabilities = (
        args.online_replay_reach_fraction,
        args.online_replay_contact_fraction,
        args.online_replay_stable_fraction,
        args.online_replay_lift_fraction,
    )
    if any(value < 0.0 for value in online_replay_phase_probabilities) or not np.isclose(
        sum(online_replay_phase_probabilities), 1.0
    ):
        raise ValueError("online replay phase fractions must be non-negative and sum to one")
    if not 0.0 <= args.her_sequence_fraction <= 1.0:
        raise ValueError("HER sequence fraction must be in [0,1]")
    if args.her_force_temperature_j <= 0.0:
        raise ValueError("HER-force temperature must be positive")
    if not 0.0 <= args.her_force_sampling_mixture <= 1.0:
        raise ValueError("HER-force sampling mixture must be in [0,1]")
    if not 0.0 <= args.demonstration_force_sampling_mixture <= 1.0:
        raise ValueError("demonstration force sampling mixture must be in [0,1]")
    if args.replay_capacity < max(args.num_envs, args.batch_size):
        raise ValueError(
            "replay_capacity must fit one vector batch and one optimizer batch"
        )
    if not 0 <= args.curriculum_evaluation_envs < args.num_envs:
        raise ValueError(
            "curriculum_evaluation_envs must be in [0,num_envs); at least one "
            "environment must remain available for replay collection"
        )
    args.output.mkdir(parents=True, exist_ok=True)

    def write_progress(stage: str, **values) -> None:
        payload = {
            "schema": "g2_visual_reverse_runtime_progress_v1",
            "stage": stage,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            **values,
        }
        temporary = args.output / "runtime_progress.json.tmp"
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(args.output / "runtime_progress.json")

    write_progress("runtime/init_started", transitions=0, episodes=0, updates=0)
    sandbox = G2LiftSandboxContract().validated()
    demonstration_grasp_ready_seed = (
        select_safe_pregrasp_seed(
            args.demonstration_dataset,
            # A 20--50 mm centre-distance sample is already inside the live
            # OmniPicker pad/table contact envelope for some randomized cube
            # positions.  Use an exact measured, pre-contact 75 mm sample;
            # contact experience still comes from the saved sequences and
            # the bounded online approach, never from an initial overlap.
            target_distance_m=0.075,
            minimum_distance_m=0.070,
            maximum_distance_m=0.085,
        )
        if args.demonstration_dataset
        else None
    )
    demonstration_reset_seed = (
        select_safe_pregrasp_seed(
            args.demonstration_dataset,
            required_dataset=demonstration_grasp_ready_seed.dataset,
            required_episode=demonstration_grasp_ready_seed.episode,
        )
        if args.demonstration_dataset
        else None
    )
    cfg = G2LiftVisualSACEnvCfg()
    if demonstration_reset_seed is not None:
        cfg.events.reverse_curriculum_arm.params["pregrasp_right_arm_q"] = (
            demonstration_reset_seed.right_arm_q_rad
        )
        cfg.events.reverse_curriculum_arm.params["grasp_ready_right_arm_q"] = (
            demonstration_grasp_ready_seed.right_arm_q_rad
        )
        # The selected pose is an exact measured, collision-attested sample.
        # Do not perturb it before the bounded Cartesian preroll has evaluated
        # the current randomized cube geometry.
        cfg.events.reverse_curriculum_arm.params["joint_perturbation_rad"] = 0.0
        # With a 20 mm vertical side-pinch offset, rho in [0.25, 0.375]
        # corresponds to a collision-clear 83--73 mm centre distance.  The
        # former [0.95, 1.0] interval requested a 31--20 mm reset and could
        # begin in pad contact for randomized cube positions.
        cfg.events.reverse_curriculum_arm.params["progress_min"] = 0.25
        cfg.events.reverse_curriculum_arm.params["progress_max"] = 0.375
    cfg.scene.num_envs = args.num_envs
    cfg.episode_length_s = args.episode_horizon_steps * sandbox.policy_dt_s
    goal_hold_s = cfg.episode_length_s + sandbox.policy_dt_s
    cfg.commands.object_pose.resampling_time_range = (goal_hold_s, goal_hold_s)
    cfg.scene.clone_in_fabric = args.clone_in_fabric
    camera_update_period_s = (
        sandbox.physics_dt_s * args.camera_capture_interval_steps
    )
    cfg.scene.head_camera.update_period = camera_update_period_s
    cfg.scene.right_wrist_camera.update_period = camera_update_period_s
    cfg.scene.head_camera.width = args.camera_width
    cfg.scene.head_camera.height = args.camera_height
    cfg.scene.right_wrist_camera.width = args.camera_width
    cfg.scene.right_wrist_camera.height = args.camera_height
    cfg.sim.render_interval = args.camera_capture_interval_steps
    cfg.actions.arm_action.maximum_joint_target_speed_rad_s = args.arm_joint_target_speed_rad_s
    cfg.actions.arm_action.maximum_joint_target_acceleration_rad_s2 = (
        args.arm_joint_target_acceleration_rad_s2
    )
    cfg.actions.gripper_action.maximum_joint_target_speed_rad_s = args.gripper_joint_target_speed_rad_s
    cfg.observations.policy.enable_corruption = False
    cfg.commands.object_pose.debug_vis = False
    env = ManagerBasedRLEnv(cfg=cfg)
    # The persistent table-boundary termination reads a runtime authority so
    # task-specific table translations cannot be compared against the generic
    # sandbox center.  Publish the exact same center as keyboard collection.
    env._g2_task_table_center_root_m = tuple(G2_KEYBOARD_TABLE_CENTER_WORLD_M)
    print("G2_VISUAL_ENV_READY", flush=True)
    write_progress("runtime/environment_ready", transitions=0, episodes=0, updates=0)
    robot, cube = env.scene["robot"], env.scene["object"]
    head, wrist = env.scene["head_camera"], env.scene["right_wrist_camera"]
    inner, outer = env.scene["right_inner_finger_contact"], env.scene["right_outer_finger_contact"]
    camera_visibility_evaluator = getattr(
        env, "_g2_camera_visibility_evaluator", None
    )
    if camera_visibility_evaluator is None:
        camera_visibility_evaluator = G2DualCameraVisibilityEvaluator(env)
        env._g2_camera_visibility_evaluator = camera_visibility_evaluator
    controlled = (*RIGHT_ARM_JOINTS, RIGHT_GRIPPER_MASTER)
    indices = torch.tensor([robot.joint_names.index(n) for n in controlled], device=env.device)
    torso_indices = torch.tensor(
        [robot.joint_names.index(n) for n in TORSO_JOINTS], device=env.device
    )
    teacher_contract = G2TeacherObservationContract()
    previous_action = torch.zeros(args.num_envs, 7, device=env.device)
    teacher_builder = G2TeacherRuntimeObservationBuilder(env)
    rng = np.random.default_rng(args.seed)
    update_schedule = G2OffPolicyUpdateSchedule(
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        replay_samples_per_transition=args.replay_samples_per_transition,
        maximum_updates_per_vector_step=args.maximum_updates_per_vector_step,
    )
    action_transform = G2TeacherStatefulActionTransform(
        maximum_arm_action_magnitude=args.maximum_arm_action_magnitude,
        arm_action_slew_per_policy_step=args.arm_action_slew_per_policy_step,
    ).validated()
    if args.bootstrap_lift_action > action_transform.maximum_arm_action_magnitude:
        raise ValueError(
            "bootstrap_lift_action is an applied controller action and must not "
            "exceed maximum_arm_action_magnitude"
        )

    def privileged():
        return teacher_builder.build(previous_action)

    student_contract = G2StudentObservationContract()

    def camera_age():
        # Camera counters and clocks reset per environment.  Use the sensor's
        # own last-update clock so a partial reset cannot turn process-global
        # elapsed time into a false stale-frame signal.
        episode_time = tensor_value(env.episode_length_buf).to(torch.float32) * float(
            env.step_dt
        )
        _, age = dual_camera_capture_timing(
            head, wrist, fallback_episode_time_s=episode_time
        )
        return age

    def proprio(state, frame_age):
        s = teacher_contract.slices
        joint_position = tensor_value(robot.data.joint_pos)
        default_position = tensor_value(robot.data.default_joint_pos)
        joint_velocity = tensor_value(robot.data.joint_vel)
        return student_contract.build(
            arm_hand_joint_position_relative_rad=state[
                :, s["controlled_joint_position_relative_rad"]
            ],
            arm_hand_joint_velocity_rad_s=state[
                :, s["controlled_joint_velocity_rad_s"]
            ],
            torso_joint_position_relative_rad=(
                joint_position.index_select(1, torso_indices)
                - default_position.index_select(1, torso_indices)
            ),
            torso_joint_velocity_rad_s=joint_velocity.index_select(1, torso_indices),
            end_effector_pose_root_xyzw=state[:, s["end_effector_pose_root_xyzw"]],
            goal_position_root_m=state[:, s["goal_position_root_m"]],
            previous_action=state[:, s["previous_action"]],
            camera_frame_age_s=frame_age,
        )

    def visual():
        return pack_rgbd(
            head.data.output["rgb"], head.data.output["distance_to_image_plane"],
            wrist.data.output["rgb"], wrist.data.output["distance_to_image_plane"],
        )

    def reverse_preroll(
        initial_previous_velocity: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Safely correct near resets after PhysX settling, outside replay.

        Direct joint teleport reaches the 2 cm FK endpoint offline but drifts
        to roughly 4.5 cm after live articulation settling.  This bounded
        Cartesian correction uses privileged geometry only as reset logic;
        it is never exposed to the deployment actor observation.
        """

        near = env._g2_reverse_last_near.clone()
        near_ids = near.nonzero(as_tuple=True)[0]
        last_action = torch.zeros(args.num_envs, 7, device=env.device)
        last_action[:, 6] = 1.0
        metrics = {
            "steps": 0.0,
            "maximum_arm_fd_velocity_rad_s": 0.0,
            "maximum_arm_fd_acceleration_rad_s2": 0.0,
            "maximum_cube_displacement_m": 0.0,
            "maximum_position_error_m": 0.0,
            "final_maximum_position_error_m": 0.0,
        }
        if not bool(near.any()):
            return last_action, metrics
        # Settling and reverse preroll are reset initialization, not replay
        # episode time.  Keep all physical/collision terminations active while
        # separating their bounded initialization budget from the RL horizon.
        env.episode_length_buf[:] = 0
        initial_cube_position = tensor_value(cube.data.root_pos_w).clone()
        aligned_at_pregrasp = (
            near.clone()
            if demonstration_reset_seed is not None
            else torch.zeros(args.num_envs, dtype=torch.bool, device=env.device)
        )
        near_converged_streak = torch.zeros(
            args.num_envs, dtype=torch.long, device=env.device
        )
        target_synchronized_at_near = torch.zeros(
            args.num_envs, dtype=torch.bool, device=env.device
        )
        best_near_error = torch.full(
            (args.num_envs,), float("inf"), dtype=torch.float32, device=env.device
        )
        best_near_error_step = torch.full(
            (args.num_envs,), -1, dtype=torch.long, device=env.device
        )
        if initial_previous_velocity.shape != (
            args.num_envs,
            len(RIGHT_ARM_JOINTS),
        ):
            raise ValueError("reverse preroll velocity baseline shape mismatch")
        # Preserve the continuous measurement time axis from settling into
        # preroll.  Initializing this to zero manufactured an acceleration
        # spike whenever the last settling step still had nonzero velocity.
        previous_velocity = initial_previous_velocity.detach().clone()
        for step in range(args.reverse_preroll_max_steps):
            reset_state = privileged()
            slices = teacher_contract.slices
            ee = reset_state[:, slices["end_effector_pose_root_xyzw"]][:, :3]
            cube_center = reset_state[:, slices["cube_pose_root_xyzw"]][:, :3]
            # Keyboard demonstrations use a horizontal side pinch.  Reusing
            # the legacy top-down ``cube_z + 0.10`` target made a valid
            # dataset-derived seed move 8--13 cm away in Y/Z before rollout.
            # The reset controller now follows the same root-frame geometry
            # that produced the demonstrations: approach along +X, with the
            # gripper center slightly above the cube center.
            pregrasp_desired = cube_center.clone()
            pregrasp_desired[:, 0] -= G2_KEYBOARD_SIDE_PREGRASP_AXIAL_OFFSET_M
            pregrasp_desired[:, 2] += G2_KEYBOARD_SIDE_GRASP_CENTER_HEIGHT_OFFSET_M
            progress = env._g2_reverse_last_progress.clamp(0.0, 1.0)
            sampled_near_distance = 0.100 + progress * (
                sandbox.reverse_initial_ee_cube_distance_m - 0.100
            )
            near_desired = cube_center.clone()
            near_desired[:, 0] -= sampled_near_distance
            near_desired[:, 2] += G2_KEYBOARD_SIDE_GRASP_CENTER_HEIGHT_OFFSET_M
            pregrasp_error = torch.linalg.vector_norm(
                pregrasp_desired - ee, dim=-1
            )
            aligned_at_pregrasp |= near & (
                pregrasp_error <= args.reverse_preroll_position_tolerance_m
            )
            desired = torch.where(
                aligned_at_pregrasp.unsqueeze(-1), near_desired, pregrasp_desired
            )
            error = desired - ee
            position_error = torch.linalg.vector_norm(error, dim=-1)
            improved = near & (position_error < best_near_error)
            best_near_error = torch.where(improved, position_error, best_near_error)
            best_near_error_step = torch.where(
                improved,
                torch.full_like(best_near_error_step, step),
                best_near_error_step,
            )
            metrics["maximum_position_error_m"] = max(
                metrics["maximum_position_error_m"],
                float(position_error[near].max()),
            )
            metrics["final_maximum_position_error_m"] = float(
                position_error[near].max()
            )
            within_near_tolerance = aligned_at_pregrasp & (
                torch.linalg.vector_norm(near_desired - ee, dim=-1)
                <= args.reverse_preroll_position_tolerance_m
            )
            near_converged_streak = torch.where(
                near & within_near_tolerance,
                near_converged_streak + 1,
                torch.zeros_like(near_converged_streak),
            )
            # One cached or transient pose sample is not sufficient evidence
            # that all vector environments physically reached the near reset.
            # Require five consecutive policy intervals without relaxing the
            # 5 mm task-space tolerance.
            converged = near_converged_streak >= 5
            target_synchronized_at_near &= within_near_tolerance
            arm_term = env.action_manager._terms["arm_action"]
            # A converged Cartesian sample may still carry velocity in both
            # the physical articulation and the target generator.  Snapping
            # either state to zero caused a measured 14.87 rad/s^2 transient.
            # With physics_dt=2 ms, the 0.015 rad/s target-velocity threshold
            # bounds that hand-off to 7.5 rad/s^2, leaving margin below the
            # unchanged 10 rad/s^2 measured-motion hard limit.
            measured_slow = previous_velocity.abs().amax(dim=-1) <= 0.05
            target_slow = (
                arm_term.current_target_velocity_rad_s.abs().amax(dim=-1)
                <= 0.015
            )
            ready_to_synchronize = (
                converged
                & measured_slow
                & target_slow
                & ~target_synchronized_at_near
            )
            if bool(ready_to_synchronize.any()):
                arm_term.synchronize_target_to_measured(
                    ready_to_synchronize.nonzero(as_tuple=True)[0]
                )
                target_synchronized_at_near |= ready_to_synchronize
            active = near & ~converged
            if bool((target_synchronized_at_near | ~near).all()) and not bool(
                ready_to_synchronize.any()
            ):
                # Re-read the live runtime geometry at the commit boundary.
                # Vector state caches have occasionally exposed a converged
                # sample immediately before one environment returned to the
                # pregrasp pose.  Re-activate only those environments; do not
                # clip or relax their task-space target.
                commit_state = privileged()
                commit_ee = commit_state[:, teacher_contract.slices[
                    "end_effector_pose_root_xyzw"
                ]][:, :3]
                commit_error = torch.linalg.vector_norm(
                    near_desired - commit_ee, dim=-1
                )
                commit_invalid = near & (
                    commit_error > args.reverse_preroll_position_tolerance_m
                )
                if bool(commit_invalid.any()):
                    near_converged_streak[commit_invalid] = 0
                    continue
                break
            last_action.zero_()
            last_action[:, 6] = 1.0
            # Preserve the existing 0.25 far-field command, then taper it in
            # the final 20 mm so the acceleration-limited joint target can
            # brake before the 5 mm commit gate instead of oscillating across
            # it.  This changes reset trajectory timing, not the training
            # action scale or any safety authority.
            endpoint_action_limit = (
                args.reverse_preroll_maximum_normalized_translation
                * torch.clamp(position_error / 0.020, min=0.0, max=1.0)
            )
            active_limit = endpoint_action_limit[active].unsqueeze(-1)
            last_action[active, :3] = torch.maximum(
                torch.minimum(
                    error[active] / 0.01,
                    active_limit,
                ),
                -active_limit,
            )
            q_before = tensor_value(robot.data.joint_pos).index_select(1, indices[:7]).clone()
            # The preroll has its own explicit max-step bound.  Do not let the
            # RL episode time-out counter (which is semantically downstream of
            # reset initialization) terminate it; all physical safety terms
            # remain active and are still fail-closed below.
            env.episode_length_buf[:] = 0
            _, _, terminated, truncated, _ = env.step(last_action)
            # No environment may silently auto-reset while another member of
            # the vector batch is being aligned.  Camera warm-up is suppressed
            # by the bounded initialization flag, so any remaining done here
            # is a physical/safety failure and applies to the whole batch.
            failed = terminated | truncated
            if bool(failed.any()):
                failed_ids = failed.nonzero(as_tuple=True)[0]
                collision_evaluator = getattr(
                    env, "_g2_forbidden_collision_evaluator", None
                )
                collision_sensor_peaks = None
                collision_body_peaks = None
                if collision_evaluator is not None:
                    collision_sensor_peaks = {
                        name: [values[index] for index in failed_ids.detach().cpu().tolist()]
                        for name, values in collision_evaluator.sensor_peak_forces_n().items()
                    }
                    collision_body_peaks = {
                        sensor_name: {
                            body_name: [
                                values[index]
                                for index in failed_ids.detach().cpu().tolist()
                            ]
                            for body_name, values in bodies.items()
                        }
                        for sensor_name, bodies in (
                            collision_evaluator.sensor_body_peak_forces_n().items()
                        )
                    }
                terms = {
                    name: env.termination_manager.get_term(name)[failed_ids]
                    .detach()
                    .cpu()
                    .tolist()
                    for name in env.termination_manager.active_terms
                    if bool(env.termination_manager.get_term(name)[failed_ids].any())
                }
                raise RuntimeError(
                    "G2_REVERSE_PREROLL_TERMINATED:"
                    + json.dumps(
                        {
                            "step": step,
                            "environment_ids": failed_ids.detach().cpu().tolist(),
                            "termination_terms": terms,
                            "collision_sensor_peak_force_n": collision_sensor_peaks,
                            "collision_body_peak_force_n": collision_body_peaks,
                            "desired_ee_position_root_m": desired[failed_ids]
                            .detach().cpu().tolist(),
                            "measured_ee_position_root_m": ee[failed_ids]
                            .detach().cpu().tolist(),
                            "arm_joint_position_rad": tensor_value(
                                robot.data.joint_pos
                            ).index_select(1, indices[:7])[failed_ids]
                            .detach().cpu().tolist(),
                        },
                        separators=(",", ":"),
                    )
                )
            q_after = tensor_value(robot.data.joint_pos).index_select(1, indices[:7])
            velocity = torch.atan2(
                torch.sin(q_after - q_before), torch.cos(q_after - q_before)
            ) / env.step_dt
            acceleration = (velocity - previous_velocity) / env.step_dt
            previous_velocity = velocity.detach().clone()
            metrics["steps"] = float(step + 1)
            metrics["maximum_arm_fd_velocity_rad_s"] = max(
                metrics["maximum_arm_fd_velocity_rad_s"], float(velocity[near].abs().max())
            )
            metrics["maximum_arm_fd_acceleration_rad_s2"] = max(
                metrics["maximum_arm_fd_acceleration_rad_s2"],
                float(acceleration[near].abs().max()),
            )
            displacement = torch.linalg.vector_norm(
                tensor_value(cube.data.root_pos_w) - initial_cube_position, dim=-1
            )
            metrics["maximum_cube_displacement_m"] = max(
                metrics["maximum_cube_displacement_m"], float(displacement[near].max())
            )
            if metrics["maximum_arm_fd_velocity_rad_s"] > 0.8:
                raise RuntimeError(f"G2_REVERSE_PREROLL_OVERSPEED:{metrics}")
            if metrics["maximum_arm_fd_acceleration_rad_s2"] > 10.0:
                raise RuntimeError(f"G2_REVERSE_PREROLL_ACCELERATION:{metrics}")
            if metrics["maximum_cube_displacement_m"] > 0.003:
                raise RuntimeError(f"G2_REVERSE_PREROLL_PUSH:{metrics}")
            if (step + 1) % 10 == 0:
                write_progress(
                    "runtime/reverse_preroll",
                    transitions=0,
                    episodes=0,
                    updates=0,
                    reverse_preroll=metrics,
                    aligned_pregrasp_environments=int(
                        (aligned_at_pregrasp & near).sum()
                    ),
                )
        else:
            # Preserve enough per-environment evidence to distinguish a
            # genuinely unreachable randomized endpoint from controller
            # oscillation or a joint-limit plateau.  This is materialized only
            # on failure, never in the per-step loop.
            timeout_state = privileged()
            timeout_slices = teacher_contract.slices
            timeout_ee = timeout_state[
                :, timeout_slices["end_effector_pose_root_xyzw"]
            ][:, :3]
            timeout_cube = timeout_state[:, timeout_slices["cube_pose_root_xyzw"]][
                :, :3
            ]
            timeout_progress = env._g2_reverse_last_progress.clamp(0.0, 1.0)
            timeout_distance = 0.100 + timeout_progress * (
                sandbox.reverse_initial_ee_cube_distance_m - 0.100
            )
            timeout_desired = timeout_cube.clone()
            timeout_desired[:, 0] -= timeout_distance
            timeout_desired[:, 2] += G2_KEYBOARD_SIDE_GRASP_CENTER_HEIGHT_OFFSET_M
            timeout_error_vector = timeout_desired - timeout_ee
            timeout_error = torch.linalg.vector_norm(timeout_error_vector, dim=-1)
            timeout_q = tensor_value(robot.data.joint_pos).index_select(
                1, indices[:7]
            )
            timeout_limits = tensor_value(robot.data.soft_joint_pos_limits).index_select(
                1, indices[:7]
            )
            timeout_limit_margin = torch.minimum(
                timeout_q - timeout_limits[..., 0],
                timeout_limits[..., 1] - timeout_q,
            )
            arm_term = env.action_manager._terms["arm_action"]
            timeout_target = arm_term.last_emitted_joint_position_target
            failing = near & (
                timeout_error > args.reverse_preroll_position_tolerance_m
            )
            raise RuntimeError(
                "G2_REVERSE_PREROLL_TIMEOUT:"
                + json.dumps(
                    {
                        "metrics": metrics,
                        "near_environment_ids": near.nonzero(as_tuple=True)[0]
                        .detach()
                        .cpu()
                        .tolist(),
                        "failing_environment_ids": failing.nonzero(as_tuple=True)[0]
                        .detach()
                        .cpu()
                        .tolist(),
                        "path_progress": timeout_progress[near]
                        .detach()
                        .cpu()
                        .tolist(),
                        "best_error_m": best_near_error[near]
                        .detach()
                        .cpu()
                        .tolist(),
                        "best_error_step": best_near_error_step[near]
                        .detach()
                        .cpu()
                        .tolist(),
                        "final_error_m": timeout_error[near]
                        .detach()
                        .cpu()
                        .tolist(),
                        "final_error_vector_m": timeout_error_vector[near]
                        .detach()
                        .cpu()
                        .tolist(),
                        "desired_ee_position_root_m": timeout_desired[near]
                        .detach()
                        .cpu()
                        .tolist(),
                        "measured_ee_position_root_m": timeout_ee[near]
                        .detach()
                        .cpu()
                        .tolist(),
                        "arm_joint_position_rad": timeout_q[near]
                        .detach()
                        .cpu()
                        .tolist(),
                        "arm_joint_limit_margin_rad": timeout_limit_margin[near]
                        .detach()
                        .cpu()
                        .tolist(),
                        "emitted_arm_target_rad": timeout_target[near]
                        .detach()
                        .cpu()
                        .tolist(),
                        "converged_streak": near_converged_streak[near]
                        .detach()
                        .cpu()
                        .tolist(),
                        "target_synchronized_at_near": (
                            target_synchronized_at_near[near]
                            .detach()
                            .cpu()
                            .tolist()
                        ),
                    },
                    separators=(",", ":"),
                )
            )

        final_state = privileged()
        slices = teacher_contract.slices
        final_distance = torch.linalg.vector_norm(
            final_state[:, slices["end_effector_to_cube_m"]], dim=-1
        )
        expected_axial_distance = 0.100 + env._g2_reverse_last_progress.clamp(0.0, 1.0) * (
            sandbox.reverse_initial_ee_cube_distance_m - 0.100
        )
        expected_distance = torch.sqrt(
            expected_axial_distance.square()
            + G2_KEYBOARD_SIDE_GRASP_CENTER_HEIGHT_OFFSET_M**2
        )
        distance_error = torch.abs(
            final_distance[near] - expected_distance[near]
        )
        if float(distance_error.max()) > sandbox.reverse_initial_distance_tolerance_m:
            raise RuntimeError(
                "G2_REVERSE_PREROLL_GEOMETRY_MISMATCH:"
                + json.dumps(
                    {
                        "distance_m": final_distance[near].detach().cpu().tolist(),
                        "metrics": metrics,
                    },
                    separators=(",", ":"),
                )
            )
        env.episode_length_buf[near_ids] = 0
        g2_lift_task_mdp.reset_task_progress_state(env, near_ids)
        return last_action.detach().clone(), metrics

    env._g2_reset_initialization_active = True
    arm_action_term = env.action_manager._terms["arm_action"]
    training_arm_target_acceleration = float(
        arm_action_term.cfg.maximum_joint_target_acceleration_rad_s2
    )
    try:
        env.reset(seed=args.seed)
        settle = torch.zeros(args.num_envs, 7, device=env.device)
        settle[:, 6] = 1.0
        settling_velocity = torch.zeros(
            (args.num_envs, len(RIGHT_ARM_JOINTS)), device=env.device
        )
        for _ in range(args.settling_steps):
            env.episode_length_buf[:] = 0
            settling_q_before = tensor_value(robot.data.joint_pos).index_select(
                1, indices[:7]
            ).clone()
            env.step(settle)
            settling_q_after = tensor_value(robot.data.joint_pos).index_select(
                1, indices[:7]
            )
            settling_velocity = torch.atan2(
                torch.sin(settling_q_after - settling_q_before),
                torch.cos(settling_q_after - settling_q_before),
            ) / env.step_dt
        arm_action_term.cfg.maximum_joint_target_acceleration_rad_s2 = (
            args.reverse_preroll_arm_target_acceleration_rad_s2
        )
        previous_action, reverse_preroll_metrics = reverse_preroll(
            settling_velocity
        )
    finally:
        arm_action_term.cfg.maximum_joint_target_acceleration_rad_s2 = (
            training_arm_target_acceleration
        )
        env._g2_reset_initialization_active = False
    state = privileged(); prop = proprio(state, camera_age()); image = visual()
    initial_camera_visibility = camera_visibility_evaluator.evaluate(
        require_rendered_depth=True
    )
    if not bool(initial_camera_visibility.reset_pass.all()):
        raise RuntimeError(
            "G2_VISUAL_DUAL_CAMERA_RESET_VISIBILITY_FAILED:"
            + json.dumps(
                initial_camera_visibility.serializable(), separators=(",", ":")
            )
        )
    state_slices = teacher_contract.slices
    ee_position = state[:, state_slices["end_effector_pose_root_xyzw"]][:, :3]
    cube_position = state[:, state_slices["cube_pose_root_xyzw"]][:, :3]
    initial_distance = torch.linalg.vector_norm(cube_position - ee_position, dim=-1)
    near_mask = env._g2_reverse_last_near
    print(
        "G2_VISUAL_INITIAL_GEOMETRY "
        + json.dumps(
            {
                "near_mask": near_mask.detach().cpu().tolist(),
                "distance_m": initial_distance.detach().cpu().tolist(),
                "path_progress": env._g2_reverse_last_progress.detach().cpu().tolist(),
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    if bool(near_mask.any()):
        near_progress = env._g2_reverse_last_progress[near_mask].clamp(0.0, 1.0)
        near_axial_distance = 0.100 + near_progress * (
            sandbox.reverse_initial_ee_cube_distance_m - 0.100
        )
        expected_near_distance = torch.sqrt(
            near_axial_distance.square()
            + G2_KEYBOARD_SIDE_GRASP_CENTER_HEIGHT_OFFSET_M**2
        )
        near_distance_error = torch.abs(
            initial_distance[near_mask] - expected_near_distance
        )
        if float(near_distance_error.max()) > sandbox.reverse_initial_distance_tolerance_m:
            raise RuntimeError(
                "G2_REVERSE_CURRICULUM_LIVE_GEOMETRY_MISMATCH:"
                + json.dumps(
                    {
                        "distance_m": initial_distance[near_mask].detach().cpu().tolist(),
                        "expected_distance_m": expected_near_distance.detach().cpu().tolist(),
                        "ee_root_m": ee_position[near_mask].detach().cpu().tolist(),
                        "cube_root_m": cube_position[near_mask].detach().cpu().tolist(),
                        "path_progress": near_progress.detach().cpu().tolist(),
                        "arm_q_rad": tensor_value(robot.data.joint_pos)[near_mask]
                        .index_select(1, indices[:7]).detach().cpu().tolist(),
                    },
                    separators=(",", ":"),
                )
            )
    ordinary_mask = ~near_mask
    if bool(ordinary_mask.any()):
        ordinary_error = torch.abs(
            initial_distance[ordinary_mask] - G2_KEYBOARD_PHOTO_EE_CUBE_DISTANCE_M
        )
        if float(ordinary_error.max()) > sandbox.ordinary_initial_distance_tolerance_m:
            raise RuntimeError(
                "G2_EXHIBITION_INITIAL_DISTANCE_MISMATCH:"
                f"{float(initial_distance[ordinary_mask].min())}:"
                f"{float(initial_distance[ordinary_mask].max())}"
            )
    print("G2_VISUAL_FIRST_OBSERVATION", flush=True)
    write_progress("runtime/first_observation", transitions=0, episodes=0, updates=0)

    recurrent_profile = load_g2_recurrent_profile(args.recurrent_profile)
    camera_encoder_channels = G2_CAMERA_ENCODER_PROFILES[
        args.camera_encoder_profile
    ]
    recurrent_contract = G2RecurrentVisualPolicyContract(
        hidden_dim=recurrent_profile.hidden_dim,
        gru_num_layers=recurrent_profile.gru_num_layers,
        sequence_length=recurrent_profile.sequence_length,
        burn_in_steps=recurrent_profile.burn_in_steps,
        sequence_stride=recurrent_profile.sequence_stride,
        camera_encoder_channels=camera_encoder_channels,
    ).validated()
    # One recurrent optimizer batch consumes only the post-burn-in timesteps.
    # Budget replay intensity in transitions, not in sequence objects.
    update_schedule = G2OffPolicyUpdateSchedule(
        learning_starts=args.learning_starts,
        batch_size=(
            args.batch_size
            * (recurrent_contract.sequence_length - recurrent_contract.burn_in_steps)
        ),
        replay_samples_per_transition=args.replay_samples_per_transition,
        maximum_updates_per_vector_step=args.maximum_updates_per_vector_step,
    )
    agent = G2RecurrentVisualAsymmetricSAC(
        G2RecurrentVisualSACConfig(
            relative_pose_weight=args.relative_pose_weight,
            pose_consistency_weight=args.pose_consistency_weight,
            pose_consistency_rotation_weight=(
                args.pose_consistency_rotation_weight
            ),
            contact_weight=args.contact_weight,
            depth_validity_weight=args.depth_validity_weight,
            cross_camera_contrastive_weight=(
                args.cross_camera_contrastive_weight
            ),
            cross_camera_contrastive_temperature=(
                args.cross_camera_contrastive_temperature
            ),
            cross_camera_false_negative_position_threshold_m=(
                args.cross_camera_false_negative_position_threshold_m
            ),
            temporal_pose_residual_weight=args.temporal_pose_residual_weight,
            temporal_pose_residual_rotation_weight=(
                args.temporal_pose_residual_rotation_weight
            ),
            share_camera_encoder_weights=args.share_camera_encoder_weights,
            camera_encoder_channels=camera_encoder_channels,
            hidden_dim=recurrent_contract.hidden_dim,
            gru_num_layers=recurrent_contract.gru_num_layers,
            sequence_length=recurrent_contract.sequence_length,
            burn_in_steps=recurrent_contract.burn_in_steps,
            sequence_stride=recurrent_contract.sequence_stride,
            gradient_clip=recurrent_profile.gradient_clip,
            random_shift_pad=args.random_shift_pad,
            demonstration_q_filter_temperature=(
                args.demonstration_q_filter_temperature
            ),
            student_head_distillation_weight=(
                args.student_head_distillation_weight
            ),
            near_contact_auxiliary_distance_m=(
                args.near_contact_auxiliary_distance_m
            ),
            near_contact_relative_pose_multiplier=(
                args.near_contact_relative_pose_multiplier
            ),
        ),
        device=env.device,
    )

    demonstration_sequences = None
    demonstration_hashes: dict[str, str] = {}
    demonstration_window_count = 0
    demonstration_original_success_window_count = 0
    demonstration_original_failure_window_count = 0
    demonstration_augmented_index_pool = None
    reference_grasp_offset_rows = 0
    reference_grasp_offset = (0.0, 0.0, -0.020)
    demonstration_metrics: dict[str, float] = {}
    demonstration_phase_codes = None
    demonstration_window_success = None
    demonstration_force_priorities = None
    demonstration_force_metrics: dict[str, float] = {}
    if args.demonstration_dataset:
        loaded = []
        stable_relative_offsets = []
        for dataset in args.demonstration_dataset:
            dataset = dataset.resolve()
            if not dataset.is_file():
                raise FileNotFoundError(f"demonstration dataset does not exist: {dataset}")
            digest = hashlib.sha256()
            with dataset.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            demonstration_hashes[str(dataset)] = digest.hexdigest()
            loaded.append(
                load_canonical_sequences(
                    dataset,
                    sequence_contract=G2StudentSequenceContract(
                        sequence_length=recurrent_contract.sequence_length,
                        burn_in_steps=recurrent_contract.burn_in_steps,
                        stride=recurrent_contract.sequence_stride,
                    ),
                    # Load both outcomes. Augmentation below uses the source
                    # episode outcome rather than a sparse terminal label.
                    successful_episodes_only=False,
                    expert_arm_action_scale=(
                        args.demonstration_expert_arm_action_scale
                    ),
                )
            )
            teacher_slices = G2TeacherObservationContract().slices
            for episode_key in list_canonical_episodes(dataset):
                episode = load_canonical_episode(dataset, episode_key)
                if not bool(episode["success"].to(torch.bool).any()):
                    continue
                safe_stable = (
                    episode["stable_grasp"].to(torch.bool).reshape(-1)
                    & episode["forbidden_collision_valid"].to(torch.bool).reshape(-1)
                    & ~episode["forbidden_collision"].to(torch.bool).reshape(-1)
                    & episode["teacher_action_compatible"].to(torch.bool).reshape(-1)
                    & (episode["expert_confidence"].reshape(-1) > 0.0)
                )
                if bool(safe_stable.any()):
                    stable_relative_offsets.append(
                        episode["teacher_observation"][
                            safe_stable,
                            teacher_slices["end_effector_to_cube_m"],
                        ].to(torch.float32)
                    )
        tensor_names = tuple(
            name for name, item in loaded[0].items() if isinstance(item, torch.Tensor)
        )
        if any(
            tuple(name for name, item in item_set.items() if isinstance(item, torch.Tensor))
            != tensor_names
            for item_set in loaded
        ):
            raise RuntimeError("demonstration sequence fields differ between datasets")
        demonstration_sequences = {
            name: torch.cat([item[name] for item in loaded], dim=0)
            for name in tensor_names
        }
        demonstration_sequences["burn_in_steps"] = recurrent_contract.burn_in_steps
        demonstration_window_count = int(
            demonstration_sequences["padding_mask"].shape[0]
        )
        if demonstration_window_count <= 0:
            raise RuntimeError("demonstration dataset has no usable recurrent windows")
        supervised_rows = float(
            demonstration_sequences["expert_confidence"][:, recurrent_contract.burn_in_steps :]
            .clamp(0.0, 1.0)
            .sum()
        )
        if supervised_rows <= 0.0:
            raise RuntimeError("demonstration dataset has no safe compatible BC rows")
        demonstration_phase_codes = classify_demonstration_sequence_phases(
            demonstration_sequences
        )
        demonstration_window_success = demonstration_sequences[
            "episode_outcome_success"
        ].to(torch.bool).reshape(-1)
        demonstration_original_success_window_count = int(
            demonstration_window_success.sum()
        )
        demonstration_original_failure_window_count = (
            demonstration_window_count - demonstration_original_success_window_count
        )
        (
            demonstration_force_priorities,
            demonstration_force_metrics,
        ) = demonstration_force_energy_priorities(
            demonstration_sequences,
            temperature_j=args.her_force_temperature_j,
        )
        if not stable_relative_offsets:
            raise RuntimeError(
                "demonstration dataset has no safe stable-grasp row in a SUCCESS episode"
            )
        calibrated = torch.cat(stable_relative_offsets, dim=0).median(dim=0).values
        reference_grasp_offset_rows = int(
            sum(value.shape[0] for value in stable_relative_offsets)
        )
        reference_grasp_offset = tuple(float(value) for value in calibrated)
        requested_augmented_windows = (
            args.demonstration_success_augmented_windows
            + args.demonstration_failure_augmented_windows
        )
        if requested_augmented_windows:
            augmentation_rng = np.random.default_rng(args.seed + 991_771)
            success_indices = torch.nonzero(
                demonstration_window_success, as_tuple=False
            ).reshape(-1).cpu().numpy()
            failure_indices = torch.nonzero(
                ~demonstration_window_success, as_tuple=False
            ).reshape(-1).cpu().numpy()
            pools = []
            if args.demonstration_success_augmented_windows:
                if not len(success_indices):
                    raise RuntimeError("no successful demonstration windows are available")
                pools.append(
                    augmentation_rng.choice(
                        success_indices,
                        size=args.demonstration_success_augmented_windows,
                        replace=(
                            args.demonstration_success_augmented_windows
                            > len(success_indices)
                        ),
                    )
                )
            if args.demonstration_failure_augmented_windows:
                if not len(failure_indices):
                    raise RuntimeError("no failed demonstration windows are available")
                pools.append(
                    augmentation_rng.choice(
                        failure_indices,
                        size=args.demonstration_failure_augmented_windows,
                        replace=(
                            args.demonstration_failure_augmented_windows
                            > len(failure_indices)
                        ),
                    )
                )
            demonstration_augmented_index_pool = np.concatenate(pools).astype(
                np.int64, copy=False
            )
            augmentation_rng.shuffle(demonstration_augmented_index_pool)
            minimum_updates = int(
                np.ceil(
                    requested_augmented_windows
                    / args.demonstration_batch_size
                )
            )
            if args.demonstration_bc_updates < minimum_updates:
                raise ValueError(
                    "demonstration BC updates cannot consume all requested augmented "
                    f"augmentations: need at least {minimum_updates} updates"
                )
        print(
            "G2_DEMONSTRATION_DATA_READY "
            + json.dumps(
                {
                    "datasets": demonstration_hashes,
                    "window_count": demonstration_window_count,
                    "safe_compatible_supervised_rows": supervised_rows,
                    "successful_source_window_count": (
                        demonstration_original_success_window_count
                    ),
                    "failed_source_window_count": (
                        demonstration_original_failure_window_count
                    ),
                    "successful_augmented_window_count": (
                        args.demonstration_success_augmented_windows
                    ),
                    "failed_augmented_window_count": (
                        args.demonstration_failure_augmented_windows
                    ),
                    "force_priority": demonstration_force_metrics,
                    "augmentation": "outcome_stratified_resampling_plus_rgbd_random_shift",
                    "reference_grasp_cube_minus_ee_root_m": reference_grasp_offset,
                    "reference_grasp_calibration_rows": reference_grasp_offset_rows,
                    "initial_pose_profile": G2_KEYBOARD_PHOTO_POSE_PROFILE,
                    "right_arm_q_rad": list(G2_KEYBOARD_PHOTO_RIGHT_ARM_Q),
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
    replay = G2RecurrentVisualReplayBuffer(
        args.replay_capacity,
        hidden_dim=recurrent_contract.hidden_dim,
        seed=args.seed,
        storage_dir=args.output / "replay",
    )
    logger = Stage2RunLogger(
        args.output / "logs", tensorboard_enabled=False, wandb_enabled=args.wandb,
        wandb_project=args.wandb_project if args.wandb else None,
        wandb_entity=args.wandb_entity, wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode, wandb_group="g2-visual-reverse",
        wandb_job_type="asymmetric-visual-sac",
        wandb_tags=("rgbd", "split-encoder", "gru", "task-auxiliary", "reverse-curriculum", "teacher-critic"),
        wandb_config={
            "num_envs": args.num_envs, "total_transitions": args.total_transitions,
            "episode_horizon_steps": args.episode_horizon_steps,
            "episode_horizon_s": cfg.episode_length_s,
            "clone_in_fabric": args.clone_in_fabric,
            "replicate_physics": cfg.scene.replicate_physics,
            "camera_sensor_resolution": f"{args.camera_width}x{args.camera_height}",
            "model_resolution": "64x48",
            "camera_renderer_contract": "ISAAC_RTX_TILED_ONE_RENDER_PRODUCT_PER_CAMERA_GROUP",
            "camera_capture_interval_physics_steps": args.camera_capture_interval_steps,
            "camera_capture_hz": 1.0 / camera_update_period_s,
            "policy_hz": 1.0 / float(env.step_dt),
            "arm_command_acceleration_limit_rad_s2": args.arm_joint_target_acceleration_rad_s2,
            "reference_arm_joint_target_speed_rad_s": (
                args.reference_arm_joint_target_speed_rad_s
            ),
            "reference_maximum_arm_action_magnitude": (
                args.reference_maximum_arm_action_magnitude
            ),
            "arm_measured_acceleration_hard_limit_rad_s2": 10.0,
            "learning_starts_transitions": args.learning_starts,
            "scripted_reverse_bootstrap_enabled": args.scripted_reverse_bootstrap,
            "scripted_reverse_bootstrap_lift_action": args.bootstrap_lift_action,
            "scripted_reverse_bootstrap_privileged_usage": (
                "OFF_POLICY_REFERENCE_COLLECTION_ONLY_NOT_ACTOR_OBSERVATION"
            ),
            "reference_gripper_close_state": (
                "OPEN_UNTIL_CALIBRATED_OFFSET_THEN_LATCHED_CLOSED_UNTIL_RESET"
            ),
            "reference_behavior_initial_probability": args.reference_behavior_initial_probability,
            "reference_behavior_minimum_probability": args.reference_behavior_minimum_probability,
            "reference_behavior_probability_decrement": args.reference_behavior_probability_decrement,
            "reference_behavior_promotion_source": "DETERMINISTIC_HELD_OUT_CURRICULUM_PROMOTION",
            "reference_behavior_sampling_boundary": "EPISODE_RESET_ONLY",
            "reference_transport_target": "EPISODE_OBJECT_GOAL_XYZ_AFTER_STABLE_GRASP",
            "action_transform": action_transform.as_dict(),
            "critic_replay_arm_action_authority": "NORMALIZED_REQUEST_PRE_TRANSFORM",
            "critic_replay_gripper_action_authority": "ACTUALLY_APPLIED_BINARY_COMMAND",
            "contact_approach_distance_m": args.contact_approach_distance_m,
            "contact_approach_arm_speed_rad_s": args.contact_approach_arm_speed_rad_s,
            "contact_approach_gripper_speed_rad_s": args.contact_approach_gripper_speed_rad_s,
            "demonstration_force_sampling_mixture": (
                args.demonstration_force_sampling_mixture
            ),
            "demonstration_force_sampling_authority": (
                "PHASE_LOCAL_UNIFORM_SUCCESS_MIXED_WITH_RECORDED_CONTACT_WORK"
            ),
            "batch_size": args.batch_size,
            "replay_samples_per_transition_target": args.replay_samples_per_transition,
            "maximum_updates_per_vector_step": args.maximum_updates_per_vector_step,
            "checkpoint_interval_transitions": args.checkpoint_interval,
            "checkpoint_retention": args.checkpoint_retention,
            "action_distribution": "6d_squashed_gaussian_plus_exact_binary_gripper",
            "actor_privileged_cube_pose": False, "critic_privileged_state_dim": 59,
            "actor_proprio_dim": 45, "actor_privileged_contact": False,
            "actor_inputs_torso_state": True,
            "actor_inputs_camera_frame_age": True,
            "rgb_depth_encoders_separate": True,
            "share_camera_encoder_weights": args.share_camera_encoder_weights,
            "full_image_reconstruction_decoder": False,
            "recurrent_policy": True,
            "recurrent_profile": recurrent_profile.serializable(),
            "recurrent_policy_contract": recurrent_contract.serializable(),
            "rollout_hidden_state_stored_in_replay": True,
            "recurrent_replay": "EPISODE_CONTIGUOUS_SEQUENCE_WITH_BURN_IN",
            "recurrent_sequence_length": recurrent_contract.sequence_length,
            "recurrent_burn_in_steps": recurrent_contract.burn_in_steps,
            "learner_random_shift_pad_pixels": args.random_shift_pad,
            "learner_random_shift_temporally_consistent": True,
            "keyboard_model_backbone_schema": G2_RECURRENT_VISUAL_POLICY_SCHEMA,
            "initial_pose_authority": "KEYBOARD_DATASET_METADATA",
            "initial_pose_profile": G2_KEYBOARD_PHOTO_POSE_PROFILE,
            "initial_right_arm_q_rad": list(G2_KEYBOARD_PHOTO_RIGHT_ARM_Q),
            "demonstration_datasets_sha256": demonstration_hashes,
            "demonstration_sequence_window_count": demonstration_window_count,
            "demonstration_success_source_window_count": (
                demonstration_original_success_window_count
            ),
            "demonstration_success_augmented_window_count": (
                args.demonstration_success_augmented_windows
            ),
            "demonstration_failure_source_window_count": (
                demonstration_original_failure_window_count
            ),
            "demonstration_failure_augmented_window_count": (
                args.demonstration_failure_augmented_windows
            ),
            "demonstration_success_augmentation_method": (
                "outcome_stratified_resampling_plus_rgbd_random_shift"
            ),
            "reference_grasp_cube_minus_ee_root_m": list(reference_grasp_offset),
            "reference_grasp_calibration_rows": reference_grasp_offset_rows,
            "reference_gripper_close_position_tolerance_m": (
                args.reference_gripper_close_position_tolerance_m
            ),
            "reference_gripper_approach_state": (
                "OPEN_UNTIL_REACHABLE_CALIBRATED_OFFSET_ENVELOPE_THEN_LATCH"
            ),
            "demonstration_bc_updates": args.demonstration_bc_updates,
            "demonstration_bc_batch_size": args.demonstration_batch_size,
            "demonstration_expert_arm_action_scale": (
                args.demonstration_expert_arm_action_scale
            ),
            "demonstration_expert_action_preprocessing_schema": (
                "g2_demonstration_bc_arm_speed_retime_v1"
            ),
            "demonstration_bc_updates_actor_only": True,
            "demonstration_bc_critics_alpha_replay_fresh": True,
            "demonstration_online_bc_actor_only": bool(args.demonstration_dataset),
            "demonstration_online_bc_initial_coefficient": (
                args.demonstration_online_bc_initial_coefficient
            ),
            "demonstration_online_bc_minimum_coefficient": (
                args.demonstration_online_bc_minimum_coefficient
            ),
            "demonstration_online_bc_decay_transitions": (
                args.demonstration_online_bc_decay_transitions
            ),
            "demonstration_online_bc_q_filter": (
                "HELD_OUT_POLICY_MASTERY_GATED_SOFT_CRITIC_ADVANTAGE_SIGMOID"
            ),
            "demonstration_online_bc_decay_authority": (
                "TIME_PROGRESS_TIMES_HELD_OUT_CONTACT_STABLE_LIFT_MASTERY"
            ),
            "demonstration_q_filter_temperature": (
                args.demonstration_q_filter_temperature
            ),
            "demonstration_phase_probabilities": dict(
                zip(
                    ("REACH", "PRE_GRASP", "CONTACT", "STABLE_GRASP", "LIFT"),
                    demonstration_phase_probabilities,
                )
            ),
            "online_replay_phase_probabilities": dict(
                zip(G2_ONLINE_REPLAY_PHASES, online_replay_phase_probabilities)
            ),
            "online_replay_sampling": (
                "NORMAL_PHASE_STRATIFIED_HER_WITH_NEAR_CONTACT_BOUNDED_FORCE_MIXTURE"
            ),
            "her_sequence_fraction": args.her_sequence_fraction,
            "her_goal_authority": "ACTUAL_FUTURE_ACHIEVED_CUBE_POSITION_ONLY",
            "her_force_temperature_j": args.her_force_temperature_j,
            "her_force_sampling_mixture": args.her_force_sampling_mixture,
            "her_force_activation_distance_m": args.contact_approach_distance_m,
            "her_force_activation_authority": (
                "EE_CUBE_DISTANCE_AND_REAL_CONTACT_STABLE_OR_LIFT_EVIDENCE"
            ),
            "her_force_noncontact_priority": 1.0,
            "demonstration_force_priority": (
                "RECORDED_TWO_FINGER_FORCE_X_CONSECUTIVE_CUBE_DISPLACEMENT"
            ),
            "demonstration_force_actor_usage": (
                "PHASE_STRATIFIED_BC_SAMPLING_ONLY_NOT_ACTOR_OBSERVATION"
            ),
            "policy_only_near_reset_probability": 0.25,
            "policy_only_near_progress_range": [0.40, 0.55],
            "gripper_minimum_hold_steps": args.gripper_minimum_hold_steps,
            "near_contact_auxiliary_distance_m": args.near_contact_auxiliary_distance_m,
            "near_contact_relative_pose_multiplier": args.near_contact_relative_pose_multiplier,
            "cross_camera_contrastive_weight": args.cross_camera_contrastive_weight,
            "cross_camera_contrastive_temperature": args.cross_camera_contrastive_temperature,
            "cross_camera_false_negative_position_threshold_m": (
                args.cross_camera_false_negative_position_threshold_m
            ),
            "temporal_pose_residual_weight": args.temporal_pose_residual_weight,
            "temporal_pose_residual_rotation_weight": (
                args.temporal_pose_residual_rotation_weight
            ),
            "demonstration_success_priority_weight": (
                args.demonstration_success_priority_weight
            ),
            "teacher_student_action_head_distillation_weight": (
                args.student_head_distillation_weight
            ),
            "teacher_student_checkpoint_export": "agent.deployment_student",
            "demonstration_reverse_reset_seed": (
                demonstration_reset_seed.serializable()
                if demonstration_reset_seed is not None
                else None
            ),
            "demonstration_grasp_ready_reset_seed": (
                demonstration_grasp_ready_seed.serializable()
                if demonstration_grasp_ready_seed is not None
                else None
            ),
            "reverse_curriculum_initial_beta": .60, "reverse_curriculum_minimum_beta": .20,
            "reverse_curriculum_promotion_source": "DETERMINISTIC_HELD_OUT_ENVS",
            "curriculum_evaluation_envs": args.curriculum_evaluation_envs,
            "replay_environment_count": (
                args.num_envs - args.curriculum_evaluation_envs
            ),
            "policy_only_environment_count": args.curriculum_evaluation_envs,
            "reverse_reset_authority": "privileged_bounded_cartesian_preroll_excluded_from_replay",
            "reverse_preroll_arm_target_acceleration_rad_s2": (
                args.reverse_preroll_arm_target_acceleration_rad_s2
            ),
            "reverse_preroll_maximum_normalized_translation": args.reverse_preroll_maximum_normalized_translation,
            "table_center_world_m": list(G2_KEYBOARD_TABLE_CENTER_WORLD_M),
            "table_surface_height_m": sandbox.table_surface_height_m,
            "cube_initial_center_world_m": list(G2_KEYBOARD_CUBE_CENTER_WORLD_M),
            "cube_size_m": list(sandbox.cube_size_m),
            "cube_mass_kg": sandbox.cube_mass_kg,
            "cube_static_friction_range": sandbox.cube_static_friction_range,
            "cube_dynamic_friction_range": sandbox.cube_dynamic_friction_range,
            "first_touch_recovery_grace_s": sandbox.first_touch_recovery_grace_s,
            "distance_frame": "robot_root",
            "distance_end_effector": "gripper_r_center_link",
            "distance_cube_source": "rigid_body_center_of_mass",
            "fixed_torso_q_rad": list(sandbox.fixed_torso_q_rad),
            "torso_action_enabled": sandbox.torso_action_enabled,
            "camera_reset_visibility": initial_camera_visibility.serializable(),
            "camera_visibility_gt_usage": "SAFETY_GATE_ONLY_NOT_ACTOR_OBSERVATION",
        },
    )

    if args.demonstration_bc_updates:
        assert demonstration_sequences is not None
        bc_rng = np.random.default_rng(args.seed + 811_009)
        write_progress(
            "runtime/demonstration_bc_started",
            transitions=0,
            episodes=0,
            updates=0,
            demonstration_bc_updates=0,
        )
        for bc_update in range(1, args.demonstration_bc_updates + 1):
            if demonstration_augmented_index_pool is None:
                selected_numpy = bc_rng.integers(
                    0,
                    demonstration_window_count,
                    size=args.demonstration_batch_size,
                )
            else:
                start = (
                    (bc_update - 1) * args.demonstration_batch_size
                ) % len(demonstration_augmented_index_pool)
                pool_positions = (
                    np.arange(args.demonstration_batch_size) + start
                ) % len(demonstration_augmented_index_pool)
                selected_numpy = demonstration_augmented_index_pool[pool_positions]
            selected = torch.as_tensor(selected_numpy, dtype=torch.long)
            bc_batch = {
                name: item.index_select(0, selected)
                for name, item in demonstration_sequences.items()
                if isinstance(item, torch.Tensor)
            }
            bc_batch["burn_in_steps"] = recurrent_contract.burn_in_steps
            demonstration_metrics = agent.behavior_clone(bc_batch)
            if bc_update == 1 or bc_update % 100 == 0:
                print(
                    "G2_DEMONSTRATION_BC_PROGRESS "
                    + json.dumps(
                        {"update": bc_update, **demonstration_metrics},
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
        logger.log_scalars(
            "demonstration_bc_complete",
            0,
            {
                "demonstration_bc/updates": args.demonstration_bc_updates,
                "demonstration_bc/window_count": demonstration_window_count,
                "demonstration_bc/success_source_windows": (
                    demonstration_original_success_window_count
                ),
                "demonstration_bc/failure_source_windows": (
                    demonstration_original_failure_window_count
                ),
                "demonstration_bc/success_augmented_windows": (
                    args.demonstration_success_augmented_windows
                ),
                "demonstration_bc/failure_augmented_windows": (
                    args.demonstration_failure_augmented_windows
                ),
                **demonstration_metrics,
            },
        )
        write_progress(
            "runtime/demonstration_bc_complete",
            transitions=0,
            episodes=0,
            updates=0,
            demonstration_bc_updates=args.demonstration_bc_updates,
        )

    transitions = replay_transitions = updates = episodes = 0
    reward_sum = action_sum = requested_action_sum = 0.0
    arm_action_sum = requested_arm_action_sum = 0.0
    contact_transitions = stable_transitions = lift_transitions = success_transitions = 0
    contact_episodes = stable_episodes = lift_episodes = success_episodes = 0
    half_success_episodes = 0
    push_events = push_before_grasp_trial_events = 0
    recovery_attempts = recovery_successes = recovery_failures = 0
    maximum_push = maximum_force = 0.0
    gripper_switches = gripper_closed_transitions = 0
    maximum_fd_velocity = maximum_fd_acceleration = 0.0
    previous_fd_velocity = torch.zeros(
        (args.num_envs, len(RIGHT_ARM_JOINTS)), device=env.device
    )
    previous_fd_velocity_valid = torch.zeros(
        args.num_envs, dtype=torch.bool, device=env.device
    )
    episode_contact = torch.zeros(args.num_envs, dtype=torch.bool, device=env.device)
    episode_stable = torch.zeros_like(episode_contact); episode_lift = torch.zeros_like(episode_contact)
    initial_cube = tensor_value(cube.data.root_pos_w).clone(); episode_age = torch.zeros(args.num_envs, dtype=torch.long, device=env.device)
    push_latched = torch.zeros_like(episode_contact)
    episode_grasp_trial = torch.zeros_like(episode_contact)
    episode_recovery_pending = torch.zeros_like(episode_contact)
    # The reset preroll's final controller command is part of the Markov state.
    # Starting the stateful transform from zero would create an artificial
    # first-rollout discontinuity.
    applied = previous_action.clone()
    recurrent_hidden = agent.initial_hidden(args.num_envs)
    recurrent_reset_mask = torch.ones(
        args.num_envs, dtype=torch.bool, device=env.device
    )
    evaluation_mask = torch.zeros(
        args.num_envs, dtype=torch.bool, device=env.device
    )
    if args.curriculum_evaluation_envs:
        evaluation_mask[-args.curriculum_evaluation_envs :] = True
    training_mask = ~evaluation_mask
    training_ids = training_mask.nonzero(as_tuple=True)[0]
    training_env_count = int(training_ids.numel())
    reference_behavior = G2EpisodeReferenceBehavior(
        training_mask.detach().cpu().numpy(), seed=args.seed + 71_003
    )
    replay_env_id = torch.arange(args.num_envs, dtype=torch.long, device=env.device)
    replay_episode_id = replay_env_id.clone()
    replay_sequence_step = torch.zeros(
        args.num_envs, dtype=torch.long, device=env.device
    )
    next_episode_id = args.num_envs
    evaluation_episodes = evaluation_contact_episodes = 0
    evaluation_stable_episodes = evaluation_lift_episodes = 0
    gripper_latch = torch.ones(args.num_envs, device=env.device)
    last_bilateral_contact = torch.zeros(
        args.num_envs, dtype=torch.bool, device=env.device
    )
    last_stable_grasp = torch.zeros_like(last_bilateral_contact)
    last_any_finger_contact = torch.zeros_like(last_bilateral_contact)
    bootstrap_transitions = 0
    bootstrap_contact_transitions = 0
    bootstrap_stable_transitions = 0
    bootstrap_lift_transitions = 0
    bootstrap_success_transitions = 0
    policy_behavior_transitions = 0
    policy_contact_transitions = 0
    policy_stable_transitions = 0
    policy_lift_transitions = 0
    policy_success_transitions = 0
    reference_episodes = reference_contact_episodes = 0
    reference_stable_episodes = reference_lift_episodes = 0
    reference_success_episodes = 0
    policy_episodes = policy_contact_episodes = 0
    policy_stable_episodes = policy_lift_episodes = 0
    policy_success_episodes = 0
    reference_behavior_probability = float(
        args.reference_behavior_initial_probability
    )
    gripper_hold_age = torch.full(
        (args.num_envs,), args.gripper_minimum_hold_steps,
        dtype=torch.long, device=env.device,
    )
    reference_close_latched = torch.zeros(
        args.num_envs, dtype=torch.bool, device=env.device
    )
    reference_stable_latched = torch.zeros_like(reference_close_latched)
    reference_close_gate_entries = 0
    reference_close_entry_episode_age_sum = 0
    reference_close_entry_episode_age_max = 0
    gripper_master_position_minimum_rad = float("inf")
    gripper_master_position_maximum_rad = -float("inf")
    last_metrics = {}
    next_checkpoint_transition = args.checkpoint_interval

    def distance_conditioned_arm_speed_limit(distance_m: torch.Tensor) -> torch.Tensor:
        """Existing G2 close-range schedule applied to final joint targets."""

        result = torch.full_like(distance_m, 0.60)
        middle = (distance_m > 0.05) & (distance_m < 0.12)
        middle_z = torch.clamp((distance_m - 0.05) / 0.07, 0.0, 1.0)
        middle_s = 3.0 * middle_z.square() - 2.0 * middle_z.pow(3)
        result = torch.where(middle, 0.20 + 0.40 * middle_s, result)
        close = (distance_m > 0.03) & (distance_m <= 0.05)
        close_z = torch.clamp((distance_m - 0.03) / 0.02, 0.0, 1.0)
        close_s = 3.0 * close_z.square() - 2.0 * close_z.pow(3)
        result = torch.where(close, 0.05 + 0.15 * close_s, result)
        result = torch.where(distance_m <= 0.03, 0.05, result)
        return result

    while transitions < args.total_transitions:
        q_before = tensor_value(robot.data.joint_pos).index_select(1, indices[:7]).clone()
        policy_request, next_recurrent_hidden = agent.select_actions(
            image,
            prop,
            recurrent_hidden,
            reset_mask=recurrent_reset_mask,
        )
        if bool(evaluation_mask.any()):
            evaluation_request, evaluation_hidden = agent.select_actions(
                image[evaluation_mask],
                prop[evaluation_mask],
                recurrent_hidden[:, evaluation_mask],
                reset_mask=recurrent_reset_mask[evaluation_mask],
                deterministic=True,
            )
            policy_request[evaluation_mask.detach().cpu().numpy()] = evaluation_request
            next_recurrent_hidden[:, evaluation_mask] = evaluation_hidden
        request = torch.as_tensor(policy_request, device=env.device)
        reference_behavior_mask = torch.as_tensor(
            reference_behavior.current_mask(),
            dtype=torch.bool,
            device=env.device,
        )
        reference_position_error = torch.linalg.vector_norm(
            state[:, state_slices["end_effector_to_cube_m"]]
            - torch.as_tensor(
                reference_grasp_offset,
                dtype=state.dtype,
                device=state.device,
            ),
            dim=-1,
        )
        if args.scripted_reverse_bootstrap:
            if bool(reference_behavior_mask.any()):
                reference_stable_latched |= (
                    reference_behavior_mask & last_stable_grasp
                )
                # Off-policy reference collection only: no behavior-cloning
                # target is created, and privileged cube geometry never enters
                # the deployable actor observation.
                controller_reference = g2_reference_controller_action(
                    state,
                    end_effector_to_cube_slice=state_slices[
                        "end_effector_to_cube_m"
                    ],
                    cube_to_goal_slice=state_slices["cube_to_goal_m"],
                    bilateral_contact=last_bilateral_contact,
                    stable_grasp=reference_stable_latched,
                    maximum_arm_action_magnitude=(
                        args.reference_maximum_arm_action_magnitude
                    ),
                    target_cube_minus_ee_m=reference_grasp_offset,
                    gripper_close_position_tolerance_m=(
                        args.reference_gripper_close_position_tolerance_m
                    ),
                )
                close_gate = reference_behavior_mask & (
                    controller_reference[:, 6] < 0.0
                )
                newly_latched = close_gate & ~reference_close_latched
                reference_close_gate_entries += int(newly_latched.sum())
                if bool(newly_latched.any()):
                    entry_ages = episode_age[newly_latched]
                    reference_close_entry_episode_age_sum += int(entry_ages.sum())
                    reference_close_entry_episode_age_max = max(
                        reference_close_entry_episode_age_max,
                        int(entry_ages.max()),
                    )
                reference_close_latched |= close_gate
                # A deterministic reference grasp is a state machine: after
                # entering the calibrated close region it must not reopen just
                # because contact motion moves the Cartesian error back across
                # the configured entry boundary.  The actuator's audited reset-open interval
                # remains authoritative before this latch can take effect.
                controller_reference[reference_close_latched, 6] = -1.0
                # Human demonstrations stop Cartesian translation while the
                # fingers traverse from open to first contact.  Continuing to
                # chase the pose target during that interval made a slow close
                # miss the cube and a fast close push it by centimetres.  Hold
                # the arm at the calibrated grasp pose until bilateral contact;
                # stable-grasp rows retain the controller's goal/lift command.
                closing_without_contact = (
                    reference_close_latched
                    & ~last_bilateral_contact
                    & ~reference_stable_latched
                )
                controller_reference[
                    closing_without_contact, :G2_VISUAL_ARM_ACTION_DIM
                ] = 0.0
                normalized_reference = controller_reference.clone()
                # The stateful transform below scales normalized requests by
                # the policy action magnitude.  Normalize with that same
                # authority so the reference retains its smaller, explicitly
                # audited Cartesian command magnitude.
                normalized_reference[:, :6] /= float(
                    action_transform.maximum_arm_action_magnitude
                )
                # The reference controller owns the binary open/close gate.
                # Overwriting this channel to -1 here previously made every
                # approach run with a closed gripper, producing outer-pad-only
                # impacts and zero bilateral contacts.
                request[reference_behavior_mask] = normalized_reference[
                    reference_behavior_mask
                ]
        elif replay_transitions < args.learning_starts:
            reference_behavior_mask.zero_()
            request = torch.as_tensor(
                rng.uniform(-1.0, 1.0, (args.num_envs, 7)),
                dtype=torch.float32,
                device=env.device,
            )
            request[:, 6] = torch.where(
                torch.rand(args.num_envs, device=env.device) > .5,
                1.0,
                -1.0,
            )
            request[evaluation_mask] = torch.as_tensor(
                policy_request[evaluation_mask.detach().cpu().numpy()],
                device=env.device,
            )
        # Actor and critic own normalized requests.  The established Teacher
        # action transform is their deterministic Markov controller wrapper;
        # its previous applied output is already part of proprioception.
        applied.copy_(action_transform.apply(request, applied))
        requested_gripper = torch.where(request[:, 6] >= 0, 1.0, -1.0)
        may_switch = requested_gripper != gripper_latch
        if args.gripper_minimum_hold_steps > 0:
            may_switch &= gripper_hold_age >= args.gripper_minimum_hold_steps
        gripper_switches += int(may_switch.sum())
        gripper_latch = torch.where(may_switch, requested_gripper, gripper_latch)
        gripper_hold_age = torch.where(
            may_switch, torch.zeros_like(gripper_hold_age), gripper_hold_age + 1
        )
        applied[:, 6] = gripper_latch
        # A close command begins the grasp trial at this policy boundary.  A
        # cube displacement observed on the same transition is therefore not
        # mislabeled as a pre-trial push.
        episode_grasp_trial |= gripper_latch < 0
        current_ee_cube_distance = torch.linalg.vector_norm(
            state[:, state_slices["end_effector_to_cube_m"]], dim=-1
        )
        arm_speed_limit = distance_conditioned_arm_speed_limit(
            current_ee_cube_distance
        )
        # Reference behavior is an online expert data source, not a policy
        # safety exemption.  A live 25-env run attributed a no-contact joint7
        # reversal of 16.65 rad/s^2 to a reference episode while the far-reach
        # cap was 0.6 rad/s.  Retime only reference rows to 0.4 rad/s; policy
        # rows retain the existing authority and all hard gates are unchanged.
        reference_speed_cap = torch.full_like(
            arm_speed_limit, args.reference_arm_joint_target_speed_rad_s
        )
        arm_speed_limit = torch.where(
            reference_behavior_mask,
            torch.minimum(arm_speed_limit, reference_speed_cap),
            arm_speed_limit,
        )
        # The failed run first touched one finger at 65.7 mm while the
        # center-distance schedule still allowed 0.251 rad/s.  A closing
        # gripper inside that observed contact envelope selects the existing
        # 0.05 rad/s contact cap; hard safety limits remain unchanged.
        contact_approach = (gripper_latch < 0) & (
            (current_ee_cube_distance <= args.contact_approach_distance_m)
            | last_any_finger_contact
        )
        arm_speed_limit = torch.where(
            contact_approach,
            torch.minimum(
                arm_speed_limit,
                torch.full_like(
                    arm_speed_limit, args.contact_approach_arm_speed_rad_s
                ),
            ),
            arm_speed_limit,
        )
        arm_term = env.action_manager._terms["arm_action"]
        arm_term.set_environment_speed_limit_rad_s(arm_speed_limit)
        gripper_term = env.action_manager._terms["gripper_action"]
        gripper_speed_limit = torch.full(
            (args.num_envs,),
            args.gripper_joint_target_speed_rad_s,
            dtype=torch.float32,
            device=env.device,
        )
        gripper_speed_limit = torch.where(
            contact_approach,
            torch.full_like(
                gripper_speed_limit, args.contact_approach_gripper_speed_rad_s
            ),
            gripper_speed_limit,
        )
        post_stable_reference = (
            reference_behavior_mask & reference_stable_latched
        )
        gripper_speed_limit = torch.where(
            post_stable_reference,
            torch.minimum(
                gripper_speed_limit,
                torch.full_like(gripper_speed_limit, 0.10),
            ),
            gripper_speed_limit,
        )
        gripper_term.set_environment_speed_limit_rad_s(gripper_speed_limit)
        # Continue closing through contact.  After stable grasp the speed cap
        # above drops to 0.10 rad/s so normal force is retained during lift
        # without the abrupt measured-position hold that released the grasp in
        # the live diagnostic.
        gripper_term.set_external_hold_mask(
            torch.zeros_like(last_stable_grasp)
        )
        next_policy, reward, terminated, truncated, _ = env.step(applied)
        del next_policy
        success_now = env.termination_manager.get_term("object_reached_goal").clone()
        done = terminated | truncated
        all_q_after = tensor_value(robot.data.joint_pos)
        q_after = all_q_after.index_select(1, indices[:7])
        gripper_master_position = all_q_after[:, indices[7]]
        gripper_master_position_minimum_rad = min(
            gripper_master_position_minimum_rad,
            float(gripper_master_position.min()),
        )
        gripper_master_position_maximum_rad = max(
            gripper_master_position_maximum_rad,
            float(gripper_master_position.max()),
        )
        previous_fd_velocity_for_event = previous_fd_velocity.detach().clone()
        fd_velocity = torch.atan2(torch.sin(q_after - q_before), torch.cos(q_after - q_before)) / env.step_dt
        inner_n, outer_n, bilateral, slip, stable = (
            g2_lift_task_mdp.contact_grasp_telemetry(env)
        )
        force = inner_n + outer_n
        continuous_env = ~done
        if bool(continuous_env.any()):
            maximum_fd_velocity = max(
                maximum_fd_velocity, float(fd_velocity[continuous_env].abs().max())
            )
        acceleration_valid = continuous_env & previous_fd_velocity_valid
        fd_acceleration = (fd_velocity - previous_fd_velocity) / env.step_dt
        if bool(acceleration_valid.any()):
            maximum_fd_acceleration = max(
                maximum_fd_acceleration,
                float(
                    fd_acceleration[acceleration_valid].abs().max()
                ),
            )
        # A reset writes a new episode state.  Its first position difference is
        # a typed reset boundary, not measured continuous motion.
        previous_fd_velocity[continuous_env] = fd_velocity[continuous_env].detach()
        previous_fd_velocity_valid.copy_(continuous_env)
        if maximum_fd_velocity > .8:
            raise RuntimeError(f"G2_VISUAL_ARM_OVERSPEED:{maximum_fd_velocity}")
        if maximum_fd_acceleration > 10.0:
            masked = fd_acceleration.abs().masked_fill(
                ~acceleration_valid.unsqueeze(-1), float("-inf")
            )
            flat_index = int(masked.argmax())
            environment_index = flat_index // len(RIGHT_ARM_JOINTS)
            joint_index = flat_index % len(RIGHT_ARM_JOINTS)
            arm_term = env.action_manager._terms["arm_action"]
            failure = {
                "schema": "g2_visual_arm_acceleration_failure_v1",
                "hard_limit_rad_s2": 10.0,
                "measured_acceleration_rad_s2": float(
                    fd_acceleration[environment_index, joint_index]
                ),
                "absolute_measured_acceleration_rad_s2": float(
                    masked[environment_index, joint_index]
                ),
                "environment_id": environment_index,
                "joint_index_in_right_arm": joint_index,
                "joint_name": RIGHT_ARM_JOINTS[joint_index],
                "transitions_before_step": transitions,
                "episode_age_steps": int(episode_age[environment_index]),
                "previous_fd_velocity_rad_s": float(
                    previous_fd_velocity_for_event[environment_index, joint_index]
                ),
                "current_fd_velocity_rad_s": float(
                    fd_velocity[environment_index, joint_index]
                ),
                "q_before_rad": float(q_before[environment_index, joint_index]),
                "q_after_rad": float(q_after[environment_index, joint_index]),
                "requested_action": request[environment_index].detach().cpu().tolist(),
                "applied_action": applied[environment_index].detach().cpu().tolist(),
                "emitted_joint_target_rad": arm_term.last_emitted_joint_position_target[
                    environment_index
                ].detach().cpu().tolist(),
                "hard_joint_limit_rad": tensor_value(
                    robot.data.joint_pos_limits
                )[environment_index, indices[joint_index]].detach().cpu().tolist(),
                "soft_joint_limit_rad": tensor_value(
                    robot.data.soft_joint_pos_limits
                )[environment_index, indices[joint_index]].detach().cpu().tolist(),
                "joint_target_limit_margin_rad": float(
                    arm_term.cfg.joint_limit_margin_rad
                ),
                "command_acceleration_limit_rad_s2": args.arm_joint_target_acceleration_rad_s2,
                "sampling_dt_s": env.step_dt,
                "done": bool(done[environment_index]),
                "bilateral_contact": bool(bilateral[environment_index]),
                "inner_contact_force_n": float(inner_n[environment_index]),
                "outer_contact_force_n": float(outer_n[environment_index]),
                "gripper_close_command": bool(gripper_latch[environment_index] < 0),
                "ee_cube_distance_m": float(
                    current_ee_cube_distance[environment_index]
                ),
                "distance_conditioned_arm_speed_limit_rad_s": float(
                    arm_speed_limit[environment_index]
                ),
                "behavior_authority": (
                    "REFERENCE"
                    if bool(reference_behavior_mask[environment_index])
                    else "POLICY"
                ),
                "reference_arm_joint_target_speed_rad_s": (
                    args.reference_arm_joint_target_speed_rad_s
                ),
                "reference_maximum_arm_action_magnitude": (
                    args.reference_maximum_arm_action_magnitude
                ),
            }
            (args.output / "safety_failure.json").write_text(
                json.dumps(failure, indent=2) + "\n"
            )
            write_progress(
                "runtime/safety_failure",
                transitions=transitions,
                episodes=episodes,
                updates=updates,
                safety_failure=failure,
            )
            raise RuntimeError(
                f"G2_VISUAL_ARM_ACCELERATION_LIMIT:{maximum_fd_acceleration}"
            )
        next_previous = applied.clone(); next_previous[done] = 0.0
        next_previous[done, 6] = 1.0
        old_previous = previous_action; previous_action = next_previous
        next_state = privileged()
        next_prop = proprio(next_state, camera_age())
        next_image = visual()
        previous_action = old_previous

        last_bilateral_contact.copy_(bilateral & ~done)
        last_stable_grasp.copy_(stable & ~done)
        last_any_finger_contact.copy_(
            ((inner_n > 1.0) | (outer_n > 1.0)) & ~done
        )
        lifted = tensor_value(cube.data.root_pos_w)[:, 2] >= (
            sandbox.table_surface_height_m + sandbox.lift_height_above_table_m
        )
        push = torch.linalg.vector_norm(tensor_value(cube.data.root_pos_w)[:, :2] - initial_cube[:, :2], dim=-1)
        push_grace_steps = int(round(sandbox.first_touch_recovery_grace_s / env.step_dt))
        new_push = (episode_age >= push_grace_steps) & (~episode_contact) & (push > .003) & (~push_latched) & (~done)
        new_push_before_trial = new_push & (~episode_grasp_trial)
        push_latched |= new_push
        episode_recovery_pending |= new_push
        recovery_success_now = (
            episode_recovery_pending & (push <= 0.003) & (~done)
        )
        episode_recovery_pending &= ~recovery_success_now
        # ``env.step`` auto-resets completed rows before the live scene tensors
        # above are sampled.  Preserve the authoritative pre-reset terminal
        # predicate in the phase telemetry: a physical success necessarily
        # implies bilateral contact, stable grasp, and lift at that transition.
        # Without this union W&B could report success=1 together with lift=0.
        bilateral_event = bilateral | success_now
        stable_event = stable | success_now
        lifted_event = lifted | success_now
        episode_contact |= bilateral_event
        episode_stable |= stable_event
        episode_lift |= lifted_event
        contact_transitions += int(bilateral_event[training_ids].sum())
        stable_transitions += int(stable_event[training_ids].sum())
        lift_transitions += int(lifted_event[training_ids].sum())
        success_transitions += int(success_now[training_ids].sum())
        reference_training_mask = reference_behavior_mask & training_mask
        policy_training_mask = (~reference_behavior_mask) & training_mask
        reference_ids = reference_training_mask.nonzero(as_tuple=True)[0]
        policy_ids = policy_training_mask.nonzero(as_tuple=True)[0]
        bootstrap_transitions += int(reference_ids.numel())
        policy_behavior_transitions += int(policy_training_mask.sum())
        if reference_ids.numel():
            bootstrap_contact_transitions += int(bilateral_event[reference_ids].sum())
            bootstrap_stable_transitions += int(stable_event[reference_ids].sum())
            bootstrap_lift_transitions += int(lifted_event[reference_ids].sum())
            bootstrap_success_transitions += int(success_now[reference_ids].sum())
        if policy_ids.numel():
            policy_contact_transitions += int(bilateral_event[policy_ids].sum())
            policy_stable_transitions += int(stable_event[policy_ids].sum())
            policy_lift_transitions += int(lifted_event[policy_ids].sum())
            policy_success_transitions += int(success_now[policy_ids].sum())
        gripper_closed_transitions += int((gripper_latch[training_ids] < 0).sum())
        push_events += int(new_push[training_ids].sum())
        push_before_grasp_trial_events += int(new_push_before_trial[training_ids].sum())
        recovery_attempts += int(new_push[training_ids].sum())
        recovery_successes += int(recovery_success_now[training_ids].sum())
        valid_push = (episode_age >= push_grace_steps) & (~episode_contact) & (~done)
        if bool(valid_push.any()):
            maximum_push = max(maximum_push, float(push[valid_push].max()))
        maximum_force = max(maximum_force, float(force.max()))

        replay_action = request.clone()
        # The gripper latch is an actuator-side dwell.  Store the binary value
        # actually applied, while arm dimensions retain normalized request
        # coordinates for exact SAC actor/critic consistency.
        replay_action[:, 6] = applied[:, 6]
        phase_code = torch.zeros(args.num_envs, dtype=torch.int8, device=env.device)
        phase_code = torch.where(
            bilateral_event, torch.ones_like(phase_code), phase_code
        )
        phase_code = torch.where(
            stable_event, torch.full_like(phase_code, 2), phase_code
        )
        phase_code = torch.where(
            lifted_event, torch.full_like(phase_code, 3), phase_code
        )
        teacher_slices = teacher_contract.slices
        current_cube_position = state[:, teacher_slices["cube_pose_root_xyzw"]][:, :3]
        next_cube_position = next_state[:, teacher_slices["cube_pose_root_xyzw"]][:, :3]
        cube_displacement = torch.linalg.vector_norm(
            next_cube_position - current_cube_position, dim=-1
        )
        cube_displacement = torch.where(done, torch.zeros_like(cube_displacement), cube_displacement)
        ee_cube_distance_for_force = torch.linalg.vector_norm(
            state[:, teacher_slices["end_effector_to_cube_m"]], dim=-1
        )
        force_near_contact = (
            ee_cube_distance_for_force <= args.contact_approach_distance_m
        )
        force_physical_evidence = bilateral_event | stable_event | lifted_event
        force_eligible = force_near_contact & force_physical_evidence
        contact_energy_j = torch.where(
            force_eligible & bilateral_event,
            force * cube_displacement,
            torch.zeros_like(force),
        )
        # HER-force is a bounded sampling priority over real physical contact
        # only.  It cannot create contact/lift labels or bypass collision
        # safety.  Stable/lift rows retain a minimum priority even when the
        # terminal auto-reset prevents post-step force materialization.
        force_priority = 1.0 + torch.clamp(
            contact_energy_j / args.her_force_temperature_j, 0.0, 4.0
        )
        force_priority = torch.where(
            force_eligible & stable_event,
            torch.maximum(force_priority, torch.full_like(force_priority, 3.0)),
            force_priority,
        )
        force_priority = torch.where(
            force_eligible & lifted_event,
            torch.maximum(force_priority, torch.full_like(force_priority, 5.0)),
            force_priority,
        )
        replay.add_batch(
            rgbd=image[training_ids].cpu().numpy(), next_rgbd=next_image[training_ids].cpu().numpy(),
            proprio=prop[training_ids].cpu().numpy(), next_proprio=next_prop[training_ids].cpu().numpy(),
            privileged=state[training_ids].cpu().numpy(), next_privileged=next_state[training_ids].cpu().numpy(),
            actions=replay_action[training_ids].cpu().numpy(), rewards=reward[training_ids].cpu().numpy()[:, None],
            # Isaac Lab auto-reset observations are not valid final states for
            # timeout bootstrapping.  Use a conservative replay terminal mask
            # while preserving the original truncation field for audit.
            terminated=(terminated | truncated)[training_ids].cpu().numpy()[:, None],
            truncated=truncated[training_ids].cpu().numpy()[:, None],
            recurrent_hidden=recurrent_hidden[:, training_ids].squeeze(0).cpu().numpy(),
            next_recurrent_hidden=next_recurrent_hidden[:, training_ids].squeeze(0).cpu().numpy(),
            env_id=replay_env_id[training_ids].cpu().numpy(),
            episode_id=replay_episode_id[training_ids].cpu().numpy(),
            sequence_step=replay_sequence_step[training_ids].cpu().numpy(),
            phase_code=phase_code[training_ids].cpu().numpy(),
            force_priority=force_priority[training_ids].cpu().numpy(),
        )
        transitions += args.num_envs
        replay_transitions += training_env_count
        reward_sum += float(reward[training_ids].sum())
        action_sum += float(torch.linalg.vector_norm(applied[training_ids], dim=-1).sum())
        arm_action_sum += float(
            torch.linalg.vector_norm(applied[training_ids, :6], dim=-1).sum()
        )
        requested_action_sum += float(
            torch.linalg.vector_norm(request[training_ids], dim=-1).sum()
        )
        requested_arm_action_sum += float(
            torch.linalg.vector_norm(request[training_ids, :6], dim=-1).sum()
        )
        if transitions == args.num_envs:
            write_progress(
                "runtime/first_transition",
                transitions=transitions,
                episodes=episodes,
                updates=updates,
            )
        episode_age += 1
        done_ids = done.nonzero(as_tuple=True)[0]
        curriculum_promoted = False
        if done_ids.numel():
            train_done_ids = done_ids[training_mask[done_ids]]
            eval_done_ids = done_ids[evaluation_mask[done_ids]]
            if train_done_ids.numel():
                env._g2_reverse_curriculum.record_training_episodes(
                    contact=episode_contact[train_done_ids].detach().cpu().tolist(),
                    stable=episode_stable[train_done_ids].detach().cpu().tolist(),
                    lift=episode_lift[train_done_ids].detach().cpu().tolist(),
                )
            if eval_done_ids.numel():
                curriculum_promoted = env._g2_reverse_curriculum.record_evaluation_episodes(
                    contact=episode_contact[eval_done_ids].detach().cpu().tolist(),
                    stable=episode_stable[eval_done_ids].detach().cpu().tolist(),
                    lift=episode_lift[eval_done_ids].detach().cpu().tolist(),
                )
                evaluation_episodes += int(eval_done_ids.numel())
                evaluation_contact_episodes += int(episode_contact[eval_done_ids].sum())
                evaluation_stable_episodes += int(episode_stable[eval_done_ids].sum())
                evaluation_lift_episodes += int(episode_lift[eval_done_ids].sum())
                if curriculum_promoted:
                    reference_behavior_probability = max(
                        args.reference_behavior_minimum_probability,
                        reference_behavior_probability
                        - args.reference_behavior_probability_decrement,
                    )
        for env_id in done_ids.tolist():
            if bool(training_mask[env_id]):
                contact_episodes += int(episode_contact[env_id])
                stable_episodes += int(episode_stable[env_id])
                lift_episodes += int(episode_lift[env_id])
                success_episodes += int(success_now[env_id])
                half_success_episodes += int(
                    bool(episode_stable[env_id]) and not bool(success_now[env_id])
                )
                recovery_failures += int(episode_recovery_pending[env_id])
                episodes += 1
                if bool(reference_behavior_mask[env_id]):
                    reference_episodes += 1
                    reference_contact_episodes += int(episode_contact[env_id])
                    reference_stable_episodes += int(episode_stable[env_id])
                    reference_lift_episodes += int(episode_lift[env_id])
                    reference_success_episodes += int(success_now[env_id])
                else:
                    policy_episodes += 1
                    policy_contact_episodes += int(episode_contact[env_id])
                    policy_stable_episodes += int(episode_stable[env_id])
                    policy_lift_episodes += int(episode_lift[env_id])
                    policy_success_episodes += int(success_now[env_id])
            episode_contact[env_id] = episode_stable[env_id] = episode_lift[env_id] = False
            episode_age[env_id] = 0; push_latched[env_id] = False
            episode_grasp_trial[env_id] = False
            episode_recovery_pending[env_id] = False
            initial_cube[env_id] = tensor_value(cube.data.root_pos_w)[env_id]
            gripper_latch[env_id] = 1.0
            gripper_hold_age[env_id] = 0
            reference_close_latched[env_id] = False
            reference_stable_latched[env_id] = False
            replay_episode_id[env_id] = next_episode_id
            next_episode_id += 1
            replay_sequence_step[env_id] = 0
        if args.scripted_reverse_bootstrap and done_ids.numel():
            reference_behavior.reset_episodes(
                done_ids.detach().cpu().numpy(),
                learning_started=replay_transitions >= args.learning_starts,
                reference_probability=reference_behavior_probability,
                force_policy_mask=env._g2_reverse_last_policy_near[done_ids]
                .detach()
                .cpu()
                .numpy(),
            )
        replay_sequence_step[~done] += 1
        next_recurrent_hidden[:, done_ids] = 0.0
        recurrent_reset_mask = done.clone()
        updates_this_vector_step = 0
        if (
            len(replay) >= args.batch_size * recurrent_contract.sequence_length
            and replay.has_sequences(recurrent_contract.sequence_length)
        ):
            for _ in range(
                update_schedule.updates_due(replay_transitions, updates)
            ):
                demonstration_batch = None
                demonstration_coefficient = 0.0
                q_filter_reliability = 1.0
                demonstration_phase_counts: dict[str, int] = {}
                if demonstration_sequences is not None:
                    assert demonstration_phase_codes is not None
                    assert demonstration_window_success is not None
                    assert demonstration_force_priorities is not None
                    policy_mastery = demonstration_policy_mastery(
                        env._g2_reverse_curriculum.evaluation_rates,
                        evaluation_episodes=evaluation_episodes,
                    )
                    demonstration_coefficient, q_filter_reliability = (
                        demonstration_bc_schedule(
                            replay_transitions=replay_transitions,
                            learning_starts=args.learning_starts,
                            decay_transitions=(
                                args.demonstration_online_bc_decay_transitions
                            ),
                            initial_coefficient=(
                                args.demonstration_online_bc_initial_coefficient
                            ),
                            minimum_coefficient=(
                                args.demonstration_online_bc_minimum_coefficient
                            ),
                            policy_mastery=policy_mastery,
                        )
                    )
                    selected_numpy, demonstration_phase_counts = (
                        sample_phase_stratified_demonstration_indices(
                            demonstration_phase_codes,
                            demonstration_window_success,
                            batch_size=args.demonstration_batch_size,
                            phase_probabilities=demonstration_phase_probabilities,
                            success_priority_weight=(
                                args.demonstration_success_priority_weight
                            ),
                            rng=rng,
                            force_priorities=demonstration_force_priorities,
                            force_priority_mixture=(
                                args.demonstration_force_sampling_mixture
                            ),
                        )
                    )
                    demonstration_batch = {
                        name: item[selected_numpy]
                        if isinstance(item, torch.Tensor)
                        else item
                        for name, item in demonstration_sequences.items()
                    }
                    selected_force_priority = demonstration_force_priorities[
                        torch.as_tensor(selected_numpy, dtype=torch.long)
                    ]
                    demonstration_force_batch_metrics = {
                        "demonstration_force/batch_priority_mean": float(
                            selected_force_priority.mean()
                        ),
                        "demonstration_force/batch_priority_max": float(
                            selected_force_priority.max()
                        ),
                        "demonstration_force/batch_prioritized_fraction": float(
                            (selected_force_priority > 1.0).to(torch.float32).mean()
                        ),
                    }
                else:
                    demonstration_force_batch_metrics = {}
                replay_batch, replay_phase_counts = replay.sample_sequences_stratified(
                    args.batch_size,
                    recurrent_contract.sequence_length,
                    phase_probabilities=online_replay_phase_probabilities,
                    force_priority_mixture=args.her_force_sampling_mixture,
                )
                replay_batch, her_metrics = (
                    relabel_recurrent_visual_batch_with_future_goals(
                        replay_batch,
                        sequence_fraction=args.her_sequence_fraction,
                        rng=rng,
                        sandbox=sandbox,
                    )
                )
                force_priority_batch = np.asarray(
                    replay_batch["force_priority"], dtype=np.float32
                )
                last_metrics = agent.update(
                    replay_batch,
                    demonstration_batch=demonstration_batch,
                    demonstration_bc_coefficient=demonstration_coefficient,
                    demonstration_q_filter_reliability=q_filter_reliability,
                )
                last_metrics.update(
                    {
                        f"demonstration_online_bc/batch_{name.lower()}_windows": float(count)
                        for name, count in demonstration_phase_counts.items()
                    }
                )
                last_metrics.update(
                    {
                        f"online_replay/batch_{name.lower()}_sequences": float(count)
                        for name, count in replay_phase_counts.items()
                    }
                )
                last_metrics.update(her_metrics)
                last_metrics.update(demonstration_force_metrics)
                last_metrics.update(demonstration_force_batch_metrics)
                last_metrics.update(
                    {
                        "her_force/priority_mean": float(force_priority_batch.mean()),
                        "her_force/priority_max": float(force_priority_batch.max()),
                        "her_force/prioritized_row_fraction": float(
                            (force_priority_batch > 1.0).mean()
                        ),
                        "her_force/temperature_j": float(args.her_force_temperature_j),
                        "her_force/sampling_mixture": float(
                            args.her_force_sampling_mixture
                        ),
                        "her_force/activation_distance_m": float(
                            args.contact_approach_distance_m
                        ),
                        "her_force/eligible_near_contact_transition_rate": float(
                            force_eligible[training_ids].float().mean()
                        ),
                    }
                )
                updates += 1
                updates_this_vector_step += 1
        if transitions % args.log_interval < args.num_envs:
            ee_cube_distance = torch.linalg.vector_norm(
                next_state[:, state_slices["end_effector_to_cube_m"]], dim=-1
            )
            ee_root = next_state[:, state_slices["end_effector_pose_root_xyzw"]][:, :3]
            cube_root = next_state[:, state_slices["cube_pose_root_xyzw"]][:, :3]
            cube_minus_ee = cube_root - ee_root
            current_near = env._g2_reverse_last_near
            values = {
                "runtime/transitions": transitions, "runtime/episodes": episodes, "runtime/gradient_updates": updates,
                "runtime/replay_collection_transitions": replay_transitions,
                "runtime/gradient_updates_this_vector_step": updates_this_vector_step,
                "runtime/replay_samples_per_transition_realized": (
                    update_schedule.realized_replay_samples_per_transition(
                        replay_transitions, updates
                    )
                ),
                "train/mean_reward": reward_sum / max(replay_transitions, 1),
                "action/applied_l2_mean": action_sum / max(replay_transitions, 1),
                "action/applied_arm_l2_mean": arm_action_sum / max(replay_transitions, 1),
                "action/requested_normalized_l2_mean": requested_action_sum / max(replay_transitions, 1),
                "action/requested_normalized_arm_l2_mean": requested_arm_action_sum / max(replay_transitions, 1),
                "curriculum/reverse_beta": env._g2_reverse_curriculum.beta,
                "curriculum/promotion_count": env._g2_reverse_curriculum.promotion_count,
                "curriculum/promoted_this_vector_step": float(curriculum_promoted),
                "curriculum/near_reset_fraction": float(env._g2_reverse_last_near.float().mean()),
                "curriculum/policy_only_near_reset_fraction": float(
                    env._g2_reverse_last_policy_near.float().mean()
                ),
                "curriculum/initial_path_progress_mean": float(env._g2_reverse_last_progress.mean()),
                "curriculum/evaluation_episode_count": evaluation_episodes,
                "curriculum/evaluation_contact_rate": evaluation_contact_episodes / max(evaluation_episodes, 1),
                "curriculum/evaluation_stable_grasp_rate": evaluation_stable_episodes / max(evaluation_episodes, 1),
                "curriculum/evaluation_lift_rate": evaluation_lift_episodes / max(evaluation_episodes, 1),
                "task/grasp_bilateral_rate": contact_transitions / max(replay_transitions, 1),
                "task/stable_grasp_rate": stable_transitions / max(replay_transitions, 1),
                "task/lift_rate": lift_transitions / max(replay_transitions, 1),
                "task/success_rate": success_transitions / max(replay_transitions, 1),
                "bootstrap/transitions": bootstrap_transitions,
                "bootstrap/contact_transitions": bootstrap_contact_transitions,
                "bootstrap/stable_grasp_transitions": bootstrap_stable_transitions,
                "bootstrap/lift_transitions": bootstrap_lift_transitions,
                "bootstrap/success_transitions": bootstrap_success_transitions,
                "behavior/reference_probability": reference_behavior_probability,
                "behavior/reference_transition_count": bootstrap_transitions,
                "behavior/policy_transition_count": policy_behavior_transitions,
                "behavior/reference_fraction": bootstrap_transitions / max(replay_transitions, 1),
                "behavior/reference_episode_count": reference_episodes,
                "behavior/reference_contact_episode_rate": (
                    reference_contact_episodes / max(reference_episodes, 1)
                ),
                "behavior/reference_stable_episode_rate": (
                    reference_stable_episodes / max(reference_episodes, 1)
                ),
                "behavior/reference_lift_episode_rate": (
                    reference_lift_episodes / max(reference_episodes, 1)
                ),
                "behavior/reference_success_episode_rate": (
                    reference_success_episodes / max(reference_episodes, 1)
                ),
                "behavior/policy_episode_count": policy_episodes,
                "behavior/policy_contact_episode_rate": (
                    policy_contact_episodes / max(policy_episodes, 1)
                ),
                "behavior/policy_stable_episode_rate": (
                    policy_stable_episodes / max(policy_episodes, 1)
                ),
                "behavior/policy_lift_episode_rate": (
                    policy_lift_episodes / max(policy_episodes, 1)
                ),
                "behavior/policy_success_episode_rate": (
                    policy_success_episodes / max(policy_episodes, 1)
                ),
                "behavior/policy_contact_transition_rate": (
                    policy_contact_transitions / max(policy_behavior_transitions, 1)
                ),
                "behavior/policy_stable_transition_rate": (
                    policy_stable_transitions / max(policy_behavior_transitions, 1)
                ),
                "behavior/policy_lift_transition_rate": (
                    policy_lift_transitions / max(policy_behavior_transitions, 1)
                ),
                "behavior/policy_success_transition_rate": (
                    policy_success_transitions / max(policy_behavior_transitions, 1)
                ),
                "task/grasp_episode_count": contact_episodes,
                "task/stable_grasp_episode_count": stable_episodes,
                "task/lift_episode_count": lift_episodes,
                "task/success_episode_count": success_episodes,
                "task/half_success_episode_count": half_success_episodes,
                "task/grasp_episode_rate": contact_episodes / max(episodes, 1),
                "task/success_episode_rate": success_episodes / max(episodes, 1),
                "task/half_success_episode_rate": (
                    half_success_episodes / max(episodes, 1)
                ),
                "runtime/replay_environment_count": training_env_count,
                "runtime/policy_only_environment_count": int(
                    evaluation_mask.sum().item()
                ),
                "task/push_event_count": push_events, "task/max_pregrasp_push_m": maximum_push,
                "task/push_before_grasp_trial_count": push_before_grasp_trial_events,
                "task/push_before_grasp_trial_rate": (
                    push_before_grasp_trial_events / max(replay_transitions, 1)
                ),
                "task/recovery_attempt_count": recovery_attempts,
                "task/recovery_success_count": recovery_successes,
                "task/recovery_fail_count": recovery_failures,
                "task/recovery_success_rate": (
                    recovery_successes / max(recovery_attempts, 1)
                ),
                "task/recovery_fail_rate": (
                    recovery_failures / max(recovery_attempts, 1)
                ),
                "gripper/command_switch_count": gripper_switches,
                "gripper/closed_transition_rate": gripper_closed_transitions / max(replay_transitions, 1),
                "reference/close_gate_entry_count": reference_close_gate_entries,
                "reference/close_gate_entry_episode_age_mean_steps": (
                    reference_close_entry_episode_age_sum
                    / max(reference_close_gate_entries, 1)
                ),
                "reference/close_gate_entry_episode_age_max_steps": (
                    reference_close_entry_episode_age_max
                ),
                "reference/close_latched_environment_count": int(
                    reference_close_latched.sum()
                ),
                "reference/grasp_offset_error_mean_m": float(
                    reference_position_error[reference_behavior_mask].mean()
                    if bool(reference_behavior_mask.any())
                    else 0.0
                ),
                "reference/grasp_offset_error_min_m": float(
                    reference_position_error[reference_behavior_mask].min()
                    if bool(reference_behavior_mask.any())
                    else 0.0
                ),
                "gripper/master_position_mean_rad": float(
                    gripper_master_position.mean()
                ),
                "gripper/master_position_min_rad": float(
                    gripper_master_position.min()
                ),
                "gripper/master_position_max_rad": float(
                    gripper_master_position.max()
                ),
                "distance/ee_to_cube_mean_m": float(ee_cube_distance.mean()),
                "distance/ee_x_root_m": float(ee_root[:, 0].mean()),
                "distance/ee_y_root_m": float(ee_root[:, 1].mean()),
                "distance/ee_z_root_m": float(ee_root[:, 2].mean()),
                "distance/cube_center_x_root_m": float(cube_root[:, 0].mean()),
                "distance/cube_center_y_root_m": float(cube_root[:, 1].mean()),
                "distance/cube_center_z_root_m": float(cube_root[:, 2].mean()),
                "distance/cube_minus_ee_x_m": float(cube_minus_ee[:, 0].mean()),
                "distance/cube_minus_ee_y_m": float(cube_minus_ee[:, 1].mean()),
                "distance/cube_minus_ee_z_m": float(cube_minus_ee[:, 2].mean()),
                "distance/cube_to_goal_mean_m": float(
                    torch.linalg.vector_norm(
                        next_state[:, state_slices["cube_to_goal_m"]], dim=-1
                    ).mean()
                ),
                "task/goal_centroid_within_5mm_transition_rate": float(
                    (
                        torch.linalg.vector_norm(
                            next_state[:, state_slices["cube_to_goal_m"]], dim=-1
                        )
                        <= sandbox.goal_tolerance_m
                    ).float().mean()
                ),
                "force/grasp_total_mean_n": float(force.mean()), "force/grasp_total_max_n": maximum_force,
                "force/inner_mean_n": float(inner_n.mean()), "force/outer_mean_n": float(outer_n.mean()),
                "force/grasp_slip_mean_m_s": float(slip.mean()), "force/grasp_slip_max_m_s": float(slip.max()),
                "safety/max_arm_fd_velocity_rad_s": maximum_fd_velocity,
                "safety/max_arm_fd_acceleration_rad_s2": maximum_fd_acceleration,
                "safety/distance_conditioned_arm_speed_limit_mean_rad_s": float(
                    arm_speed_limit.mean()
                ),
                "safety/distance_conditioned_arm_speed_limit_min_rad_s": float(
                    arm_speed_limit.min()
                ),
                "safety/contact_approach_environment_count": int(
                    contact_approach.sum()
                ),
                "safety/gripper_speed_limit_min_rad_s": float(
                    gripper_speed_limit.min()
                ),
                "camera/head_valid_depth_ratio": float((image[:, 0, 5] > 0).float().mean()),
                "camera/wrist_valid_depth_ratio": float((image[:, 1, 5] > 0).float().mean()),
                # The recurrent learner performs future-achieved-goal HER on
                # sampled replay sequences.  Keep the environment-native
                # transition relabeler distinguished instead of reporting the
                # whole HER path as disabled.
                "her/enabled": float(args.her_sequence_fraction > 0.0),
                "her/recurrent_sequence_enabled": float(
                    args.her_sequence_fraction > 0.0
                ),
                "her/environment_native_enabled": 0.0,
                **last_metrics,
            }
            if bool(current_near.any()):
                values["distance/ee_to_cube_near_mean_m"] = float(
                    ee_cube_distance[current_near].mean()
                )
            if bool((~current_near).any()):
                values["distance/ee_to_cube_home_mean_m"] = float(
                    ee_cube_distance[~current_near].mean()
                )
            for term_index, term_name in enumerate(env.reward_manager.active_terms):
                values[f"reward/{term_name}_transition_mean"] = float(
                    env.reward_manager._step_reward[:, term_index].mean()
                    * float(env.step_dt)
                )
            logger.log_scalars("visual_train", transitions, values)
            write_progress(
                "runtime/heartbeat",
                transitions=transitions,
                episodes=episodes,
                updates=updates,
                reverse_beta=env._g2_reverse_curriculum.beta,
                wandb=logger.wandb_run_identity,
            )
        if transitions >= next_checkpoint_transition:
            replay.flush()
            checkpoint_path = args.output / f"checkpoint_{transitions:012d}.pt"
            checkpoint_payload = {
                "schema": "g2_recurrent_visual_reverse_sac_checkpoint_v4",
                "visual_policy_schema": G2_RECURRENT_VISUAL_POLICY_SCHEMA,
                "recurrent_policy_contract": recurrent_contract.serializable(),
                "visual_replay_schema": G2_VISUAL_REPLAY_SCHEMA,
                "teacher_observation_schema": G2_TEACHER_OBSERVATION_SCHEMA,
                "camera_shape": G2_VISUAL_CAMERA_SHAPE,
                "agent": agent.state_dict(),
                "transitions": transitions, "updates": updates, "episodes": episodes,
                "replay": {"capacity": replay.capacity, "size": replay.size, "position": replay.position,
                           "hidden_dim": replay.hidden_dim,
                           "sequence_length": recurrent_contract.sequence_length,
                           "burn_in_steps": recurrent_contract.burn_in_steps,
                           "storage": str(args.output / "replay")},
                "reverse_beta": env._g2_reverse_curriculum.beta,
                "update_schedule": {
                    "learning_starts": args.learning_starts,
                    "batch_size": args.batch_size,
                    "replay_samples_per_transition": args.replay_samples_per_transition,
                    "maximum_updates_per_vector_step": args.maximum_updates_per_vector_step,
                },
                "action_transform": action_transform.as_dict(),
                "reference_behavior_probability": reference_behavior_probability,
                "reference_behavior": reference_behavior.serializable(),
                "initial_pose_authority": "KEYBOARD_DATASET_METADATA",
                "initial_pose_profile": G2_KEYBOARD_PHOTO_POSE_PROFILE,
                "initial_right_arm_q_rad": list(G2_KEYBOARD_PHOTO_RIGHT_ARM_Q),
                "demonstration_datasets_sha256": demonstration_hashes,
                "demonstration_bc_updates": args.demonstration_bc_updates,
                "demonstration_success_source_window_count": (
                    demonstration_original_success_window_count
                ),
                "demonstration_success_augmented_window_count": (
                    args.demonstration_success_augmented_windows
                ),
                "demonstration_failure_source_window_count": (
                    demonstration_original_failure_window_count
                ),
                "demonstration_failure_augmented_window_count": (
                    args.demonstration_failure_augmented_windows
                ),
                "demonstration_online_bc": {
                    "actor_only": True,
                    "expert_arm_action_scale": (
                        args.demonstration_expert_arm_action_scale
                    ),
                    "initial_coefficient": (
                        args.demonstration_online_bc_initial_coefficient
                    ),
                    "minimum_coefficient": (
                        args.demonstration_online_bc_minimum_coefficient
                    ),
                    "decay_transitions": (
                        args.demonstration_online_bc_decay_transitions
                    ),
                    "q_filter": "SOFT_CRITIC_ADVANTAGE_SIGMOID",
                    "q_filter_temperature": (
                        args.demonstration_q_filter_temperature
                    ),
                    "phase_probabilities": dict(
                        zip(
                            ("REACH", "PRE_GRASP", "CONTACT", "STABLE_GRASP", "LIFT"),
                            demonstration_phase_probabilities,
                        )
                    ),
                    "success_priority_weight": (
                        args.demonstration_success_priority_weight
                    ),
                    "teacher_student_policy_contract": (
                        recurrent_contract.serializable()
                    ),
                    "student_head_distillation_weight": (
                        args.student_head_distillation_weight
                    ),
                    "cross_camera_contrastive_weight": (
                        args.cross_camera_contrastive_weight
                    ),
                    "temporal_pose_residual_weight": (
                        args.temporal_pose_residual_weight
                    ),
                },
                "reference_grasp_cube_minus_ee_root_m": list(
                    reference_grasp_offset
                ),
            }
            temporary_checkpoint = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(checkpoint_payload, temporary_checkpoint)
            temporary_checkpoint.replace(checkpoint_path)
            checkpoints = sorted(args.output.glob("checkpoint_*.pt"))
            for expired in checkpoints[:-args.checkpoint_retention]:
                expired.unlink()
            next_checkpoint_transition = (
                transitions // args.checkpoint_interval + 1
            ) * args.checkpoint_interval
        previous_action, state, prop, image = next_previous, next_state, next_prop, next_image
        recurrent_hidden = next_recurrent_hidden.detach()
        applied[done, :6] = 0.0; applied[done, 6] = 1.0

    replay.flush()
    final_distance = torch.linalg.vector_norm(
        state[:, state_slices["end_effector_to_cube_m"]], dim=-1
    )
    result = {
        "status": "COMPLETE",
        "transitions": transitions,
        "episode_horizon_steps": args.episode_horizon_steps,
        "replay_collection_transitions": replay_transitions,
        "updates": updates,
        "episodes": episodes,
        "wandb": logger.wandb_run_identity,
        "reverse_beta": env._g2_reverse_curriculum.beta,
        "student_actor_uses_rgbd": True,
        "teacher_actor_uses_rgbd": True,
        "teacher_actor_uses_gru": True,
        "keyboard_teacher_shared_policy_contract": recurrent_contract.serializable(),
        "initial_pose_authority": "KEYBOARD_DATASET_METADATA",
        "initial_pose_profile": G2_KEYBOARD_PHOTO_POSE_PROFILE,
        "initial_right_arm_q_rad": list(G2_KEYBOARD_PHOTO_RIGHT_ARM_Q),
        "demonstration_datasets_sha256": demonstration_hashes,
        "demonstration_sequence_window_count": demonstration_window_count,
        "demonstration_success_source_window_count": (
            demonstration_original_success_window_count
        ),
        "demonstration_success_augmented_window_count": (
            args.demonstration_success_augmented_windows
        ),
        "demonstration_failure_source_window_count": (
            demonstration_original_failure_window_count
        ),
        "demonstration_failure_augmented_window_count": (
            args.demonstration_failure_augmented_windows
        ),
        "demonstration_success_augmentation_method": (
            "outcome_stratified_resampling_plus_rgbd_random_shift"
        ),
        "reference_grasp_cube_minus_ee_root_m": list(reference_grasp_offset),
        "reference_grasp_calibration_rows": reference_grasp_offset_rows,
        "demonstration_bc_updates": args.demonstration_bc_updates,
        "demonstration_expert_arm_action_scale": (
            args.demonstration_expert_arm_action_scale
        ),
        "demonstration_bc_final_metrics": demonstration_metrics,
        "demonstration_force_priority_metrics": demonstration_force_metrics,
        "demonstration_online_bc_initial_coefficient": (
            args.demonstration_online_bc_initial_coefficient
        ),
        "demonstration_online_bc_minimum_coefficient": (
            args.demonstration_online_bc_minimum_coefficient
        ),
        "demonstration_online_bc_decay_transitions": (
            args.demonstration_online_bc_decay_transitions
        ),
        "demonstration_phase_probabilities": dict(
            zip(
                ("REACH", "PRE_GRASP", "CONTACT", "STABLE_GRASP", "LIFT"),
                demonstration_phase_probabilities,
            )
        ),
        "demonstration_reverse_reset_seed": (
            demonstration_reset_seed.serializable()
            if demonstration_reset_seed is not None
            else None
        ),
        "demonstration_grasp_ready_reset_seed": (
            demonstration_grasp_ready_seed.serializable()
            if demonstration_grasp_ready_seed is not None
            else None
        ),
        "critic_uses_privileged_state": True,
        "teacher_observation_dim": teacher_contract.observation_dim,
        "initial_near_mask": near_mask.detach().cpu().tolist(),
        "initial_ee_cube_distance_m": initial_distance.detach().cpu().tolist(),
        "final_ee_cube_distance_m": final_distance.detach().cpu().tolist(),
        "contact_transitions": contact_transitions,
        "stable_grasp_transitions": stable_transitions,
        "lift_transitions": lift_transitions,
        "success_transitions": success_transitions,
        "push_event_count": push_events,
        "push_before_grasp_trial_count": push_before_grasp_trial_events,
        "recovery_attempt_count": recovery_attempts,
        "recovery_success_count": recovery_successes,
        "recovery_fail_count": recovery_failures,
        "maximum_pregrasp_push_m": maximum_push,
        "maximum_grasp_force_n": maximum_force,
        "maximum_arm_fd_velocity_rad_s": maximum_fd_velocity,
        "maximum_arm_fd_acceleration_rad_s2": maximum_fd_acceleration,
        "arm_command_acceleration_limit_rad_s2": args.arm_joint_target_acceleration_rad_s2,
        "reference_arm_joint_target_speed_rad_s": (
            args.reference_arm_joint_target_speed_rad_s
        ),
        "reference_maximum_arm_action_magnitude": (
            args.reference_maximum_arm_action_magnitude
        ),
        "scripted_reverse_bootstrap_enabled": args.scripted_reverse_bootstrap,
        "bootstrap_transitions": bootstrap_transitions,
        "bootstrap_contact_transitions": bootstrap_contact_transitions,
        "bootstrap_stable_grasp_transitions": bootstrap_stable_transitions,
        "bootstrap_lift_transitions": bootstrap_lift_transitions,
        "bootstrap_success_transitions": bootstrap_success_transitions,
        "policy_behavior_transitions": policy_behavior_transitions,
        "policy_contact_transitions": policy_contact_transitions,
        "policy_stable_grasp_transitions": policy_stable_transitions,
        "policy_lift_transitions": policy_lift_transitions,
        "policy_success_transitions": policy_success_transitions,
        "reference_episode_count": reference_episodes,
        "reference_contact_episode_count": reference_contact_episodes,
        "reference_stable_grasp_episode_count": reference_stable_episodes,
        "reference_lift_episode_count": reference_lift_episodes,
        "reference_success_episode_count": reference_success_episodes,
        "policy_episode_count": policy_episodes,
        "policy_contact_episode_count": policy_contact_episodes,
        "policy_stable_grasp_episode_count": policy_stable_episodes,
        "policy_lift_episode_count": policy_lift_episodes,
        "policy_success_episode_count": policy_success_episodes,
        "reference_behavior_probability": reference_behavior_probability,
        "reference_behavior": reference_behavior.serializable(),
        "reference_close_gate_entry_count": reference_close_gate_entries,
        "reference_close_gate_entry_episode_age_mean_steps": (
            reference_close_entry_episode_age_sum
            / max(reference_close_gate_entries, 1)
        ),
        "reference_close_gate_entry_episode_age_max_steps": (
            reference_close_entry_episode_age_max
        ),
        "gripper_master_position_minimum_rad": (
            gripper_master_position_minimum_rad
        ),
        "gripper_master_position_maximum_rad": (
            gripper_master_position_maximum_rad
        ),
        "action_transform": action_transform.as_dict(),
        "reverse_preroll": reverse_preroll_metrics,
        "last_update_metrics": last_metrics,
        "cube_mass_kg": sandbox.cube_mass_kg,
        "cube_static_friction_range": sandbox.cube_static_friction_range,
        "cube_dynamic_friction_range": sandbox.cube_dynamic_friction_range,
        "camera_capture_interval_physics_steps": args.camera_capture_interval_steps,
        "camera_sensor_resolution": [args.camera_width, args.camera_height],
        "camera_renderer_head": type(getattr(head, "_renderer", None)).__name__,
        "camera_renderer_right_wrist": type(getattr(wrist, "_renderer", None)).__name__,
        "clone_in_fabric": args.clone_in_fabric,
        "initial_camera_visibility": initial_camera_visibility.serializable(),
        "replay_samples_per_transition_target": args.replay_samples_per_transition,
        "replay_samples_per_transition_realized": (
            update_schedule.realized_replay_samples_per_transition(
                replay_transitions, updates
            )
        ),
        "recurrent_replay_sequence_length": recurrent_contract.sequence_length,
        "recurrent_replay_burn_in_steps": recurrent_contract.burn_in_steps,
        "recurrent_profile": recurrent_profile.serializable(),
        "random_shift_pad_pixels": args.random_shift_pad,
        "curriculum_evaluation_envs": args.curriculum_evaluation_envs,
        "replay_environment_count": training_env_count,
        "policy_only_environment_count": int(evaluation_mask.sum().item()),
        "half_success_episode_count": half_success_episodes,
        "curriculum_evaluation_episodes": evaluation_episodes,
    }
    (args.output / "result_pre_close.json").write_text(json.dumps(result, indent=2) + "\n")
    write_progress(
        "runtime/complete",
        transitions=transitions,
        episodes=episodes,
        updates=updates,
        wandb=logger.wandb_run_identity,
    )
    logger.close(); env.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recurrent-profile",
        choices=recurrent_profile_names(),
        default=default_recurrent_profile_name(),
    )
    parser.add_argument(
        "--camera-encoder-profile",
        choices=("baseline_4layer", "cnn5_160"),
        default="baseline_4layer",
    )
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--total-transitions", type=int, default=2000); parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--episode-horizon-steps", type=int, default=1600)
    parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--replay-capacity", type=int, default=50000)
    parser.add_argument("--checkpoint-interval", type=int, default=2000); parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--checkpoint-retention", type=int, default=5)
    parser.add_argument("--settling-steps", type=int, default=100); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--camera-capture-interval-steps", type=int, default=10)
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument("--camera-height", type=int, default=192)
    parser.add_argument("--clone-in-fabric", action="store_true")
    parser.add_argument("--replay-samples-per-transition", type=float, default=1.0)
    parser.add_argument("--maximum-updates-per-vector-step", type=int, default=32)
    parser.add_argument("--arm-joint-target-speed-rad-s", type=float, default=.6)
    parser.add_argument(
        "--reference-arm-joint-target-speed-rad-s", type=float, default=.4
    )
    parser.add_argument(
        "--arm-joint-target-acceleration-rad-s2", type=float, default=2.0
    )
    parser.add_argument(
        "--reverse-preroll-arm-target-acceleration-rad-s2",
        type=float,
        default=3.0,
    )
    parser.add_argument("--gripper-joint-target-speed-rad-s", type=float, default=.8)
    parser.add_argument("--gripper-minimum-hold-steps", type=int, default=5)
    parser.add_argument(
        "--reference-gripper-close-position-tolerance-m",
        type=float,
        # Across the accepted 4x4x6 cm demonstrations, closing starts between
        # 1.3 and 15.7 mm from the calibrated relative pose.  A temporary
        # 30-mm gate froze the arm outside this demonstrated grasp manifold and
        # caused 12--31 mm pushes.  Round the measured upper envelope to 16 mm.
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
    # 0.10 rad/s consumed the contact window, while 0.80 rad/s produced a
    # 309-N spike in the live diagnostic.  The reference state machine now
    # holds Cartesian motion during closure, allowing this conservative middle
    # speed without changing the actuator's 0.8-rad/s hard authority.
    parser.add_argument("--contact-approach-gripper-speed-rad-s", type=float, default=0.40)
    parser.add_argument("--reverse-preroll-max-steps", type=int, default=80)
    parser.add_argument("--reverse-preroll-position-tolerance-m", type=float, default=.002)
    parser.add_argument("--reverse-preroll-maximum-normalized-translation", type=float, default=.25)
    parser.add_argument("--relative-pose-weight", type=float, default=.2)
    parser.add_argument("--pose-consistency-weight", type=float, default=.1)
    parser.add_argument(
        "--pose-consistency-rotation-weight", type=float, default=.1
    )
    parser.add_argument("--contact-weight", type=float, default=.1)
    parser.add_argument("--depth-validity-weight", type=float, default=.05)
    parser.add_argument("--cross-camera-contrastive-weight", type=float, default=.02)
    parser.add_argument("--cross-camera-contrastive-temperature", type=float, default=.1)
    parser.add_argument(
        "--cross-camera-false-negative-position-threshold-m",
        type=float,
        default=.01,
    )
    parser.add_argument("--temporal-pose-residual-weight", type=float, default=.05)
    parser.add_argument(
        "--temporal-pose-residual-rotation-weight", type=float, default=.1
    )
    parser.add_argument("--near-contact-auxiliary-distance-m", type=float, default=.10)
    parser.add_argument("--near-contact-relative-pose-multiplier", type=float, default=4.0)
    parser.add_argument("--share-camera-encoder-weights", action="store_true")
    parser.add_argument("--random-shift-pad", type=int, default=4)
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
    parser.add_argument(
        "--demonstration-q-filter-temperature", type=float, default=0.1
    )
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
    parser.add_argument(
        "--her-force-sampling-mixture",
        type=float,
        default=0.25,
        help=(
            "mixture weight for physical near-contact force priority; normal "
            "phase-stratified replay and HER retain the remaining authority"
        ),
    )
    parser.add_argument(
        "--demonstration-force-sampling-mixture", type=float, default=0.25
    )
    parser.add_argument("--student-head-distillation-weight", type=float, default=0.1)
    parser.add_argument(
        "--demonstration-success-augmented-windows",
        type=int,
        default=0,
        help=(
            "BC window count sampled only from physically successful episodes; "
            "duplicates receive the existing temporally consistent RGB-D random shift"
        ),
    )
    parser.add_argument(
        "--demonstration-failure-augmented-windows",
        type=int,
        default=0,
        help=(
            "BC windows sampled from failed physical episodes; episode boundaries, "
            "failure labels, and collision authority remain unchanged"
        ),
    )
    parser.add_argument("--curriculum-evaluation-envs", type=int, default=1)
    parser.add_argument("--wandb", action="store_true"); parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    parser.add_argument("--wandb-project", default="geniesim-g2-visual-sac"); parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument(
        "--shutdown-mode",
        choices=G2_SHUTDOWN_MODES,
        default="graceful-audit",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "runtime_progress.json").write_text(
        json.dumps(
            {
                "schema": "g2_visual_reverse_runtime_progress_v1",
                "stage": "runtime/app_import_begin",
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "transitions": 0,
                "episodes": 0,
                "updates": 0,
            },
            indent=2,
        )
        + "\n"
    )
    print("RUNTIME_APP_IMPORT_BEGIN", flush=True)
    from isaaclab.app import AppLauncher
    print("RUNTIME_APP_IMPORT_DONE", flush=True)
    print("RUNTIME_APP_LAUNCH_BEGIN", flush=True)
    launcher = AppLauncher(
        headless=True,
        enable_cameras=True,
        fast_shutdown=use_official_fast_shutdown(args.shutdown_mode),
    )
    print("RUNTIME_APP_LAUNCH_DONE", flush=True)
    exit_code = 0
    failure_path = args.output / "failure_pre_close.json"
    try:
        result = run(args)
        print(json.dumps(result, indent=2), flush=True)
    except BaseException as error:
        exit_code = 1
        traceback.print_exc()
        failure_path.write_text(
            json.dumps(
                {
                    "schema": "g2_visual_reverse_failure_pre_close_v1",
                    "status": "FAILED",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            )
            + "\n"
        )
    artifacts = []
    result_path = args.output / "result_pre_close.json"
    if result_path.is_file():
        artifacts.append(result_path)
    if failure_path.is_file():
        artifacts.append(failure_path)
    progress = args.output / "runtime_progress.json"
    if progress.is_file():
        artifacts.append(progress)
    artifacts.extend(sorted(args.output.glob("checkpoint_*.pt")))
    commit_pre_close_artifacts(
        output_dir=args.output,
        shutdown_mode=args.shutdown_mode,
        artifact_paths=artifacts,
    )
    print("PRE_CLOSE_DURABILITY_COMMITTED", flush=True)
    print("APP_CLOSE_BEGIN", flush=True)
    launcher.app.close(exit_code=exit_code)
    print("APP_CLOSE_RETURNED", flush=True)
    print("MAIN_RETURNED", flush=True)
    return exit_code


if __name__ == "__main__":
    _exit_code = main()
    if _exit_code != 0:
        raise SystemExit(_exit_code)
