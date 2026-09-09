import json

import h5py
import pytest
import torch

from geniesim.rl.isaaclab.g2_keyboard_teacher_dataset import (
    G2EpisodeOutcome,
    G2KeyboardTeacherTransitionContract,
)
from geniesim.rl.isaaclab.g2_student_training import (
    G2RecurrentStudentTrainer,
    G2StudentDeploymentPolicy,
    G2StudentTrainingConfig,
    estimate_student_rgbd_storage,
    list_canonical_episodes,
    load_canonical_episode,
    load_canonical_sequences,
)
from geniesim.rl.isaaclab.g2_teacher_sac import G2TeacherObservationContract
from geniesim.rl.isaaclab.g2_keyboard_pose import (
    G2_KEYBOARD_PHOTO_POSE_PROFILE,
    G2_KEYBOARD_PHOTO_RIGHT_ARM_Q,
)


def _canonical_episode(
    *, steps: int = 6, height: int = 6, width: int = 8, success: bool = False
):
    contract = G2KeyboardTeacherTransitionContract(height=height, width=width)
    teacher_contract = G2TeacherObservationContract()
    teacher = torch.zeros(steps, teacher_contract.observation_dim)
    for pose_name in ("end_effector_pose_root_xyzw", "cube_pose_root_xyzw"):
        teacher[:, teacher_contract.slices[pose_name].stop - 1] = 1.0
    keyboard = torch.zeros(steps, 8)
    keyboard[:, 7] = torch.where(
        torch.arange(steps) % 2 == 0,
        torch.ones(steps),
        -torch.ones(steps),
    )
    rgb = torch.arange(
        steps * height * width * 3, dtype=torch.int64
    ).reshape(steps, height, width, 3).remainder(256).to(torch.uint8)
    depth = torch.linspace(
        0.1, 1.5, steps * height * width, dtype=torch.float32
    ).reshape(steps, height, width, 1)
    terminated = torch.zeros(steps, 1, dtype=torch.bool)
    truncated = torch.zeros(steps, 1, dtype=torch.bool)
    if success:
        terminated[-1] = True
    else:
        truncated[-1] = True
    outcome = torch.full(
        (steps, 1), int(G2EpisodeOutcome.ONGOING), dtype=torch.int16
    )
    outcome[-1] = int(
        G2EpisodeOutcome.SUCCESS if success else G2EpisodeOutcome.TIMEOUT
    )
    timestamp = torch.arange(steps, dtype=torch.float32).unsqueeze(-1) * 0.02
    return contract.build(
        teacher_observation=teacher,
        next_teacher_observation=teacher.clone(),
        keyboard_physical_command=keyboard,
        teleop_action=keyboard,
        applied_teleop_action=keyboard,
        processed_arm_command=torch.zeros(steps, 7),
        nullspace_joint_delta=torch.zeros(steps, 7),
        nullspace_task_leakage=torch.zeros(steps, 1),
        nullspace_limit_clipped=torch.zeros(steps, 1, dtype=torch.bool),
        head_rgb=rgb,
        head_depth=depth,
        right_wrist_rgb=rgb.flip(2),
        right_wrist_depth=depth.clone(),
        contact_force_n=torch.zeros(steps, 2),
        stable_grasp=torch.zeros(steps, 1, dtype=torch.bool),
        slip_speed_m_s=torch.zeros(steps, 1),
        reward_components=torch.zeros(steps, 6),
        reward=torch.zeros(steps, 1),
        terminated=terminated,
        truncated=truncated,
        success=terminated.clone() if success else torch.zeros(steps, 1, dtype=torch.bool),
        outcome_code=outcome,
        timestamp=timestamp,
        camera_timestamp=timestamp.repeat(1, 2),
        camera_frame_age=torch.zeros(steps, 2),
        torso_joint_position_relative_rad=torch.zeros(steps, 5),
        torso_joint_velocity_rad_s=torch.zeros(steps, 5),
        teacher_q_value=torch.linspace(0.0, 1.0, steps).unsqueeze(-1),
        episode_id=torch.full((steps, 1), 7, dtype=torch.int64),
        sequence_step=torch.arange(steps, dtype=torch.int64).unsqueeze(-1),
    )


def _write_dataset(path, *, schema_location: str = "root", steps: int = 6):
    episode = _canonical_episode(steps=steps)
    episode = dict(episode)
    episode["actions"] = episode["applied_teleop_action"].clone()
    with h5py.File(path, "w") as file:
        data = file.create_group("data")
        if schema_location == "root":
            from geniesim.rl.isaaclab.g2_keyboard_teacher_dataset import (
                G2_KEYBOARD_TEACHER_DATASET_SCHEMA,
            )

            file.attrs["schema"] = G2_KEYBOARD_TEACHER_DATASET_SCHEMA
        elif schema_location == "env_args":
            from geniesim.rl.isaaclab.g2_keyboard_teacher_dataset import (
                G2_KEYBOARD_TEACHER_DATASET_SCHEMA,
            )
            from geniesim.rl.isaaclab.g2_visual_sac import (
                G2RecurrentVisualPolicyContract,
            )

            data.attrs["env_args"] = json.dumps(
                {
                    "dataset_schema": G2_KEYBOARD_TEACHER_DATASET_SCHEMA,
                    "initial_pose_contract": {
                        "profile": G2_KEYBOARD_PHOTO_POSE_PROFILE,
                        "right_arm_q_rad": list(G2_KEYBOARD_PHOTO_RIGHT_ARM_Q),
                    },
                    "keyboard_task_geometry": {
                        "cube_size_m": [0.04, 0.04, 0.06],
                        "cube_center_world_m": [0.50, -0.23, 0.760],
                    },
                    "recurrent_visual_policy_contract": (
                        G2RecurrentVisualPolicyContract().validated().serializable()
                    ),
                }
            )
        else:
            file.attrs["schema"] = schema_location
        demo = data.create_group("demo_0")
        for name, value in episode.items():
            demo.create_dataset(name, data=value.cpu().numpy())
    return episode


@pytest.mark.parametrize("schema_location", ["root", "env_args"])
def test_canonical_loader_validates_both_supported_schema_locations(tmp_path, schema_location):
    dataset = tmp_path / f"teacher_{schema_location}.hdf5"
    expected = _write_dataset(dataset, schema_location=schema_location)
    assert list_canonical_episodes(dataset) == ("demo_0",)
    loaded = load_canonical_episode(dataset, "demo_0")
    assert set(loaded) == set(expected)
    assert torch.equal(loaded["head_rgb"], expected["head_rgb"])

    sequence = load_canonical_sequences(
        dataset,
        sequence_contract=G2StudentTrainingConfig(
            sequence_length=4, burn_in_steps=1, stride=3
        ).sequence_contract(),
    )
    assert sequence["rgbd_u8"].shape == (2, 4, 2, 6, 48, 64)
    assert sequence["deployable_proprioception"].shape == (2, 4, 45)
    assert sequence["expert_action_target"].shape == (2, 4, 7)
    assert sequence["sequence_lengths"].tolist() == [4, 3]
    assert sequence["episode_outcome_success"].tolist() == [False, False]
    assert sequence["burn_in_steps"] == 1


def test_canonical_episode_can_be_rewindowed_from_legacy_temporal_contract(tmp_path):
    dataset = tmp_path / "teacher_old_temporal_window.hdf5"
    _write_dataset(dataset, schema_location="env_args", steps=40)
    with h5py.File(dataset, "r+") as file:
        environment = json.loads(file["data"].attrs["env_args"])
        recurrent = environment["recurrent_visual_policy_contract"]
        recurrent["hidden_dim"] = 256
        recurrent["sequence_length"] = 16
        recurrent["burn_in_steps"] = 4
        recurrent.pop("sequence_stride", None)
        recurrent.pop("gru_num_layers", None)
        file["data"].attrs["env_args"] = json.dumps(environment)

    sequence = load_canonical_sequences(dataset)
    assert sequence["rgbd_u8"].shape == (1, 48, 2, 6, 48, 64)
    assert sequence["sequence_lengths"].tolist() == [40]
    assert sequence["burn_in_steps"] == 12


def test_student_rejects_superseded_30mm_cube_dataset(tmp_path):
    dataset = tmp_path / "teacher_old_cube.hdf5"
    _write_dataset(dataset, schema_location="env_args")
    with h5py.File(dataset, "r+") as file:
        environment = json.loads(file["data"].attrs["env_args"])
        environment["keyboard_task_geometry"]["cube_size_m"] = [0.03, 0.03, 0.03]
        environment["keyboard_task_geometry"]["cube_center_world_m"] = [
            0.50, -0.23, 0.745
        ]
        file["data"].attrs["env_args"] = json.dumps(environment)
    with pytest.raises(ValueError, match="object dimensions differ"):
        list_canonical_episodes(dataset)


def test_canonical_loader_rejects_legacy_or_unknown_schema(tmp_path):
    dataset = tmp_path / "legacy.hdf5"
    _write_dataset(dataset, schema_location="legacy_keyboard_teacher_v9")
    with pytest.raises(ValueError, match="schema differs"):
        list_canonical_episodes(dataset)


def test_canonical_loader_rejects_old_initial_pose_distribution(tmp_path):
    dataset = tmp_path / "old_pose.hdf5"
    _write_dataset(dataset, schema_location="env_args")
    with h5py.File(dataset, "r+") as file:
        metadata = json.loads(file["data"].attrs["env_args"])
        metadata["initial_pose_contract"]["profile"] = "ARM_ONLY_TOP_DOWN_90_FRESH_V7_L_SHAPE"
        file["data"].attrs["env_args"] = json.dumps(metadata)
    with pytest.raises(ValueError, match="initial pose differs"):
        list_canonical_episodes(dataset)


def test_canonical_loader_requires_and_cross_checks_public_hdf_actions(tmp_path):
    missing = tmp_path / "missing_actions.hdf5"
    _write_dataset(missing)
    with h5py.File(missing, "r+") as file:
        del file["data/demo_0/actions"]
    with pytest.raises(ValueError, match="missing canonical fields"):
        load_canonical_episode(missing, "demo_0")

    mismatched = tmp_path / "mismatched_actions.hdf5"
    _write_dataset(mismatched)
    with h5py.File(mismatched, "r+") as file:
        file["data/demo_0/actions"][0, 0] = 0.5
    with pytest.raises(ValueError, match="public_actions_applied_request_mismatch"):
        load_canonical_episode(mismatched, "demo_0")


def test_50m_storage_estimate_is_an_explicit_rgbd_only_floor():
    estimate = estimate_student_rgbd_storage(50_000_000)
    assert estimate["compact_rgbd_bytes"] == 1_843_200_000_000
    assert estimate["raw_rgbd_bytes"] == 39_321_600_000_000
    assert estimate["compact_rgbd_tib"] > 1.6
    assert estimate["raw_rgbd_tib"] > 35.0


def test_sequence_loader_drops_tail_windows_with_no_post_burn_in_target(tmp_path):
    dataset = tmp_path / "short_tail.hdf5"
    _write_dataset(dataset, steps=9)
    config = G2StudentTrainingConfig(
        sequence_length=8, burn_in_steps=4, stride=8
    )
    sequence = load_canonical_sequences(
        dataset, sequence_contract=config.sequence_contract()
    )
    assert sequence["sequence_lengths"].tolist() == [8]


def test_sequence_loader_can_restrict_bc_to_successful_episodes(tmp_path):
    dataset = tmp_path / "success_only.hdf5"
    successful = _canonical_episode(steps=6, success=True)
    unsuccessful = _canonical_episode(steps=6, success=False)
    unsuccessful["episode_id"][:] = 8
    for episode in (successful, unsuccessful):
        episode["actions"] = episode["applied_teleop_action"].clone()
    from geniesim.rl.isaaclab.g2_keyboard_teacher_dataset import (
        G2_KEYBOARD_TEACHER_DATASET_SCHEMA,
    )
    with h5py.File(dataset, "w") as file:
        file.attrs["schema"] = G2_KEYBOARD_TEACHER_DATASET_SCHEMA
        data = file.create_group("data")
        for index, episode in enumerate((unsuccessful, successful)):
            demo = data.create_group(f"demo_{index}")
            for name, value in episode.items():
                demo.create_dataset(name, data=value.cpu().numpy())
    result = load_canonical_sequences(
        dataset,
        sequence_contract=G2StudentTrainingConfig(
            sequence_length=4, burn_in_steps=1, stride=3
        ).sequence_contract(),
        successful_episodes_only=True,
    )
    assert result["successful_episodes_only"] is True
    assert bool(result["episode_outcome_success"].all())
    assert set(torch.unique(result["episode_id"]).tolist()) <= {0, 7}
    assert 7 in set(torch.unique(result["episode_id"]).tolist())


def test_student_update_checkpoint_resume_and_metrics(tmp_path):
    dataset = tmp_path / "teacher.hdf5"
    _write_dataset(dataset)
    config = G2StudentTrainingConfig(
        hidden_dim=32,
        sequence_length=4,
        burn_in_steps=1,
        stride=3,
        seed=19,
    )
    batch = load_canonical_sequences(
        dataset, sequence_contract=config.sequence_contract()
    )
    trainer = G2RecurrentStudentTrainer(config)
    metrics = trainer.update(batch)
    assert metrics["student/update"] == 1.0
    assert metrics["student/burn_in_steps"] == 1.0
    assert metrics["student/teacher_value_valid_fraction"] == 1.0
    assert metrics["student/target_bilateral_contact_fraction"] == 0.0
    assert metrics["student/target_pregrasp_push_fraction"] == 0.0
    assert metrics["student/camera_frame_age_mean_s"] == 0.0
    assert "student/target_phase_miss_fraction" in metrics
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())

    checkpoint = trainer.save_checkpoint(tmp_path / "student.pt")
    restored = G2RecurrentStudentTrainer.from_checkpoint(checkpoint)
    assert restored.update_count == 1
    assert restored.config == config
    for name, value in trainer.model.state_dict().items():
        assert torch.equal(value, restored.model.state_dict()[name])


def test_deployment_policy_has_explicit_vector_hidden_reset_and_binary_gripper():
    model = G2StudentTrainingConfig(hidden_dim=32).model()
    policy = G2StudentDeploymentPolicy(model)
    rgbd = torch.zeros(2, 2, 6, 48, 64, dtype=torch.uint8)
    proprio = torch.zeros(2, 45)
    first = policy.step(rgbd, proprio, reset_mask=torch.tensor([True, True]))
    assert first["raw_student_action"].shape == (2, 7)
    assert first["controller_action"][:, 6].tolist() == [1.0, 1.0]
    policy.reset(torch.tensor([1]))
    assert torch.count_nonzero(policy.hidden[:, 1]) == 0
    assert torch.count_nonzero(policy.hidden[:, 0]) > 0
    with pytest.raises(ValueError, match="batch changed"):
        policy.step(rgbd[:1], proprio[:1])
