import math

import numpy as np
import pytest
import torch

from geniesim.rl.isaaclab.g2_visual_sac import (
    G2_CAMERA_ENCODER_PROFILES,
    G2_DEMONSTRATION_PHASES,
    G2_ONLINE_REPLAY_PHASES,
    G2EpisodeReferenceBehavior,
    G2RecurrentVisualAsymmetricSAC,
    G2RecurrentVisualPolicyContract,
    G2RecurrentVisualReplayBuffer,
    G2RecurrentVisualStudent,
    G2ReverseCurriculum,
    G2StudentObservationContract,
    G2TaskRelevantVisualEncoder,
    G2VisualAsymmetricSAC,
    G2VisualSACConfig,
    G2VisualReplayBuffer,
    classify_demonstration_sequence_phases,
    cross_camera_contrastive_loss,
    demonstration_bc_schedule,
    demonstration_force_energy_priorities,
    demonstration_policy_mastery,
    g2_reference_controller_action,
    pack_rgbd,
    preprocess_demonstration_expert_actions,
    random_shift_rgbd_sequences,
    relative_pose_auxiliary_loss,
    relative_pose_consistency_loss,
    relative_pose_target_from_teacher_state,
    sample_phase_stratified_demonstration_indices,
    temporal_pose_residual_loss,
)
from geniesim.rl.isaaclab.g2_teacher_sac import G2TeacherObservationContract


def test_visual_sac_entropy_target_matches_hybrid_action_contract():
    assert G2VisualSACConfig().resolved_target_entropy == pytest.approx(
        -(6.0 + math.log(2.0))
    )
    assert G2VisualSACConfig(target_entropy=-3.5).resolved_target_entropy == -3.5


def test_demonstration_speed_preprocessing_only_retimes_arm_labels():
    action = torch.tensor(
        [[[1.0, -0.8, 0.6, -0.4, 0.2, -0.1, -1.0],
          [-0.5, 0.4, -0.3, 0.2, -0.1, 0.0, 1.0]]]
    )
    original = action.clone()
    processed, metadata = preprocess_demonstration_expert_actions(
        action, arm_action_scale=0.75
    )
    torch.testing.assert_close(processed[..., :6], original[..., :6] * 0.75)
    assert torch.equal(processed[..., 6], original[..., 6])
    assert torch.equal(action, original)
    assert metadata["arm_action_scale"] == pytest.approx(0.75)
    assert metadata["usage"] == "BEHAVIOR_CLONING_LABEL_ONLY_NOT_RL_TRANSITION_REWRITE"


def test_demonstration_speed_preprocessing_preserves_continuous_gripper_fixture():
    action = torch.zeros(2, 3, 7)
    action[..., :6] = 0.4
    action[..., 6] = torch.tensor([[0.0, 0.5, -0.5], [1.0, -1.0, 0.25]])
    processed, _ = preprocess_demonstration_expert_actions(
        action, arm_action_scale=0.5
    )
    assert torch.equal(processed[..., 6], action[..., 6])
    torch.testing.assert_close(processed[..., :6], torch.full((2, 3, 6), 0.2))


def test_pack_rgbd_shape_depth_and_invalid_pixels():
    rgb = torch.rand(3, 192, 256, 4)
    depth = torch.full((3, 192, 256), 1.2345)
    depth[0, 0, 0] = torch.inf
    packed = pack_rgbd(rgb, depth, rgb, depth)
    assert packed.shape == (3, 2, 6, 48, 64)
    assert packed.dtype == torch.uint8
    assert set(int(value) for value in torch.unique(packed[:, :, 5])) <= {0, 255}
    code = packed[1, 0, 3].float() * 256.0 + packed[1, 0, 4].float()
    reconstructed = code * (2.0 / 65535.0)
    assert float((reconstructed - 1.2345).abs().max()) <= 2.0 / 65535.0


def test_relative_pose_uses_xyzw_and_cube_symmetry():
    contract = G2TeacherObservationContract()
    state = torch.zeros(1, contract.observation_dim)
    slices = contract.slices
    # EE is +90 degrees around Z. Cube is +1 m along root X.
    half = 2.0**-0.5
    state[:, slices["end_effector_pose_root_xyzw"]] = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, half, half]]
    )
    state[:, slices["cube_pose_root_xyzw"]] = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
    )
    target = relative_pose_target_from_teacher_state(state)
    torch.testing.assert_close(target[:, :3], torch.tensor([[0.0, -1.0, 0.0]]), atol=1e-6, rtol=0)

    identity = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    cube_quarter_turn = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, half, half]])
    assert relative_pose_auxiliary_loss(identity, cube_quarter_turn).item() < 1.0e-10


def test_visual_asymmetric_sac_updates_encoder_decoder_and_critics():
    replay = G2VisualReplayBuffer(32)
    rng = np.random.default_rng(7)
    count = 16
    actions = rng.uniform(-1, 1, (count, 7)).astype(np.float32)
    actions[:, 6] = np.where(actions[:, 6] >= 0.0, 1.0, -1.0)
    replay.add_batch(
        rgbd=rng.integers(0, 256, (count, 2, 6, 48, 64), dtype=np.uint8),
        next_rgbd=rng.integers(0, 256, (count, 2, 6, 48, 64), dtype=np.uint8),
        proprio=rng.normal(size=(count, 45)).astype(np.float32),
        next_proprio=rng.normal(size=(count, 45)).astype(np.float32),
        privileged=rng.normal(size=(count, 59)).astype(np.float32),
        next_privileged=rng.normal(size=(count, 59)).astype(np.float32),
        actions=actions,
        rewards=rng.normal(size=(count, 1)).astype(np.float32),
        terminated=np.zeros((count, 1), np.float32),
        truncated=np.zeros((count, 1), np.float32),
    )
    agent = G2VisualAsymmetricSAC()
    before = agent.visual.rgb_encoders[0][0].weight.detach().clone()
    metrics = agent.update(replay.sample(8))
    assert not torch.equal(before, agent.visual.rgb_encoders[0][0].weight)
    assert all(np.isfinite(value) for value in metrics.values())
    action = agent.select_actions(replay.rgbd[:2], replay.proprio[:2])
    assert action.shape == (2, 7)
    assert set(np.unique(action[:, 6])).issubset({-1.0, 1.0})


def test_visual_replay_rejects_nonbinary_gripper_action(tmp_path):
    replay = G2VisualReplayBuffer(2, storage_dir=tmp_path)
    batch = {
        "rgbd": np.zeros((1, 2, 6, 48, 64), np.uint8),
        "next_rgbd": np.zeros((1, 2, 6, 48, 64), np.uint8),
        "proprio": np.zeros((1, 45), np.float32),
        "next_proprio": np.zeros((1, 45), np.float32),
        "privileged": np.zeros((1, 59), np.float32),
        "next_privileged": np.zeros((1, 59), np.float32),
        "actions": np.zeros((1, 7), np.float32),
        "rewards": np.zeros((1, 1), np.float32),
        "terminated": np.zeros((1, 1), np.float32),
        "truncated": np.zeros((1, 1), np.float32),
    }
    with pytest.raises(ValueError, match=r"exactly -1 or \+1"):
        replay.add_batch(**batch)


def test_named_recurrent_profiles_preserve_current_and_legacy_settings():
    from geniesim.rl.isaaclab.g2_recurrent_profile import load_g2_recurrent_profile

    current = load_g2_recurrent_profile("temporal_128x48")
    intermediate = load_g2_recurrent_profile("temporal_128x32")
    legacy = load_g2_recurrent_profile("legacy_256x16")
    assert (
        current.hidden_dim,
        current.sequence_length,
        current.burn_in_steps,
        current.sequence_stride,
        current.gradient_clip,
    ) == (128, 48, 12, 36, 1.0)
    assert (
        intermediate.hidden_dim,
        intermediate.sequence_length,
        intermediate.burn_in_steps,
        intermediate.sequence_stride,
        intermediate.gradient_clip,
    ) == (128, 32, 8, 24, 1.0)
    assert (
        legacy.hidden_dim,
        legacy.sequence_length,
        legacy.burn_in_steps,
        legacy.sequence_stride,
        legacy.gradient_clip,
    ) == (256, 16, 4, 12, 5.0)
    assert current.gru_num_layers == legacy.gru_num_layers == 1


def test_recurrent_visual_teacher_uses_keyboard_student_backbone_and_hidden_replay():
    production = G2RecurrentVisualPolicyContract().validated()
    assert production.hidden_dim == 256
    assert production.gru_num_layers == 1
    assert production.sequence_length == 16
    assert production.burn_in_steps == 4
    assert production.sequence_stride == 12

    contract = G2RecurrentVisualPolicyContract(hidden_dim=32).validated()
    assert contract.camera_shape == (2, 6, 48, 64)
    assert contract.proprioception_dim == 45
    assert contract.action_dim == 7

    from geniesim.rl.isaaclab.g2_visual_sac import G2RecurrentVisualSACConfig

    assert G2RecurrentVisualSACConfig().gradient_clip == 5.0
    assert G2RecurrentVisualSACConfig().resolved_sequence_stride == 12

    agent = G2RecurrentVisualAsymmetricSAC(
        G2RecurrentVisualSACConfig(hidden_dim=32)
    )
    assert isinstance(agent.actor.policy, G2RecurrentVisualStudent)
    replay = G2RecurrentVisualReplayBuffer(8, hidden_dim=32)
    rng = np.random.default_rng(31)
    count = 4
    actions = rng.uniform(-1.0, 1.0, (count, 7)).astype(np.float32)
    actions[:, 6] = np.where(actions[:, 6] >= 0.0, 1.0, -1.0)
    replay.add_batch(
        rgbd=rng.integers(0, 256, (count, 2, 6, 48, 64), dtype=np.uint8),
        next_rgbd=rng.integers(0, 256, (count, 2, 6, 48, 64), dtype=np.uint8),
        proprio=rng.normal(size=(count, 45)).astype(np.float32),
        next_proprio=rng.normal(size=(count, 45)).astype(np.float32),
        privileged=rng.normal(size=(count, 59)).astype(np.float32),
        next_privileged=rng.normal(size=(count, 59)).astype(np.float32),
        actions=actions,
        rewards=rng.normal(size=(count, 1)).astype(np.float32),
        terminated=np.zeros((count, 1), np.float32),
        truncated=np.zeros((count, 1), np.float32),
        recurrent_hidden=np.zeros((count, 32), np.float32),
        next_recurrent_hidden=rng.normal(size=(count, 32)).astype(np.float32),
    )
    before_rgb = agent.actor.policy.visual.rgb_encoders[0][0].weight.detach().clone()
    before_gru = agent.actor.policy.gru.weight_ih_l0.detach().clone()
    metrics = agent.update(replay.sample(2))
    assert not torch.equal(
        before_rgb, agent.actor.policy.visual.rgb_encoders[0][0].weight
    )
    assert not torch.equal(before_gru, agent.actor.policy.gru.weight_ih_l0)
    assert all(np.isfinite(value) for value in metrics.values())
    hidden = agent.initial_hidden(2)
    action, next_hidden = agent.select_actions(
        replay.rgbd[:2], replay.proprio[:2], hidden
    )
    assert action.shape == (2, 7)
    assert next_hidden.shape == (1, 2, 32)


def test_recurrent_visual_teacher_behavior_clone_updates_only_actor_path():
    from geniesim.rl.isaaclab.g2_visual_sac import G2RecurrentVisualSACConfig

    agent = G2RecurrentVisualAsymmetricSAC(
        G2RecurrentVisualSACConfig(
            hidden_dim=16, sequence_length=4, burn_in_steps=1, random_shift_pad=0
        )
    )
    batch_size, steps = 2, 4
    target = torch.zeros(batch_size, steps, 7)
    target[..., 0] = 0.5
    target[..., 6] = 1.0
    relative = torch.zeros(batch_size, steps, 7)
    relative[..., 6] = 1.0
    batch = {
        "rgbd_u8": torch.zeros(batch_size, steps, 2, 6, 48, 64, dtype=torch.uint8),
        "deployable_proprioception": torch.zeros(batch_size, steps, 45),
        "expert_action_target": target,
        "expert_confidence": torch.ones(batch_size, steps, 1),
        "episode_id": torch.arange(batch_size).unsqueeze(1).expand(-1, steps),
        "sequence_step": torch.arange(steps).unsqueeze(0).expand(batch_size, -1),
        "padding_mask": torch.ones(batch_size, steps, dtype=torch.bool),
        "hidden_reset_mask": torch.tensor([[True, False, False, False]]).expand(batch_size, -1),
        "sequence_lengths": torch.full((batch_size,), steps),
        "relative_pose_target": relative,
        "contact_target": torch.zeros(batch_size, steps, 3),
        "depth_validity_target": torch.ones(batch_size, steps, 2),
        "burn_in_steps": 1,
    }
    actor_before = agent.actor.mean.weight.detach().clone()
    critic_before = agent.q1.net[0].weight.detach().clone()
    metrics = agent.behavior_clone(batch)
    assert not torch.equal(actor_before, agent.actor.mean.weight)
    assert torch.equal(critic_before, agent.q1.net[0].weight)
    assert metrics["demonstration_bc/supervised_rows"] == pytest.approx(6.0)
    assert metrics["demonstration_bc/loss_cross_camera_contrastive"] >= 0.0
    assert metrics["demonstration_bc/loss_temporal_pose_residual"] >= 0.0
    assert all(np.isfinite(value) for value in metrics.values())


def test_online_demonstration_regularizer_is_actor_only_and_phase_compatible():
    from geniesim.rl.isaaclab.g2_visual_sac import G2RecurrentVisualSACConfig

    agent = G2RecurrentVisualAsymmetricSAC(
        G2RecurrentVisualSACConfig(
            hidden_dim=16, sequence_length=4, burn_in_steps=1,
            random_shift_pad=0, demonstration_q_filter_temperature=0.1,
        )
    )
    batch_size, steps = 2, 4
    target = torch.zeros(batch_size, steps, 7)
    target[..., 0] = 0.4
    target[..., 6] = -1.0
    teacher_state = torch.zeros(batch_size, steps, 59)
    batch = {
        "rgbd_u8": torch.zeros(batch_size, steps, 2, 6, 48, 64, dtype=torch.uint8),
        "deployable_proprioception": torch.zeros(batch_size, steps, 45),
        "expert_action_target": target,
        "expert_confidence": torch.ones(batch_size, steps, 1),
        "teacher_state_target": teacher_state,
        "episode_id": torch.arange(batch_size).unsqueeze(1).expand(-1, steps),
        "sequence_step": torch.arange(steps).unsqueeze(0).expand(batch_size, -1),
        "padding_mask": torch.ones(batch_size, steps, dtype=torch.bool),
        "hidden_reset_mask": torch.tensor([[True, False, False, False]]).expand(batch_size, -1),
        "sequence_lengths": torch.full((batch_size,), steps),
        "burn_in_steps": 1,
    }
    loss, metrics = agent._demonstration_regularization_loss(batch)
    agent.actor_optimizer.zero_grad(set_to_none=True)
    agent.critic_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert any(parameter.grad is not None for parameter in agent.actor.parameters())
    assert any(
        parameter.grad is not None
        for parameter in agent.actor.policy.arm_action_head.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in agent.actor.policy.gripper_head.parameters()
    )
    assert all(parameter.grad is None for parameter in agent.q1.parameters())
    assert all(parameter.grad is None for parameter in agent.q2.parameters())
    assert 0.0 <= metrics["demonstration_online_bc/q_filter_weight_mean"] <= 1.0
    assert metrics["demonstration_online_bc/supervised_rows"] == pytest.approx(6.0)


def test_demonstration_phase_sampler_prioritizes_physical_phase_and_success():
    batch, steps = 5, 4
    state = torch.zeros(batch, steps, 59)
    phase_slice = G2TeacherObservationContract().slices["curriculum_phase_features"]
    action = torch.zeros(batch, steps, 7)
    action[..., 6] = 1.0
    bilateral = torch.zeros(batch, steps, 1)
    stable = torch.zeros(batch, steps, 1)
    # One window for every ordered phase.
    action[1, 1:, 6] = -1.0
    state[2, 1:, phase_slice.start + 1] = 1.0
    bilateral[2, 1:] = 1.0
    state[3, 1:, phase_slice.start + 1] = 1.0
    stable[3, 1:] = 1.0
    state[4, 1:, phase_slice.start + 2] = 1.0
    sequences = {
        "teacher_state_target": state,
        "expert_action_target": action,
        "bilateral_contact_target": bilateral,
        "stable_grasp_target": stable,
        "padding_mask": torch.ones(batch, steps, dtype=torch.bool),
        "burn_in_steps": 1,
    }
    codes = classify_demonstration_sequence_phases(sequences)
    torch.testing.assert_close(codes, torch.arange(5))
    indices, counts = sample_phase_stratified_demonstration_indices(
        codes,
        torch.tensor([False, False, False, True, True]),
        batch_size=100,
        phase_probabilities=(0.25, 0.20, 0.20, 0.20, 0.15),
        success_priority_weight=2.0,
        rng=np.random.default_rng(42),
    )
    assert indices.shape == (100,)
    assert sum(counts.values()) == 100
    assert set(counts) == set(G2_DEMONSTRATION_PHASES)


def test_demonstration_force_sampler_uses_bounded_mixture_not_full_replacement():
    codes = torch.zeros(2, dtype=torch.long)
    success = torch.ones(2, dtype=torch.bool)
    force = torch.tensor([1.0, 1000.0])
    uniform_indices, _ = sample_phase_stratified_demonstration_indices(
        codes,
        success,
        batch_size=4000,
        phase_probabilities=(1.0, 0.0, 0.0, 0.0, 0.0),
        success_priority_weight=1.0,
        rng=np.random.default_rng(7),
        force_priorities=force,
        force_priority_mixture=0.0,
    )
    mixed_indices, _ = sample_phase_stratified_demonstration_indices(
        codes,
        success,
        batch_size=4000,
        phase_probabilities=(1.0, 0.0, 0.0, 0.0, 0.0),
        success_priority_weight=1.0,
        rng=np.random.default_rng(7),
        force_priorities=force,
        force_priority_mixture=0.25,
    )
    uniform_high = float(np.mean(uniform_indices == 1))
    mixed_high = float(np.mean(mixed_indices == 1))
    assert uniform_high == pytest.approx(0.5, abs=0.04)
    assert 0.60 < mixed_high < 0.70


def test_demonstration_force_sampler_rejects_invalid_mixture():
    with pytest.raises(ValueError, match="force_priority_mixture"):
        sample_phase_stratified_demonstration_indices(
            torch.zeros(1, dtype=torch.long),
            torch.zeros(1, dtype=torch.bool),
            batch_size=1,
            phase_probabilities=(1.0, 0.0, 0.0, 0.0, 0.0),
            success_priority_weight=1.0,
            rng=np.random.default_rng(1),
            force_priorities=torch.ones(1),
            force_priority_mixture=1.1,
        )


def test_demonstration_phase_sampler_redistributes_missing_stratum():
    codes = torch.tensor([0, 2, 4], dtype=torch.long)
    indices, counts = sample_phase_stratified_demonstration_indices(
        codes,
        torch.tensor([False, False, True]),
        batch_size=1000,
        phase_probabilities=(0.25, 0.0, 0.20, 0.40, 0.15),
        success_priority_weight=1.0,
        rng=np.random.default_rng(13),
    )
    assert set(indices.tolist()) <= {0, 1, 2}
    assert counts["STABLE_GRASP"] == 0
    assert counts["REACH"] > 0
    assert counts["CONTACT"] > 0
    assert counts["LIFT"] > 0


def test_demonstration_bc_schedule_waits_for_held_out_policy_mastery():
    assert demonstration_policy_mastery(
        (1.0, 1.0, 1.0), evaluation_episodes=9
    ) == 0.0
    no_success = demonstration_policy_mastery(
        (0.8, 0.4, 0.0), evaluation_episodes=20
    )
    coefficient, reliability = demonstration_bc_schedule(
        replay_transitions=76000,
        learning_starts=1000,
        decay_transitions=50000,
        initial_coefficient=1.0,
        minimum_coefficient=0.1,
        policy_mastery=no_success,
    )
    assert coefficient == pytest.approx(1.0)
    assert reliability == pytest.approx(0.0)
    full_mastery = demonstration_policy_mastery(
        (0.5, 0.3, 0.2), evaluation_episodes=100
    )
    coefficient, reliability = demonstration_bc_schedule(
        replay_transitions=76000,
        learning_starts=1000,
        decay_transitions=50000,
        initial_coefficient=1.0,
        minimum_coefficient=0.1,
        policy_mastery=full_mastery,
    )
    assert coefficient == pytest.approx(0.1)
    assert reliability == pytest.approx(1.0)


def test_recurrent_replay_samples_contiguous_episode_sequences_with_burn_in():
    from geniesim.rl.isaaclab.g2_visual_sac import G2RecurrentVisualSACConfig

    replay = G2RecurrentVisualReplayBuffer(64, hidden_dim=16, seed=9)
    rng = np.random.default_rng(9)
    for step in range(6):
        count = 2
        actions = rng.uniform(-1.0, 1.0, (count, 7)).astype(np.float32)
        actions[:, 6] = 1.0
        rgbd = np.full((count, 2, 6, 48, 64), step, dtype=np.uint8)
        replay.add_batch(
            rgbd=rgbd,
            next_rgbd=np.full_like(rgbd, step + 1),
            proprio=np.full((count, 45), step, np.float32),
            next_proprio=np.full((count, 45), step + 1, np.float32),
            privileged=rng.normal(size=(count, 59)).astype(np.float32),
            next_privileged=rng.normal(size=(count, 59)).astype(np.float32),
            actions=actions,
            rewards=np.zeros((count, 1), np.float32),
            terminated=np.zeros((count, 1), np.float32),
            truncated=np.zeros((count, 1), np.float32),
            recurrent_hidden=np.zeros((count, 16), np.float32),
            next_recurrent_hidden=np.zeros((count, 16), np.float32),
            env_id=np.arange(count, dtype=np.int64),
            episode_id=np.arange(count, dtype=np.int64),
            sequence_step=np.full(count, step, dtype=np.int64),
        )
    batch = replay.sample_sequences(3, 6)
    assert batch["rgbd"].shape == (3, 6, 2, 6, 48, 64)
    assert np.all(np.diff(batch["sequence_step"], axis=1) == 1)
    assert np.all(batch["episode_id"] == batch["episode_id"][:, :1])
    assert np.all(batch["next_rgbd"][:, :-1] == batch["rgbd"][:, 1:])
    agent = G2RecurrentVisualAsymmetricSAC(
        G2RecurrentVisualSACConfig(
            hidden_dim=16, sequence_length=6, burn_in_steps=2,
            random_shift_pad=1,
        )
    )
    metrics = agent.update(batch)
    assert metrics["replay/sequence_length"] == 6
    assert metrics["replay/burn_in_steps"] == 2
    assert metrics["loss/visual_cross_camera_contrastive"] >= 0.0
    assert metrics["loss/visual_temporal_pose_residual"] >= 0.0
    assert all(np.isfinite(value) for value in metrics.values())


def test_recurrent_online_replay_phase_stratification_selects_lift_sequences():
    replay = G2RecurrentVisualReplayBuffer(64, hidden_dim=8, seed=17)
    rng = np.random.default_rng(17)
    for step in range(6):
        count = len(G2_ONLINE_REPLAY_PHASES)
        actions = np.zeros((count, 7), np.float32)
        actions[:, 6] = -1.0
        rgbd = np.full((count, 2, 6, 48, 64), step, dtype=np.uint8)
        replay.add_batch(
            rgbd=rgbd,
            next_rgbd=np.full_like(rgbd, step + 1),
            proprio=np.zeros((count, 45), np.float32),
            next_proprio=np.zeros((count, 45), np.float32),
            privileged=rng.normal(size=(count, 59)).astype(np.float32),
            next_privileged=rng.normal(size=(count, 59)).astype(np.float32),
            actions=actions,
            rewards=np.zeros((count, 1), np.float32),
            terminated=np.zeros((count, 1), np.float32),
            truncated=np.zeros((count, 1), np.float32),
            recurrent_hidden=np.zeros((count, 8), np.float32),
            next_recurrent_hidden=np.zeros((count, 8), np.float32),
            env_id=np.arange(count, dtype=np.int64),
            episode_id=np.arange(count, dtype=np.int64),
            sequence_step=np.full(count, step, dtype=np.int64),
            phase_code=np.arange(count, dtype=np.int8),
            force_priority=np.arange(1, count + 1, dtype=np.float32),
        )
    batch, counts = replay.sample_sequences_stratified(
        8, 6, phase_probabilities=(0.0, 0.0, 0.0, 1.0)
    )
    assert counts == {"REACH": 0, "CONTACT": 0, "STABLE_GRASP": 0, "LIFT": 8}
    assert np.all(batch["phase_code"][:, -1] == 3)
    assert np.all(np.diff(batch["sequence_step"], axis=1) == 1)
    with pytest.raises(ValueError, match="force priority mixture"):
        replay.sample_sequences_stratified(
            1,
            6,
            phase_probabilities=(1.0, 0.0, 0.0, 0.0),
            force_priority_mixture=1.1,
        )


def test_demonstration_force_priority_uses_only_safe_bilateral_motion():
    contract = G2TeacherObservationContract()
    state = torch.zeros(2, 4, contract.observation_dim)
    cube = contract.slices["cube_pose_root_xyzw"]
    state[..., cube.stop - 1] = 1.0
    state[0, :, cube.start] = torch.tensor([0.0, 0.001, 0.003, 0.006])
    sequences = {
        "teacher_state_target": state,
        "contact_force_target_n": torch.tensor(
            [[[0.0, 0.0], [10.0, 10.0], [10.0, 10.0], [10.0, 10.0]],
             [[50.0, 50.0], [50.0, 50.0], [50.0, 50.0], [50.0, 50.0]]]
        ),
        "bilateral_contact_target": torch.ones(2, 4, 1),
        "safe_for_her_force": torch.tensor(
            [[[1], [1], [1], [1]], [[0], [0], [0], [0]]], dtype=torch.bool
        ),
        "stable_grasp_target": torch.zeros(2, 4, 1),
        "padding_mask": torch.ones(2, 4, dtype=torch.bool),
        "burn_in_steps": 1,
    }
    original_padding = sequences["padding_mask"].clone()
    priority, metrics = demonstration_force_energy_priorities(
        sequences, temperature_j=0.02
    )
    assert priority[0] > 1.0
    assert priority[1] == 1.0
    assert metrics["demonstration_force/prioritized_window_fraction"] == 0.5
    assert torch.equal(sequences["padding_mask"], original_padding)


def test_reference_behavior_forces_policy_only_at_marked_near_reset():
    scheduler = G2EpisodeReferenceBehavior(
        np.array([True, True, False]), seed=5
    )
    scheduler.reset_episodes(
        np.array([0, 1, 2]),
        learning_started=True,
        reference_probability=1.0,
        force_policy_mask=np.array([True, False, True]),
    )
    assert scheduler.current_mask().tolist() == [False, True, False]


def test_random_shift_keeps_modalities_and_time_aligned():
    # Every channel/time plane carries the same spatial ramp plus an offset.
    base = torch.arange(48 * 64, dtype=torch.int64).reshape(48, 64) % 128
    sequence = torch.empty((2, 3, 2, 6, 48, 64), dtype=torch.uint8)
    for batch in range(2):
        for step in range(3):
            for camera in range(2):
                for channel in range(6):
                    sequence[batch, step, camera, channel] = (
                        base + 7 * step + 3 * channel
                    ).to(torch.uint8)
    shifted = random_shift_rgbd_sequences(sequence, pad=4)
    assert shifted.shape == sequence.shape
    assert shifted.dtype == torch.uint8
    # A single spatial transform per (batch,camera) preserves channel and
    # temporal offsets away from uint8 wrap-around.
    assert torch.equal(
        shifted[:, 1, :, 0].to(torch.int16) - shifted[:, 0, :, 0].to(torch.int16),
        torch.full((2, 2, 48, 64), 7, dtype=torch.int16),
    )
    assert torch.equal(
        shifted[:, :, :, 1].to(torch.int16) - shifted[:, :, :, 0].to(torch.int16),
        torch.full((2, 3, 2, 48, 64), 3, dtype=torch.int16),
    )


def test_heavier_gated_cross_camera_encoder_and_pose_consistency_gradients():
    model = G2RecurrentVisualStudent()
    assert model.hidden_dim == 256
    assert model.visual.rgb_encoders[0][0].out_channels == 32
    assert model.visual.rgb_encoders[0][2].out_channels == 64
    assert model.visual.rgb_encoders[0][4].out_channels == 96
    assert model.visual.rgb_encoders[0][6].out_channels == 128
    assert model.visual.cross_camera_attention.num_heads == 4

    original = torch.tensor(
        [[0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=True
    )
    shifted = torch.tensor(
        [[0.02, 0.0, 0.0, 0.0, 0.0, 0.01, 0.99995]], requires_grad=True
    )
    loss = relative_pose_consistency_loss(original, shifted)
    assert loss > 0.0
    loss.backward()
    assert original.grad is not None
    assert shifted.grad is not None


def test_cross_camera_contrastive_loss_aligns_same_timestep_camera_tokens():
    head = torch.eye(4, requires_grad=True)
    aligned_wrist = torch.eye(4, requires_grad=True)
    aligned = torch.stack((head, aligned_wrist), dim=1).reshape(1, 4, 2, 4)
    valid = torch.ones(1, 4, dtype=torch.bool)
    aligned_loss = cross_camera_contrastive_loss(
        aligned, valid, temperature=0.1
    )
    permuted = torch.stack((head, aligned_wrist.roll(1, dims=0)), dim=1).reshape(
        1, 4, 2, 4
    )
    permuted_loss = cross_camera_contrastive_loss(
        permuted, valid, temperature=0.1
    )
    assert aligned_loss < permuted_loss
    aligned_loss.backward()
    assert head.grad is not None
    assert aligned_wrist.grad is not None


def test_temporal_pose_residual_uses_motion_not_absolute_pose_offset():
    target = torch.zeros(1, 4, 7)
    target[..., 0] = torch.tensor((0.00, 0.01, 0.03, 0.06))
    target[..., 6] = 1.0
    prediction = target.clone()
    prediction[..., :3] += torch.tensor((0.2, -0.1, 0.05))
    prediction.requires_grad_()
    valid = torch.ones(1, 4, dtype=torch.bool)
    matching_motion = temporal_pose_residual_loss(prediction, target, valid)
    wrong_motion_prediction = prediction.detach().clone()
    wrong_motion_prediction[:, 2:, 0] += 0.1
    wrong_motion = temporal_pose_residual_loss(
        wrong_motion_prediction, target, valid
    )
    assert matching_motion < 1.0e-3
    assert wrong_motion > matching_motion
    matching_motion.backward()
    assert prediction.grad is not None


def test_random_shift_returns_exact_k_prime_principal_point_delta():
    torch.manual_seed(7)
    sequence = torch.zeros((2, 3, 2, 6, 48, 64), dtype=torch.uint8)
    sequence[:, :, :, :, 24, 32] = 211
    shifted, principal_delta = random_shift_rgbd_sequences(
        sequence, pad=4, return_principal_point_shift=True
    )
    assert principal_delta.shape == (2, 3, 2, 2)
    assert torch.equal(principal_delta[:, 1:], principal_delta[:, :1].expand(-1, 2, -1, -1))
    for batch in range(2):
        for camera in range(2):
            du, dv = principal_delta[batch, 0, camera].to(torch.int64)
            assert shifted[batch, 0, camera, 0, 24 + dv, 32 + du] == 211


def test_heavier_gated_cross_camera_encoder_and_pose_consistency_gradients():
    model = G2RecurrentVisualStudent()
    assert model.hidden_dim == 256
    assert [model.visual.rgb_encoders[0][index].out_channels for index in (0, 2, 4, 6)] == [
        32, 64, 96, 128
    ]
    assert model.visual.cross_camera_attention.num_heads == 4
    original = torch.tensor(
        [[0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=True
    )
    shifted = torch.tensor(
        [[0.02, 0.0, 0.0, 0.0, 0.0, 0.01, 0.99995]], requires_grad=True
    )
    loss = relative_pose_consistency_loss(original, shifted)
    assert torch.isfinite(loss)
    assert loss > 0.0
    loss.backward()
    assert original.grad is not None
    assert shifted.grad is not None


def test_reverse_curriculum_requires_a_full_success_window():
    curriculum = G2ReverseCurriculum(window=4)
    for _ in range(3):
        assert not curriculum.record_episode(contact=True, stable=True, lift=True)
    assert curriculum.record_episode(contact=True, stable=True, lift=True)
    assert curriculum.beta == pytest.approx(0.55)
    sampled = curriculum.sample_near(np.random.default_rng(42), 1000)
    assert 0.45 < sampled.mean() < 0.65


def test_reverse_curriculum_vector_batch_promotes_at_most_once():
    curriculum = G2ReverseCurriculum(window=100)
    values = [True] * 2000
    assert curriculum.record_episodes(contact=values, stable=values, lift=values)
    assert curriculum.beta == pytest.approx(0.55)
    assert curriculum.promotion_count == 1


def test_reverse_curriculum_promotes_only_from_held_out_evaluation():
    curriculum = G2ReverseCurriculum(window=4)
    curriculum.record_training_episodes(
        contact=[True] * 20, stable=[True] * 20, lift=[True] * 20
    )
    assert curriculum.beta == pytest.approx(0.60)
    assert curriculum.training_rates == pytest.approx((1.0, 1.0, 1.0))
    assert not curriculum.record_evaluation_episodes(
        contact=[True] * 3, stable=[True] * 3, lift=[True] * 3
    )
    assert curriculum.record_evaluation_episodes(
        contact=[True], stable=[True], lift=[True]
    )
    assert curriculum.beta == pytest.approx(0.55)


def test_reference_behavior_is_latched_until_episode_reset():
    scheduler = G2EpisodeReferenceBehavior(
        [True, True, False], seed=42
    )
    np.testing.assert_array_equal(scheduler.current_mask(), [True, True, False])
    # Merely querying the scheduler cannot resample authority mid-episode.
    for _ in range(20):
        np.testing.assert_array_equal(
            scheduler.current_mask(), [True, True, False]
        )
    scheduler.reset_episodes(
        [0, 2], learning_started=True, reference_probability=0.0
    )
    np.testing.assert_array_equal(scheduler.current_mask(), [False, True, False])
    scheduler.reset_episodes(
        [0, 1, 2], learning_started=False, reference_probability=0.0
    )
    np.testing.assert_array_equal(scheduler.current_mask(), [True, True, False])


def test_reference_controller_transports_stable_grasp_to_xyz_goal():
    contract = G2TeacherObservationContract()
    state = torch.zeros(2, contract.observation_dim)
    slices = contract.slices
    state[:, slices["end_effector_to_cube_m"]] = torch.tensor(
        [[0.03, -0.02, -0.01], [0.01, 0.01, -0.02]]
    )
    state[:, slices["cube_to_goal_m"]] = torch.tensor(
        [[0.006, -0.003, 0.080], [-0.004, 0.002, 0.060]]
    )
    result = g2_reference_controller_action(
        state,
        end_effector_to_cube_slice=slices["end_effector_to_cube_m"],
        cube_to_goal_slice=slices["cube_to_goal_m"],
        bilateral_contact=torch.tensor([True, True]),
        stable_grasp=torch.tensor([True, False]),
        maximum_arm_action_magnitude=0.10,
    )
    torch.testing.assert_close(
        result[0, :3], torch.tensor([0.10, -0.10, 0.10])
    )
    torch.testing.assert_close(result[1, :6], torch.zeros(6))
    torch.testing.assert_close(result[:, 6], -torch.ones(2))


def test_reference_controller_tracks_calibrated_side_pinch_offset():
    contract = G2TeacherObservationContract()
    state = torch.zeros(2, contract.observation_dim)
    slices = contract.slices
    calibrated = (-0.0247, 0.0001, -0.0251)
    state[:, slices["end_effector_to_cube_m"]] = torch.tensor(
        [calibrated, (-0.0097, 0.0001, -0.0251)]
    )
    result = g2_reference_controller_action(
        state,
        end_effector_to_cube_slice=slices["end_effector_to_cube_m"],
        cube_to_goal_slice=slices["cube_to_goal_m"],
        bilateral_contact=torch.tensor([False, False]),
        stable_grasp=torch.tensor([False, False]),
        maximum_arm_action_magnitude=0.10,
        target_cube_minus_ee_m=calibrated,
    )
    torch.testing.assert_close(result[0, :6], torch.zeros(6), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        result[1, :3], torch.tensor([0.10, 0.0, 0.0]), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(result[:, 6], torch.tensor([-1.0, 1.0]))


def test_recurrent_student_requires_contiguous_episode_sequences():
    model = G2RecurrentVisualStudent(hidden_dim=32)
    rgbd = torch.randint(0, 256, (2, 3, 2, 6, 48, 64), dtype=torch.uint8)
    proprio = torch.randn(2, 3, 45)
    episode = torch.tensor([[1, 1, 1], [2, 2, 2]])
    steps = torch.tensor([[0, 1, 2], [4, 5, 6]])
    relative_target = torch.zeros(2, 3, 7)
    relative_target[..., 6] = 1.0
    total, metrics, hidden = model.distillation_loss(
        rgbd_u8=rgbd,
        deployable_proprioception=proprio,
        expert_action=torch.zeros(2, 3, 7),
        expert_confidence=torch.ones(2, 3, 1),
        teacher_value=torch.zeros(2, 3, 1),
        teacher_value_valid=torch.ones(2, 3, 1, dtype=torch.bool),
        relative_pose_target=relative_target,
        contact_target=torch.zeros(2, 3, 3),
        stable_grasp_target=torch.zeros(2, 3, 1),
        slip_speed_target_m_s=torch.zeros(2, 3, 1),
        depth_validity_target=torch.ones(2, 3, 2),
        failure_target=torch.zeros(2, 3, dtype=torch.long),
        padding_mask=torch.ones(2, 3, dtype=torch.bool),
        sequence_lengths=torch.tensor([3, 3]),
        hidden_reset_mask=torch.tensor([[1, 0, 0], [1, 0, 0]], dtype=torch.bool),
        burn_in_steps=1,
        episode_id=episode,
        sequence_step=steps,
    )
    assert torch.isfinite(total)
    total.backward()
    assert model.visual.rgb_encoders[0][0].weight.grad is not None
    assert model.visual.depth_encoders[1][0].weight.grad is not None
    assert model.state_encoder[0].weight.grad is not None
    assert model.gru.weight_ih_l0.grad is not None
    assert model.arm_action_head[0].weight.grad is not None
    assert model.gripper_head[0].weight.grad is not None
    assert model.relative_pose_head[0].weight.grad is not None
    assert model.contact_head[0].weight.grad is not None
    assert model.stable_grasp_head[0].weight.grad is not None
    assert model.slip_speed_head[0].weight.grad is not None
    assert model.depth_validity_head[0].weight.grad is not None
    assert model.failure_head[0].weight.grad is not None
    assert model.cross_camera_projection[0].weight.grad is not None
    assert hidden.shape == (1, 2, 32)
    assert set(metrics) == {
        "loss/student_imitation", "loss/student_value",
        "loss/student_auxiliary", "loss/student_relative_pose",
        "loss/student_pose_consistency",
        "loss/student_cross_camera_contrastive",
        "loss/student_temporal_pose_residual",
        "loss/student_contact", "loss/student_depth_validity",
        "loss/student_stable_grasp", "loss/student_slip_speed",
        "loss/student_failure", "loss/student_total",
    }
    output, _ = model.forward_sequence(rgbd, proprio)
    assert output["arm_action"].shape == (2, 3, 6)
    assert output["gripper_action"].shape == (2, 3, 1)
    assert output["action"].shape == (2, 3, 7)
    assert not output["torso_action"].bool().any()
    broken_episode = episode.clone(); broken_episode[0, 2] = 9
    with pytest.raises(ValueError, match="episode boundary"):
        model.distillation_loss(
            rgbd_u8=rgbd,
            deployable_proprioception=proprio,
            expert_action=torch.zeros(2, 3, 7),
            expert_confidence=torch.ones(2, 3, 1),
            episode_id=broken_episode,
            sequence_step=steps,
        )


def test_student_observation_and_camera_weight_sharing_are_explicit():
    contract = G2StudentObservationContract()
    assert contract.observation_dim == 45
    independent = G2TaskRelevantVisualEncoder(share_camera_encoder_weights=False)
    shared = G2TaskRelevantVisualEncoder(share_camera_encoder_weights=True)
    assert len(independent.rgb_encoders) == len(independent.depth_encoders) == 2
    assert len(shared.rgb_encoders) == len(shared.depth_encoders) == 1


def test_cnn5_encoder_is_heavier_without_changing_visual_contract():
    baseline = G2TaskRelevantVisualEncoder(
        encoder_channels=G2_CAMERA_ENCODER_PROFILES["baseline_4layer"]
    )
    heavier = G2TaskRelevantVisualEncoder(
        encoder_channels=G2_CAMERA_ENCODER_PROFILES["cnn5_160"]
    )
    baseline_conv = sum(isinstance(module, torch.nn.Conv2d) for module in baseline.modules())
    heavier_conv = sum(isinstance(module, torch.nn.Conv2d) for module in heavier.modules())
    baseline_parameters = sum(parameter.numel() for parameter in baseline.parameters())
    heavier_parameters = sum(parameter.numel() for parameter in heavier.parameters())
    assert baseline_conv == 16
    assert heavier_conv == 20
    assert heavier_parameters > baseline_parameters
    rgbd = torch.randint(0, 256, (2, 2, 6, 48, 64), dtype=torch.uint8)
    assert baseline.encode(rgbd).shape == heavier.encode(rgbd).shape == (2, 64)
    contract = G2RecurrentVisualPolicyContract(
        camera_encoder_channels=G2_CAMERA_ENCODER_PROFILES["cnn5_160"]
    ).validated()
    assert contract.serializable()["camera_encoder_channels"] == [32, 64, 96, 128, 160]


def test_recurrent_student_padding_does_not_advance_hidden_state():
    torch.manual_seed(17)
    model = G2RecurrentVisualStudent(hidden_dim=16)
    rgbd = torch.randint(0, 256, (1, 3, 2, 6, 48, 64), dtype=torch.uint8)
    proprio = torch.randn(1, 3, 45)
    valid_output, valid_hidden = model.forward_sequence(
        rgbd[:, :2],
        proprio[:, :2],
        padding_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    padded_output, padded_hidden = model.forward_sequence(
        rgbd,
        proprio,
        padding_mask=torch.tensor([[True, True, False]]),
    )
    torch.testing.assert_close(padded_hidden, valid_hidden)
    torch.testing.assert_close(
        padded_output["action"][:, :2], valid_output["action"]
    )
