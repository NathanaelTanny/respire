"""Encoder, verifier, flow head and the analytic pressure model."""

from __future__ import annotations

import torch

from respire.models.attention import CrossChannelAttention
from respire.models.deterministic_head import DeterministicHead
from respire.models.encoder import EncoderStandardiser, LatentEncoder
from respire.models.flow import FlowPolicy
from respire.models.pressure import OneStepPressureModel, state_features
from respire.models.ssm import SelectiveSSMBlock, SelectiveStack
from respire.models.verifier import ReturnVerifier
from respire.settings import EncoderConfig, PolicyConfig, VerifierConfig


def _window(batch: int, steps: int = 12, channels: int = 12) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    return torch.randn(batch, steps, channels, generator=generator)


def test_encoder_shapes_follow_the_figure() -> None:
    cfg = EncoderConfig()
    model = LatentEncoder(cfg)
    output = model(_window(5), torch.zeros(5, cfg.window_steps, cfg.action_channels))
    assert output.latent.shape == (5, 16)
    assert output.axes.shape == (5, 3)
    assert output.hidden.shape == (5, 32)
    assert output.reconstruction.shape == (5, 12)
    assert output.sparsity.shape == (5,)


def test_encoder_has_six_selective_blocks() -> None:
    model = LatentEncoder(EncoderConfig())
    assert isinstance(model.stack, SelectiveStack)
    assert model.stack.n_blocks == 6
    assert all(isinstance(block, SelectiveSSMBlock) for block in model.stack.blocks)


def test_sparse_readout_uses_the_requested_number_of_atoms() -> None:
    cfg = EncoderConfig(sparse_topk=3)
    model = LatentEncoder(cfg)
    output = model(_window(4), torch.zeros(4, cfg.window_steps, 2))
    expected = 3 / 32
    assert abs(float(output.sparsity.mean()) - expected) < 1e-6


def test_cross_channel_attention_returns_channel_weights() -> None:
    attention = CrossChannelAttention(12, 12, 32, 4)
    pooled, weights = attention(_window(6))
    assert pooled.shape == (6, 32)
    assert weights.shape == (6, 12)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(6), atol=1e-5)


def test_encoder_gradients_reach_every_block() -> None:
    cfg = EncoderConfig()
    model = LatentEncoder(cfg)
    output = model(_window(3), torch.zeros(3, cfg.window_steps, 2))
    output.latent.sum().backward()
    touched = [block for block in model.stack.blocks if block.in_proj.weight.grad is not None]
    assert len(touched) == 6


def test_verifier_parameter_count_is_the_reported_order() -> None:
    cfg = VerifierConfig()
    model = ReturnVerifier(cfg, 54)
    count = model.parameter_count()
    relative = abs(count - cfg.target_params) / cfg.target_params
    assert relative <= cfg.param_tolerance, count


def test_verifier_returns_a_positive_variance_head() -> None:
    model = ReturnVerifier(VerifierConfig(hidden=(64, 32), feature_dim=32), 54)
    features = torch.randn(7, 54)
    mean, log_variance = model(features)
    assert mean.shape == (7,)
    assert bool(torch.isfinite(log_variance).all())
    assert model.predict(features).shape == (7,)


def test_flow_sampling_density_is_consistent() -> None:
    cfg = PolicyConfig(hidden=32, depth=2, time_features=8)
    policy = FlowPolicy(cfg, 48)
    policy.configure_action_scale(12.0, 0.5)
    condition = torch.randn(24, 48)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(2)
    sample = policy.sample(condition, generator)
    explicit = policy.log_prob(sample.action, condition)
    assert torch.allclose(sample.log_prob, explicit, atol=0.6)


def test_flow_transports_on_the_pump_box() -> None:
    cfg = PolicyConfig(hidden=32, depth=2, time_features=8, euler_steps=4)
    policy = FlowPolicy(cfg, 48)
    policy.configure_action_scale(12.0, 0.5)
    unit = torch.rand(16, 2)
    raw = policy.to_raw(unit)
    assert torch.allclose(raw[:, 0], 12.0 * unit[:, 0], atol=1e-5)
    assert torch.allclose(policy.to_unit(raw), unit, atol=1e-6)
    sample = policy.sample(torch.randn(32, 48))
    assert bool(torch.isfinite(sample.action).all())
    assert sample.action.shape == (32, 2)


def test_pathwise_sampling_is_differentiable() -> None:
    policy = FlowPolicy(PolicyConfig(hidden=16, depth=2, time_features=8), 48)
    policy.configure_action_scale(12.0, 0.5)
    action = policy.sample_pathwise(torch.randn(4, 48))
    action.sum().backward()
    assert policy.field.net[0].weight.grad is not None


def test_deterministic_head_matches_the_flow_interface() -> None:
    head = DeterministicHead(PolicyConfig(hidden=16, depth=2), 48)
    condition = torch.randn(3, 48)
    sample = head.sample(condition)
    assert sample.action.shape == (3, 2)
    assert float(head.log_prob(sample.action, condition).abs().sum()) == 0.0


def test_pressure_model_recovers_a_linear_surface() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(3)
    state = torch.randn(400, 5, generator=generator) * 3.0
    action = torch.rand(400, 2, generator=generator)
    target = 80.0 + 1.5 * state[:, 0] - 2.0 * action[:, 0] - 4.0 * action[:, 1]
    model = OneStepPressureModel(ridge=1e-6)
    model.fit(state, action, target)
    predicted = model.predict(state, action)
    assert float((predicted - target).abs().max()) < 1e-2
    assert model.lipschitz > 4.0


def test_state_features_extract_slopes() -> None:
    window = torch.zeros(2, 12, 12)
    window[:, :-1, 2] = 70.0
    window[:, -1, 2] = 74.0
    features = state_features(window)
    assert features.shape == (2, 5)
    assert float(features[0, 3]) == 4.0


def test_standardiser_round_trip() -> None:
    standardiser = EncoderStandardiser(12)
    windows = _window(8)
    standardiser.fit(windows)
    restored = standardiser.inverse(standardiser.transform(windows))
    assert torch.allclose(restored, windows, atol=1e-5)
