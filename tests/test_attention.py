import math

import pytest
import torch

from gpt2_from_scratch.config import Config
from gpt2_from_scratch.attention import attention


@pytest.fixture
def cfg():
    return Config(
        d_model=8,
        n_heads=2,
        d_head=4,
        n_ctx=16,
    )


@pytest.fixture
def model(cfg):
    torch.manual_seed(42)
    return attention(cfg)


# TEST 1: Output shape
def test_attention_output_shape(model):
    x = torch.randn(2, 5, 8)

    output = model(x)

    assert output.shape == (2, 5, 8)
    assert torch.isfinite(output).all()


# TEST 2: Causal masking
def test_causal_mask(model):
    torch.manual_seed(42)

    x = torch.randn(2, 5, 8)
    x_modified = x.clone()

    # Modify future tokens only
    x_modified[:, 3:, :] += 100.0

    original_output = model(x)
    modified_output = model(x_modified)

    # Earlier tokens must not be affected
    torch.testing.assert_close(
        original_output[:, :3, :],
        modified_output[:, :3, :],
        atol=1e-6,
        rtol=1e-6,
    )


# TEST 3: Gradient propagation
def test_attention_gradients(model):
    x = torch.randn(2, 5, 8, requires_grad=True)

    output = model(x)
    loss = output.square().sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, (
            f"No gradient for {name}"
        )
        assert torch.isfinite(parameter.grad).all(), (
            f"Invalid gradient for {name}"
        )


# TEST 4: Compare against an independent reference
def test_attention_numerical_correctness(model, cfg):
    torch.manual_seed(123)

    x = torch.randn(2, 5, cfg.d_model)

    actual = model(x)

    # Independently calculate Q, K, V
    q = torch.einsum("btd,hdf->bhtf", x, model.W_Q)
    k = torch.einsum("btd,hdf->bhtf", x, model.W_K)
    v = torch.einsum("btd,hdf->bhtf", x, model.W_V)

    q = q + model.B_Q[None, :, None, :]
    k = k + model.B_K[None, :, None, :]
    v = v + model.B_V[None, :, None, :]

    # Attention scores
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(cfg.d_head)

    # Independent causal mask
    seq_len = x.shape[1]

    mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool),
        diagonal=1,
    )

    scores = scores.masked_fill(mask, float("-inf"))

    # Softmax and weighted values
    weights = torch.softmax(scores, dim=-1)
    z = weights @ v

    # Combine heads
    z = z.transpose(1, 2).contiguous()
    z = z.reshape(x.shape[0], seq_len, cfg.d_model)

    # Output projection
    expected = z @ model.W_O + model.B_O

    torch.testing.assert_close(
        actual,
        expected,
        atol=1e-6,
        rtol=1e-5,
    )