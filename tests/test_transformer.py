
import pytest
import torch

from gpt2_from_scratch.config import Config
from gpt2_from_scratch.transformer import Transformer


@pytest.fixture
def model():
    torch.manual_seed(42)

    cfg = Config(
        d_model=8,
        n_heads=2,
        d_head=4,
        n_ctx=16,
    )

    return Transformer(cfg)


# TEST 1: Output shape
@pytest.mark.parametrize("batch,tokens", [
    (1, 1),
    (2, 5),
    (3, 10),
])
def test_output_shape(model, batch, tokens):
    x = torch.randn(batch, tokens, 8)

    output = model(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


# TEST 2: Correct Pre-LN and residual calculations
def test_transformer_correctness(model):
    x = torch.randn(2, 5, 8)

    actual = model(x)

    # Independent calculation of the block
    normalized_1 = model.LayerNorm_a(x)
    attention_output = model.attention(normalized_1)

    after_attention = x + attention_output

    normalized_2 = model.LayerNorm_b(after_attention)
    mlp_output = model.MLP(normalized_2)

    expected = after_attention + mlp_output

    torch.testing.assert_close(
        actual,
        expected,
        atol=1e-6,
        rtol=1e-5,
    )


# TEST 3: Residual stream is preserved
def test_residual_connections(model):
    x = torch.randn(2, 5, 8)

    # Make Attention and MLP produce zero outputs
    with torch.no_grad():
        for parameter in model.attention.parameters():
            parameter.zero_()

        for parameter in model.MLP.parameters():
            parameter.zero_()

    output = model(x)

    # With both branches zero, output must equal input
    torch.testing.assert_close(
        output,
        x,
        atol=1e-6,
        rtol=1e-6,
    )


# TEST 4: Future tokens cannot affect earlier tokens
def test_causal_behavior(model):
    torch.manual_seed(123)

    x = torch.randn(2, 6, 8)
    modified = x.clone()

    # Change future tokens
    modified[:, 3:, :] += 100.0

    original_output = model(x)
    modified_output = model(modified)

    # Earlier tokens should stay unchanged
    torch.testing.assert_close(
        original_output[:, :3, :],
        modified_output[:, :3, :],
        atol=1e-6,
        rtol=1e-5,
    )


# TEST 5: Gradients propagate through the block
def test_gradients(model):
    x = torch.randn(2, 5, 8, requires_grad=True)

    output = model(x)
    loss = output.square().sum()

    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, (
            f"Missing gradient: {name}"
        )

        assert torch.isfinite(parameter.grad).all(), (
            f"Invalid gradient: {name}"
        )


# TEST 6: LayerNorm modules are independent
def test_independent_layernorms(model):
    assert model.LayerNorm_a is not model.LayerNorm_b

    params_a = list(model.LayerNorm_a.parameters())
    params_b = list(model.LayerNorm_b.parameters())

    assert len(params_a) == len(params_b)
    assert len(params_a) > 0

    for a, b in zip(params_a, params_b):
        assert a is not b
        assert a.data_ptr() != b.data_ptr()
