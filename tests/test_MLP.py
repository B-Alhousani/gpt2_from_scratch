
import pytest
import torch
import torch.nn.functional as F

from gpt2_from_scratch.config import Config
from gpt2_from_scratch.mlp import MLP


@pytest.fixture
def model():
    torch.manual_seed(42)

    cfg = Config(
        d_model=8,
        n_ctx=16,
        n_heads=2,
        d_head=4,
    )

    return MLP(cfg)


# TEST 1: Output shape
@pytest.mark.parametrize("batch,tokens", [
    (1, 1),
    (2, 5),
    (3, 10),
])
def test_output_shape(model, batch, tokens):
    x = torch.randn(batch, tokens, 8)

    output = model(x)

    assert output.shape == (batch, tokens, 8)
    assert torch.isfinite(output).all()


# TEST 2: Correct mathematical calculation
def test_numerical_correctness(model):
    x = torch.randn(2, 5, 8)

    actual = model(x)

    # Independent reference using matrix multiplication
    hidden = x @ model.W_1 + model.B_1
    activated = F.gelu(hidden, approximate="tanh")
    expected = activated @ model.W_2 + model.B_2

    torch.testing.assert_close(
        actual,
        expected,
        atol=1e-6,
        rtol=1e-5,
    )


# TEST 3: Tokens are processed independently
def test_token_independence(model):
    x = torch.randn(2, 5, 8)
    modified = x.clone()

    # Change only token 3 in batch 0
    modified[0, 3, :] += 10

    original_output = model(x)
    modified_output = model(modified)

    # All other tokens must remain unchanged
    torch.testing.assert_close(
        original_output[0, :3, :],
        modified_output[0, :3, :],
    )

    torch.testing.assert_close(
        original_output[0, 4:, :],
        modified_output[0, 4:, :],
    )

    torch.testing.assert_close(
        original_output[1],
        modified_output[1],
    )

    # The changed token should produce a different result
    assert not torch.allclose(
        original_output[0, 3],
        modified_output[0, 3],
    )


# TEST 4: All parameters are registered
def test_parameters(model):
    parameters = dict(model.named_parameters())

    assert set(parameters.keys()) == {
        "W_1", "B_1", "W_2", "B_2"
    }

    assert model.W_1.shape == (8, 32)
    assert model.B_1.shape == (32,)
    assert model.W_2.shape == (32, 8)
    assert model.B_2.shape == (8,)


# TEST 5: Backpropagation works
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
