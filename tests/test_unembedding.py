
import pytest
import torch

from gpt2_from_scratch.config import Config
from gpt2_from_scratch.embedding import embedding
from gpt2_from_scratch.unembedding import unembedding


@pytest.fixture
def modules():
    torch.manual_seed(42)

    cfg = Config(
        d_model=8,
        d_vocab=17,
        n_ctx=16,
        n_heads=2,
        d_head=4,
    )

    token_embedding = embedding(cfg)
    decoder = unembedding(token_embedding)

    return cfg, token_embedding, decoder


# TEST 1: Output shape
@pytest.mark.parametrize("batch,tokens", [
    (1, 1),
    (2, 5),
    (3, 10),
])
def test_output_shape(modules, batch, tokens):
    cfg, token_embedding, decoder = modules

    x = torch.randn(batch, tokens, cfg.d_model)

    logits = decoder(x)

    assert logits.shape == (batch, tokens, cfg.d_vocab)
    assert torch.isfinite(logits).all()


# TEST 2: Mathematical correctness
def test_numerical_correctness(modules):
    cfg, token_embedding, decoder = modules

    x = torch.randn(2, 5, cfg.d_model)

    actual = decoder(x)

    # Independent reference: dot product with each
    # vocabulary token's embedding vector.
    weights = token_embedding.embedding_matrix

    expected = torch.stack(
        [
            (x * weights[i]).sum(dim=-1)
            for i in range(cfg.d_vocab)
        ],
        dim=-1,
    )

    torch.testing.assert_close(
        actual,
        expected,
        atol=1e-6,
        rtol=1e-5,
    )


# TEST 3: Same embedding object is reused
def test_weight_tying(modules):
    cfg, token_embedding, decoder = modules

    assert decoder.unmbedding is token_embedding

    assert (
        decoder.unmbedding.embedding_matrix
        is token_embedding.embedding_matrix
    )

    # No independent unembedding weight matrix
    parameters = list(decoder.parameters())

    assert len(parameters) == 1
    assert parameters[0] is token_embedding.embedding_matrix


# TEST 4: Changes to embedding weights affect logits
def test_shared_weight_updates(modules):
    cfg, token_embedding, decoder = modules

    x = torch.ones(2, 3, cfg.d_model)

    before = decoder(x).detach().clone()

    with torch.no_grad():
        token_embedding.embedding_matrix[0].add_(1.0)

    after = decoder(x)

    # Only vocabulary token 0's score should change
    torch.testing.assert_close(
        after[..., 0] - before[..., 0],
        torch.full((2, 3), float(cfg.d_model)),
        atol=1e-6,
        rtol=1e-6,
    )

    torch.testing.assert_close(
        after[..., 1:],
        before[..., 1:],
    )


# TEST 5: Gradients propagate to shared weights
def test_gradients(modules):
    cfg, token_embedding, decoder = modules

    x = torch.randn(
        2, 5, cfg.d_model,
        requires_grad=True,
    )

    logits = decoder(x)

    loss = logits.square().sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    weight = token_embedding.embedding_matrix

    assert weight.grad is not None
    assert torch.isfinite(weight.grad).all()
    assert torch.count_nonzero(weight.grad) > 0
