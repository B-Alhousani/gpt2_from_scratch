import torch
from gpt2_from_scratch.positional_embedding import positional_embedding


def test_positional_embedding():
    model = positional_embedding(1024, 768)

    ids = torch.tensor([
        [10, 20, 30, 40, 50],
        [60, 70, 80, 90, 100]
    ])

    output = model(ids)

    # Check shape
    assert output.shape == (5, 768)

    # Check correct position lookup
    torch.testing.assert_close(
        output, model.positional_embedding[:5]
    )

    # Check gradients
    output.sum().backward()
    assert model.positional_embedding.grad is not None