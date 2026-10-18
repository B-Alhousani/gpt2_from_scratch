import torch
import torch.nn as nn

from gpt2_from_scratch.embedding import embedding


def test_embedding():
    torch.manual_seed(42)

    mine = embedding(100, 768)
    official = nn.Embedding(100, 768)

    with torch.no_grad():
        official.weight.copy_(mine.embedding_matrix)

    ids = torch.tensor([[1, 5, 1], [3, 7, 2]])

    my_output = mine(ids)
    official_output = official(ids)

    assert my_output.shape == (2, 3, 768)
    torch.testing.assert_close(my_output, official_output)

    my_output.sum().backward()
    assert mine.embedding_matrix.grad is not None