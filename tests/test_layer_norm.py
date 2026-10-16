import torch
from gpt2_from_scratch.layer_norm import LayerNorm

torch.manual_seed(42)

x = torch.randn(2, 4, 768)

mine = LayerNorm(features=768)
official = torch.nn.LayerNorm(768, eps=1e-5)

my_output = mine(x)
official_output = official(x)

torch.testing.assert_close(
    my_output,
    official_output,
    rtol=1e-4,
    atol=1e-5
)

print("PASS: LayerNorm outputs match!")