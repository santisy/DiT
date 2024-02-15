import sys
sys.path.insert(0, ".")
import torch
from models import PreviousNodeEmbedder

node = torch.randn(4, 64, 32)
PNE = PreviousNodeEmbedder(64, 32, 128)
out = PNE(node)

print(out.shape)
