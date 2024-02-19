import sys
sys.path.insert(0, ".")
import torch
from models import DiT

model = DiT(in_channels=3, hidden_size=128, depth=4, num_heads=8, mlp_ratio=2,
            condition_node_dim=3, condition_node_num=16, 
            cross_layers=[2,], aligned_gen=True, sibling_num=4,
            num_classes=-1)
input_x = torch.randn(4, 64, 3)
cond_x = torch.randn(4, 16, 3)
t = torch.rand(4,)

out = model(input_x, t, None, cond_x)
print(out.shape)
