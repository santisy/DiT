import sys
sys.path.insert(0, ".")
import torch
from models import DiT
import gc

# Testing level 0

model = DiT(in_channels=64,
            hidden_size=256,
            depth=4,
            num_heads=8,
            mlp_ratio=2,
            condition_node_dim=[],
            condition_node_num=[], 
            cross_layers=[2,3],
            aligned_gen=False,
            add_inject=False,
            num_classes=-1)

input_x = torch.randn(4, 16, 64)
cond_x = []
pos_x = []
t = torch.rand(4,)
out = model(input_x, t, None, cond_x, pos_x)
print("level 0 test passed (not aligned, no cond), output shape is", out.shape)

del model
gc.collect()

model = DiT(in_channels=64,
            hidden_size=256,
            depth=4,
            num_heads=8,
            mlp_ratio=2,
            condition_node_dim=[32],
            condition_node_num=[2,], 
            cross_layers=[2,3],
            aligned_gen=True,
            add_inject=False,
            num_classes=-1)
input_x = torch.randn(4, 16, 64)
cond_x = [torch.randn(4, 2, 32),]
pos_x = [torch.randn(4, 2, 3),]
t = torch.rand(4,)
out = model(input_x, t, None, cond_x, pos_x)
print("level 1 test passed (aligned, 1 cond), output shape is", out.shape)


del model
gc.collect()

model = DiT(in_channels=64,
            hidden_size=256,
            depth=4,
            num_heads=8,
            mlp_ratio=2,
            condition_node_dim=[32, 64],
            condition_node_num=[2, 4], 
            cross_layers=[2,3],
            aligned_gen=True,
            add_inject=False,
            num_classes=-1)
input_x = torch.randn(4, 32, 64)
cond_x = [torch.randn(4, 2, 32), torch.randn(4, 4, 64)]
pos_x = [torch.randn(4, 2, 3), torch.randn(4, 4, 3)]
t = torch.rand(4,)
out = model(input_x, t, None, cond_x, pos_x)
print("level 2 test passed (aligned, 2 cond), output shape is", out.shape)
