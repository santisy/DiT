import sys
sys.path.insert(0, ".")
import torch
from models import Attention
from torch.cuda.amp import autocast


input_var = torch.randn(4, 4096, 128).cuda()
attn_model = Attention(128, 16).cuda()
with autocast():
    out = attn_model(input_var)
print(out.shape)
