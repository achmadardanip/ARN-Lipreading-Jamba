from .video_cnn import VideoCNN
import torch
import torch.nn as nn
import random
from torch.cuda.amp import autocast, GradScaler
import math
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from fightingcv_attention.attention.SEAttention import SEAttention

import torch
from mamba_ssm import Mamba

batch, length, dim = 2, 64, 16
seq_model = Mamba(
    d_model=dim, # Model dimension d_model
    d_state=16,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
).to("cuda")

class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5,
                 device: str ='cuda'):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output

class VideoModel(nn.Module):
    def __init__(self, args, dropout=0.2):
        super(VideoModel, self).__init__()
        self.args = args
        self.video_cnn = VideoCNN(se=self.args.se)

        in_dim = 512 + 1 if self.args.border else 512
        in_dim_transformer = 512
        out_dim_gru = 1024
        self.gru1 = nn.GRU(in_dim, out_dim_gru, 3, batch_first=True, bidirectional=True, dropout=0.2)        
        self.v_cls = nn.Linear(in_dim+2*out_dim_gru, self.args.n_class)
        self.dropout = nn.Dropout(p=dropout)
        self.seq_model1 = Mamba(d_model=in_dim, d_state=16, d_conv=4, expand=2)

        
    def forward(self, v, border=None):
        f_v = self.video_cnn(v)
        f_v = self.dropout(f_v)
        f_v = f_v.float()

        if self.args.border:
            border = border[:, :, None]
            f_v = torch.cat([f_v, border], -1)
            
        h1 = self.seq_model1(f_v)
        h_bigru, _ = self.gru1(f_v)
        h = torch.cat([h_bigru,h1], dim = -1)
        y_v = self.v_cls(self.dropout(h.mean(1)))

        return y_v

