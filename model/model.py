# Path: ARN-Lipreadings/model/model.py

from .video_cnn import VideoCNN
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import math
from transformers import JambaModel, JambaConfig

class VideoModel(nn.Module):
    def __init__(self, args, dropout=0.2):
        super(VideoModel, self).__init__()
        self.args = args
        self.video_cnn = VideoCNN(se=self.args.se)

        cnn_out_dim = 512
        out_dim_gru = 1024

        # --- PERBAIKAN: Hitung dimensi input yang sudah di-padding ---
        if self.args.border:
            # 512 (cnn) + 1 (border) + 7 (padding) = 520. 520 adalah kelipatan 8.
            in_dim = cnn_out_dim + 8 
        else:
            in_dim = cnn_out_dim

        # Pastikan dimensi input untuk Jamba adalah kelipatan 8
        assert in_dim % 8 == 0, f"Input dimension {in_dim} is not divisible by 8"

        self.gru1 = nn.GRU(in_dim, out_dim_gru, 3, batch_first=True, bidirectional=True, dropout=0.2)
        
        # Gunakan in_dim yang sudah benar untuk JambaConfig
        self.jamba_config = JambaConfig(
            hidden_size=in_dim,
            intermediate_size=2 * in_dim, 
            num_hidden_layers=1,
            ssm_d_state=16,
            ssm_d_conv=4,
            ssm_expand=2,
            vocab_size=1
            # Tidak ada 'use_mamba_kernels=False' agar kernel cepat digunakan
        )
        
        self.seq_model1 = JambaModel(self.jamba_config)
        
        # Sesuaikan ukuran layer klasifikasi terakhir
        # Inputnya adalah: output GRU (2 * 1024) + output Jamba (in_dim)
        self.v_cls = nn.Linear(in_dim + 2*out_dim_gru, self.args.n_class)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, v, border=None):
        # f_v dari CNN memiliki dimensi [B, T, 512]
        f_v = self.video_cnn(v)
        f_v = self.dropout(f_v)
        f_v = f_v.float()

        if self.args.border:
            # --- PERBAIKAN: Lakukan padding pada fitur 'border' ---
            border = border[:, :, None] # Shape: [B, T, 1]
            # Buat padding nol sebanyak 7 fitur
            padding = torch.zeros(border.shape[0], border.shape[1], 7, device=v.device)
            # Gabungkan: 512 + 1 + 7 = 520
            f_v = torch.cat([f_v, border, padding], dim=-1)
            # --- PERBAIKAN SELESAI ---
        
        # Sekarang f_v memiliki dimensi channel 520, yang habis dibagi 8
        jamba_outputs = self.seq_model1(inputs_embeds=f_v)
        h1 = jamba_outputs.last_hidden_state
        
        h_bigru, _ = self.gru1(f_v)
        h = torch.cat([h_bigru, h1], dim=-1)
        y_v = self.v_cls(self.dropout(h.mean(1)))
        
        return y_v