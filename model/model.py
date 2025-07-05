# # Path: ARN-Lipreadings/model/model.py

# from .video_cnn import VideoCNN
# import torch
# import torch.nn as nn
# from torch.cuda.amp import autocast, GradScaler
# import math
# from transformers import JambaModel, JambaConfig

# class VideoModel(nn.Module):
#     def __init__(self, args, dropout=0.2):
#         super(VideoModel, self).__init__()
#         self.args = args
#         self.video_cnn = VideoCNN(se=self.args.se)

#         cnn_out_dim = 512
#         out_dim_gru = 1024

#         # --- PERBAIKAN: Hitung dimensi input yang sudah di-padding ---
#         if self.args.border:
#             # 512 (cnn) + 1 (border) + 7 (padding) = 520. 520 adalah kelipatan 8.
#             in_dim = cnn_out_dim + 8 
#         else:
#             in_dim = cnn_out_dim

#         # Pastikan dimensi input untuk Jamba adalah kelipatan 8
#         assert in_dim % 8 == 0, f"Input dimension {in_dim} is not divisible by 8"

#         self.gru1 = nn.GRU(in_dim, out_dim_gru, 3, batch_first=True, bidirectional=True, dropout=0.2)
        
#         # Gunakan in_dim yang sudah benar untuk JambaConfig
#         self.jamba_config = JambaConfig(
#             hidden_size=in_dim,
#             intermediate_size=2 * in_dim, 
#             num_hidden_layers=1,
#             ssm_d_state=16,
#             ssm_d_conv=4,
#             ssm_expand=2,
#             vocab_size=1
#             # Tidak ada 'use_mamba_kernels=False' agar kernel cepat digunakan
#         )
        
#         self.seq_model1 = JambaModel(self.jamba_config)
        
#         # Sesuaikan ukuran layer klasifikasi terakhir
#         # Inputnya adalah: output GRU (2 * 1024) + output Jamba (in_dim)
#         self.v_cls = nn.Linear(in_dim + 2*out_dim_gru, self.args.n_class)
#         self.dropout = nn.Dropout(p=dropout)

#     def forward(self, v, border=None):
#         # f_v dari CNN memiliki dimensi [B, T, 512]
#         f_v = self.video_cnn(v)
#         f_v = self.dropout(f_v)
#         f_v = f_v.float()

#         if self.args.border:
#             # --- PERBAIKAN: Lakukan padding pada fitur 'border' ---
#             border = border[:, :, None] # Shape: [B, T, 1]
#             # Buat padding nol sebanyak 7 fitur
#             padding = torch.zeros(border.shape[0], border.shape[1], 7, device=v.device)
#             # Gabungkan: 512 + 1 + 7 = 520
#             f_v = torch.cat([f_v, border, padding], dim=-1)
#             # --- PERBAIKAN SELESAI ---
        
#         # Sekarang f_v memiliki dimensi channel 520, yang habis dibagi 8
#         jamba_outputs = self.seq_model1(inputs_embeds=f_v)
#         h1 = jamba_outputs.last_hidden_state
        
#         h_bigru, _ = self.gru1(f_v)
#         h = torch.cat([h_bigru, h1], dim=-1)
#         y_v = self.v_cls(self.dropout(h.mean(1)))
        
#         return y_v


#new script

# Path: model/model.py
# Path: model/model.py

import torch
import torch.nn as nn
from mamba_ssm import Mamba
# Impor JambaConfig untuk membuat objek konfigurasi
from transformers.models.jamba.modeling_jamba import JambaAttention, JambaMLP
from transformers import JambaConfig
from .video_cnn import VideoCNN

# =================================================================================
# == Blok Penyusun untuk ARN-Jamba
# =================================================================================

class MambaLayer(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)

    def forward(self, x):
        return self.mamba(x)

class TransformerMoELayer(nn.Module):
    # --- PERBAIKAN ---
    # Sekarang menerima objek 'config' sebagai argumen utama
    def __init__(self, config):
        super().__init__()
        # Inisialisasi JambaAttention dan JambaMlp menggunakan objek config
        # Ini adalah cara yang benar dan akan mengatasi TypeError
        self.attention = JambaAttention(config, layer_idx=0) # layer_idx bisa 0 karena ini blok yang berdiri sendiri
        self.mlp = JambaMLP(config)
        self.ln_attn = nn.LayerNorm(config.hidden_size)
        self.ln_mlp = nn.LayerNorm(config.hidden_size)

    def forward(self, x):
        # Alur forward tidak berubah
        # Koneksi residual pertama
        residual = x
        attn_output, _, _ = self.attention(x)
        x = self.ln_attn(residual + attn_output)
        
        # Koneksi residual kedua
        residual = x
        mlp_output = self.mlp(x)
        x = self.ln_mlp(residual + mlp_output)
        return x

class ARNJambaSequential(nn.Module):
    # --- PERBAIKAN ---
    # Sekarang menerima objek 'config' untuk diteruskan ke lapisan Transformer
    def __init__(self, config, num_layers=32, mamba_per_transformer=7):
        super().__init__()
        self.layers = nn.ModuleList()
        layers_per_block = mamba_per_transformer + 1
        d_model = config.hidden_size
        print(f"Membangun ARNJambaSequential: {num_layers} lapisan...")
        for i in range(num_layers):
            if i % layers_per_block == 0:
                print(f"  - Lapisan {i+1}: TransformerMoELayer")
                # Teruskan objek config ke TransformerMoELayer
                self.layers.append(TransformerMoELayer(config))
            else:
                print(f"  - Lapisan {i+1}: MambaLayer")
                self.layers.append(MambaLayer(d_model))
                
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# =================================================================================
# == BAGIAN UTAMA: VideoModel dengan Arsitektur Hybrid Paralel
# =================================================================================

class VideoModel(nn.Module):
    def __init__(self, args, dropout=0.2):
        super(VideoModel, self).__init__()
        self.args = args
        
        self.video_cnn = VideoCNN(se=self.args.se)

        cnn_feature_dim = 512

        # --- PERBAIKAN: Buat satu objek JambaConfig terpusat ---
        jamba_config = JambaConfig(
            hidden_size=cnn_feature_dim,
            num_attention_heads=8,
            num_hidden_layers=32, # Meskipun tidak digunakan langsung, ini adalah bagian dari config
            # Parameter untuk MoE di dalam JambaMlp
            num_experts=16,
            num_experts_per_tok=2,
            intermediate_size=cnn_feature_dim * 2,
            # Parameter lain yang mungkin diperlukan
            vocab_size=1, # Dummy
            rms_norm_eps=1e-5
        )

        # --- JALUR PARALEL 1: Modul ARN-Jamba ---
        # Teruskan objek jamba_config ke ARNJambaSequential
        self.jamba_backbone = ARNJambaSequential(
            config=jamba_config,
            num_layers=32,
            mamba_per_transformer=7
        )
        
        # --- JALUR PARALEL 2: Modul Bi-GRU ---
        gru_hidden_dim = 512
        self.gru_backbone = nn.GRU(
            cnn_feature_dim, 
            gru_hidden_dim, 
            num_layers=3,
            batch_first=True, 
            bidirectional=True, 
            dropout=0.2
        )
        
        # TAHAP AKHIR: Penggabungan dan Klasifikasi
        combined_dim = cnn_feature_dim + (2 * gru_hidden_dim)
        self.v_cls = nn.Linear(combined_dim, self.args.n_class)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, v, border=None):
        # Alur forward tidak berubah
        f_v = self.video_cnn(v)
        f_v = self.dropout(f_v)
        f_v = f_v.float()

        jamba_out = self.jamba_backbone(f_v)
        h_jamba = torch.mean(jamba_out, dim=1)

        gru_out, _ = self.gru_backbone(f_v)
        h_gru = torch.mean(gru_out, dim=1)

        h_combined = torch.cat([h_jamba, h_gru], dim=-1)
        
        y_v = self.v_cls(self.dropout(h_combined))
        
        return y_v