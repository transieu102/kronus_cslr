import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class GCNFeatureExtractor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_index, num_joints):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, out_channels)
        self.register_buffer('edge_index', edge_index)
        self.num_joints = num_joints

    def forward(self, x):
        # x: [B, T, J, D]
        B, T, J, D = x.shape
        x = x.view(B * T, J, D)  # [B*T, J, D]
        x = x.permute(1, 0, 2)   # [J, B*T, D]
        x = x.reshape(J, B * T, D)
        x = x.permute(1, 0, 2)   # [B*T, J, D]
        x = x.reshape(-1, D)     # [(B*T)*J, D]
        # Apply GCNConv
        x = self.gcn1(x, self.edge_index)
        x = F.gelu(x)
        x = self.gcn2(x, self.edge_index)
        x = F.gelu(x)
        x = x.view(B, T, J, -1)  # [B, T, J, out_channels]
        return x

class CSLRTransformerGCN(nn.Module):
    def __init__(
        self,
        input_dim=3,  # số chiều của mỗi joint (x, y, z)
        gcn_hidden=128,
        gcn_out=256,
        num_joints=25,
        hidden_dim=512,
        num_layers=2,
        num_heads=8,
        conv_channels=512,
        mlp_hidden=512,
        num_classes=100,
        dropout=0.1,
        max_input_len=1000,
        max_output_len=100,
        num_decoder_layers=1
    ):
        super().__init__()
        # Định nghĩa edge_index cho skeleton 25 joints (ví dụ: chuỗi liên tiếp, cần thay bằng edge_index thực tế nếu có)
        edge_index = torch.tensor([
            [i for i in range(num_joints-1)] + [i+1 for i in range(num_joints-1)],
            [i+1 for i in range(num_joints-1)] + [i for i in range(num_joints-1)]
        ], dtype=torch.long)
        self.gcn_feat = GCNFeatureExtractor(input_dim, gcn_hidden, gcn_out, edge_index, num_joints)
        # Positional embedding cho joints
        self.joint_pos_enc = nn.Parameter(torch.randn(1, num_joints, gcn_out))
        # Transformer encoder cho joint-level feature
        self.joint_encoder_layer = TransformerEncoderLayer(d_model=gcn_out, nhead=4, batch_first=True)
        self.joint_encoder = TransformerEncoder(self.joint_encoder_layer, num_layers=1)
        self.temporal_pooling = nn.Sequential(
            nn.Conv1d(gcn_out, conv_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
        )
        self.mlp = nn.Sequential(
            nn.Linear(conv_channels, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.token_embedding = nn.Embedding(num_classes, mlp_hidden)
        decoder_layer = nn.TransformerDecoderLayer(d_model=mlp_hidden, nhead=num_heads, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.out_proj = nn.Linear(mlp_hidden, num_classes)
        self.max_output_len = max_output_len
        self.pos_queries = nn.Parameter(torch.randn(1, self.max_output_len + 1, mlp_hidden))

    def _encode_memory(self, poses):
        # poses: [B, T, J, D]
        x = F.gelu(self.gcn_feat(poses))  # [B, T, J, gcn_out]
        B, T, J, F_ = x.shape
        # Thêm positional embedding cho joints
        x = x + self.joint_pos_enc[:, :J, :]  # [B, T, J, gcn_out]
        # Tổng hợp joint-level feature thành frame-level feature bằng transformer encoder
        x = x.view(B * T, J, F_)  # [B*T, J, gcn_out]
        x = self.joint_encoder(x)  # [B*T, J, gcn_out]
        x = x.mean(dim=1)         # [B*T, gcn_out] (mean pooling trên joints)
        x = x.view(B, T, F_)      # [B, T, gcn_out]
        # Tiếp tục pipeline
        x = x.permute(0, 2, 1)    # [B, gcn_out, T]
        x = self.temporal_pooling(x)  # [B, C, T']
        x = x.permute(0, 2, 1)    # [B, T', C]
        memory = self.mlp(x)      # [B, T', mlp_hidden]
        return memory

    def forward(self, tokenizer, poses, max_length=None):
        device = poses.device
        memory = self._encode_memory(poses)
        B = poses.shape[0]
        if max_length is None:
            max_length = self.max_output_len
        L = max_length + 1
        tgt_in = torch.full((B, L), tokenizer.pad_id, dtype=torch.long, device=device)
        tgt_in[:, 0] = tokenizer.bos_id
        tgt_emb = self.token_embedding(tgt_in)
        query = tgt_emb + self.pos_queries[:, :L]
        out = self.decoder(query, memory)
        logits = self.out_proj(out)
        return logits 