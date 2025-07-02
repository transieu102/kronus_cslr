import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class CustomDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        # Cross attention
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Feed forward
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.activation = nn.GELU()

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Cross attention
        tgt2 = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Feed forward
        tgt2 = self.linear2(self.activation(self.linear1(tgt)))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        return tgt

class CustomTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask, memory_mask)
        return output

class CSLRTransformerBaseline(nn.Module):
    def __init__(
        self,
        input_dim=86,
        hidden_dim=512,
        num_layers=2,
        num_heads=8,
        conv_channels=512,
        mlp_hidden=512,
        num_classes=100,
        dropout=0.1,
        max_input_len=1000,   # Độ dài sequence đầu vào (encoder)
        max_output_len=100,   # Độ dài sequence đầu ra (decode/inference)
        num_decoder_layers=1
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=max_input_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.temporal_pooling = nn.Sequential(
            nn.Conv1d(hidden_dim, conv_channels, kernel_size=3, padding=1),
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
        self.max_output_len = max_output_len
        # Learnable positional embedding cho decoder query
        self.pos_queries = nn.Parameter(torch.randn(1, self.max_output_len + 1, mlp_hidden))
        decoder_layer = CustomDecoderLayer(d_model=mlp_hidden, nhead=num_heads, dropout=dropout)
        self.decoder = CustomTransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.out_proj = nn.Linear(mlp_hidden, num_classes)
        self.num_classes = num_classes

    def _encode_memory(self, poses):
        B, T, J, D = poses.shape
        x = poses.view(B, T, J * D)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        x = self.temporal_pooling(x)
        x = x.permute(0, 2, 1)
        memory = self.mlp(x)
        return memory

    def forward(self, tokenizer, poses, max_length=None):
        device = poses.device
        memory = self._encode_memory(poses)
        B = poses.shape[0]
        if max_length is None:
            max_length = self.max_output_len
        L = max_length + 1  # +1 cho <eos>

        # Chỉ sử dụng positional embedding cho query
        query = self.pos_queries[:, :L].expand(B, -1, -1)  # [B, L, D]
        out = self.decoder(query, memory)
        logits = self.out_proj(out)  # [B, L, C]
        return logits
