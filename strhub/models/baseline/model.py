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
        self.token_embedding = nn.Embedding(num_classes, mlp_hidden)
        decoder_layer = nn.TransformerDecoderLayer(d_model=mlp_hidden, nhead=num_heads, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.out_proj = nn.Linear(mlp_hidden, num_classes)
        self.num_classes = num_classes
        self.max_output_len = max_output_len
        # Learnable positional embedding cho decoder query (giống PARSeq)
        self.pos_queries = nn.Parameter(torch.randn(1, self.max_output_len + 1, mlp_hidden))

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

        # Tạo input cho decoder: BOS + PAD cho toàn bộ sequence
        tgt_in = torch.full((B, L), tokenizer.pad_id, dtype=torch.long, device=device)
        tgt_in[:, 0] = tokenizer.bos_id
        tgt_emb = self.token_embedding(tgt_in)  # [B, L, D]
        # Cộng positional embedding vào token embedding
        query = tgt_emb + self.pos_queries[:, :L]
        out = self.decoder(query, memory)
        logits = self.out_proj(out)  # [B, L, C]
        return logits
