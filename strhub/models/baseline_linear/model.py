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

class PositionExpert(nn.Module):
    def __init__(self, mlp_hidden, position_idx, max_len, dropout=0.2):
        super().__init__()
        self.position_idx = position_idx
        self.attention = nn.MultiheadAttention(mlp_hidden, num_heads=1, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(mlp_hidden)
        self.linear = nn.Linear(mlp_hidden, mlp_hidden)
        
        # Position-specific positional encoding
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1, mlp_hidden))
        nn.init.normal_(self.pos_embedding, std=0.02)
        
    def forward(self, memory, return_attention=False):
        # Add position-specific bias
        memory = memory + self.pos_embedding
        
        # Self-attention with position-specific focus
        context, attn_weights = self.attention(memory, memory, memory)
        context = context.mean(dim=1)  # [B, D]
        
        # Regularization
        context = self.dropout(context)
        context = self.norm(context)
        
        # Position-specific linear transformation
        context = self.linear(context)
        
        if return_attention:
            return context, attn_weights
        return context

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
        max_input_len=1000,
        max_output_len=100,
        num_decoder_layers=1,
        position_dropout=0.2,  # Dropout cho position experts
        position_l2=0.01,     # L2 regularization cho position experts
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
        self.num_classes = num_classes
        
        # Position experts với regularization
        self.position_experts = nn.ModuleList([
            PositionExpert(mlp_hidden, i, max_output_len + 1, position_dropout)
            for i in range(max_output_len + 1)
        ])
        
        # Shared final head với layer norm
        self.final_norm = nn.LayerNorm(mlp_hidden)
        self.final_head = nn.Linear(mlp_hidden, num_classes)
        
        self.position_l2 = position_l2

    def _encode_memory(self, poses):
        B, T, J, D = poses.shape
        x = poses.view(B, T, J * D)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        x = self.temporal_pooling(x)
        x = x.permute(0, 2, 1)
        memory = self.mlp(x)  # [B, T', D]
        return memory

    def forward(self, tokenizer, poses, max_length=None, return_attention=False):
        device = poses.device
        memory = self._encode_memory(poses)  # [B, T', D]
        B = poses.shape[0]
        if max_length is None:
            max_length = self.max_output_len
        max_length = max_length + 1  # +1 cho <eos>
        
        logits = []
        attention_maps = [] if return_attention else None
        
        for expert in self.position_experts[:max_length]:
            # Position-specific attention và linear
            if return_attention:
                context, attn_weights = expert(memory, return_attention=True)
                attention_maps.append(attn_weights)
            else:
                context = expert(memory)
                
            # Normalize trước khi vào final head
            context = self.final_norm(context)
            # Shared final head
            logit = self.final_head(context)
            logits.append(logit)
            
        logits = torch.stack(logits, dim=1)  # [B, max_output_len, num_classes]
        
        if return_attention:
            attention_maps = torch.stack(attention_maps, dim=1)  # [B, max_output_len, T', T']
            return logits, attention_maps
        return logits

   
