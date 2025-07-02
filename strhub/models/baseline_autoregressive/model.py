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

class CSLRTransformerAutoregressive(nn.Module):
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

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, tokenizer, poses, max_length=None):
        device = poses.device
        memory = self._encode_memory(poses)
        B = poses.shape[0]
        if max_length is None:
            max_length = self.max_output_len
        L = max_length + 1  # +1 for <eos>

        # Initialize with BOS token
        tgt_in = torch.full((B, 1), tokenizer.bos_id, dtype=torch.long, device=device)
        outputs = []
        
        # Generate tokens autoregressively
        for i in range(L):
            # Create mask for autoregressive decoding
            tgt_mask = self._generate_square_subsequent_mask(tgt_in.size(1)).to(device)
            
            # Get embeddings for current sequence
            tgt_emb = self.token_embedding(tgt_in)
            query = tgt_emb + self.pos_queries[:, :tgt_in.size(1)]
            
            # Decode
            out = self.decoder(query, memory, tgt_mask=tgt_mask)
            logits = self.out_proj(out[:, -1:])  # Only take the last token's prediction
            outputs.append(logits)
            
            # Get the predicted token
            pred = logits.argmax(dim=-1)
            
            # Stop if we predict EOS
            if (pred == tokenizer.eos_id).any():
                break
                
            # Append predicted token to input sequence
            tgt_in = torch.cat([tgt_in, pred], dim=1)
        
        # Concatenate all outputs
        logits = torch.cat(outputs, dim=1)
        return logits 