import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from .modules import Decoder, DecoderLayer, TokenEmbedding
from functools import partial
from typing import Optional, Sequence
from torch import Tensor
from strhub.data.utils import GlossTokenizer as Tokenizer
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

class CSLRTransformerAutoregressivePermutate(nn.Module):
    def __init__(
        self,
        input_dim=86,
        hidden_dim=512,
        num_layers=2,
        encoder_num_heads=8,
        decoder_num_heads=12,
        conv_channels=512,
        mlp_hidden=512,
        num_classes=100,
        dropout=0.1,
        max_input_len=1000,
        max_output_len=100,
        num_decoder_layers=1,
        dec_mlp_ratio=4, 
        refine_iters=3
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=max_input_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=encoder_num_heads, dropout=dropout, batch_first=True)
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
        self.token_embedding = TokenEmbedding(num_classes, mlp_hidden)
        decoder_layer = DecoderLayer(mlp_hidden, decoder_num_heads, mlp_hidden * dec_mlp_ratio, dropout) 
        # decoder_layer = nn.TransformerDecoderLayer(d_model=mlp_hidden, nhead=num_heads, dropout=dropout, batch_first=True)
        self.decoder = Decoder(decoder_layer, num_layers=num_decoder_layers, norm=nn.LayerNorm(mlp_hidden))
        # self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.out_proj = nn.Linear(mlp_hidden, num_classes)
        self.num_classes = num_classes
        self.max_output_len = max_output_len
        self.pos_queries = nn.Parameter(torch.randn(1, self.max_output_len + 1, mlp_hidden))
        self.refine_iters = refine_iters
        self.dropout = nn.Dropout(p=dropout)
    @property
    def _device(self) -> torch.device:
        return next(self.out_proj.parameters(recurse=False)).device
    def encode(self, poses):
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
    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[Tensor] = None,
        tgt_padding_mask: Optional[Tensor] = None,
        tgt_query: Optional[Tensor] = None,
        tgt_query_mask: Optional[Tensor] = None,
    ):
        N, L = tgt.shape
        # <bos> stands for the null context. We only supply position information for characters after <bos>.
        null_ctx = self.token_embedding(tgt[:, :1])
        tgt_emb = self.pos_queries[:, : L - 1] + self.token_embedding(tgt[:, 1:])
        tgt_emb = self.dropout(torch.cat([null_ctx, tgt_emb], dim=1))
        if tgt_query is None:
            tgt_query = self.pos_queries[:, :L].expand(N, -1, -1)
        tgt_query = self.dropout(tgt_query)
        return self.decoder(tgt_query, tgt_emb, memory, tgt_query_mask, tgt_mask, tgt_padding_mask)
    
    def forward(self, tokenizer: Tokenizer, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        testing = max_length is None
        max_length = self.max_output_len if max_length is None else min(max_length, self.max_output_len)
        bs = images.shape[0]
        # +1 for <eos> at end of sequence.
        num_steps = max_length + 1
        memory = self.encode(images)

        # Query positions up to `num_steps`
        pos_queries = self.pos_queries[:, :num_steps].expand(bs, -1, -1)

        # Special case for the forward permutation. Faster than using `generate_attn_masks()`
        tgt_mask = query_mask = torch.triu(torch.ones((num_steps, num_steps), dtype=torch.bool, device=self._device), 1)


        tgt_in = torch.full((bs, num_steps), tokenizer.pad_id, dtype=torch.long, device=self._device)
        tgt_in[:, 0] = tokenizer.bos_id

        logits = []
        for i in range(num_steps):
            j = i + 1  # next token index
            # Efficient decoding:
            # Input the context up to the ith token. We use only one query (at position = i) at a time.
            # This works because of the lookahead masking effect of the canonical (forward) AR context.
            # Past tokens have no access to future tokens, hence are fixed once computed.
            tgt_out = self.decode(
                tgt_in[:, :j],
                memory,
                tgt_mask[:j, :j],
                tgt_query=pos_queries[:, i:j],
                tgt_query_mask=query_mask[i:j, :j],
            )
            # the next token probability is in the output's ith token position
            p_i = self.out_proj(tgt_out)
            logits.append(p_i)
            if j < num_steps:
                # greedy decode. add the next token index to the target input
                tgt_in[:, j] = p_i.squeeze().argmax(-1)
                # Efficient batch decoding: If all output words have at least one EOS token, end decoding.
                if testing and (tgt_in == tokenizer.eos_id).any(dim=-1).all():
                    break

        logits = torch.cat(logits, dim=1)
        if self.refine_iters:
            # For iterative refinement, we always use a 'cloze' mask.
            # We can derive it from the AR forward mask by unmasking the token context to the right.
            query_mask[torch.triu(torch.ones(num_steps, num_steps, dtype=torch.bool, device=self._device), 2)] = 0
            bos = torch.full((bs, 1), tokenizer.bos_id, dtype=torch.long, device=self._device)
            for i in range(self.refine_iters):
                # Prior context is the previous output.
                tgt_in = torch.cat([bos, logits[:, :-1].argmax(-1)], dim=1)
                # Mask tokens beyond the first EOS token.
                tgt_padding_mask = (tgt_in == tokenizer.eos_id).int().cumsum(-1) > 0
                tgt_out = self.decode(
                    tgt_in, memory, tgt_mask, tgt_padding_mask, pos_queries, query_mask[:, : tgt_in.shape[1]]
                )
                logits = self.out_proj(tgt_out)

        return logits
    
    