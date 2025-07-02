# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Optional

import torch
from torch import Tensor, nn as nn
from torch.nn import functional as F
from torch.nn.modules import transformer
import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomMultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, memory_mask=None):
        """
        query: [B, T_q, D]
        key, value: [B, T_k, D]
        memory_mask: [B, T_k] or [B, 1, T_k] or [B, T_q, T_k] (True = mask)
        """
        B, T_q, D = query.size()
        T_k = key.size(1)

        # Linear projection
        Q = self.q_proj(query)  # [B, T_q, D]
        K = self.k_proj(key)    # [B, T_k, D]
        V = self.v_proj(value)  # [B, T_k, D]

        # Reshape to multihead: [B, H, T, d_k]
        Q = Q.view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores: [B, H, T_q, T_k]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if memory_mask is not None:
            if memory_mask.dim() == 2:
                memory_mask = memory_mask[:, None, None, :]  # [B, 1, 1, T_k]
            elif memory_mask.dim() == 3:
                memory_mask = memory_mask[:, None, :, :]     # [B, 1, T_q, T_k]
            # mask = True nghĩa là bị chặn (giống nn.MultiheadAttention)
            attn_scores = attn_scores.masked_fill(memory_mask, float('-inf'))

            # Ensure at least one unmasked element per row
            all_masked = torch.all(memory_mask, dim=-1, keepdim=True)  # [B, 1, 1, T_k]
            attn_scores = attn_scores.masked_fill(all_masked, -1e9)


        # Softmax attention
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, H, T_q, T_k]
        attn_weights = self.dropout(attn_weights)

        # Attention output
        attn_output = torch.matmul(attn_weights, V)  # [B, H, T_q, d_k]
        # print(attn_output.shape, attn_weights.shape, V.shape)
        # input()
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T_q, D)  # [B, T_q, D]

        # Output projection
        output = self.out_proj(attn_output)

        return output, attn_weights  # [B, T_q, D], [B, H, T_q, T_k]


class DecoderLayer(nn.Module):
    """A Transformer decoder layer supporting two-stream attention (XLNet)
    This implements a pre-LN decoder, as opposed to the post-LN default in PyTorch."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='gelu', layer_norm_eps=1e-5):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # self.cross_attn = CustomMultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_q = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_c = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = transformer._get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.gelu
        super().__setstate__(state)

    def forward_stream(
        self,
        tgt: Tensor,
        tgt_norm: Tensor,
        tgt_kv: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor],
        tgt_key_padding_mask: Optional[Tensor],
    ):
        """Forward pass for a single stream (i.e. content or query)
        tgt_norm is just a LayerNorm'd tgt. Added as a separate parameter for efficiency.
        Both tgt_kv and memory are expected to be LayerNorm'd too.
        memory is LayerNorm'd by ViT.
        """
        # tgt2, sa_weights = self.self_attn(
        #     tgt_norm, tgt_kv, tgt_kv, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        # )
        # tgt = tgt + self.dropout1(tgt2)

        tgt2, ca_weights = self.cross_attn(self.norm1(tgt), memory, memory)
        tgt = self.dropout2(tgt2)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(tgt)))))
        tgt = tgt + self.dropout3(tgt2)
        # B, T, D = tgt.shape
        # S = memory.size(1)
        # threshold = 1.0 / S

        # used_mask = torch.zeros(B, S, device=tgt.device)  # [B, S]
        # outputs = []
        # all_ca_weights = []

        # for t in range(T):
        #     tgt_step = tgt[:, t:t+1, :]  # [B, 1, D]
        #     tgt_step_norm = self.norm1(tgt_step)

        #     # memory_mask: True tại vị trí cần mask
        #     memory_mask = used_mask > threshold  # [B, S]

        #     # Gọi attention có mask
        #     tgt2_step, ca_weights = self.cross_attn(
        #         query=tgt_step_norm,
        #         key=memory,
        #         value=memory,
        #         memory_mask=memory_mask  # [B, S]
        #     )  # tgt2_step: [B, 1, D], ca_weights: [B, num_heads, 1, S]

        #     # Lấy attention trung bình giữa các head
        #     ca_weights_mean = ca_weights.mean(dim=1).squeeze(1)  # [B, S]
        #     # print("ca_weights_mean", ca_weights_mean)
        #     # print("tgt2_step", tgt2_step.shape)
        #     # input()
        #     all_ca_weights.append(ca_weights_mean)
        #     # input()
        #     # Cập nhật used_mask dựa trên ca_weights vượt threshold
        #     used_mask = torch.max(used_mask, (ca_weights_mean > threshold).float())  # [B, S]

        #     # FFN
        #     tgt2_step = self.dropout2(tgt2_step)
        #     tgt_ff = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(tgt2_step)))))
        #     tgt_out = tgt2_step + self.dropout3(tgt_ff)

        #     outputs.append(tgt_out)  # [B, 1, D]

        # # Kết quả: [B, T, D]
        # tgt = torch.cat(outputs, dim=1)

        # ca_weights = torch.stack(all_ca_weights, dim=1)

        # Plot heatmap of cross-attention weights
        # import matplotlib.pyplot as plt
        # import seaborn as sns
        
        # # Get the first attention head's weights
        # for i in range(ca_weights.shape[0]):
        #     attn_weights = ca_weights[i].detach().cpu().numpy()

        #     print(ca_weights.shape, attn_weights.shape)
        #     # Create figure and plot heatmap
        #     plt.figure(figsize=(10, 8))
        #     sns.heatmap(attn_weights, cmap='viridis')
        #     plt.title('Cross-Attention Weights Heatmap')
        #     plt.xlabel('Memory Sequence Length')
        #     plt.ylabel('Query Sequence Length')
            
        #     # Save the plot
        #     plt.savefig(f'/home/sieut/kronus/ca_weights_{i}.png')
        #     plt.close()
        
        return tgt, None, ca_weights

    def forward(
        self,
        query,
        content,
        memory,
        query_mask: Optional[Tensor] = None,
        content_mask: Optional[Tensor] = None,
        content_key_padding_mask: Optional[Tensor] = None,
        update_content: bool = True,
    ):
        query_norm = self.norm_q(query)
        content_norm = self.norm_c(content)
        query, sa_weights, ca_weights = self.forward_stream(query, query_norm, content_norm, memory, query_mask, content_key_padding_mask)
        if update_content:
            content, sa_weights, ca_weights = self.forward_stream(
                content, content_norm, content_norm, memory, content_mask, content_key_padding_mask
            )
        return query, content, sa_weights, ca_weights


class Decoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm):
        super().__init__()
        self.layers = transformer._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        query,
        content,
        memory,
        query_mask: Optional[Tensor] = None,
        content_mask: Optional[Tensor] = None,
        content_key_padding_mask: Optional[Tensor] = None,
    ):
        for i, mod in enumerate(self.layers):
            last = i == len(self.layers) - 1
            query, content, sa_weights, ca_weights = mod(
                query, content, memory, query_mask, content_mask, content_key_padding_mask, update_content=not last
            )
        query = self.norm(query)
        return query, sa_weights, ca_weights



class TokenEmbedding(nn.Module):

    def __init__(self, charset_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(charset_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, tokens: torch.Tensor):
        return math.sqrt(self.embed_dim) * self.embedding(tokens)
