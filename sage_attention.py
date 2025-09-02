from torch import nn
import torch
from wan.modules.attention import flash_attention
from utils import WanRMSNorm, rope_apply

class WanSageAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.window_size = window_size
        
        # Additional linear layer for qkv -> attention score shape
        # Input: concatenated q, k, v -> Output: attention score shape
        self.qkv_to_attn = nn.Linear(dim * 3, num_heads)

    # def forward(self, x, seq_lens, grid_sizes, freqs):
    def forward(self, q, k, v, seq_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        # b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        (b, s, n, d) = q.shape
        
        # Prepare qkv for additional linear layer
        # Flatten q, k, v and concatenate them
        q_flat = q.view(b, s, -1)  # [B, L, num_heads * head_dim]
        k_flat = k.view(b, s, -1)  # [B, L, num_heads * head_dim]
        v_flat = v.view(b, s, -1)  # [B, L, num_heads * head_dim]
        qkv_concat = torch.cat([q_flat, k_flat, v_flat], dim=-1)  # [B, L, 3 * dim]
        
        additional_scores = self.qkv_to_attn(qkv_concat)  # [B, L, num_heads]

        from sageattention import sageattn

        attn_output = sageattn(
            q=q,
            k=k,
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        additional_scores_expanded = additional_scores.unsqueeze(-1)  # [B, L, num_heads, 1]
        
        # Add to attention output
        x = attn_output + additional_scores_expanded
        return x