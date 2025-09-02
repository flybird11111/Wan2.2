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
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        
        # Additional linear layer for qkv -> attention score shape
        # Input: concatenated q, k, v -> Output: attention score shape
        self.qkv_to_attn = nn.Linear(dim * 3, num_heads)

    # def forward(self, x, seq_lens, grid_sizes, freqs):
    def forward(self, q, k, v, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        # b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        b, s, n, d = *q.shape[:2], self.num_heads, self.head_dim

        # # query, key, value function
        # def qkv_fn(x):
        #     q = self.norm_q(self.q(x)).view(b, s, n, d)
        #     k = self.norm_k(self.k(x)).view(b, s, n, d)
        #     v = self.v(x).view(b, s, n, d)
        #     return q, k, v

        # q, k, v = qkv_fn(x)
        
        # Prepare qkv for additional linear layer
        # Flatten q, k, v and concatenate them
        q_flat = q.view(b, s, -1)  # [B, L, num_heads * head_dim]
        k_flat = k.view(b, s, -1)  # [B, L, num_heads * head_dim]
        v_flat = v.view(b, s, -1)  # [B, L, num_heads * head_dim]
        qkv_concat = torch.cat([q_flat, k_flat, v_flat], dim=-1)  # [B, L, 3 * dim]
        
        # Generate additional attention-like scores
        # This will have shape [B, L, num_heads]
        additional_scores = self.qkv_to_attn(qkv_concat)  # [B, L, num_heads]

        # Apply RoPE to q and k
        q_rope = rope_apply(q, grid_sizes, freqs)
        k_rope = rope_apply(k, grid_sizes, freqs)

        # Get attention output from flash_attention
        # Note: flash_attention typically returns [B, L, num_heads, head_dim]

        attn_output = flash_attention(
            q=q_rope,
            k=k_rope,
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # Add the additional scores to the attention output
        # We need to expand additional_scores to match attn_output shape
        # additional_scores: [B, L, num_heads] -> [B, L, num_heads, 1]
        additional_scores_expanded = additional_scores.unsqueeze(-1)  # [B, L, num_heads, 1]
        
        # Add to attention output
        x = attn_output + additional_scores_expanded

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x