import torch
from torch import nn
from utils import WanRMSNorm, rope_apply
class LoRALinear(nn.Module):
    """LoRA (Low-Rank Adaptation) Linear Layer"""
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 rank: int = 4, 
                 alpha: float = 1.0, 
                 dropout: float = 0.0,
                 bias: bool = True):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.in_features = in_features
        self.out_features = out_features
        
        # Original linear layer (frozen during LoRA training)
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        if rank > 0:
            # LoRA matrices: B @ A
            # A: [rank, in_features], B: [out_features, rank]
            self.lora_A = nn.Parameter(torch.randn(rank, in_features) * (1 / rank))
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.scaling = alpha / rank
        else:
            self.lora_A = None
            self.lora_B = None
            self.scaling = 1.0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original linear transformation
        result = self.linear(x)
        
        # Add LoRA adaptation if enabled
        if self.rank > 0 and self.lora_A is not None and self.lora_B is not None:
            # x @ A^T @ B^T = x @ (B @ A)^T
            lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
            result += lora_out * self.scaling
            
        return result
    
    def freeze_original(self):
        """Freeze the original linear layer parameters"""
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
    
    def unfreeze_original(self):
        """Unfreeze the original linear layer parameters"""
        self.linear.weight.requires_grad = True
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = True


class WanRMSNorm(nn.Module):
    """RMS Normalization"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class WanSelfAttentionLoRA(nn.Module):
    """WanSelfAttention with LoRA parameters and SageAttention"""
    
    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                 # LoRA parameters
                 lora_rank=4,
                 lora_alpha=1.0,
                 lora_dropout=0.0,
                 lora_targets=None):  # Can be ['q', 'k', 'v', 'o'] or None for all
        
        assert dim % num_heads == 0
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        
        # LoRA configuration
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        # Default to applying LoRA to all linear layers
        if lora_targets is None:
            lora_targets = ['q', 'k', 'v', 'o']
        self.lora_targets = lora_targets

        # Linear layers with LoRA
        self.q = LoRALinear(
            dim, dim, 
            rank=lora_rank if 'q' in lora_targets else 0,
            alpha=lora_alpha, 
            dropout=lora_dropout, 
            bias=False
        )
        
        self.k = LoRALinear(
            dim, dim, 
            rank=lora_rank if 'k' in lora_targets else 0,
            alpha=lora_alpha, 
            dropout=lora_dropout, 
            bias=False
        )
        
        self.v = LoRALinear(
            dim, dim, 
            rank=lora_rank if 'v' in lora_targets else 0,
            alpha=lora_alpha, 
            dropout=lora_dropout, 
            bias=False
        )
        
        self.o = LoRALinear(
            dim, dim, 
            rank=lora_rank if 'o' in lora_targets else 0,
            alpha=lora_alpha, 
            dropout=lora_dropout, 
            bias=False
        )

        # Normalization layers
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, C] or [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        try:
            from sageattention import sageattn
        except ImportError:
            raise ImportError("Please install sageattention: pip install sageattention")
        
        # Handle input shape
        if x.dim() == 3:  # [B, L, C]
            b, s = x.shape[:2]
            x_flat = x
        else:  # [B, L, num_heads, C / num_heads]
            b, s = x.shape[:2]
            x_flat = x.flatten(2)  # [B, L, C]
        
        n, d = self.num_heads, self.head_dim

        # Query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x_flat)

        # Apply RoPE (Rotary Position Embedding)
        q_rope = rope_apply(q, grid_sizes, freqs)
        k_rope = rope_apply(k, grid_sizes, freqs)

        q_sage = q_rope.transpose(1, 2)  # [B, L, H, D] -> [B, H, L, D]
        k_sage = k_rope.transpose(1, 2)  # [B, L, H, D] -> [B, H, L, D]
        v_sage = v.transpose(1, 2)       # [B, L, H, D] -> [B, H, L, D]

        x = sageattn(q=q_sage, k=k_sage, v=v_sage, tensor_layout="HND")

        x = x.transpose(1, 2) 

        # Output projection
        x = x.flatten(2)  # [B, L, C]
        x = self.o(x)
        return x
    
    def enable_lora_training(self):
            """Enable LoRA training mode: freeze original weights, enable LoRA parameters"""
            lora_layers = [
                ('q', self.q), ('k', self.k), ('v', self.v), ('o', self.o)
            ]
            
            for name, layer in lora_layers:
                if isinstance(layer, LoRALinear) and name in self.lora_targets:
                    layer.freeze_original()
                    print(f"Frozen original weights for {name}, enabled LoRA training")
    
    def disable_lora_training(self):
        """Disable LoRA training mode: unfreeze all parameters"""
        lora_layers = [
            ('q', self.q), ('k', self.k), ('v', self.v), ('o', self.o)
        ]
        
        for name, layer in lora_layers:
            if isinstance(layer, LoRALinear) and name in self.lora_targets:
                layer.unfreeze_original()
                print(f"Unfrozen all weights for {name}")
    
    def get_lora_parameters(self):
        """Get only LoRA parameters for optimization"""
        lora_params = []
        lora_layers = [
            ('q', self.q), ('k', self.k), ('v', self.v), ('o', self.o)
        ]
        
        for name, layer in lora_layers:
            if isinstance(layer, LoRALinear) and name in self.lora_targets:
                if layer.lora_A is not None:
                    lora_params.extend([layer.lora_A, layer.lora_B])
        
        return lora_params
    
    def count_lora_parameters(self):
        """Count the number of LoRA parameters"""
        total_params = 0
        lora_layers = [
            ('q', self.q), ('k', self.k), ('v', self.v), ('o', self.o)
        ]
        
        for name, layer in lora_layers:
            if isinstance(layer, LoRALinear) and name in self.lora_targets and layer.rank > 0:
                # A: [rank, in_features], B: [out_features, rank]
                params_A = layer.rank * layer.in_features
                params_B = layer.out_features * layer.rank
                layer_params = params_A + params_B
                total_params += layer_params
                print(f"LoRA {name}: {layer_params:,} parameters")
        
        return total_params
    
    def merge_lora_weights(self):
        """Merge LoRA weights into the original linear layers"""
        with torch.no_grad():
            lora_layers = [
                ('q', self.q), ('k', self.k), ('v', self.v), ('o', self.o)
            ]
            
            for name, layer in lora_layers:
                if isinstance(layer, LoRALinear) and name in self.lora_targets:
                    if layer.rank > 0 and layer.lora_A is not None and layer.lora_B is not None:
                        # Merge LoRA weights: W = W_original + α/r * B * A
                        delta_w = layer.lora_B @ layer.lora_A * layer.scaling
                        layer.linear.weight.data += delta_w
                        
                        # Reset LoRA parameters
                        layer.lora_A.data.zero_()
                        layer.lora_B.data.zero_()
                        
                        print(f"Merged LoRA weights for {name}")