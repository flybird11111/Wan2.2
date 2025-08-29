import torch
from torch import nn
import torch.nn.functional as F
from sage_attention import WanSageAttention
from selfattention import WanSelfAttention
    

class WanAttentionTrainer(nn.Module):
    """
    封装 FlashAttention 和 SageAttention，同时训练共享参数，
    并约束两者输出一致
    """
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        super().__init__()
        # Flash Attention 模块
        self.num_heads = num_heads
        self.flash_attn = WanSelfAttention(
            dim=dim, num_heads=num_heads,
            window_size=window_size, qk_norm=qk_norm, eps=eps
        )
        # Sage Attention 模块
        self.sage_attn = WanSageAttention(
            dim=dim, num_heads=num_heads,
            window_size=window_size, qk_norm=qk_norm, eps=eps
        )

    def forward(self, x, seq_lens, grid_sizes, freqs, return_loss=True):
        # forward
        out_flash = self.flash_attn(x, seq_lens, grid_sizes, freqs)
        
        # 阻止 out_flash 的梯度传播，使其不参与训练
        with torch.no_grad():
            out_flash_detached = out_flash.detach()
        
        out_sage = self.sage_attn(x, seq_lens, grid_sizes, freqs)
        
        if return_loss:
            # 使用 detached 的 flash 输出计算一致性 loss
            # 梯度只会传播到 sage attention
            loss = F.mse_loss(out_sage, out_flash_detached)
            return loss, out_flash_detached, out_sage
        else:
            return out_flash, out_sage

def main():
    # 输入

    B = 2
    L = 512
    C = 2048
    num_heads = 8
    head_dim = C // num_heads

    model = WanAttentionTrainer(dim=C, num_heads=num_heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 构造输入
    for i in range(500):
        x = torch.randn(B, L, C).cuda()  # [B, L, n, d]
        seq_lens = torch.tensor([L, L]).cuda()             # 每个 batch 的有效长度
        grid_sizes = torch.tensor([[8, 8, 2], [8, 8, 2]]).cuda()  # [B, 3]，随便造个立方网格 (F,H,W)
        freqs = torch.randn(L, head_dim // 2).cuda()

        # 前向 + 反向
        model.cuda()
        loss, out_flash, out_sage = model(x, seq_lens, grid_sizes, freqs)
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    main()

