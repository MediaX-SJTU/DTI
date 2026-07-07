import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. Video 3D RoPE (保持不变)
# ==========================================
def get_3d_rotary_pos_embed(embed_dim, T, H, W, theta=10000.0):
    dim_t = embed_dim // 4
    dim_h = embed_dim * 3 // 8
    dim_w = embed_dim - dim_t - dim_h

    freqs_t = 1.0 / (theta ** (torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t))
    freqs_h = 1.0 / (theta ** (torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h))
    freqs_w = 1.0 / (theta ** (torch.arange(0, dim_w, 2)[: (dim_w // 2)].float() / dim_w))

    t_grid = torch.arange(T).float()
    h_grid = torch.arange(H).float()
    w_grid = torch.arange(W).float()

    pos_t = torch.einsum('t, f -> t f', t_grid, freqs_t)
    pos_h = torch.einsum('h, f -> h f', h_grid, freqs_h)
    pos_w = torch.einsum('w, f -> w f', w_grid, freqs_w)

    pos_t = pos_t.view(T, 1, 1, -1).expand(T, H, W, -1)
    pos_h = pos_h.view(1, H, 1, -1).expand(T, H, W, -1)
    pos_w = pos_w.view(1, 1, W, -1).expand(T, H, W, -1)

    freqs = torch.cat([pos_t, pos_h, pos_w], dim=-1)
    freqs = freqs.reshape(-1, freqs.shape[-1])
    
    freqs_cos = freqs.cos().unsqueeze(0).unsqueeze(2) 
    freqs_sin = freqs.sin().unsqueeze(0).unsqueeze(2)
    return freqs_cos, freqs_sin

def apply_rotary_emb(x, freqs_cos, freqs_sin):
    x1, x2 = x.chunk(2, dim=-1)
    x_rot = torch.cat([-x2, x1], dim=-1)
    freqs_cos = torch.cat([freqs_cos, freqs_cos], dim=-1)
    freqs_sin = torch.cat([freqs_sin, freqs_sin], dim=-1)
    return x * freqs_cos + x_rot * freqs_sin

# ==========================================
# 2. 支持 RoPE 的 Attention 和 Block (保持不变)
# ==========================================
class RoPEAttention(nn.Module):
    def __init__(self, dim, num_heads=6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, freqs_cos, freqs_sin):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        q = apply_rotary_emb(q, freqs_cos, freqs_sin)
        k = apply_rotary_emb(k, freqs_cos, freqs_sin)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class RoPEBlock(nn.Module):
    def __init__(self, dim, num_heads=6):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.attn = RoPEAttention(dim, num_heads=num_heads)
        self.norm2 = nn.RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(approximate="tanh"), nn.Linear(dim * 4, dim)
        )

    def forward(self, x, freqs_cos, freqs_sin):
        x = x + self.attn(self.norm1(x), freqs_cos, freqs_sin)
        x = x + self.mlp(self.norm2(x))
        return x

# ==========================================
# 3. 升级版 Stem：支持 1x1x1 Conv 升维
# ==========================================
class WanAlignedStem(nn.Module):
    def __init__(self, in_channels, base_dim, embed_dim, groups=32):
        super().__init__()        
        self.groups = groups
        
        # 1. 基础特征提取 (输出 base_dim = 1024)
        self.stem_first = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1), 
            nn.GroupNorm(self.groups, 64), nn.SiLU(),
            nn.Conv2d(64, 256, kernel_size=4, stride=2, padding=1), 
            nn.GroupNorm(self.groups, 256), nn.SiLU(),
            nn.Conv2d(256, base_dim, kernel_size=4, stride=2, padding=1), 
            nn.GroupNorm(self.groups, base_dim), nn.SiLU()
        )
        
        self.stem_rest = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)), 
            nn.GroupNorm(self.groups, 64), nn.SiLU(),
            nn.Conv3d(64, 256, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)), 
            nn.GroupNorm(self.groups, 256), nn.SiLU(),
            nn.Conv3d(256, base_dim, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)), 
            nn.GroupNorm(self.groups, base_dim), nn.SiLU()
        )

        # 2. 核心：1x1x1 Conv 升维投影 (1024 -> 2048)
        self.channel_proj = nn.Conv3d(base_dim, embed_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x_first = x[:, :, 0, :, :]      
        x_rest = x[:, :, 1:, :, :]      

        out_first = self.stem_first(x_first).unsqueeze(2)  
        out_rest = self.stem_rest(x_rest)                  

        out = torch.cat([out_first, out_rest], dim=2)      
        return self.channel_proj(out)
    
class WanAlignedDistributionShifter(nn.Module):
    def __init__(
        self, in_channels=3, latent_channels=16, base_dim=1024, embed_dim=2048, 
        depth=20, num_heads=16, groups=32  # <--- 直接合并为单一个 depth 参数
    ):
        super().__init__()
        self.head_dim = embed_dim // num_heads
        self.latent_channels = latent_channels
        self.total_layers = depth  # 用于残差缩放
        
        # 1. 扩维版 CNN Stem
        self.stem = WanAlignedStem(in_channels, base_dim=base_dim, embed_dim=embed_dim, groups=groups)
        
        # 2. 纯粹的单流 Transformer 主干
        # <--- 彻底干掉 shared 和 ds 的区分，就叫 blocks
        self.blocks = nn.ModuleList([RoPEBlock(embed_dim, num_heads) for _ in range(depth)])
        
        # 3. 双分支输出头 (Dual-Branch DS Head)
        self.ds_norm = nn.RMSNorm(embed_dim)
        self.ds_mlp =  nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(approximate="tanh"))
        self.ds_head = nn.Conv3d(embed_dim, latent_channels * 2, kernel_size=3, padding=1)

        self.apply(self._init_base_weights)
        self._apply_residual_scaling()
        self._init_heads()

    def forward(self, x):
        B, C, T_in, H_in, W_in = x.shape
        
        x_feat = self.stem(x) 
        _, embed_dim, T_out, H_out, W_out = x_feat.shape
        
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(self.head_dim, T_out, H_out, W_out, theta=10000.0)
        freqs_cos, freqs_sin = freqs_cos.to(x.device), freqs_sin.to(x.device)
        
        x_seq = x_feat.flatten(2).transpose(1, 2) 
        
        # <--- 极简的前向传播：只遍历一个 List
        for block in self.blocks:
            x_seq = block(x_seq, freqs_cos, freqs_sin)
            
        x_ds = self.ds_mlp(self.ds_norm(x_seq))
        x_ds_3d = x_ds.transpose(1, 2).view(B, -1, T_out, H_out, W_out)
        
        out = self.ds_head(x_ds_3d)
        j_raw, delta_z = torch.chunk(out, 2, dim=1) 
        J = torch.sigmoid(j_raw)
        
        return J, delta_z

    def _init_base_weights(self, m):
        """
        1. 基础遍历初始化：给网络刷上底色
        """
        if isinstance(m, nn.Linear):
            # Transformer 核心秘籍：绝对不能用 Xavier，必须用 std=0.02 的截断正态分布
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
        elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
            # 针对 Stem 中的 SiLU/GELU 激活函数，Kaiming Normal (Fan-Out) 是数学最优解
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _apply_residual_scaling(self):
        """
        2. Transformer 保命神技：残差分支衰减
        防止深层网络 (depth=20) 训练初期方差爆炸，导致梯度消失或 NaN。
        """
        # 现在的 self.total_layers 就是单一的 depth
        scale_factor = 1.0 / math.sqrt(2.0 * self.total_layers)
        
        for name, p in self.named_parameters():
            # 严格锁定每个 RoPEBlock 里的两个残差汇入点：
            # 1. Attention 的输出投影层
            # 2. MLP 的最后一层投影层
            if name.endswith('attn.proj.weight') or name.endswith('mlp.2.weight'):
                with torch.no_grad():
                    p.mul_(scale_factor)
                    # 打印一条日志可以让你安心（实际训练中可注释掉）
                    # print(f"Applied residual scaling to: {name}")

    def _init_heads(self):
        """
        3. 精确制导：零初始化末端网络 (Zero-Initialization)
        这是保障 DS 模型在第一步不会毁掉 HQ 先验的绝对防线！
        """
        # 1. 斩断所有随机特征预测 (将最后的 Conv3d 权重设为 0)
        nn.init.constant_(self.ds_head.weight, 0)
        
        # 2. 偏置魔改 (Bias Trick)
        # self.ds_head.bias 的长度是 latent_channels * 2 (比如 32)
        
        # [前一半通道] - 对应 J 矩阵预测
        # 设为 -2.0，经过 Sigmoid(-2.0) ≈ 0.119
        # 物理意义：网络初始状态极度保守，全图只敢预测 11.9% 的加噪比例。
        nn.init.constant_(self.ds_head.bias[:self.latent_channels], -2.0)
        
        # [后一半通道] - 对应 delta_z (残差偏移)
        # 设为 0.0，不经过任何激活函数
        # 物理意义：网络初始状态认为不需要任何非线性修正，退化为纯正的 Identity Mapping。
        nn.init.constant_(self.ds_head.bias[self.latent_channels:], 0.0)

# 测试用例
if __name__ == "__main__":
    model = WanAlignedDistributionShifter().cuda()
    # T=17 满足 T=4n+1，输出 T_out 应该是 1 + 16/4 = 5
    dummy_x = torch.randn(2, 3, 17, 256, 256).cuda()
    J, delta_z = model(dummy_x)
    
    print(f"J_gt Matrix Shape [2, 16, 5, 32, 32]: {J.shape} | Range: [{J.min().item():.3f}, {J.max().item():.3f}]")
    print(f"delta_z Matrix Shape [2, 16, 5, 32, 32]: {delta_z.shape} | Range: [{delta_z.min().item():.3f}, {delta_z.max().item():.3f}]")


