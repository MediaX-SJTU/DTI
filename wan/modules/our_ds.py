import torch
import torch.nn as nn
import torch.nn.functional as F
import math



# class RMSNorm(nn.Module):
#     def __init__(self, dim, eps=1e-6):
#         super().__init__()
#         self.eps = eps
#         # 只需要一个缩放参数 weight，不需要 bias (beta)
#         self.weight = nn.Parameter(torch.ones(dim))

#     def forward(self, x):
#         # x: [B, N, C] 或者任何形状，只要在最后一维做 norm
#         # 1. 计算均方根的平方 (即方差的变体)
#         variance = x.pow(2).mean(dim=-1, keepdim=True)
#         # 2. 乘以平方根的倒数 (rsqrt 在 GPU 上极其高效)
#         x_norm = x * torch.rsqrt(variance + self.eps)
#         # 3. 乘以可学习权重
#         return self.weight * x_norm

# class RMSNorm(nn.Module):

#     def __init__(self, dim, channel_first=True, images=True, bias=False):
#         super().__init__()
#         broadcastable_dims = (1, 1, 1) if not images else (1, 1)
#         shape = (dim, *broadcastable_dims) if channel_first else (dim,)

#         self.channel_first = channel_first
#         self.scale = dim**0.5
#         self.gamma = nn.Parameter(torch.ones(shape))
#         self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.

#     def forward(self, x):
#         return F.normalize(
#             x, dim=(1 if self.channel_first else
#                     -1)) * self.scale * self.gamma + self.bias
    

# ==========================================
# 1. Video 3D RoPE 
# ==========================================
def get_3d_rotary_pos_embed(embed_dim, T, H, W, theta=10000.0):
    """
    为时间、高度、宽度分别生成旋转频率
    假设 embed_dim = 64 (每头维度), 我们按 T:16, H:24, W:24 的比例分配
    """
    dim_t = embed_dim // 4
    dim_h = embed_dim * 3 // 8
    dim_w = embed_dim - dim_t - dim_h

    # 生成各维度的频率
    freqs_t = 1.0 / (theta ** (torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t))
    freqs_h = 1.0 / (theta ** (torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h))
    freqs_w = 1.0 / (theta ** (torch.arange(0, dim_w, 2)[: (dim_w // 2)].float() / dim_w))

    # 生成网格坐标
    t_grid = torch.arange(T).float()
    h_grid = torch.arange(H).float()
    w_grid = torch.arange(W).float()

    # 外积得到具体位置的频率
    pos_t = torch.einsum('t, f -> t f', t_grid, freqs_t)  # [T, dim_t/2]
    pos_h = torch.einsum('h, f -> h f', h_grid, freqs_h)  # [H, dim_h/2]
    pos_w = torch.einsum('w, f -> w f', w_grid, freqs_w)  # [W, dim_w/2]

    # 扩展到 3D 空间
    pos_t = pos_t.view(T, 1, 1, -1).expand(T, H, W, -1)
    pos_h = pos_h.view(1, H, 1, -1).expand(T, H, W, -1)
    pos_w = pos_w.view(1, 1, W, -1).expand(T, H, W, -1)

    # 拼接 [T, H, W, embed_dim/2]
    freqs = torch.cat([pos_t, pos_h, pos_w], dim=-1)
    # 展平为 [T*H*W, embed_dim/2]
    freqs = freqs.reshape(-1, freqs.shape[-1])
    
    # 复制一份，变成 [T*H*W, embed_dim] 以匹配复数旋转
    freqs_cos = freqs.cos().unsqueeze(0).unsqueeze(2) # [1, N, 1, dim/2]
    freqs_sin = freqs.sin().unsqueeze(0).unsqueeze(2)
    return freqs_cos, freqs_sin

def apply_rotary_emb(x, freqs_cos, freqs_sin):
    # x: [B, N, num_heads, head_dim]
    # 把最后一维分成两半
    x1, x2 = x.chunk(2, dim=-1)
    x_rot = torch.cat([-x2, x1], dim=-1)
    # 频率维度是 dim/2，需要重复一次匹配 dim
    freqs_cos = torch.cat([freqs_cos, freqs_cos], dim=-1)
    freqs_sin = torch.cat([freqs_sin, freqs_sin], dim=-1)
    return x * freqs_cos + x_rot * freqs_sin

# ==========================================
# 2. 支持 RoPE 的 Attention 和 Block
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
        # [B, N, 3, heads, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        # 注入 3D RoPE (只对 q 和 k)
        q = apply_rotary_emb(q, freqs_cos, freqs_sin)
        k = apply_rotary_emb(k, freqs_cos, freqs_sin)

        # 转置准备 Attention: [B, heads, N, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Flash Attention / Scaled Dot Product (PyTorch 2.0+)
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
# 3. 对齐 Wan2.1 的非对称 CNN Stem
# ==========================================
class WanAlignedStem(nn.Module):
    def __init__(self, in_channels, embed_dim, groups=32):
        super().__init__()        
        self.groups = groups
        
        # ==========================================
        # 1. 首帧处理分支 (2D 卷积)
        # ==========================================
        self.stem_first = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1), 
            nn.GroupNorm(self.groups, 64),  # <--- 加入 GroupNorm
            nn.SiLU(),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), 
            nn.GroupNorm(self.groups, 128), # <--- 加入 GroupNorm
            nn.SiLU(),
            
            nn.Conv2d(128, embed_dim, kernel_size=4, stride=2, padding=1), 
            nn.GroupNorm(self.groups, embed_dim), # <--- 加入 GroupNorm
            nn.SiLU()
        )
        
        # ==========================================
        # 2. 后续帧处理分支 (3D 卷积)
        # ==========================================
        self.stem_rest = nn.Sequential(
            # 步长为 (1, 2, 2)，时间不缩放
            nn.Conv3d(in_channels, 64, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)), 
            nn.GroupNorm(self.groups, 64),  # <--- GroupNorm 同样适用 3D 张量
            nn.SiLU(),
            
            # 步长为 (2, 2, 2)，时间缩小一半
            nn.Conv3d(64, 128, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)), 
            nn.GroupNorm(self.groups, 128),
            nn.SiLU(),
            
            # 步长为 (2, 2, 2)，时间再缩小一半
            nn.Conv3d(128, embed_dim, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)), 
            nn.GroupNorm(self.groups, embed_dim),
            nn.SiLU()
        )

    def forward(self, x):
        # x: [B, C_in, T_in, H_in, W_in]
        x_first = x[:, :, 0, :, :]      # 首帧: [B, C, H, W]
        x_rest = x[:, :, 1:, :, :]      # 后续: [B, C, T-1, H, W]

        # 穿过带 Norm 的网络
        out_first = self.stem_first(x_first).unsqueeze(2)  # [B, dim, 1, H/8, W/8]
        out_rest = self.stem_rest(x_rest)                  # [B, dim, (T-1)/4, H/8, W/8]

        return torch.cat([out_first, out_rest], dim=2)

# ==========================================
# 4. 主网络 (Y-Shaped + Video RoPE)
# ==========================================
class WanAlignedDistributionShifter(nn.Module):
    def __init__(
        self, in_channels=3, latent_channels=16, embed_dim=1024, 
        shared_depth=16, time_depth=4, jitter_depth=4, num_heads=16, groups=32
    ):
        super().__init__()
        self.head_dim = embed_dim // num_heads
        
        # 1. 严格对齐的 CNN Stem
        self.stem = WanAlignedStem(in_channels, embed_dim, groups)
        
        # 2. Y 型架构 Blocks
        self.shared_blocks = nn.ModuleList([RoPEBlock(embed_dim, num_heads) for _ in range(shared_depth)])
        self.time_blocks = nn.ModuleList([RoPEBlock(embed_dim, num_heads) for _ in range(time_depth)])
        self.jitter_blocks = nn.ModuleList([RoPEBlock(embed_dim, num_heads) for _ in range(jitter_depth)])
        
        # 3. 输出头
        self.time_norm = nn.RMSNorm(embed_dim)
        self.timestep_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(approximate="tanh"), nn.Linear(embed_dim, 1000)
        )
        
        self.jitter_norm = nn.RMSNorm(embed_dim)
        self.jitter_mlp =  nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(approximate="tanh"))
        self.jitter_head = nn.Sequential(
            nn.Conv3d(embed_dim, latent_channels, kernel_size=3, padding=1), nn.Sigmoid()
        )

        # self.apply(self._init_weights)
        
        # 1. 启动流水线，刷上基础权重 (Truncated Normal)
        self.apply(self._init_base_weights)
        # 2. 极其重要：执行残差缩放，防止方差爆炸
        self._apply_residual_scaling()
        # 3. 最后：精准打击，零初始化输出头
        self._init_heads()

    def forward(self, x):
        B, C, T_in, H_in, W_in = x.shape
        
        # 1. 获取物理对齐的时空特征
        x_feat = self.stem(x) # [B, embed_dim, T_out, H_out, W_out]
        _, embed_dim, T_out, H_out, W_out = x_feat.shape
        
        # 2. 动态生成 3D RoPE
        # (因为不需要保存参数，放在 forward 里动态生成最安全，不占显存)
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(self.head_dim, T_out, H_out, W_out, theta=10000.0)
        freqs_cos, freqs_sin = freqs_cos.to(x.device), freqs_sin.to(x.device)
        
        # 3. 展平为 Token
        x_seq = x_feat.flatten(2).transpose(1, 2) # [B, N, embed_dim]
        
        # 4. 主干传递 (注入 RoPE)
        for block in self.shared_blocks:
            x_seq = block(x_seq, freqs_cos, freqs_sin)
            
        # 5. 分支 A: Time Logits
        x_time = x_seq
        for block in self.time_blocks:
            x_time = block(x_time, freqs_cos, freqs_sin)
        x_time = self.time_norm(x_time)
        time_logits = self.timestep_head(x_time.mean(dim=1))
        
        # 6. 分支 B: Jitter Matrix
        x_jitter = x_seq
        for block in self.jitter_blocks:
            x_jitter = block(x_jitter, freqs_cos, freqs_sin)
        x_jitter = self.jitter_mlp(self.jitter_norm(x_jitter))
        # 还原物理形状
        x_jitter_3d = x_jitter.transpose(1, 2).view(B, -1, T_out, H_out, W_out)
        jitter_matrix = self.jitter_head(x_jitter_3d)
        
        return time_logits, jitter_matrix
    
    # def _init_weights(self, m):
    #     # ==========================================
    #     # 线性层 (Transformer 主干)
    #     # ==========================================
    #     if isinstance(m, nn.Linear):
    #         nn.init.xavier_uniform_(m.weight)
    #         if m.bias is not None:
    #             nn.init.zeros_(m.bias)
                
    #     # ==========================================
    #     # 卷积层 (CNN Stem)
    #     # ==========================================
    #     elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
    #         # 针对 SiLU/GELU 激活函数，使用 Kaiming Normal 是最优的
    #         torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #         if m.bias is not None:
    #             nn.init.constant_(m.bias, 0)

    #     # ==========================================
    #     # 零初始化输出头 (Zero-Initialization)
    #     # ==========================================
    #     # 分支 A: Time Logits (分类输出)
    #     if hasattr(self, 'timestep_head'):
    #         # 最后一层 Linear 权重设为 0，bias 设为 0
    #         # 效果：初始状态下，模型对 1000 个时间步的预测概率完全均等 (最大信息熵)
    #         nn.init.constant_(self.timestep_head[-1].weight, 0)
    #         nn.init.constant_(self.timestep_head[-1].bias, 0)

    #     # 分支 B: Jitter Matrix (矩阵输出)
    #     if hasattr(self, 'jitter_head'):
    #         # 最后一层 Conv3d 权重设为 0
    #         nn.init.constant_(self.jitter_head[-2].weight, 0)
    #         # 【极其精妙的设定】：Bias 设为 -2.0
    #         # 经过 Sigmoid(-2.0) 约为 0.119
    #         # 效果：初始状态下，模型非常保守，对所有区域只敢加 11.9% 的轻微噪声，绝不乱加！
    #         nn.init.constant_(self.jitter_head[-2].bias, -2.0)

    def _init_base_weights(self, m):
        """基础遍历初始化"""
        if isinstance(m, nn.Linear):
            # 绝对不要用 Xavier！必须用标准差 0.02 的截断正态分布
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
        elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def _apply_residual_scaling(self):
        """ Transformer 保命神技：残差分支衰减 """
        # 假设你的总 Block 深度大约是 20
        num_layers = 20 
        for name, p in self.named_parameters():
            if name.endswith('attn.proj.weight') or name.endswith('mlp.2.weight'):
                with torch.no_grad():
                    p.mul_(1.0 / math.sqrt(2.0 * num_layers))
    
    def _init_heads(self):
        """ 精确制导：重置末端网络 """
        if hasattr(self, 'timestep_head'):
            nn.init.constant_(self.timestep_head[-1].weight, 0)
            nn.init.constant_(self.timestep_head[-1].bias, 0)

        if hasattr(self, 'jitter_head'): # 或你改名后的 jitter_conv
            nn.init.constant_(self.jitter_head[-2].weight, 0)
            nn.init.constant_(self.jitter_head[-2].bias, -2.0)

# 测试用例
if __name__ == "__main__":
    model = WanAlignedDistributionShifter().cuda()
    # T=17 满足 T=4n+1，输出 T_out 应该是 1 + 16/4 = 5
    dummy_x = torch.randn(2, 3, 17, 256, 256).cuda()
    time_out, jitter_out = model(dummy_x)
    print(f"Jitter 形状 (期望 [2, 16, 5, 32, 32]): {jitter_out.shape}")