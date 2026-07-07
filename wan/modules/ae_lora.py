import math
import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# __all__ = [
#     'AdaEncoder',
# ]


# class RMS_norm(nn.Module):

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


# class ResidualBlock3D(nn.Module):
#     def __init__(self, in_channels, out_channels, down_sample = False, kernel_size=[3,3,1], stride=[1,1,1], padding=[1,1,0]):
#         super().__init__()
#         self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size[2], stride=stride[2], padding=padding[2]) if in_channels != out_channels or down_sample else nn.Identity()
#         self.residual = nn.Sequential(
#             RMS_norm(in_channels, True, False, False), nn.SiLU(),
#             nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size[0], stride=stride[0], padding=padding[0]),
#             RMS_norm(out_channels, True, False, False), nn.SiLU(), 
#             nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size[1], stride=stride[1], padding=padding[1]))

#     def forward(self, x):
#         s = self.shortcut(x)
#         r = self.residual(x)
#         x  = s + r
#         return x
    
# class ResidualBlock2D(nn.Module):
#     def __init__(self, in_channels, out_channels, down_sample = False, kernel_size=[3,3,1], stride=[1,1,1], padding=[1,1,0]):
#         super().__init__()
#         self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size[2], stride=stride[2], padding=padding[2]) if in_channels != out_channels or down_sample else nn.Identity()
#         self.residual = nn.Sequential(
#             RMS_norm(in_channels, True, True, False), nn.SiLU(),
#             nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size[0], stride=stride[0], padding=padding[0]),
#             RMS_norm(out_channels, True, True, False), nn.SiLU(), 
#             nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size[1], stride=stride[1], padding=padding[1]))

#     def forward(self, x):
#         s = self.shortcut(x)
#         r = self.residual(x)
#         x  = s + r
#         return x

# class Attention2D(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim

#         # layers
#         self.norm = RMS_norm(dim)
#         self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
#         self.proj = nn.Conv2d(dim, dim, 1)

#         # zero out the last layer params
#         nn.init.zeros_(self.proj.weight)

#     def forward(self, x):
#         identity = x
#         b, c, t, h, w = x.size()
#         x = rearrange(x, 'b c t h w -> (b t) c h w')
#         x = self.norm(x)
#         # compute query, key, value
#         q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3,
#                                          -1).permute(0, 1, 3,
#                                                      2).contiguous().chunk(
#                                                          3, dim=-1)

#         # apply attention
#         x = F.scaled_dot_product_attention(
#             q,
#             k,
#             v,
#         )
#         x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)

#         # output
#         x = self.proj(x)
#         x = rearrange(x, '(b t) c h w-> b c t h w', t=t)
#         return x + identity

# class QueryHead(nn.Module):
#     def __init__(self, input_dim, output_dim, num_heads, avg_size = 64):
#         super().__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.num_heads = num_heads
#         self.avg_size = avg_size
    
#         # 假设使用简单的自注意力机制
#         self.attn = nn.MultiheadAttention(avg_size * avg_size, num_heads)
    
#         # 输出维度调整
#         self.fc = nn.Sequential(nn.LayerNorm(input_dim, elementwise_affine=False), nn.Linear(input_dim, output_dim))

#     def forward(self, x):
#         # x: (B, C, T, H, W)
#         b, c, t, h, w = x.shape

#         x = F.adaptive_avg_pool3d(x, (1, self.avg_size, self.avg_size))     # b, c, t = 1, h = 64, w = 64

#         x = rearrange(x, 'b c t h w -> b c (t h w)')

#         # 自注意力
#         attn_output, _ = self.attn(x, x, x)  # 进行自注意力计算
#         attn_output = attn_output.mean(dim=2)  # 这里对空间维度的输出做聚合

#         # 投影到最终的输出维度
#         output = self.fc(attn_output)
#         return output

# class AdaEncoder(nn.Module):
#     def __init__(self, in_channels = 3, out_channels = 16, hidden_channels=64, timestep_range = 1000, stride = [[1,1,1], [1,2,2], [1,(1,2,2),(1,2,2)], [1,2,2]]):
#         super().__init__()

#         # 3D卷积层 + 残差块
#         self.resblock1 = ResidualBlock3D(in_channels, hidden_channels, down_sample = False, stride=stride[0])
#         self.resblock2 = ResidualBlock3D(hidden_channels, hidden_channels * 2, down_sample = True, stride=stride[1])
#         self.attention1 = Attention2D(hidden_channels*2)
#         self.resblock3 = ResidualBlock3D(hidden_channels * 2, hidden_channels * 4, down_sample = True, stride=stride[2])
#         self.attention2 = Attention2D(hidden_channels*4)
#         self.resblock4 = ResidualBlock3D(hidden_channels * 4, hidden_channels * 8, down_sample = True, stride=stride[3])

#         self.output1 = nn.Sequential(
#             ResidualBlock3D(hidden_channels * 8, hidden_channels * 8), Attention2D(hidden_channels * 8), RMS_norm(hidden_channels * 8, images=False), nn.SiLU(),
#             nn.Conv3d(hidden_channels * 8, out_channels, 3, padding=1))
        
#         self.output2 = nn.Sequential(ResidualBlock3D(out_channels * 2, out_channels * 4), Attention2D(out_channels * 4), RMS_norm(out_channels * 4, images=False), nn.SiLU(), 
#                                      ResidualBlock3D(out_channels * 4, out_channels * 8), Attention2D(out_channels * 8), RMS_norm(out_channels * 8, images=False), nn.SiLU())
                                     
#         self.latent_predictor = nn.Conv3d(out_channels * 8, out_channels, 3, padding=1)
#         self.timestep_predictor = QueryHead(out_channels * 8, timestep_range, 16)

#     def forward(self, x, vae_output):
#         # 输入 x: (B, C, T, H, W)
#         x = self.resblock2(self.resblock1(x))
#         x = self.resblock3(self.attention1(x))
#         x = self.resblock4(self.attention2(x))
#         x = self.output1(x)

#         input = torch.cat([vae_output, x], dim = 1)
#         output = self.output2(input)

#         latent = self.latent_predictor(output)
#         timestep_prediction = self.timestep_predictor(output)
#         # probs = F.softmax(timestep_prediction - timestep_prediction.max(dim=-1, keepdim=True).values, dim=-1)
#         # timestep = torch.argmax(probs, dim=-1)

#         return latent, timestep_prediction


CACHE_T = 2

# class CausalConv3d(nn.Conv3d):
#     """
#     Causal 3d convolusion.
#     """

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._padding = (self.padding[2], self.padding[2], self.padding[1],
#                          self.padding[1], 2 * self.padding[0], 0)
#         self.padding = (0, 0, 0)

#     def forward(self, x, cache_x=None):
#         padding = list(self._padding)
#         if cache_x is not None and self._padding[4] > 0:
#             cache_x = cache_x.to(x.device)
#             x = torch.cat([cache_x, x], dim=2)
#             padding[4] -= cache_x.shape[2]
#         x = F.pad(x, padding)

#         return super().forward(x)
    
class CausalConv3d(nn.Conv3d):
    """
    Causal 3d convolusion with LoRA support.
    """

    def __init__(self, *args, lora_rank=0, lora_alpha=16, **kwargs):
        super().__init__(*args, **kwargs)
        # 原有 Padding 逻辑
        self._padding = (self.padding[2], self.padding[2], self.padding[1],
                         self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0) # 这里的 padding 设为 0 很关键，因为 forward 里手动 pad 了

        # ## LoRA Added: 初始化 LoRA 权重
        self.lora_rank = lora_rank
        self.lora_scaling = lora_alpha / lora_rank if lora_rank > 0 else 1.0

        if lora_rank > 0:
            # LoRA A: 
            # 关键点：padding 必须设为 0。
            # 因为 forward 中 x 已经被 F.pad 处理过了，这里的卷积不需要再 pad。
            self.lora_A = nn.Conv3d(
                self.in_channels, lora_rank, self.kernel_size,
                self.stride, 0, self.dilation, self.groups, bias=False
            )
            # LoRA B: 1x1x1 卷积恢复维度
            self.lora_B = nn.Conv3d(lora_rank, self.out_channels, 1, 1, 0, bias=False)

            # LoRA 初始化：A 高斯/凯明，B 全零
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        
        # 1. 显式 Padding (处理因果关系)
        x = F.pad(x, padding)

        # 2. 原权重计算 (frozen weights)
        out = super().forward(x)

        # 3. ## LoRA Added: LoRA 分支计算
        # 注意：这里直接使用 pad 过的 x 输入给 lora_A
        if self.lora_rank > 0:
            lora_out = self.lora_A(x)
            lora_out = self.lora_B(lora_out)
            out = out + lora_out * self.lora_scaling

        return out

class LoRAConv2d(nn.Conv2d):
    def __init__(self, *args, lora_rank=0, lora_alpha=16, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_rank = lora_rank
        self.lora_scaling = lora_alpha / lora_rank if lora_rank > 0 else 1.0

        if lora_rank > 0:
            # LoRA A: 降维，保持卷积核大小
            self.lora_A = nn.Conv2d(
                self.in_channels, lora_rank, self.kernel_size,
                self.stride, self.padding, self.dilation, self.groups, bias=False
            )
            # LoRA B: 升维，1x1 卷积
            self.lora_B = nn.Conv2d(lora_rank, self.out_channels, 1, 1, 0, bias=False)
            
            # 初始化
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        # 原有权重路径
        out = super().forward(x)
        
        # LoRA 路径
        if self.lora_rank > 0:
            lora_out = self.lora_A(x)
            lora_out = self.lora_B(lora_out)
            out = out + lora_out * self.lora_scaling
        return out


class RMS_norm(nn.Module):

    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.

    def forward(self, x):
        return F.normalize(
            x, dim=(1 if self.channel_first else
                    -1)) * self.scale * self.gamma + self.bias


class Upsample(nn.Upsample):

    def forward(self, x):
        """
        Fix bfloat16 support for nearest neighbor interpolation.
        """
        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):

    def __init__(self, dim, mode, lora_rank=256, lora_alpha=512):
        assert mode in ('none', 'upsample2d', 'upsample3d', 'downsample2d',
                        'downsample3d')
        super().__init__()
        self.dim = dim
        self.mode = mode
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        # layers
        if mode == 'upsample2d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                # nn.Conv2d(dim, dim // 2, 3, padding=1))
                LoRAConv2d(dim, dim // 2, 3, padding=1, lora_rank=lora_rank, lora_alpha=lora_alpha))
        elif mode == 'upsample3d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                # nn.Conv2d(dim, dim // 2, 3, padding=1))
                LoRAConv2d(dim, dim // 2, 3, padding=1, lora_rank=lora_rank, lora_alpha=lora_alpha))
            self.time_conv = CausalConv3d(
                # dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
                dim, dim*2, (3, 1, 1), padding=(1, 0, 0), lora_rank=lora_rank, lora_alpha=lora_alpha)

        elif mode == 'downsample2d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                # nn.Conv2d(dim, dim, 3, stride=(2, 2)))
                LoRAConv2d(dim, dim, 3, stride=(2, 2), lora_rank=lora_rank, lora_alpha=lora_alpha))
        elif mode == 'downsample3d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                # nn.Conv2d(dim, dim, 3, stride=(2, 2)))
                LoRAConv2d(dim, dim, 3, stride=(2, 2), lora_rank=lora_rank, lora_alpha=lora_alpha))
            self.time_conv = CausalConv3d(
                # dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0), lora_rank=lora_rank, lora_alpha=lora_alpha)

        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == 'upsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = 'Rep'
                    feat_idx[0] += 1
                else:

                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[
                            idx] is not None and feat_cache[idx] != 'Rep':
                        # cache last frame of last two chunk
                        cache_x = torch.cat([
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                                cache_x.device), cache_x
                        ],
                                            dim=2)
                    if cache_x.shape[2] < 2 and feat_cache[
                            idx] is not None and feat_cache[idx] == 'Rep':
                        cache_x = torch.cat([
                            torch.zeros_like(cache_x).to(cache_x.device),
                            cache_x
                        ],
                                            dim=2)
                    if feat_cache[idx] == 'Rep':
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]),
                                    3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.resample(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)

        if self.mode == 'downsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:

                    cache_x = x[:, :, -1:, :, :].clone()
                    # if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx]!='Rep':
                    #     # cache last frame of last two chunk
                    #     cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)

                    x = self.time_conv(
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x

    def init_weight(self, conv):
        conv_weight = conv.weight
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        one_matrix = torch.eye(c1, c2)
        init_matrix = one_matrix
        nn.init.zeros_(conv_weight)
        #conv_weight.data[:,:,-1,1,1] = init_matrix * 0.5
        conv_weight.data[:, :, 1, 0, 0] = init_matrix  #* 0.5
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def init_weight2(self, conv):
        conv_weight = conv.weight.data
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        init_matrix = torch.eye(c1 // 2, c2)
        #init_matrix = repeat(init_matrix, 'o ... -> (o 2) ...').permute(1,0,2).contiguous().reshape(c1,c2)
        conv_weight[:c1 // 2, :, -1, 0, 0] = init_matrix
        conv_weight[c1 // 2:, :, -1, 0, 0] = init_matrix
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)


class ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, lora_rank=256, lora_alpha=512, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        # layers
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False), nn.SiLU(),
            # CausalConv3d(in_dim, out_dim, 3, padding=1),
            CausalConv3d(in_dim, out_dim, 3, padding=1, lora_rank=lora_rank, lora_alpha=lora_alpha),
            RMS_norm(out_dim, images=False), nn.SiLU(), nn.Dropout(dropout),
            # CausalConv3d(out_dim, out_dim, 3, padding=1))
            CausalConv3d(out_dim, out_dim, 3, padding=1, lora_rank=lora_rank, lora_alpha=lora_alpha))
        # self.shortcut = CausalConv3d(in_dim, out_dim, 1) \
        self.shortcut = CausalConv3d(in_dim, out_dim, 1, lora_rank=lora_rank, lora_alpha=lora_alpha) \
            if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h


class AttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.
    """

    def __init__(self, dim, lora_rank=256, lora_alpha=512):
        super().__init__()
        self.dim = dim
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        # layers
        self.norm = RMS_norm(dim)
        # self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.to_qkv = LoRAConv2d(dim, dim * 3, 1, lora_rank=lora_rank, lora_alpha=lora_alpha)
        # self.proj = nn.Conv2d(dim, dim, 1)
        self.proj = LoRAConv2d(dim, dim, 1, lora_rank=lora_rank, lora_alpha=lora_alpha)

        # zero out the last layer params
        # 注意：如果使用了 LoRA，这里 zero 的是 base weight，LoRA weight B 已经在 init 里 zero 了
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.norm(x)
        # compute query, key, value
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3,
                                         -1).permute(0, 1, 3,
                                                     2).contiguous().chunk(
                                                         3, dim=-1)

        # apply attention
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
        )
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)

        # output
        x = self.proj(x)
        x = rearrange(x, '(b t) c h w-> b c t h w', t=t)
        return x + identity

def count_conv3d(model):
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
    return count

class Encoder(nn.Module):
    def __init__(self, dim=128,
                z_dim=32,
                dim_mult=[1, 2, 4, 4],
                num_res_blocks=2,
                attn_scales=[],
                temperal_downsample=[True, True, False],
                lora_rank=256,
                lora_alpha=512,
                dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        # self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1, lora_rank=lora_rank, lora_alpha=lora_alpha)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, lora_rank, lora_alpha, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim, lora_rank, lora_alpha))
                in_dim = out_dim

            # downsample block
            if i != len(dim_mult) - 1:
                mode = 'downsample3d' if temperal_downsample[
                    i] else 'downsample2d'
                downsamples.append(Resample(out_dim, mode=mode, lora_rank=lora_rank, lora_alpha=lora_alpha))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, lora_rank, lora_alpha, dropout), AttentionBlock(out_dim, lora_rank, lora_alpha),
            ResidualBlock(out_dim, out_dim, lora_rank, lora_alpha, dropout))

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1, lora_rank=lora_rank, lora_alpha=lora_alpha))
    
    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                        cache_x.device), cache_x
                ],
                                    dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## downsamples
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x

class DistributionHead(nn.Module):
    def __init__(self, c_in, c_out, hidden_dim=256):
        super().__init__()
        self.cv3d = nn.Conv3d(c_in, hidden_dim, kernel_size=3)
        self.proj = nn.Linear(hidden_dim, hidden_dim * 2)
        self.time_proj = nn.Linear(hidden_dim * 2, c_out)
        self.jitter = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        self.gn = nn.GroupNorm(16, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim * 2, elementwise_affine=False)

        self.init_weights()

    def forward(self, x):
        # x: [B, C, T, H, W]
        # 输入卷积
        x = F.silu(self.gn(self.cv3d(x)))

        # 1. 空间平均
        x = x.mean(dim=(3,4))    # [B, C, T]
        # 2. 调整维度供 Linear 使用
        x = x.permute(0, 2, 1)   # [B, T, C]
        
        # 3. 投影 + 激活
        x = self.ln(F.silu(self.proj(x))) # [B, T, C]

        # 4. 时间平均 (注意现在的 T 是 dim=1)
        x = x.mean(dim=1)        # [B, C]

        # 5. 输出时间步投影
        time_logits = self.time_proj(x)        # [B, C_out]
        jitter_scalar = torch.sigmoid(self.jitter(x))

        return time_logits, jitter_scalar
    
    def init_weights(self):
        # 1. 遍历所有模块进行通用初始化 (General Initialization)
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                # 对于 SiLU (Swish) 激活函数，Kaiming Normal (He init) 是最佳选择
                # mode='fan_in' 保持前向传播的方差
                # nonlinearity='relu' 是因为 SiLU 在正半轴近似 ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            # elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            #     # Norm 层标准初始化：weight=1, bias=0
            #     # 注意：你的 self.ln 是 elementwise_affine=False，没有权重，这里会自动跳过或无效，不会报错
            #     if hasattr(m, "elementwise_affine") and m.elementwise_affine:
            #         nn.init.constant_(m.weight, 1)
            #         nn.init.constant_(m.bias, 0)
            #     elif hasattr(m, "affine") and m.affine:
            #         nn.init.constant_(m.weight, 1)
            #         nn.init.constant_(m.bias, 0)

        # 2. 特殊初始化：时间分类头 (Time Projection Head)
        # 目标：让初始输出的 logits 接近 0，使得 Softmax 后概率分布接近均匀 (1/1000)
        # 这样可以最大化初始熵，避免模型在开始时“盲目自信”地偏向某个时间步
        nn.init.normal_(self.time_proj.weight, std=0.02)
        # nn.init.constant_(self.time_proj.bias, 0)

        # 在初始化函数中
        # 假设 output dim = 1000
        # 我们希望初始 t 集中在 400~600 之间
        nn.init.constant_(self.time_proj.bias, -10.0) # 先全设为极小

        # 把中间段的 bias 设为 0 或正数
        center_idx = 500
        window = 100
        self.time_proj.bias.data[center_idx-window : center_idx+window] = 0.0

        # 3. 特殊初始化：Jitter 预测头 (Jitter Regression Head)
        # 这里的 self.jitter 是一个 Sequential，我们需要取最后一层 Linear
        last_jitter_layer = self.jitter[-1]
        
        # 权重设得很小：让初始输出对输入的依赖很弱
        nn.init.normal_(last_jitter_layer.weight, std=0.01)
        
        # 【关键技巧】偏置设为负数：
        # Jitter 经过 Sigmoid 输出。如果 Bias=0，Sigmoid(0)=0.5。
        # 一开始就给 lq_video 混合 50% 的噪声通常太大了，会破坏特征，导致训练初期震荡。
        # 设为 -2.0 -> Sigmoid(-2.0) ≈ 0.12
        # 设为 -3.0 -> Sigmoid(-3.0) ≈ 0.047
        # 建议让模型从“微小扰动”开始学习，逐渐增加抖动幅度。
        nn.init.constant_(last_jitter_layer.bias, -2.0)

class DistributionShifter(nn.Module):
    def __init__(self,
                dtype=torch.float,
                device="cuda",
                dim=96,
                z_dim=16,
                dim_mult=[1, 2, 4, 4],
                num_res_blocks=2,
                attn_scales=[],
                temperal_downsample=[False, True, True],
                dropout=0.0, 
                time_range = 1000,
                lora_rank=128, lora_alpha=512):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.time_range = time_range
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        self.encoder = Encoder(dim, z_dim * 2, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_downsample, lora_rank, lora_alpha, dropout).to(device)
        
        
        # self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1).to(device)
        # self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1, lora_rank=lora_rank, lora_alpha=lora_alpha).to(device)
        self.distribution_head = DistributionHead(z_dim * 2, time_range).to(device)

        # mean = [
        #     -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        #     0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        # ]
        # std = [
        #     2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        #     3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        # ]
        # self.mean = torch.tensor(mean, dtype=dtype, device=device)
        # self.std = torch.tensor(std, dtype=dtype, device=device)
        # self.scale = [self.mean, 1.0 / self.std]


    def forward(self, x):
        self.clear_cache()
        ## cache
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        ## 对encode输入的x，按时间拆分为1、4、4、4....
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(
                    x[:, :, :1, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx)
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx)
                out = torch.cat([out, out_], 2)
        # mu, log_var = self.conv1(out).chunk(2, dim=1)
        time_logits, jitter_scalar = self.distribution_head(out)
        # if isinstance(self.scale[0], torch.Tensor):
        #     mu = (mu - self.scale[0].view(1, self.z_dim, 1, 1, 1)) * self.scale[1].view(
        #         1, self.z_dim, 1, 1, 1)
        # else:
        #     mu = (mu - self.scale[0]) * self.scale[1]
        self.clear_cache()
        return time_logits, jitter_scalar
    
    def clear_cache(self):
        #cache encode
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num