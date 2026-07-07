import math

import torch
import torch.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention

__all__ = ['VRModel']

T5_CONTEXT_TOKEN_NUMBER = 512

class LoRALinear(nn.Module):
    def __init__(self, in_dim, out_dim, r=8, alpha=16.0):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.in_dim = in_dim
        self.out_dim = out_dim

        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros((r, in_dim)))
            self.lora_B = nn.Parameter(torch.zeros((out_dim, r)))
            # nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            # nn.init.zeros_(self.lora_B)
        else:
            self.lora_A = nn.Identity()
            self.lora_B = nn.Identity()

    def forward(self, x):
        if self.r > 0:
            delta = (self.lora_B @ self.lora_A) * self.alpha / self.r
            return x @ delta.T
        else:
            return 0


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast("cuda", enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@amp.autocast("cuda", enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()

class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
    
class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)

class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                 r = 8, 
                 alpha = 16.0):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.r = r
        self.alpha = alpha

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.qlora = LoRALinear(dim, dim, r, alpha)
        self.klora = LoRALinear(dim, dim, r, alpha)
        self.vlora = LoRALinear(dim, dim, r, alpha)
        self.olora = LoRALinear(dim, dim, r, alpha)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x) + self.qlora(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x) + self.klora(x)).view(b, s, n, d)
            v = (self.v(x) + self.vlora(x)).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x) + self.olora(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x) + self.qlora(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context) + self.klora(context)).view(b, -1, n, d)
        v = (self.v(context) + self.vlora(context)).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x) + self.olora(x)
        return x
    
class ConditionSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, qk_norm = True, eps = 1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v
        q, k, v = qkv_fn(x)
        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens)
        # output
        x = x.flatten(2)
        x = self.o(x)
        return x
    
class ConditionCrossAttention(ConditionSelfAttention):
    def forward(self, x, c1, c2, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            condition1(Tensor): Shape [B, L, C]
            condition2(Tensor): Shape [B, L, C]
        """
        b, s, n,d = *x.shape[:2], self.num_heads, self.head_dim
        
        condition = torch.cat([c1,c2],dim=1)
        k = self.norm_k(self.k(condition)).view(b,-1,n,d)
        
        c1 = rope_apply(k[:,:s], grid_sizes, freqs)
        c2 = rope_apply(k[:,s:], grid_sizes, freqs)

        k = torch.cat([c1,c2], dim = 1)

        q = self.norm_q(self.q(x)).view(b,-1,n,d)
        v = self.v(condition).view(b,-1,n,d)
        x = flash_attention(q=rope_apply(q, grid_sizes, freqs),
                            k=k,
                            v=v)
        x = x.flatten(2)
        x = self.o(x)
        return x

class WanAttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 r=8,
                 alpha=16.0):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.r = r
        self.alpha = alpha

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps, r, alpha)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanT2VCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps, r, alpha)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))
        self.flora = nn.Sequential(LoRALinear(dim, ffn_dim, r, alpha), nn.GELU(approximate='tanh'), LoRALinear(ffn_dim, dim, r, alpha))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        with amp.autocast("cuda", dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1]) + e[0], seq_lens, grid_sizes,
            freqs)
        with amp.autocast("cuda", dtype=torch.float32):
            x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.norm2(x).float() * (1 + e[4]) + e[3]
            y = self.ffn(y) + self.flora(y)
            with amp.autocast("cuda", dtype=torch.float32):
                x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x

class ConditionFirstBlock(nn.Module):
    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 qk_norm=True,
                 eps=1e-6                
                 ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        # self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 12, dim) / dim**0.5)

        # layers
        self.norm_c1 = WanLayerNorm(dim, eps)
        self.norm_c2 = WanLayerNorm(dim, eps)
        self.self_attn_c1 = ConditionSelfAttention(dim, num_heads, qk_norm, eps)
        self.self_attn_c2 = ConditionSelfAttention(dim, num_heads, qk_norm, eps)
        
        self.norm3 = WanLayerNorm(dim, eps) 
        self.norm4 = WanLayerNorm(dim, eps)
        self.cross_attn = ConditionCrossAttention(dim, num_heads, qk_norm, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))
        
    def forward(self, x, c1, c2, e, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            c1(Tensor): Shape [B, L, C]
            c2(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        with amp.autocast("cuda", dtype=torch.float32):
            modulation = self.modulation.expand(e.shape[0], 12, -1)
            modulation = torch.cat([modulation[:,:6],modulation[:,6:] + e], dim = 1).chunk(12, dim =1)
            # e = (self.modulation + e).chunk(6, dim=1)
        assert modulation[0].dtype == torch.float32

        # self-attn for conditions
        y_c1 = self.self_attn_c1(self.norm_c1(c1).float() * (1 + modulation[1]) + modulation[0], seq_lens, grid_sizes,
            freqs)
        y_c2 = self.self_attn_c2(self.norm_c2(c2).float() * (1 + modulation[4]) + modulation[3], seq_lens, grid_sizes,
            freqs)
        with amp.autocast("cuda", dtype=torch.float32):
            c1 = c1 + y_c1 * modulation[2]
            c2 = c2 + y_c2 * modulation[5]
        

        # cross-attn for x attending (x,c1,c2)
        y = self.cross_attn(self.norm3(x).float() * (1 + modulation[7]) + modulation[6], c1, c2, grid_sizes, freqs)
        with amp.autocast("cuda", dtype=torch.float32):
            x = x + y * modulation[8]
        y = self.ffn(self.norm4(x).float() * (1 + modulation[10]) + modulation[9])
        with amp.autocast("cuda", dtype=torch.float32):
            x = x + y * modulation[11]

        return x, c1, c2
    
class ConditionAttentionBlock(nn.Module):
    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 qk_norm=True,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        self.eps = eps


        self.norm1 = WanLayerNorm(dim, eps)

        self.cross_attn = ConditionCrossAttention(dim, num_heads, qk_norm, eps)

        self.norm2 = WanLayerNorm(dim, eps)

        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x, c1, c2, e, grid_sizes, freqs):
        assert e.dtype == torch.float32
        with amp.autocast("cuda", dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32
        y = self.cross_attn(self.norm1(x).float() * (1 + e[1]) + e[0], c1, c2, grid_sizes, freqs)
        with amp.autocast("cuda", dtype=torch.float32):
            x = x + y * e[2]
        
        y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
        with amp.autocast("cuda", dtype=torch.float32):
            x = x + y * e[5]
        
        return x

class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6, r=8, alpha=16.0):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps
        self.r = r
        self.alpha = alpha

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)
        self.hlora = LoRALinear(dim, out_dim, r, alpha)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        assert e.dtype == torch.float32
        with amp.autocast("cuda", dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            y = self.norm(x) * (1 + e[1]) + e[0]
            x = self.head(y) + self.hlora(y)
        return x
    

# class MLPProj(torch.nn.Module):

#     def __init__(self, in_dim, out_dim):
#         super().__init__()

#         self.proj = torch.nn.Sequential(
#             torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
#             torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
#             torch.nn.LayerNorm(out_dim))

#     def forward(self, image_embeds):
#         clip_extra_context_tokens = self.proj(image_embeds)
#         return clip_extra_context_tokens
    
class VRModel(ModelMixin, ConfigMixin):
    r"""
    Video Restoration model with Wan diffusion backbone supporting text-to-video.
    """

    ignore_for_config = [
        'model_type', 'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock', 'ConditionFirstBlock', 'ConditionAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 condition_dim=1280,
                 dim=1536,
                 ffn_dim=8960,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=12,
                 num_layers=30,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 r=256,
                 alpha=512.0):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video) or 'flf2v' (first-last-frame-to-video) or 'vace'
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            condition_dim ('int', *optional", defaults to 1280):
                Condition embeddings' channels (dim_C2)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
            r ('int', *optional*, defaults to 8):
                LoRA rank for finetuning linear layers
            alpha ('float', *optional*, defaults to 16.0):
                coefficient for calculating LoRA impact as output * alpha/r
        """

        super().__init__()

        assert model_type in ['t2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.condition_dim = condition_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.r = r
        self.alpha = alpha


        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.anchor_stem = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.visual_embedding = nn.Sequential(nn.Linear(condition_dim, dim), nn.GELU(approximate='tanh'), nn.Linear(dim,dim))
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.first_layer = ConditionFirstBlock(dim, ffn_dim, num_heads, qk_norm, eps)
        blocks = []
        for i in range(num_layers):
            blocks.append(WanAttentionBlock(dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps, r, alpha))
            if (i + 1) % 4 == 0:
                blocks.append(ConditionAttentionBlock(dim, ffn_dim, num_heads, qk_norm, eps))

        self.blocks = nn.ModuleList(blocks)

        # head
        self.head = Head(dim, out_dim, patch_size, eps, r, alpha)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], dim=1)

        # initialize weights
        self.init_weights()

    def forward(
        self,
        x,
        t,
        context,
        conditions
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            conditions (Dict[Tensor], *optional*):
                Conditional video inputs, each has same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        lq_latent = conditions["lq"]
        anchor_latent = conditions["anchor"]
        dino_embedding = conditions["dino"]                       # tensor (B, T*H'*W', C_dino), H' = H/2, W'= W/2

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]      # list of tensor with (1, dim, T/1, H/2, W/2)
        lq_latent = [self.patch_embedding(u.unsqueeze(0)) for u in lq_latent]
        anchor_latent = [self.anchor_stem(u.unsqueeze(0)) for u in anchor_latent]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in lq_latent])
        
        x = [u.flatten(2).transpose(1, 2) for u in x]      # list of tensor with (1, T*H/2*W/2, dim)
        lq_latent = [u.flatten(2).transpose(1, 2) for u in lq_latent]
        anchor_latent = [u.flatten(2).transpose(1, 2) for u in anchor_latent]

        seq_lens = torch.tensor([u.size(1) for u in lq_latent], dtype=torch.long)

        x, lq_latent, anchor_latent = torch.cat(x), torch.cat(lq_latent), torch.cat(anchor_latent) # B, L, C
        lq_latent = lq_latent + anchor_latent

        # visual embeddings
        dino_embedding = self.visual_embedding(dino_embedding)  # B, L, C=C_model

        # time embeddings
        with amp.autocast("cuda", dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))
        
        x, c1, c2 = self.first_layer(x, lq_latent, dino_embedding, e0, seq_lens, grid_sizes, self.freqs)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens)
        
        iter_blocks = iter(self.blocks)
        for i in range(self.num_layers):
            x = next(iter_blocks)(x, **kwargs)
            if (i + 1) % 4 == 0:
                x = next(iter_blocks)(x, c1, c2, e0, grid_sizes, self.freqs)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack([u.float() for u in x])

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization, and set LoRA parameters to zero.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            if hasattr(m, 'lora_A'):  # Assuming LoRA uses 'lora_A' as the low-rank matrix A
            #     nn.init.zeros_(m.lora_A)  # Initialize LoRA A matrix to zero
                nn.init.kaiming_uniform_(m.lora_A, a=math.sqrt(5))
            if hasattr(m, 'lora_B'):  # Assuming LoRA uses 'lora_B' as the low-rank matrix B
                nn.init.zeros_(m.lora_B)  # Initialize LoRA B matrix to zero

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)

        # init second patch embedding
        nn.init.zeros_(self.anchor_stem.weight)
        if self.anchor_stem.bias is not None:
            nn.init.zeros_(self.anchor_stem.bias)
