import sys
sys.path.append('..')
sys.path.append('../realesrgan_degradation')

import math
import torch
import torch.nn.functional as F
import random
import torchvision
from tqdm import tqdm
from torchvision.transforms import v2
from torch.amp import autocast
from transformers.image_utils import load_images
from collections import defaultdict
from torch.distributed.tensor import DTensor
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP2
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor.placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
)
from wan.configs import WAN_CONFIGS
from wan.modules.GE_model_V1 import GEModel
# from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from fm_solvers_unipc_new import FlowUniPCMultistepScheduler
from data import save_video

config = WAN_CONFIGS["t2v-1.3B"]
n_prompt = config.sample_neg_prompt
num_train_timesteps = config.num_train_timesteps
param_dtype = config.param_dtype            # BF-16
vae_stride = config.vae_stride
patch_size = config.patch_size
num_layers = config.num_layers

class EMA:
    def __init__(self, model, beta):
        self.model = model
        self.beta = beta
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                # In-place EMA update to avoid reallocations.
                self.shadow[name].mul_(self.beta).add_(param.detach(), alpha=1.0 - self.beta)
    
    @torch.no_grad()
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.detach().clone()
                param.copy_(self.shadow[name])
    
    @torch.no_grad()
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.copy_(self.backup[name])
        self.backup = {}


class ShardedEMA:
    def __init__(self, fsdp_model: FSDP2, beta: float):
        self.model = fsdp_model
        self.beta = beta
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.shadow[name].mul_(self.beta).add_(param.detach(), alpha = 1.0 - self.beta)
    
    @torch.no_grad()
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.detach().clone()
                param.copy_(self.shadow[name])
    
    @torch.no_grad()
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.copy_(self.backup[name])
        self.backup = {}


def dinov3(video, model, device):
    '''
    Input: video: tensor(B,3,T,H,W)
    Return: output: tensor(B, T*H/16*W/16, C_dino)
    '''
    b,c,t,h,w = video.shape
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    video_tensor = (video.transpose(1,2) + 1.0) / 2.0
    image_tensor = normalize(video_tensor).to(device)
    
    output = []
    with torch.inference_mode():
        for i in image_tensor:
            outputs = model(i)
            output.append(outputs.last_hidden_state.clone()[:,5:])
    # outputs = model(image_tensor.reshape(-1, c, h, w))
    # output = outputs.last_hidden_state.clone()[:,5:]
    # c_dino, L_dino = output.shape[2], output.shape[1]
    # output = output.reshape(b, t*L_dino, c_dino)
    c_dino = output[0].shape[2]
    output = torch.stack(output).reshape(b, -1, c_dino)
    return output

@torch.no_grad()
def clip_encode_patches(video, model, processor, device):
    """
    Input:
        video: tensor [B, 3, T, H, W]
    Return:
        output: tensor [B, T * H/16 * W/16, 768]
    """
    b, c, t, h, w = video.shape
    assert c == 3, "CLIP encoder expects 3-channel RGB input."
    assert h % 16 == 0 and w % 16 == 0, "H and W must be divisible by 16."

    video_tensor = video.transpose(1, 2).contiguous()
    video_tensor = (video_tensor + 1.0) / 2.0

    mean = torch.tensor(processor.image_mean, device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor(processor.image_std, device=device).view(1, 1, 3, 1, 1)

    image_tensor = (video_tensor.to(device).float() - mean) / std

    output = []
    expected_patch_num = (h // 16) * (w // 16)

    with torch.inference_mode():
        for i in image_tensor:
            # i: [T, 3, H, W]
            outputs = model(pixel_values=i, interpolate_pos_encoding=True)
            patch_tokens = outputs.last_hidden_state[:, 1:]   # [T, L_patch, 768]

            assert patch_tokens.shape[1] == expected_patch_num, (
                f"Unexpected patch number: got {patch_tokens.shape[1]}, expected {expected_patch_num}"
            )
            output.append(patch_tokens)

    c_clip = output[0].shape[2]
    output = torch.stack(output).reshape(b, -1, c_clip)

    return output


def evalGE(save_path, model, vae, dino_embeddings, c_null, device, seed = -1, sampling_steps = 50, shift = 5.0):
    """
    model: generative model
    dino_embedding: L = T' * H' * W', C_dino
    """
    torch.cuda.empty_cache()
    seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=device)
    seed_g.manual_seed(seed)

    generated_videos = []
    context_null = [c_null.to(device)]

    model.eval()
    for dino_embedding, latent_length in zip(dino_embeddings, [7,7,4,5,5]):
        latent_shape = (16, latent_length, 64, 64)
        assert dino_embedding.shape[0] == latent_shape[1] * latent_shape[2] / 2 * latent_shape[3] / 2, "Dino embeddings should have same length as latent patch token sequences."
        dino_embedding = dino_embedding.to(device)
        noise = torch.randn(
            1,
            latent_shape[0],
            latent_shape[1],
            latent_shape[2],
            latent_shape[3],
            dtype=torch.float32,
            device=device)
        with autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
            sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                    sampling_steps, device=device, shift=shift)
            timesteps = sample_scheduler.timesteps
            latent = noise
            for _, t in enumerate(tqdm(timesteps)):
                timestep = torch.stack([t])
                v_pred = model(x = latent, c = dino_embedding, t = timestep, context = context_null)
                temp_x0 = sample_scheduler.step(
                    v_pred,
                    t,
                    latent,
                    return_dict=False,
                    generator=seed_g)[0]
                latent = temp_x0
            x0 = latent
            generated_videos.append(vae.decode(x0)[0].to("cpu"))                                 # C, T, H, W

    for i, video in enumerate(generated_videos):
        torch.save(video, save_path + f"{i+1}.pt")
        save_video(video, save_path + f"{i+1}.mp4", fps = 3)

    del generated_videos, sample_scheduler


def evalGEAE(save_path, model, vae, dino_embeddings, ae_latents, timesteps, c_null, device, seed = -1, sampling_steps = 50, shift = 5.0):
    torch.cuda.empty_cache()
    seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=device)
    seed_g.manual_seed(seed)

    generated_videos = []
    context_null = [c_null.to(device)]

    model.eval()
    for dino_embedding, ae_latent, timestep, latent_length in zip(dino_embeddings, ae_latents, timesteps, [7,7,4,5,5]):
        latent_shape = (16, latent_length, 64, 64)
        assert dino_embedding.shape[0] == latent_shape[1] * latent_shape[2] / 2 * latent_shape[3] / 2, "Dino embeddings should have same length as latent patch token sequences."
        dino_embedding, ae_latent, timestep = dino_embedding.to(device), ae_latent.to(device), timestep.to(device)
        # noise = torch.randn(
        #     1,
        #     latent_shape[0],
        #     latent_shape[1],
        #     latent_shape[2],
        #     latent_shape[3],
        #     dtype=torch.float32,
        #     device=device)
        with autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
            sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
            sampling_steps = math.ceil((1.0 - timestep.item() / 1000.0) * sampling_steps)
            sample_scheduler.set_timesteps_2(
                    start_step = timestep, num_inference_steps = sampling_steps, device=device, shift=shift)
            timesteps = sample_scheduler.timesteps
            latent = ae_latent
            for _, t in enumerate(tqdm(timesteps)):
                timestep = torch.stack([t])
                v_pred = model(x = latent, c = dino_embedding, t = timestep, context = context_null)
                temp_x0 = sample_scheduler.step(
                    v_pred,
                    t,
                    latent,
                    return_dict=False,
                    generator=seed_g)[0]
                latent = temp_x0
            x0 = latent
            generated_videos.append(vae.decode(x0)[0].to("cpu"))                                 # C, T, H, W

    for i, video in enumerate(generated_videos):
        torch.save(video, save_path + f"{i+1}.pt")
        save_video(video, save_path + f"{i+1}.mp4", fps = 3)

    del generated_videos, sample_scheduler


def low_pass(x, spatial_scale=0.5):
    """
    使用 2D bicubic + antialias 提取空间低频

    x: [B, C, T, H, W]
    """
    B, C, T, H, W = x.shape
    
    # === 第一步：空间低通滤波 (H, W) ===
    # 将 T 合并到 B，变成 [B*T, C, H, W]
    x_spatial = x.reshape(B * T, C, H, W)
    
    # 空间下采样 (必须 antialias=True)
    x_down = F.interpolate(x_spatial, scale_factor=spatial_scale, mode='bicubic', antialias=True)
    # 空间上采样
    x_up = F.interpolate(x_down, size=(H, W), mode='bicubic', antialias=True)
    
    # 恢复视频形状
    x_spatial_low = x_up.view(B, C, T, H, W)
    
    return x_spatial_low

def high_pass(x, spatial_scale=0.5):
    return x - low_pass(x, spatial_scale=spatial_scale)


