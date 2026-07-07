import sys
sys.path.append('..')
sys.path.append('../realesrgan_degradation')
import argparse
import math
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.amp import autocast
from torch.nn import functional as F
from tqdm import tqdm
import torchvision

from safetensors.torch import load_model
from modelscope import AutoModel
import wan
from wan.configs import WAN_CONFIGS
from wan.modules.vae import WanVAE
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.modules.our_model import VRModel as VR1

from common import dinov3
from data import save_video


# ---------- helpers ----------
def _video_to_tensor_chw_t(video_thwc_uint8: torch.Tensor) -> torch.Tensor:

    v = video_thwc_uint8.permute(0, 3, 1, 2).contiguous().float() / 255.0
    v = v * 2.0 - 1.0
    v = v.permute(1, 0, 2, 3).contiguous()
    return v


def read_lq_video(video_path: str) -> Tuple[torch.Tensor, float]:

    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    try:
        v_thwc, _, info = torchvision.io.read_video(str(video_path), pts_unit="sec")
    except Exception as e:
        raise RuntimeError(f"Failed to read video {video_path}: {e}")

    if v_thwc.numel() == 0:
        raise ValueError(f"Empty video: {video_path}")

    fps = 24.0
    if isinstance(info, dict):
        fps = float(info.get("video_fps", 24.0))

    v = _video_to_tensor_chw_t(v_thwc)
    return v, fps


def pad_to_constraints(
    x_cthw: torch.Tensor,
    spatial_multiple: int = 16,
    temporal_group: int = 4,
) -> Tuple[torch.Tensor, dict]:

    C, T, H, W = x_cthw.shape

    # 空域对齐（16 倍数）
    new_H = ((H + spatial_multiple - 1) // spatial_multiple) * spatial_multiple
    new_W = ((W + spatial_multiple - 1) // spatial_multiple) * spatial_multiple
    pad_top = (new_H - H) // 2
    pad_bottom = new_H - H - pad_top
    pad_left = (new_W - W) // 2
    pad_right = new_W - W - pad_left

    if pad_top or pad_bottom or pad_left or pad_right:
        try:
            x_cthw = F.pad(
                x_cthw, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect"
            )
        except Exception:
            # 当尺寸过小时 reflect 可能失败，回退到 replicate
            x_cthw = F.pad(
                x_cthw, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate"
            )

    # 时域对齐（4N+1）
    target_T = 1 + ((T - 1 + temporal_group - 1) // temporal_group) * temporal_group
    pad_T = target_T - T
    if pad_T > 0:
        # 用最后一帧复制补齐，避免引入不存在的运动
        x_cthw = F.pad(x_cthw, (0, 0, 0, 0, 0, pad_T), mode="replicate")

    crop_info = {
        "orig_T": T,
        "orig_H": H,
        "orig_W": W,
        "pad_top": pad_top,
        "pad_left": pad_left,
    }
    return x_cthw, crop_info


def crop_to_original(x_cthw: torch.Tensor, crop_info: dict) -> torch.Tensor:

    T = crop_info["orig_T"]
    H = crop_info["orig_H"]
    W = crop_info["orig_W"]
    top = crop_info["pad_top"]
    left = crop_info["pad_left"]
    return x_cthw[:, :T, top:top + H, left:left + W]


# ---------- core sampler ----------
@torch.inference_mode()
def restore_one_video_cthw(
    lq_cthw: torch.Tensor,                 # [C,T,H,W] in [-1,1]
    rs_model,
    dinov3_model,
    vae,
    c_null: torch.Tensor,                  # [L, C]
    device: torch.device,
    *,
    sampling_steps: int = 50,
    shift: float = 5.0,
    seed: int = 2025,
    num_train_timesteps: int = 1000,
    param_dtype=torch.bfloat16,
    vae_stride=(4, 8, 8),
    patch_size=(1, 2, 2),
):

    rs_model.eval()
    dinov3_model.eval()

    # [B,C,T,H,W]
    lq_bcthw = lq_cthw.unsqueeze(0).to(device)

    B, C, T, H, W = lq_bcthw.shape
    assert (T - 1) % 4 == 0, "Need (T-1)%4==0"

    # context_null: list length B
    context_null = [c_null.to(device)] * B

    latent_length = 1 + (T - 1) // 4
    indices = [0] + torch.arange(3, T, 4).tolist()
    dino_in = lq_bcthw[:, :, indices]                      # [B,C,T',H,W]
    dino_emb = dinov3(dino_in, dinov3_model, device)       # [B,L,1280]

    # VAE encode
    lq_latent = torch.stack(vae.encode(lq_bcthw))           # [B,Cz,T',H',W']

    # 采样噪声 latent
    latent_shape = (
        vae.model.z_dim,
        latent_length,
        H // vae_stride[1],
        W // vae_stride[2],
    )

    # token length 对齐
    H_ = latent_shape[2]
    W_ = latent_shape[3]
    spatial_tokens = (H_ // patch_size[1]) * (W_ // patch_size[2])
    seq_len = math.ceil(spatial_tokens * (latent_length / patch_size[0]))
    assert dino_emb.shape[1] == seq_len, f"dino_len={dino_emb.shape[1]} != seq_len={seq_len}"

    g = torch.Generator(device=device)
    g.manual_seed(seed)

    latent = torch.randn(
        B, latent_shape[0], latent_shape[1], latent_shape[2], latent_shape[3],
        device=device, dtype=torch.float32, generator=g
    )

    sample_scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=num_train_timesteps,
        shift=shift,
    )
    sample_scheduler.set_timesteps(sampling_steps, device=device)

    timesteps_int = sample_scheduler.timesteps.to(device)
    sigmas = sample_scheduler.sigmas[:-1].to(device).to(torch.float32)
    timesteps_float = sigmas * float(num_train_timesteps)

    conditions = {"lq": lq_latent, "dino": dino_emb}

    with autocast("cuda", dtype=param_dtype):
        torch.cuda.synchronize()
        for i, t in enumerate(tqdm(timesteps_int, desc="Sampling", leave=False)):
            t_model = timesteps_float[i].expand(B)
            v_pred = rs_model(x=latent, t=t_model, context=context_null, conditions=conditions)
            latent = sample_scheduler.step(v_pred, t, latent, return_dict=False, generator=g)[0]
        torch.cuda.synchronize()

    restored_list = vae.decode(latent)
    return restored_list[0].detach().float().cpu()


# ---------- single-video inference ----------
def run_inference_on_video(
    input_path: str,
    output_path: str,
    rs_model,
    dinov3_model,
    vae,
    c_null: torch.Tensor,
    device: torch.device,
    *,
    sampling_steps: int = 50,
    shift: float = 5.0,
    seed: int = 2025,
    fps: Optional[float] = None,
    num_train_timesteps: int = 1000,
    param_dtype=torch.bfloat16,
    vae_stride=(4, 8, 8),
    patch_size=(1, 2, 2),
):

    print(f"[INFO] Reading input video: {input_path}")
    lq_cthw, input_fps = read_lq_video(input_path)
    C, T, H, W = lq_cthw.shape
    print(f"[INFO] Original video: T={T}, H={H}, W={W}, fps={input_fps:.2f}")

    # 健壮性检查与提示
    if H % 16 != 0 or W % 16 != 0:
        print(
            f"[WARN] Resolution {H}x{W} is not a multiple of 16; "
            "padding upward and cropping back."
        )
    if (T - 1) % 4 != 0:
        print(
            f"[WARN] Frame count {T} is not 4N+1; "
            "padding upward and trimming back."
        )

    lq_padded, crop_info = pad_to_constraints(lq_cthw, spatial_multiple=16, temporal_group=4)
    print(f"[INFO] Padded shape for model: {tuple(lq_padded.shape)}")

    restored = restore_one_video_cthw(
        lq_padded,
        rs_model,
        dinov3_model,
        vae,
        c_null,
        device,
        sampling_steps=sampling_steps,
        shift=shift,
        seed=seed,
        num_train_timesteps=num_train_timesteps,
        param_dtype=param_dtype,
        vae_stride=vae_stride,
        patch_size=patch_size,
    )

    restored = crop_to_original(restored, crop_info)
    print(f"[INFO] Output shape after cropping back: {tuple(restored.shape)}")

    output_fps = fps if fps is not None else input_fps
    save_video(restored, output_path, fps=int(round(output_fps)))
    print(f"[INFO] Saved output video: {output_path}")

    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="DTI single-video inference")
    parser.add_argument("--input", "-i", required=True, help="输入低质量视频路径")
    parser.add_argument("--output", "-o", required=True, help="输出修复后视频路径")
    parser.add_argument("--sampling_steps", type=int, default=50)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=6, help="输出 fps，默认使用输入视频 fps")
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    device = torch.device("cuda")
    checkpoint_dir = "/remote-home/share/yingweitang/Wan21_code/Wan2.1-T2V-1.3B/"
    config = WAN_CONFIGS["t2v-1.3B"]
    num_train_timesteps = config.num_train_timesteps
    param_dtype = config.param_dtype            # BF-16
    vae_stride = config.vae_stride
    patch_size = config.patch_size

    c_null = torch.load("../Exposition/context_null.pt").to("cpu")
    print("success load c_null")
    rs_model = VR1(dim=1536, ffn_dim=8960, num_heads=12, num_layers=30).to(device)
    print("success initilize rs_model")
    load_model(rs_model, "/remote-home/share/yingweitang/checkpoints/V1/VR_V1_010.safetensors")
    print("success load model")

    # VAE & DINOv3
    vae = WanVAE(vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint), device=device)
    print("success load vae")
    dinov3_model = AutoModel.from_pretrained(
        "facebook/dinov3-vith16plus-pretrain-lvd1689m",
        device_map=device
    )
    print("success load dinov3")

    run_inference_on_video(
        input_path=args.input,
        output_path=args.output,
        rs_model=rs_model,
        dinov3_model=dinov3_model,
        vae=vae,
        c_null=c_null,
        device=device,
        sampling_steps=args.sampling_steps,
        shift=args.shift,
        seed=args.seed,
        fps=args.fps,
        num_train_timesteps=num_train_timesteps,
        param_dtype=param_dtype,
        vae_stride=vae_stride,
        patch_size=patch_size,
    )


if __name__ == "__main__":
    main()
