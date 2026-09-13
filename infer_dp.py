import sys
sys.path.append('..')
sys.path.append('../realesrgan_degradation')
import argparse
import os
import math
import random
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

from wan.modules.our_model import VRModel as VR1
from wan.modules.our_ds2 import WanAlignedDistributionShifter as ODS
from fm_solvers_unipc_new import FlowUniPCMultistepScheduler as OurScheduler
from common import dinov3
from data import save_video


checkpoint_dir = "/remote-home/share/yingweitang/Wan21_code/Wan2.1-T2V-1.3B/"
ckpt_path = "/remote-home/share/yingweitang/checkpoints"
config = WAN_CONFIGS["t2v-1.3B"]
num_train_timesteps = config.num_train_timesteps
param_dtype = config.param_dtype            # BF-16
vae_stride = config.vae_stride
patch_size = config.patch_size


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
            x_cthw = F.pad(
                x_cthw, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate"
            )

    target_T = 1 + ((T - 1 + temporal_group - 1) // temporal_group) * temporal_group
    pad_T = target_T - T
    if pad_T > 0:
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


# ---------- dynamic prediction helpers ----------
def get_dynamic_start_time(t_pred, perception_ratio=0.0):

    omega = max(0.0, min(1.0, perception_ratio))
    t_max = 0.9998
    t_start = (1.0 - omega) * t_pred + omega * t_max
    return t_start


# ---------- core DP + DiT sampler ----------
def DSVRpipeline(
    lq_tensors,
    c_null,
    vr_model,
    dinov3_model,
    vae,
    ds,
    device,
    shift=5.0,
    sampling_steps=50,
    seed=-1,
    perception_ratio=0.5,
):
    torch.cuda.empty_cache()
    seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=device)
    seed_g.manual_seed(seed)

    original_shape = lq_tensors.shape                # B, C, T, H, W
    assert (original_shape[2] - 1) % 4 == 0, "Wrong frame length."

    context_null = [c_null.to(device)] * original_shape[0]
    length = original_shape[2]
    latent_length = 1 + (length - 1) // 4
    indices = [0] + torch.arange(3, length, 4).tolist()
    dino_inputs = lq_tensors[:, :, indices]
    dino_embeddings = dinov3(dino_inputs, dinov3_model, device)             # B, L, C_dino
    latent_shape = (
        vae.model.z_dim,
        latent_length,
        original_shape[3] // vae_stride[1],
        original_shape[4] // vae_stride[2],
    )
    seq_len = math.ceil(
        (latent_shape[2] * latent_shape[3]) / (patch_size[1] * patch_size[2]) * latent_shape[1] / patch_size[0]
    )
    assert dino_embeddings.shape[1] == seq_len, (
        "Dino embeddings should have same length as latent patch token sequences."
    )

    noise = torch.randn(
        original_shape[0],
        latent_shape[0],
        latent_shape[1],
        latent_shape[2],
        latent_shape[3],
        dtype=torch.float32,
        device=device,
    )

    lq_tensors = lq_tensors.to(device)

    with torch.inference_mode():
        lq_latents = torch.stack(vae.encode(lq_tensors))
        J_pred, delta_z_pred = ds(lq_tensors)
        mse_j_pred = ((J_pred / (1 - J_pred)) ** 2).mean(dim=[1, 2, 3, 4])
        t_pred = torch.sqrt(mse_j_pred)
        t_start = get_dynamic_start_time(t_pred, perception_ratio=perception_ratio)

    sample_scheduler = OurScheduler(num_train_timesteps=num_train_timesteps, shift=shift)

    with autocast("cuda", dtype=param_dtype), torch.inference_mode():
        conditions = {"lq": lq_latents, "dino": dino_embeddings}
        sample_scheduler.set_timesteps_3(t_start, sampling_steps, device=device, shift=shift)
        anchor_latent = lq_latents + delta_z_pred
        latent = (1 - t_start) * anchor_latent + t_start * noise
        timesteps = sample_scheduler.timesteps
        num_inference_steps = sample_scheduler.num_inference_steps
        for _, t in enumerate(tqdm(timesteps, leave=False, position=1)):
            timestep = torch.stack([t])
            v_pred = vr_model(x=latent, t=timestep, context=context_null, conditions=conditions)
            temp_x0 = sample_scheduler.step(
                v_pred,
                t,
                latent,
                return_dict=False,
                generator=seed_g,
            )[0]
            latent = temp_x0
        x0 = latent
        restored_tensor = vae.decode(x0)             # list of C, T, H, W

    return restored_tensor[0], num_inference_steps, 1000 * t_start[0].item()


@torch.inference_mode()
def restore_one_video_dp(
    lq_cthw: torch.Tensor,
    c_null: torch.Tensor,
    vr_model,
    dinov3_model,
    vae,
    ds,
    device: torch.device,
    *,
    shift: float = 5.0,
    sampling_steps: int = 50,
    seed: int = -1,
    perception_ratio: float = 0.5,
):

    vr_model.eval()
    dinov3_model.eval()
    ds.eval()

    lq_bcthw = lq_cthw.unsqueeze(0).to(device)
    restored, num_steps, t_start = DSVRpipeline(
        lq_bcthw,
        c_null,
        vr_model,
        dinov3_model,
        vae,
        ds,
        device,
        shift=shift,
        sampling_steps=sampling_steps,
        seed=seed,
        perception_ratio=perception_ratio,
    )
    print(f"[INFO] DP predicted start timestep: {t_start:.2f}, inference steps: {num_steps}")
    return restored.detach().float().cpu()


# ---------- single-video inference ----------
def run_inference_on_video(
    input_path: str,
    output_path: str,
    c_null: torch.Tensor,
    vr_model,
    dinov3_model,
    vae,
    ds,
    device: torch.device,
    *,
    sampling_steps: int = 50,
    shift: float = 5.0,
    seed: int = -1,
    fps: Optional[float] = None,
    perception_ratio: float = 0.5,
):

    print(f"[INFO] Reading input video: {input_path}")
    lq_cthw, input_fps = read_lq_video(input_path)
    C, T, H, W = lq_cthw.shape
    print(f"[INFO] Original video: T={T}, H={H}, W={W}, fps={input_fps:.2f}")

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

    restored = restore_one_video_dp(
        lq_padded,
        c_null,
        vr_model,
        dinov3_model,
        vae,
        ds,
        device,
        shift=shift,
        sampling_steps=sampling_steps,
        seed=seed,
        perception_ratio=perception_ratio,
    )

    restored = crop_to_original(restored, crop_info)
    print(f"[INFO] Output shape after cropping back: {tuple(restored.shape)}")

    output_fps = fps if fps is not None else input_fps
    save_video(restored, output_path, fps=int(round(output_fps)))
    print(f"[INFO] Saved output video: {output_path}")

    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="DP + DiT single-video inference")
    parser.add_argument("--input", "-i", required=True, help="input path")
    parser.add_argument("--output", "-o", required=True, help="output path")
    parser.add_argument("--perception_ratio", "-p", type=float, default=0.5, help="DP ratio")
    parser.add_argument("--sampling_steps", type=int, default=50)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=6, help="fps")
    parser.add_argument("--seed", type=int, default=-1, help="seed")
    args = parser.parse_args()

    device = torch.device("cuda")

    c_null = torch.load("../Exposition/context_null.pt").to("cpu")
    print("success load c_null")

    vr_model = VR1().to(device)
    ds_model = ODS().to(device)
    load_model(vr_model, f"{ckpt_path}/V1/VR_V1_010.safetensors")
    state_dict = torch.load(f"{ckpt_path}/OurDS/ODS_V3_007.pt")
    ds_model.load_state_dict(state_dict["model_state_dict"])
    print("success load vr_model & ds_model")

    vae = WanVAE(vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint), device=device)
    print("success load vae")

    dinov3_model = AutoModel.from_pretrained(
        "facebook/dinov3-vith16plus-pretrain-lvd1689m",
        device_map=device,
    )
    print("success load dinov3")

    run_inference_on_video(
        input_path=args.input,
        output_path=args.output,
        c_null=c_null,
        vr_model=vr_model,
        dinov3_model=dinov3_model,
        vae=vae,
        ds=ds_model,
        device=device,
        sampling_steps=args.sampling_steps,
        shift=args.shift,
        seed=args.seed,
        fps=args.fps,
        perception_ratio=args.perception_ratio,
    )


if __name__ == "__main__":
    main()
