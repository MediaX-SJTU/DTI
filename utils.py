import sys
sys.path.append('..')
sys.path.append('../realesrgan_degradation')

import re
import os
import glob
import torch
import random
from collections import Counter
from PIL import Image

from tqdm import tqdm
from safetensors import safe_open
from safetensors.torch import load_file, load_model, save_file, save_model
from torchvision import transforms
import wan
from wan.configs import WAN_CONFIGS
import lpips as lp
from skimage.metrics import structural_similarity as ssim
from torchmetrics.multimodal import CLIPImageQualityAssessment

from data import RVSR_D, temporal_corrupt, save_video

checkpoint_dir = "./Wan2.1-T2V-1.3B/"
LQ_ROOT = "../../../share/yingweitang/VFHQ_Test/LQ/"
LQ_ROOT2 = "../../../share/yingweitang/VFHQ_Test/LQ_2/"
sample_guidance_scale = 5
# sp_size = 1
config = WAN_CONFIGS["t2v-1.3B"]
n_prompt = config.sample_neg_prompt
num_train_timesteps = config.num_train_timesteps
param_dtype = config.param_dtype            # BF-16
vae_stride = config.vae_stride
patch_size = config.patch_size
num_layers = config.num_layers

def Exact_DP(real_length):
    """
    Get the combination of legal lengths([1,25]) that can exactly compose real_length of video.

    Return: 
            dp[m]: the number of intervals; 
            combination: the smallest combination possible 
    """
    assert real_length > 0, "Video length can't be zero."
    m = real_length
    mi_list = []
    for n in range(7):
        mi_list.append(1 + 4 * n)
    dp = [float('inf')] * (m + 1)
    last_choice = [-1] * (m + 1)  
    dp[0] = 0  
    for mi in mi_list:
        for i in range(mi, m + 1):
            if dp[i - mi] + 1 < dp[i]:
                dp[i] = dp[i - mi] + 1
                last_choice[i] = mi
    if dp[m] == float('inf'):
        return -1, []  
    else:
        combination = []
        current = m
        while current > 0:
            mi = last_choice[current]
            combination.append(mi)
            current -= mi
        return dp[m], combination  

def parrallel_counter(combination):
    composition = Counter(combination)
    division = []
    first_frame = 0
    for k in composition:
        assert 100 // k * k + 100 % k == 100, "Math Error."
        number = 100 // k
        group = composition[k] // number
        rest = composition[k] % number
        for g in range(group):
            subset = []
            for n in range(number):
                subset.append([first_frame, first_frame + k])
                first_frame += k
            division.append(subset)
        if rest > 0 :
            subset = []
            for r in range(rest):
                subset.append([first_frame, first_frame + k])
                first_frame += k
            division.append(subset)
    return division


def generate_lq(gt_root, lq_root, temporal_degrade_threshold = 0.5):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    # gt_clip_paths = []
    i = 1
    for clip in tqdm(sorted(os.listdir(gt_root))):
        clip_path = os.path.join(gt_root, clip)
        if os.path.isdir(clip_path):
            # gt_clip_paths.append(clip_path)
            information = {"GT_path": clip_path,}
            files = sorted(glob.glob(os.path.join(clip_path, '*.png')))
            frames = [transform(Image.open(f).convert('RGB')) for f in files] # [C,H,W]
            length = len(frames)
            # truncation_length = 1 + (length - 1) // 4 * 4
            truncation_length = length
            information["Effective_length"] = str(truncation_length)
            frames = frames[:truncation_length]
            video = torch.stack(frames, dim = 1)                              # C, T, H, W
            degraded_video = RVSR_D(video)
            p = random.random()
            if p < temporal_degrade_threshold:
                degraded_video = temporal_corrupt(degraded_video)
            save_video(degraded_video, f"../Exposition/LQ_Videos_2/{i:03d}.mp4", fps = 20)
            tensor_dict = {"LQ_tensor": degraded_video.clone()}
            save_path = os.path.join(lq_root, clip, "lq.safetensors")
            os.makedirs(os.path.dirname(save_path), exist_ok= True)
            save_file(tensor_dict, save_path, information)
            i += 1

def generate_gt(gt_root):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    i = 1
    for clip in tqdm(sorted(os.listdir(gt_root))):
        clip_path = os.path.join(gt_root, clip)
        if os.path.isdir(clip_path):
            files = sorted(glob.glob(os.path.join(clip_path, '*.png')))
            frames = [transform(Image.open(f).convert('RGB')) for f in files] # [C,H,W]
            video = torch.stack(frames, dim = 1)                              # C, T, H, W
            save_video(video, f"../Exposition/GT_Videos_20/{i:03d}.mp4", fps = 20, quality=6)
            i += 1

def read_video(lq_root):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])    
    gt_videos = []
    lq_videos = []
    captions = []
    clip_paths = []
    for clip in sorted(os.listdir(lq_root)):
        clip_path = os.path.join(lq_root, clip)
        if os.path.isdir(clip_path):
            with safe_open(os.path.join(clip_path, "lq.safetensors"), framework="pt") as f:
                metadata = f.metadata()
                lq_videos.append(f.get_tensor("LQ_tensor"))                               # C, T, H, W
            gt_path = metadata["GT_path"]
            effective_length = int(metadata["Effective_length"])
            # lq_dict, metadata = load_file(os.path.join(clip_path, "lq.safetensors"))
            # lq_videos.append(lq_dict["LQ_tensor"])                            
            files = sorted(glob.glob(os.path.join(gt_path, '*.png')))
            frames = [transform(Image.open(f).convert('RGB')) for f in files]
            frames = frames[:effective_length]
            gt_videos.append(torch.stack(frames, dim=1))
            caption = load_file(os.path.join(clip_path, "caption.safetensors"))["T5Tensor"]
            captions.append(caption)
            clip_paths.append(clip_path)
    return gt_videos, lq_videos, captions, clip_paths

def read_video2(lq_root):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])    
    gt_videos = []
    lq_videos = []
    clip_names = []
    for clip in sorted(os.listdir(lq_root)):
        clip_path = os.path.join(lq_root, clip)
        if os.path.isdir(clip_path):
            with safe_open(os.path.join(clip_path, "lq.safetensors"), framework="pt") as f:
                metadata = f.metadata()
                lq_videos.append(f.get_tensor("LQ_tensor"))                               # C, T, H, W
            gt_path = metadata["GT_path"]
            gt_path = gt_path.replace("../../../share", "/remote-home/share")
            effective_length = int(metadata["Effective_length"])                          
            files = sorted(glob.glob(os.path.join(gt_path, '*.png')))
            frames = [transform(Image.open(f).convert('RGB')) for f in files]
            frames = frames[:effective_length]
            gt_videos.append(torch.stack(frames, dim=1))
            clip_names.append(clip)
    return gt_videos, lq_videos, clip_names


def PSNR_V(video1, video2):
    """
    videos: tensors in shape C, T, H, W
    """
    assert video1.shape == video2.shape, "Videos must have the same shape."
    length = video1.shape[1]

    psnr_values = []
    for t in range(length):
        mse = torch.mean((video1[:, t, :, :] - video2[:, t, :, :]) ** 2)
        psnr_value = 10 * torch.log10(1.0 / mse)
        psnr_values.append(psnr_value)

    psnr = torch.mean(torch.tensor(psnr_values)).item()
    return psnr

def LPIPS_V(video1, video2, loss_fn):
    """
    videos: tensors in shape C, T, H, W
    """
    assert video1.shape == video2.shape, "Videos must have the same shape."
    # loss_fn = lpips.LPIPS(net='alex')
    loss_fn.to(video1.device)
    lpips_values = loss_fn(video1.transpose(0,1), video2.transpose(0,1))
    lpips_score = torch.mean(lpips_values).item()
    return lpips_score

def OC_LPIPS(video1, video2, loss_fn):
    assert video1.device == video2.device, "Two tensors should be on the same device."
    groups1, groups2 = [], []
    final_score = 0.0
    for i in [0, 144, 288]:
        for j in [0, 144, 288]:
            groups1.append(video1[:,:, i:i+224, j:j+224])
            groups2.append(video2[:,:, i:i+224, j:j+224])
    for v1, v2 in zip(groups1, groups2):
        final_score += LPIPS_V(v1, v2, loss_fn)
    return final_score / 9.0

def SSIM_V(video1, video2):
    """
    videos: tensors in shape C, T, H, W
    """
    assert video1.shape == video2.shape, "Videos must have the same shape."
    length = video1.shape[1]
    ssim_values = []
    for t in range(length):
        ssim_values.append(ssim(video1[:,t,:,:].cpu().numpy(), video2[:,t,:,:].cpu().numpy(), data_range=2.0, channel_axis=0))
    ssim_value = sum(ssim_values) / len(ssim_values)
    return ssim_value

def CLIPIQA_V(video):
    """
    video: tensors in shape C, T, H, W
    """
    metric = CLIPImageQualityAssessment(data_range = 2, prompts = ('quality', 'natural'))
    clipiqa_dict = metric(video.transpose(0,1))
    quality, natural = torch.mean(clipiqa_dict['quality']).float().item(), torch.mean(clipiqa_dict['natural']).float().item() 
    return quality, natural

def verify(tensors):
    for i, tensor in enumerate(tensors):
        if i < 2:
            assert tensor.shape[1] == 25, "First two tensors' length wrong."
        elif i >= 3:
            assert tensor.shape[1] == 17, "Last two tensors' length wrong."
        else:
            assert tensor.shape[1] == 13, "Middle tensor's length wrong."
            
def RSC(generator, device):
    loss_fn = lp.LPIPS(net='alex')
    save_path = "/remote-home/share/yingweitang/VFHQ_EVAL"
    clips = []
    if generator == "flashvsr":
        save_path_t, save_path_v = save_path + "/Tensors/FlashVSR", save_path + "/Videos/FlashVSR"
        os.makedirs(save_path_t, exist_ok=True)
        os.makedirs(save_path_v, exist_ok=True)
        path = "/remote-home/huqiang/chenyan/VFHQ/flashvsr_results/tensor"
        for filename in sorted(os.listdir(path)):
            with safe_open(os.path.join(path, filename), framework="pt") as f:
                video_tensor = f.get_tensor("video")
            if video_tensor.dtype != torch.float:
                video_tensor = video_tensor.type(torch.float)
            match = re.match(r"([A-Za-z0-9\+\-_]+)_([0-9]+)_seed([0-9]+)", filename)
            assert match, "The tensor name must match."
            clip_part = match.group(1)  
            index_part = match.group(2)  
            # seed_part = match.group(3)
            clips.append(clip_part)
            save_t = save_path_t + f"/{clip_part}/{index_part}.pt"
            save_v = save_path_v + f"/{clip_part}/{index_part}.mp4"
            os.makedirs(os.path.dirname(save_t), exist_ok=True)
            os.makedirs(os.path.dirname(save_v), exist_ok=True)
            torch.save(video_tensor, save_t)
            save_video(video_tensor, save_v, fps=6)
        gt_path = save_path + "/Tensors/GT"
        PSNR, SSIM, LPIPS = [], [], []
        for clip in sorted(os.listdir(gt_path)):
            psnr, ssim, lpips = 0.0, 0.0, 0.0
            for i, f in enumerate(sorted(os.listdir(os.path.join(gt_path, clip)))):
                file = os.path.join(gt_path, clip, f)
                corresponding = file.replace("GT", "FlashVSR")
                if i < 2:
                    gt_tensor = torch.load(file)[:,:21].to(device)
                else:
                    gt_tensor = torch.load(file).to(device)
                rs_tensor = torch.load(corresponding).to(device)
                assert gt_tensor.shape[1] == rs_tensor.shape[1], "Tensors' length must equal."
                psnr += PSNR_V(rs_tensor, gt_tensor)
                ssim += SSIM_V(rs_tensor, gt_tensor)
                lpips += LPIPS_V(rs_tensor, gt_tensor, loss_fn)
            psnr, ssim, lpips = psnr / 5.0, ssim / 5.0, lpips / 5.0
            PSNR.append(psnr)
            SSIM.append(ssim)
            LPIPS.append(lpips)
    else:
        if generator == "seedvr2":
            save_path_t, save_path_v = save_path + "/Tensors/SeedVR2", save_path + "/Videos/SeedVR2"
            path = "/remote-home/huqiang/chenyan/VFHQ/seedvr_results/tensor"
        if generator == "vividvr":
            save_path_t, save_path_v = save_path + "/Tensors/VividVR", save_path + "/Videos/VividVR"
            path = "/remote-home/huqiang/chenyan/VFHQ/Vivid-VR_results/tensors"
        os.makedirs(save_path_t, exist_ok=True)
        os.makedirs(save_path_v, exist_ok=True)
        for filename in sorted(os.listdir(path)):
            with safe_open(os.path.join(path, filename), framework="pt") as f:
                video_tensor = f.get_tensor("video")
            if video_tensor.dtype != torch.float:
                video_tensor = video_tensor.type(torch.float)
            match = re.match(r"([A-Za-z0-9\+\-_]+)_([0-9]+)", filename)
            assert match, "The tensor name must match."
            clip_part = match.group(1)  
            index_part = match.group(2)
            clips.append(clip_part)  
            save_t = save_path_t + f"/{clip_part}/{index_part}.pt"
            save_v = save_path_v + f"/{clip_part}/{index_part}.mp4"
            os.makedirs(os.path.dirname(save_t), exist_ok=True)
            os.makedirs(os.path.dirname(save_v), exist_ok=True)
            torch.save(video_tensor, save_t)
            save_video(video_tensor, save_v, fps=6)
        gt_path = save_path + "/Tensors/GT"
        PSNR, SSIM, LPIPS = [], [], []
        for clip in sorted(os.listdir(gt_path)):
            psnr, ssim, lpips = 0.0, 0.0, 0.0
            for i, f in enumerate(sorted(os.listdir(os.path.join(gt_path, clip)))):
                file = os.path.join(gt_path, clip, f)
                if generator == "seedvr2":
                    corresponding = file.replace("GT", "SeedVR2")
                if generator == "vividvr":
                    corresponding = file.replace("GT", "VividVR")                   
                gt_tensor, rs_tensor = torch.load(file).to(device), torch.load(corresponding).to(device)
                assert gt_tensor.shape[1] == rs_tensor.shape[1], "Tensors' length must equal."
                psnr += PSNR_V(rs_tensor, gt_tensor)
                ssim += SSIM_V(rs_tensor, gt_tensor)
                lpips += LPIPS_V(rs_tensor, gt_tensor, loss_fn)
            psnr, ssim, lpips = psnr / 5.0, ssim / 5.0, lpips / 5.0
            PSNR.append(psnr)
            SSIM.append(ssim)
            LPIPS.append(lpips)
    clips = clips[::5]
    return PSNR, SSIM, LPIPS, clips
    

def load_pretrained_weight(old_state_dict, new_state_dict, num_layers):
    frozen_parameters = []
    new_state_dict["patch_embedding.weight"] = old_state_dict["patch_embedding.weight"]
    new_state_dict["patch_embedding.bias"] = old_state_dict["patch_embedding.bias"]
    new_state_dict["text_embedding.0.weight"] = old_state_dict["text_embedding.0.weight"]
    new_state_dict["text_embedding.0.bias"] = old_state_dict["text_embedding.0.bias"]
    new_state_dict["text_embedding.2.weight"] = old_state_dict["text_embedding.2.weight"]
    new_state_dict["text_embedding.2.bias"] = old_state_dict["text_embedding.2.bias"]
    new_state_dict["time_embedding.0.weight"] = old_state_dict["time_embedding.0.weight"]
    new_state_dict["time_embedding.0.bias"] = old_state_dict["time_embedding.0.bias"]
    new_state_dict["time_embedding.2.weight"] = old_state_dict["time_embedding.2.weight"]
    new_state_dict["time_embedding.2.bias"] = old_state_dict["time_embedding.2.bias"]
    new_state_dict["time_projection.1.weight"] = old_state_dict["time_projection.1.weight"]
    new_state_dict["time_projection.1.bias"] = old_state_dict["time_projection.1.bias"]
    frozen_parameters.extend(["patch_embedding.weight", "patch_embedding.bias", "text_embedding.0.weight", "text_embedding.0.bias", "text_embedding.2.weight", 
    "text_embedding.2.bias", "time_embedding.0.weight", "time_embedding.0.bias", "time_embedding.2.weight", "time_embedding.2.bias", "time_projection.1.weight",
    "time_projection.1.bias"])    

    for i in range(num_layers):
        new_state_dict[f"blocks.{(i // 4) + i}.modulation"] = old_state_dict[f"blocks.{i}.modulation"]
        new_state_dict[f"blocks.{(i // 4) + i}.self_attn.q.weight"] = old_state_dict[f"blocks.{i}.self_attn.q.weight"]
        new_state_dict[f"blocks.{(i // 4) + i}.self_attn.q.bias"] = old_state_dict[f"blocks.{i}.self_attn.q.bias"]
        new_state_dict[f"blocks.{(i // 4) + i}.self_attn.k.weight"] = old_state_dict[f"blocks.{i}.self_attn.k.weight"]
        new_state_dict[f"blocks.{(i // 4) + i}.self_attn.k.bias"] = old_state_dict[f"blocks.{i}.self_attn.k.bias"]
        new_state_dict[f"blocks.{(i // 4) + i}.self_attn.v.weight"] = old_state_dict[f"blocks.{i}.self_attn.v.weight"]
        new_state_dict[f"blocks.{(i // 4) + i}.self_attn.v.bias"] = old_state_dict[f"blocks.{i}.self_attn.v.bias"]
        new_state_dict[f"blocks.{(i // 4) + i}.self_attn.o.weight"] = old_state_dict[f"blocks.{i}.self_attn.o.weight"]
        new_state_dict[f"blocks.{(i // 4) + i}.self_attn.o.bias"] = old_state_dict[f"blocks.{i}.self_attn.o.bias"]
        new_state_dict[f"blocks.{(i // 4) + i}.self_attn.norm_q.weight"] = old_state_dict[f"blocks.{i}.self_attn.norm_q.weight"]
        new_state_dict[f"blocks.{(i // 4) + i}.self_attn.norm_k.weight"] = old_state_dict[f"blocks.{i}.self_attn.norm_k.weight"]
        new_state_dict[f"blocks.{(i // 4) + i}.norm3.weight"] = old_state_dict[f"blocks.{i}.norm3.weight"]
        new_state_dict[f"blocks.{(i // 4) + i}.norm3.bias"] = old_state_dict[f"blocks.{i}.norm3.bias"]
        new_state_dict[f"blocks.{(i // 4) + i}.cross_attn.q.weight"] = old_state_dict[f"blocks.{i}.cross_attn.q.weight"]
        new_state_dict[f"blocks.{(i // 4) + i}.cross_attn.q.bias"] = old_state_dict[f"blocks.{i}.cross_attn.q.bias"]
        new_state_dict[f"blocks.{(i // 4) + i}.cross_attn.k.weight"] = old_state_dict[f"blocks.{i}.cross_attn.k.weight"]
        new_state_dict[f"blocks.{(i // 4) + i}.cross_attn.k.bias"] = old_state_dict[f"blocks.{i}.cross_attn.k.bias"]
        new_state_dict[f"blocks.{(i // 4) + i}.cross_attn.v.weight"] = old_state_dict[f"blocks.{i}.cross_attn.v.weight"]
        new_state_dict[f"blocks.{(i // 4) + i}.cross_attn.v.bias"] = old_state_dict[f"blocks.{i}.cross_attn.v.bias"]
        new_state_dict[f"blocks.{(i // 4) + i}.cross_attn.o.weight"] = old_state_dict[f"blocks.{i}.cross_attn.o.weight"]
        new_state_dict[f"blocks.{(i // 4) + i}.cross_attn.o.bias"] = old_state_dict[f"blocks.{i}.cross_attn.o.bias"]
        new_state_dict[f"blocks.{(i // 4) + i}.cross_attn.norm_q.weight"] = old_state_dict[f"blocks.{i}.cross_attn.norm_q.weight"]
        new_state_dict[f"blocks.{(i // 4) + i}.cross_attn.norm_k.weight"] = old_state_dict[f"blocks.{i}.cross_attn.norm_k.weight"]
        new_state_dict[f"blocks.{(i // 4) + i}.ffn.0.weight"] = old_state_dict[f"blocks.{i}.ffn.0.weight"]
        new_state_dict[f"blocks.{(i // 4) + i}.ffn.0.bias"] = old_state_dict[f"blocks.{i}.ffn.0.bias"]
        new_state_dict[f"blocks.{(i // 4) + i}.ffn.2.weight"] = old_state_dict[f"blocks.{i}.ffn.2.weight"]
        new_state_dict[f"blocks.{(i // 4) + i}.ffn.2.bias"] = old_state_dict[f"blocks.{i}.ffn.2.bias"]
        frozen_parameters.extend([f"blocks.{(i // 4) + i}.modulation", f"blocks.{(i // 4) + i}.self_attn.q.weight", f"blocks.{(i // 4) + i}.self_attn.q.bias",
        f"blocks.{(i // 4) + i}.self_attn.k.weight", f"blocks.{(i // 4) + i}.self_attn.k.bias", f"blocks.{(i // 4) + i}.self_attn.v.weight",
        f"blocks.{(i // 4) + i}.self_attn.v.bias", f"blocks.{(i // 4) + i}.self_attn.o.weight", f"blocks.{(i // 4) + i}.self_attn.o.bias",
        f"blocks.{(i // 4) + i}.self_attn.norm_q.weight", f"blocks.{(i // 4) + i}.self_attn.norm_k.weight", f"blocks.{(i // 4) + i}.norm3.weight",
        f"blocks.{(i // 4) + i}.norm3.bias", f"blocks.{(i // 4) + i}.cross_attn.q.weight", f"blocks.{(i // 4) + i}.cross_attn.q.bias",
        f"blocks.{(i // 4) + i}.cross_attn.k.weight", f"blocks.{(i // 4) + i}.cross_attn.k.bias", f"blocks.{(i // 4) + i}.cross_attn.v.weight",
        f"blocks.{(i // 4) + i}.cross_attn.v.bias", f"blocks.{(i // 4) + i}.cross_attn.o.weight", f"blocks.{(i // 4) + i}.cross_attn.o.bias",
        f"blocks.{(i // 4) + i}.cross_attn.norm_q.weight", f"blocks.{(i // 4) + i}.cross_attn.norm_k.weight", f"blocks.{(i // 4) + i}.ffn.0.weight",
        f"blocks.{(i // 4) + i}.ffn.0.bias", f"blocks.{(i // 4) + i}.ffn.2.weight", f"blocks.{(i // 4) + i}.ffn.2.bias"])
    
    new_state_dict["head.modulation"] = old_state_dict["head.modulation"]
    new_state_dict["head.head.weight"] = old_state_dict["head.head.weight"]
    new_state_dict["head.head.bias"] = old_state_dict["head.head.bias"]
    frozen_parameters.extend(["head.modulation", "head.head.weight", "head.head.bias"])
    
    return new_state_dict, frozen_parameters

def load_pretrained_weight_T2(old_state_dict, new_state_dict, num_layers):
    frozen_parameters = []
    new_state_dict["patch_embedding.weight"] = old_state_dict["patch_embedding.weight"]
    new_state_dict["patch_embedding.bias"] = old_state_dict["patch_embedding.bias"]
    new_state_dict["text_embedding.0.weight"] = old_state_dict["text_embedding.0.weight"]
    new_state_dict["text_embedding.0.bias"] = old_state_dict["text_embedding.0.bias"]
    new_state_dict["text_embedding.2.weight"] = old_state_dict["text_embedding.2.weight"]
    new_state_dict["text_embedding.2.bias"] = old_state_dict["text_embedding.2.bias"]
    new_state_dict["time_embedding.0.weight"] = old_state_dict["time_embedding.0.weight"]
    new_state_dict["time_embedding.0.bias"] = old_state_dict["time_embedding.0.bias"]
    new_state_dict["time_embedding.2.weight"] = old_state_dict["time_embedding.2.weight"]
    new_state_dict["time_embedding.2.bias"] = old_state_dict["time_embedding.2.bias"]
    new_state_dict["time_projection.1.weight"] = old_state_dict["time_projection.1.weight"]
    new_state_dict["time_projection.1.bias"] = old_state_dict["time_projection.1.bias"]
    frozen_parameters.extend(["patch_embedding.weight", "patch_embedding.bias", "text_embedding.0.weight", "text_embedding.0.bias", "text_embedding.2.weight", 
    "text_embedding.2.bias","time_embedding.0.weight", "time_embedding.0.bias", "time_embedding.2.weight", "time_embedding.2.bias", "time_projection.1.weight", "time_projection.1.bias"])   

    for i in range(num_layers):
        new_state_dict[f"blocks.{(i // 4) + i}.modulation"] = old_state_dict[f"blocks.{i}.modulation"]
        new_state_dict[f"blocks.{(i // 4) + i}.self_attn.q.weight"] = old_state_dict[f"blocks.{i}.self_attn.q.weight"]
        new_state_dict[f"blocks.{(i // 4) + i}.self_attn.q.bias"] = old_state_dict[f"blocks.{i}.self_attn.q.bias"]
        new_state_dict[f"blocks.{(i // 4) + i}.self_attn.k.weight"] = old_state_dict[f"blocks.{i}.self_attn.k.weight"]
        new_state_dict[f"blocks.{(i // 4) + i}.self_attn.k.bias"] = old_state_dict[f"blocks.{i}.self_attn.k.bias"]
        new_state_dict[f"blocks.{(i // 4) + i}.self_attn.v.weight"] = old_state_dict[f"blocks.{i}.self_attn.v.weight"]
        new_state_dict[f"blocks.{(i // 4) + i}.self_attn.v.bias"] = old_state_dict[f"blocks.{i}.self_attn.v.bias"]
        new_state_dict[f"blocks.{(i // 4) + i}.self_attn.o.weight"] = old_state_dict[f"blocks.{i}.self_attn.o.weight"]
        new_state_dict[f"blocks.{(i // 4) + i}.self_attn.o.bias"] = old_state_dict[f"blocks.{i}.self_attn.o.bias"]
        new_state_dict[f"blocks.{(i // 4) + i}.self_attn.norm_q.weight"] = old_state_dict[f"blocks.{i}.self_attn.norm_q.weight"]
        new_state_dict[f"blocks.{(i // 4) + i}.self_attn.norm_k.weight"] = old_state_dict[f"blocks.{i}.self_attn.norm_k.weight"]
        new_state_dict[f"blocks.{(i // 4) + i}.norm3.weight"] = old_state_dict[f"blocks.{i}.norm3.weight"]
        new_state_dict[f"blocks.{(i // 4) + i}.norm3.bias"] = old_state_dict[f"blocks.{i}.norm3.bias"]
        new_state_dict[f"blocks.{(i // 4) + i}.cross_attn.q.weight"] = old_state_dict[f"blocks.{i}.cross_attn.q.weight"]
        new_state_dict[f"blocks.{(i // 4) + i}.cross_attn.q.bias"] = old_state_dict[f"blocks.{i}.cross_attn.q.bias"]
        new_state_dict[f"blocks.{(i // 4) + i}.cross_attn.k.weight"] = old_state_dict[f"blocks.{i}.cross_attn.k.weight"]
        new_state_dict[f"blocks.{(i // 4) + i}.cross_attn.k.bias"] = old_state_dict[f"blocks.{i}.cross_attn.k.bias"]
        new_state_dict[f"blocks.{(i // 4) + i}.cross_attn.v.weight"] = old_state_dict[f"blocks.{i}.cross_attn.v.weight"]
        new_state_dict[f"blocks.{(i // 4) + i}.cross_attn.v.bias"] = old_state_dict[f"blocks.{i}.cross_attn.v.bias"]
        new_state_dict[f"blocks.{(i // 4) + i}.cross_attn.o.weight"] = old_state_dict[f"blocks.{i}.cross_attn.o.weight"]
        new_state_dict[f"blocks.{(i // 4) + i}.cross_attn.o.bias"] = old_state_dict[f"blocks.{i}.cross_attn.o.bias"]
        new_state_dict[f"blocks.{(i // 4) + i}.cross_attn.norm_q.weight"] = old_state_dict[f"blocks.{i}.cross_attn.norm_q.weight"]
        new_state_dict[f"blocks.{(i // 4) + i}.cross_attn.norm_k.weight"] = old_state_dict[f"blocks.{i}.cross_attn.norm_k.weight"]
        new_state_dict[f"blocks.{(i // 4) + i}.ffn.0.weight"] = old_state_dict[f"blocks.{i}.ffn.0.weight"]
        new_state_dict[f"blocks.{(i // 4) + i}.ffn.0.bias"] = old_state_dict[f"blocks.{i}.ffn.0.bias"]
        new_state_dict[f"blocks.{(i // 4) + i}.ffn.2.weight"] = old_state_dict[f"blocks.{i}.ffn.2.weight"]
        new_state_dict[f"blocks.{(i // 4) + i}.ffn.2.bias"] = old_state_dict[f"blocks.{i}.ffn.2.bias"]
        frozen_parameters.extend([f"blocks.{(i // 4) + i}.modulation", f"blocks.{(i // 4) + i}.self_attn.q.weight", f"blocks.{(i // 4) + i}.self_attn.q.bias",
        f"blocks.{(i // 4) + i}.self_attn.k.weight", f"blocks.{(i // 4) + i}.self_attn.k.bias", f"blocks.{(i // 4) + i}.self_attn.v.weight",
        f"blocks.{(i // 4) + i}.self_attn.v.bias", f"blocks.{(i // 4) + i}.self_attn.o.weight", f"blocks.{(i // 4) + i}.self_attn.o.bias",
        f"blocks.{(i // 4) + i}.self_attn.norm_q.weight", f"blocks.{(i // 4) + i}.self_attn.norm_k.weight", f"blocks.{(i // 4) + i}.norm3.weight",
        f"blocks.{(i // 4) + i}.norm3.bias", f"blocks.{(i // 4) + i}.cross_attn.q.weight", f"blocks.{(i // 4) + i}.cross_attn.q.bias",
        f"blocks.{(i // 4) + i}.cross_attn.k.weight", f"blocks.{(i // 4) + i}.cross_attn.k.bias", f"blocks.{(i // 4) + i}.cross_attn.v.weight",
        f"blocks.{(i // 4) + i}.cross_attn.v.bias", f"blocks.{(i // 4) + i}.cross_attn.o.weight", f"blocks.{(i // 4) + i}.cross_attn.o.bias",
        f"blocks.{(i // 4) + i}.cross_attn.norm_q.weight", f"blocks.{(i // 4) + i}.cross_attn.norm_k.weight", f"blocks.{(i // 4) + i}.ffn.0.weight",
        f"blocks.{(i // 4) + i}.ffn.0.bias", f"blocks.{(i // 4) + i}.ffn.2.weight", f"blocks.{(i // 4) + i}.ffn.2.bias"])
    
    new_state_dict["head.modulation"] = old_state_dict["head.modulation"]
    new_state_dict["head.head.weight"] = old_state_dict["head.head.weight"]
    new_state_dict["head.head.bias"] = old_state_dict["head.head.bias"]
    frozen_parameters.extend(["head.modulation", "head.head.weight", "head.head.bias"])
    
    return new_state_dict, frozen_parameters


def load_pretrained_weight_ae(old_state_dict, new_state_dict):
    frozen_parameters = []
    counter = 0
    for name in new_state_dict:
        if "lora_" not in name:
            frozen_parameters.append(name)
    for name in old_state_dict:
        if ("encoder." + name) in new_state_dict:
            new_state_dict["encoder." + name] = old_state_dict[name]
            # print("encoder." + name)
            counter += 1
        elif ("conv1." + name) in new_state_dict:
            new_state_dict["conv1." + name] = old_state_dict[name]
            # print("conv1." + name)
            counter += 1
        else:
            print("There are unused weights!")

    return new_state_dict, frozen_parameters, counter
