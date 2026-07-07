import os
import cv2
import glob
import random
import numpy as np
from PIL import Image
from typing import List, Tuple
from pathlib import Path
import imageio
import ffmpeg
from safetensors.torch import load_file
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF

class LatentDataset(Dataset):
    def __init__(self, target_tensor, reference_tensor, monte_carlo = None):
        super().__init__()
        if monte_carlo is not None:
            assert monte_carlo > 0, "Monte Carlo takes positive argument."
        self.monte_carlo = monte_carlo
        self.effective_length = target_tensor.shape[0] // monte_carlo * monte_carlo
        if (monte_carlo is not None) and (monte_carlo > 1): 
            self.target = torch.split(target_tensor[:self.effective_length], monte_carlo, dim=0)
            self.reference = torch.split(reference_tensor[:self.effective_length], monte_carlo, dim=0)
        else:
            self.target = [t.unsqueeze(0) for t in target_tensor]
            self.reference = [r.unsqueeze(0) for r in reference_tensor]
        assert len(self.target) == len(self.reference) and self.target[0].shape == self.reference[0].shape

    def __getitem__(self, index):
        return self.target[index], self.reference[index]
    
    def __len__(self):
        return len(self.target)

    
class VideoDataset(Dataset):
    def __init__(self, root, max_frames = None, transform = None):
        """
        root: VFHQ根目录

        return: target_video: [C,T,H,W], caption: str
        """
        self.root = root

        # 遍历所有 Clip 文件夹
        self.clip_paths = []
        for group in sorted(os.listdir(root)):
            group_path = os.path.join(root, group)
            if not os.path.isdir(group_path):
                continue
            for clip in sorted(os.listdir(group_path)):
                clip_path = os.path.join(group_path, clip)
                if os.path.isdir(clip_path):
                    self.clip_paths.append(clip_path)

        self.max_frames = max_frames
        self.transform = transform

        if not self.transform:
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]) #transforms.Lambda(lambda x: x * 2 - 1)

    def _load_clip(self, clip_path):
        # 按顺序读取所有 png
        files = sorted(glob.glob(os.path.join(clip_path, '*.png')))
        frames = [self.transform(Image.open(f).convert('RGB')) for f in files] # [C,H,W]
        if self.max_frames and self.max_frames < len(frames):
            start_index = random.randint(0, len(frames) - self.max_frames)
            frames = frames[start_index:start_index + self.max_frames] 
        video = torch.stack(frames, dim=1)  # [C, T, H, W]
        return video

    def __len__(self):
        return len(self.clip_paths)

    def __getitem__(self, idx):
        clip_path = self.clip_paths[idx]
        target_video = self._load_clip(clip_path)  # [C, T, H, W]
        # caption_path = os.path.join(clip_path, "caption.txt")
        # caption = open(caption_path, "r").read().strip() if os.path.exists(caption_path) else ""
        caption_path = os.path.join(clip_path, "caption.safetensors")
        caption = load_file(caption_path)["T5Tensor"]    # L, C_text
        return target_video, caption 
class CropShortEdgeFromTop:
    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size 

        if h < w:
            return TF.crop(img, top=0, left=0, height=512, width=w)
        else:
            return TF.crop(img, top=0, left=0, height=h, width=512)
class Train4KDataset(Dataset):
    def __init__(
        self,
        root: str,
        segment_len: int = 13,
        resize_short: int = 513,
        transform=None,
        caption_mode: str = "dummy",
    ):
        self.root = root
        self.segment_len = segment_len
        self.resize_short = resize_short
        self.caption_mode = caption_mode

        # 默认：短边 resize 到 512，长边等比缩放；然后 ToTensor + Normalize
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(
                    size=self.resize_short,  # short edge
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                CropShortEdgeFromTop(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = transform

        self.video_dirs: List[str] = []
        for name in sorted(os.listdir(root)):
            p = os.path.join(root, name)
            if os.path.isdir(p):
                self.video_dirs.append(p)

        # index: (video_dir, start_frame)
        self.index: List[Tuple[str, int]] = []

        for vd in self.video_dirs:
            frames = sorted(glob.glob(os.path.join(vd, "*.png")))
            n_segments = len(frames) // self.segment_len  # floor(n/13)
            if n_segments <= 0:
                continue

            for seg_id in range(n_segments):
                start = seg_id * self.segment_len
                self.index.append((vd, start))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        video_dir, start = self.index[idx]

        frame_paths = sorted(glob.glob(os.path.join(video_dir, "*.png")))
        clip_paths = frame_paths[start: start + self.segment_len]
        if len(clip_paths) != self.segment_len:
            raise RuntimeError(
                f"{video_dir}: bad clip slice start={start}, "
                f"got {len(clip_paths)} frames"
            )

        frames = []
        for fp in clip_paths:
            img = Image.open(fp).convert("RGB")
            frames.append(self.transform(img))

        # [C, T, H, W]
        video = torch.stack(frames, dim=1)

        if self.caption_mode == "safetensors":
            from safetensors.torch import load_file
            cap_path = os.path.join(video_dir, "caption.safetensors")
            caption = load_file(cap_path)["T5Tensor"]
        else:
            caption = torch.empty(0)

        return video, caption


def collate_fn(batch):
    """
    batch: list of tuples (target_video, caption)
    取最短长度作为边界, stack成 [B,C,T,H,W]

    return: tensor of videos [B,C,T,H,W], list of captions [B,str]
    """
    min_len = min([v[0].shape[1] for v in batch])
    min_len = 1 + (min_len - 1) // 4 * 4
    videos = torch.stack([v[0][:,:min_len] for v in batch], dim=0)
    captions = [v[1] for v in batch]
    return videos, captions, min_len


def save_video(tensor, filename, fps=24, quality=5):
    """
    tensor: (C, T, H, W), float in [-1,1]
    filename: str, 保存路径
    fps: int, 帧率
    """
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    # [-1,1] -> [0,1]
    tensor = (tensor.clone() + 1.0) / 2.0
    # [0,1] -> [0,255]
    tensor = (tensor * 255).clamp(0, 255).byte()
    tensor = tensor.cpu().numpy()
    
    # C,T,H,W -> T,H,W,C
    tensor = np.transpose(tensor, (1,2,3,0))
    
    # 写视频
    writer = imageio.get_writer(filename, fps=fps, codec='libx264', quality=quality)
    for frame in tensor:
        writer.append_data(frame)
    writer.close()

def save_video2(tensor, filename, fps=24):
    """
    tensor: (C, T, H, W), float in [-1,1]
    filename: str, 保存路径
    fps: int, 帧率
    """
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    # [-1,1] -> [0,1]
    tensor = (tensor.clone() + 1) * 127.5  # 将 [-1, 1] 映射到 [0, 255]
    tensor = tensor.clamp(0, 255).byte()    # tensor.to(torch.uint8)
    frames = tensor.permute(1, 2, 3, 0).numpy()  # 维度变成 (T, H, W, 3)
    # 使用 ffmpeg 编码
    ffmpeg.input('pipe:0', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(frames.shape[2], frames.shape[1]), framerate=24).output(filename, vcodec='ffv1').run(input=frames.tobytes())

