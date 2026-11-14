import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import tv_tensors
from torchvision.transforms import v2
from torch.utils.data import Dataset

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class NormalizationWrapper(nn.Module):
    def __init__(self, backbone, num_frames=1):
        super().__init__()
        self.backbone = backbone
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        if num_frames > 1:
            mean = torch.cat([mean] * num_frames, dim=1)
            std = torch.cat([std] * num_frames, dim=1)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x):
        x = x.float() / 255.0
        x = (x - self.mean) / self.std
        return self.backbone(x)

def get_resnet18_model(num_frames=1):
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if num_frames > 1:
        in_channels = 3 * num_frames
        old_conv = resnet.conv1
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            resnet.conv1.weight.data = torch.cat([old_conv.weight.data] * num_frames, dim=1) / float(num_frames)
    resnet.fc = nn.Linear(resnet.fc.in_features, 1)
    return NormalizationWrapper(resnet, num_frames)

class MemmapCommaDataset(Dataset):
    def __init__(self, csv_file, memmap_path, split="Training", cv_fold=None, num_frames=2, frame_gap=2, subsample_ratio=1.0, is_eval=False, use_aug=False, use_hue=True):
        self.memmap_path = memmap_path
        self.num_frames = num_frames
        self.frame_gap = frame_gap
        self.subsample_ratio = subsample_ratio
        self.is_eval = is_eval
        self.use_aug = use_aug

        self.file_handle = None
        self.frame_bytes = 128 * 384 * 3

        if self.use_aug and not self.is_eval:
            if use_hue:
                self.aug = v2.Compose([
                    v2.RandomHorizontalFlip(p=0.5),
                    v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
                ])
            else:
                self.aug = v2.Compose([
                    v2.RandomHorizontalFlip(p=0.5),
                    v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)
                ])

        df = pd.read_csv(csv_file)
        
        # --- Dynamic Fold Subset Logic ---
        if cv_fold is not None:
            if split == "Testing":
                df_split = df[df["split"] == "Testing"].reset_index(drop=True)
            elif split == "Training":
                df_split = df[(df['cv_fold'].notna()) & (df['cv_fold'] != -1) & (df['cv_fold'] != cv_fold)].reset_index(drop=True)
            elif split == "Validation":
                df_split = df[df['cv_fold'] == cv_fold].reset_index(drop=True)
        else:
            df_split = df[df["split"] == split].reset_index(drop=True)
            
        df_dict = {(r["set_id"], r["frame_id"]): r["memmap_idx"] for _, r in df_split.iterrows()}

        self.valid_samples = []
        first_valid_per_set = {}
        if num_frames > 1 and is_eval:
            for set_id, group in df_split.groupby("set_id"):
                frames = sorted(group["frame_id"].tolist())
                valid_start = frames[0]
                for f in frames:
                    has_all = True
                    for i in range(1, num_frames):
                        if (set_id, f - i * frame_gap) not in df_dict:
                            has_all = False
                            break
                    if has_all:
                        valid_start = f
                        break
                first_valid_per_set[set_id] = valid_start

        for _, row in df_split.iterrows():
            set_id = row["set_id"]
            frame_id = row["frame_id"]
            idx0 = row["memmap_idx"]

            if num_frames == 1:
                self.valid_samples.append({"indices": [idx0], "speed": row["speed"], "set_id": set_id, "frame_id": frame_id})
                continue

            indices = []
            is_valid = True
            for i in reversed(range(1, num_frames)):
                past_f = frame_id - i * frame_gap
                if (set_id, past_f) in df_dict:
                    indices.append(df_dict[(set_id, past_f)])
                else:
                    is_valid = False
                    break

            if is_valid:
                indices.append(idx0)
                self.valid_samples.append({"indices": indices, "speed": row["speed"], "set_id": set_id, "frame_id": frame_id})
            else:
                if self.is_eval:
                    fv = first_valid_per_set[set_id]
                    fb_indices = []
                    for i in reversed(range(1, num_frames)):
                        fb_indices.append(df_dict[(set_id, fv - i * frame_gap)])
                    fb_indices.append(df_dict[(set_id, fv)])
                    self.valid_samples.append({"indices": fb_indices, "speed": row["speed"], "set_id": set_id, "frame_id": frame_id})
                else:
                    continue

        if self.is_eval:
            assert len(self.valid_samples) == len(df_split), "Eval dropped frames!"

        self.active_samples = self.valid_samples.copy()

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'file_handle' in state and state['file_handle'] is not None:
            state['file_handle'].close()
        state['file_handle'] = None
        return state

    def _read_frame(self, idx):
        if self.file_handle is None:
            self.file_handle = open(self.memmap_path, 'rb')
        self.file_handle.seek(idx * self.frame_bytes)
        bytes_data = self.file_handle.read(self.frame_bytes)
        return np.frombuffer(bytes_data, dtype=np.uint8).reshape((128, 384, 3))

    def resample(self):
        if not self.is_eval and self.subsample_ratio < 1.0:
            k = int(len(self.valid_samples) * self.subsample_ratio)
            self.active_samples = random.sample(self.valid_samples, k)
        else:
            self.active_samples = self.valid_samples.copy()

    def __len__(self):
        return len(self.active_samples)

    def __getitem__(self, idx):
        sample = self.active_samples[idx]
        target = torch.tensor([sample["speed"] / 100.0], dtype=torch.float32)

        imgs = [torch.from_numpy(self._read_frame(i).copy()).permute(2, 0, 1) for i in sample["indices"]]

        if self.use_aug and not self.is_eval:
            tv_imgs = tuple(tv_tensors.Image(img) for img in imgs)
            if self.num_frames == 1:
                imgs = [self.aug(tv_imgs[0])]
            else:
                imgs = list(self.aug(*tv_imgs))

        stacked_tensor = torch.cat(imgs, dim=0)
        return stacked_tensor, target, sample["set_id"], sample["frame_id"]
