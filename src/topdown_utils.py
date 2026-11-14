import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.ops import StochasticDepth

class TopDownDataset(Dataset):
    def __init__(self, csv_file, memmap_path, total_frames=91200, split="Training", cv_fold=0, offsets=[-2, -1, 0, 1, 2], is_eval=False, use_aug=True, speed_aug_prob=0.0, speed_aug_max=135.0, speed_aug_strides=(2,), speed_aug_label='center', erase_prob=0.0, zoom_range=0.0):
        self.df = pd.read_csv(csv_file)

        if split == "Training":
            self.df = self.df[(self.df['cv_fold'] != cv_fold) & (self.df['cv_fold'] != -1)].reset_index(drop=True)
        elif split == "Validation":
            self.df = self.df[self.df['cv_fold'] == cv_fold].reset_index(drop=True)
        elif split == "Testing":
            self.df = self.df[self.df['split'] == 'Testing'].reset_index(drop=True)

        self.memmap_path = memmap_path
        self.total_frames = total_frames
        self.memmap = None # Lazy load

        self.offsets = offsets
        self.is_eval = is_eval
        self.use_aug = use_aug
        # Speed augmentation: sample the window at stride 2 and double the label (a slow clip seen at half
        # the frame rate looks like a clip at twice the speed). Only used in training, only when 2v <= speed_aug_max.
        self.speed_aug_prob = speed_aug_prob
        self.speed_aug_max = speed_aug_max
        # speed_aug_strides: strides to pick from at random; speed_aug_label: 'center' = stride x centre-frame speed,
        # 'mean' = stride x mean speed over the frames the strided window spans
        self.speed_aug_strides = speed_aug_strides
        self.speed_aug_label = speed_aug_label
        # Random erasing: blank one rectangle (2-10% of the image) at the same place in every frame of the stack
        self.erase_prob = erase_prob
        # Zoom augmentation: rescale the whole stack by s in [1-zoom_range, 1+zoom_range] about the centre with the
        # label unchanged, so pixel displacement alone no longer fixes the speed (simulates camera-scale differences)
        self.zoom_range = zoom_range

        self.set_ids = self.df['set_id'].values
        self.memmap_indices = self.df['memmap_idx'].values
        self.speeds = self.df['speed'].values
        self.frame_ids = self.df['frame_id'].values

    def __len__(self):
        return len(self.set_ids)

    def __getitem__(self, idx):
        if self.memmap is None:
            self.memmap = np.memmap(self.memmap_path, dtype='uint8', mode='r',
                                    shape=(self.total_frames, 192, 96, 3))

        target_set = self.set_ids[idx]
        frames = []
        speed = self.speeds[idx]
        offsets = self.offsets
        if self.use_aug and not self.is_eval and self.speed_aug_prob > 0 and random.random() < self.speed_aug_prob:
            k = random.choice(self.speed_aug_strides)
            lo, hi = idx + k * min(self.offsets), idx + k * max(self.offsets)
            if lo >= 0 and hi < len(self.set_ids) and self.set_ids[lo] == target_set and self.set_ids[hi] == target_set:
                base = self.speeds[lo:hi + 1].mean() if self.speed_aug_label == 'mean' else speed
                if k * self.speeds[lo:hi + 1].max() <= self.speed_aug_max if self.speed_aug_label == 'mean' else k * speed <= self.speed_aug_max:
                    offsets = [k * o for o in self.offsets]
                    speed = k * base

        for off in offsets:
            test_idx = idx + off
            valid_idx = idx

            step = -1 if off > 0 else 1
            for fallback_idx in range(test_idx, idx, step):
                if 0 <= fallback_idx < len(self.set_ids) and self.set_ids[fallback_idx] == target_set:
                    valid_idx = fallback_idx
                    break

            m_idx = self.memmap_indices[valid_idx]
            # Convert to [3, 192, 96]
            img = torch.from_numpy(np.copy(self.memmap[m_idx])).permute(2, 0, 1).float() / 255.0
            frames.append(img)

        combined = torch.cat(frames, dim=0) # [15, 192, 96]

        # Consistent Temporal Augmentation
        if self.use_aug and not self.is_eval:
            # Horizontal Flip (applies to all channels identically)
            if random.random() > 0.5:
                combined = torch.flip(combined, dims=[2])

            # Consistent Color Jitter
            if random.random() > 0.5:
                brightness = random.uniform(0.8, 1.2)
                contrast = random.uniform(0.8, 1.2)
                saturation = random.uniform(0.8, 1.2)
                hue = random.uniform(-0.1, 0.1)

                # Split back to [3, H, W] to apply standard color ops, then recombine
                split_frames = torch.split(combined, 3, dim=0)
                aug_frames = []
                for f in split_frames:
                    f = TF.adjust_brightness(f, brightness)
                    f = TF.adjust_contrast(f, contrast)
                    f = TF.adjust_saturation(f, saturation)
                    f = TF.adjust_hue(f, hue)
                    aug_frames.append(f)
                combined = torch.cat(aug_frames, dim=0)

        if self.use_aug and not self.is_eval and self.zoom_range > 0:
            z = random.uniform(1 - self.zoom_range, 1 + self.zoom_range)
            theta = torch.tensor([[1 / z, 0, 0], [0, 1 / z, 0]], dtype=combined.dtype).unsqueeze(0)
            grid = torch.nn.functional.affine_grid(theta, (1,) + tuple(combined.shape), align_corners=False)
            combined = torch.nn.functional.grid_sample(combined.unsqueeze(0), grid, mode='bilinear',
                                                       padding_mode='reflection', align_corners=False).squeeze(0)

        if self.use_aug and not self.is_eval and random.random() < self.erase_prob:
            H, W = combined.shape[1:]
            area = random.uniform(0.02, 0.10) * H * W
            ratio = random.uniform(0.3, 3.3)
            h, w = min(H, int(round((area * ratio) ** 0.5))), min(W, int(round((area / ratio) ** 0.5)))
            y0, x0 = random.randint(0, H - h), random.randint(0, W - w)
            combined[:, y0:y0 + h, x0:x0 + w] = 0.0

        target = torch.tensor([speed], dtype=torch.float32) / 100.0
        return combined, target, target_set, self.frame_ids[idx]

class TopDownWrapper(nn.Module):
    def __init__(self, base_model, num_frames=5):
        super().__init__()
        self.base_model = base_model
        # ImageNet normalization applied across the stack
        mean = torch.tensor([0.485, 0.456, 0.406] * num_frames).view(1, -1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225] * num_frames).view(1, -1, 1, 1)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x):
        return self.base_model((x - self.mean) / self.std)

def get_convnext_tiny_topdown(num_frames=5, drop_path_scale=1.0):
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    # Stochastic depth: torchvision's default rises linearly to 0.1; scale it for stronger regularization
    for m in model.modules():
        if isinstance(m, StochasticDepth):
            m.p = m.p * drop_path_scale
    old_conv = model.features[0][0]
    new_in_channels = num_frames * 3

    new_conv = nn.Conv2d(new_in_channels, old_conv.out_channels,
                         kernel_size=old_conv.kernel_size, stride=old_conv.stride,
                         padding=old_conv.padding, bias=old_conv.bias is not None)

    with torch.no_grad():
        scaled_weight = old_conv.weight.data / num_frames
        for i in range(num_frames):
            new_conv.weight.data[:, i*3:(i+1)*3, :, :] = scaled_weight
        if old_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.data

    model.features[0][0] = new_conv
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 1)

    return TopDownWrapper(model, num_frames)
