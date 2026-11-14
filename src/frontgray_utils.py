import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset

# Front-view grayscale stack: 5 grey frames 0.1 s apart (frame gap 2 at 20 Hz)
# + 4 neighbouring-frame differences + 1 long difference (last - first) = 10 channels, from the 144x256 front-view cache.
BASE_GAP = 2
STEPS = np.array([-2, -1, 0, 1, 2])

class FrontGrayDataset(Dataset):
    def __init__(self, csv_file, memmap_path, total_frames=91200, split="Training", cv_fold=0, is_eval=False,
                 stretch_prob=0.75, stretch_gaps=(1, 2, 3, 4), max_label=140.0, still_prob=0.04, shift=8):
        df = pd.read_csv(csv_file)
        if split == "Training":
            df = df[(df['cv_fold'] != cv_fold) & (df['cv_fold'] != -1)]
        elif split == "Validation":
            df = df[df['cv_fold'] == cv_fold]
        elif split == "Testing":
            df = df[df['split'] == 'Testing']
        self.df = df.sort_values(['set_id', 'frame_id']).reset_index(drop=True)
        self.memmap_path, self.total_frames, self.memmap = memmap_path, total_frames, None
        self.is_eval = is_eval
        # Time-stretch: with prob stretch_prob pick a gap g from stretch_gaps (else the base gap 2);
        # label = mean speed over the frames the window spans x g/2 (a clip sampled at g looks g/2 times as fast)
        self.stretch_prob, self.stretch_gaps, self.max_label = stretch_prob, stretch_gaps, max_label
        self.still_prob, self.shift = still_prob, shift

        self.set_ids = self.df['set_id'].values
        self.memmap_indices = self.df['memmap_idx'].values
        self.speeds = self.df['speed'].values.astype(np.float32)
        self.frame_ids = self.df['frame_id'].values
        # first/last row of each drive, for clamping windows at the drive edges
        codes = pd.factorize(self.set_ids)[0]
        starts = np.r_[0, np.flatnonzero(np.diff(codes)) + 1]
        ends = np.r_[starts[1:] - 1, len(codes) - 1]
        self.lo, self.hi = starts[codes], ends[codes]

    def __len__(self):
        return len(self.set_ids)

    def _load(self, rows):
        if self.memmap is None:
            self.memmap = np.memmap(self.memmap_path, dtype='uint8', mode='r', shape=(self.total_frames, 144, 256, 3))
        rgb = torch.from_numpy(np.stack([self.memmap[self.memmap_indices[r]] for r in rows])).float() / 255.0
        return (rgb * torch.tensor([0.299, 0.587, 0.114])).sum(-1)    # 5,144,256

    def __getitem__(self, idx):
        lo, hi = self.lo[idx], self.hi[idx]
        g = BASE_GAP
        if not self.is_eval and random.random() < self.stretch_prob:
            g = random.choice(self.stretch_gaps)
        rows = np.clip(idx + g * STEPS, lo, hi)
        label = self.speeds[rows[0]:rows[-1] + 1].mean() * g / BASE_GAP
        if not self.is_eval and label > self.max_label:
            g = BASE_GAP
            rows = np.clip(idx + g * STEPS, lo, hi)
            label = self.speeds[rows[0]:rows[-1] + 1].mean()
        if self.is_eval:
            label = self.speeds[idx]                                   # score against the frame's own speed
        elif random.random() < self.still_prob:
            rows, label = np.full(5, idx), 0.0                          # one frame repeated = standing still

        x = self._load(rows)
        if not self.is_eval:
            c, b = random.uniform(0.8, 1.2), random.uniform(-0.1, 0.1)
            x = ((x - x.mean()) * c + x.mean() + b).clamp(0, 1)
            if self.shift:
                dy, dx = random.randint(-self.shift, self.shift), random.randint(-self.shift, self.shift)
                x = torch.nn.functional.pad(x.unsqueeze(0), (self.shift,) * 4, mode='replicate').squeeze(0)
                x = x[:, self.shift + dy:self.shift + dy + 144, self.shift + dx:self.shift + dx + 256]
            if random.random() < 0.5:
                x = torch.flip(x, dims=[2])
        stack = torch.cat([(x - 0.45) / 0.25, (x[1:] - x[:-1]) * 5.0, (x[-1:] - x[:1]) * 5.0], 0)   # 10,144,256
        return stack, torch.tensor([label / 100.0], dtype=torch.float32), self.set_ids[idx], self.frame_ids[idx]

def get_convnext_tiny_frontgray(in_ch=10, drop_path=0.1):
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1, stochastic_depth_prob=drop_path)
    old = model.features[0][0]
    new = nn.Conv2d(in_ch, old.out_channels, kernel_size=old.kernel_size, stride=old.stride, padding=old.padding, bias=True)
    with torch.no_grad():
        grey = old.weight.mean(1, keepdim=True)                         # RGB filters -> one grey filter
        new.weight.copy_(grey.repeat(1, in_ch, 1, 1) * 3.0 / in_ch)
        new.bias.copy_(old.bias)
    model.features[0][0] = new
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 1)
    return model

def bin_weights(speeds, width=10.0):
    # 1/sqrt(count) per 10 km/h bin, normalised to mean 1 over the training frames
    bins = (np.asarray(speeds) // width).astype(int)
    counts = np.bincount(bins, minlength=20)
    w = 1.0 / np.sqrt(np.maximum(counts, 1))
    w = w / w[bins].mean()
    return torch.tensor(w, dtype=torch.float32)
