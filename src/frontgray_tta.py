import numpy as np
import torch
from frontgray_utils import FrontGrayDataset, STEPS

# Temporal test-time augmentation for the FG model: feed the 5 frames g frames apart instead of the base 2.
# The model was trained with time-stretch (a clip sampled at gap g looks g/2 times as fast), so its output
# is rescaled by 2/g to get back to the true speed. Label = the frame's own speed, as in normal evaluation.
class FrontGrayGapDataset(FrontGrayDataset):
    def __init__(self, *args, eval_gap=2, **kwargs):
        super().__init__(*args, is_eval=True, **kwargs)
        self.eval_gap = eval_gap

    def __getitem__(self, idx):
        rows = np.clip(idx + self.eval_gap * STEPS, self.lo[idx], self.hi[idx])
        x = self._load(rows)
        stack = torch.cat([(x - 0.45) / 0.25, (x[1:] - x[:-1]) * 5.0, (x[-1:] - x[:1]) * 5.0], 0)
        return stack, torch.tensor([self.speeds[idx] / 100.0], dtype=torch.float32), self.set_ids[idx], self.frame_ids[idx]
