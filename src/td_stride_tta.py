# Temporal test-time augmentation for the top-down models: the S/SZ models were trained with stride-2 windows
# labelled 2 x speed, so feeding a stride-2 window (offsets -16..16 step 2) and halving the output gives a second
# speed estimate. Only valid where 2 x speed stayed inside training (<= 135 km/h), so the blend gates it by speed later.
# Usage: py -3.10 td_stride_tta.py TAG SPLIT [FOLD]  -> outputs/stridetta_{TAG}_{SPLIT}.csv (set_id, frame_id, true_speed, pred_s2)
import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from topdown_utils import TopDownDataset, get_convnext_tiny_topdown

if __name__ == '__main__':
    tag, split = sys.argv[1], sys.argv[2]
    fold = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    ds = TopDownDataset(os.path.join('processed_data', 'master_index_cv.csv'), os.path.join('processed_data', 'topdown_memmap.dat'),
                        split=split, cv_fold=fold, offsets=list(range(-16, 17, 2)), is_eval=True, use_aug=False)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    model = get_convnext_tiny_topdown(num_frames=17).cuda()
    model.load_state_dict(torch.load(os.path.join('outputs', f'ckpt_{tag}_latest.pth'), map_location='cuda')['model_state_dict'])
    model.eval()
    preds, sids, fids, ys = [], [], [], []
    with torch.no_grad():
        for x, y, s, f in loader:
            with torch.amp.autocast('cuda'):
                preds.append((model(x.cuda()) * 100.0 / 2).float().cpu().numpy().ravel())
            sids += list(s); fids += f.tolist(); ys.append(y.numpy().ravel() * 100.0)
    d = pd.DataFrame({'set_id': sids, 'frame_id': fids, 'true_speed': np.concatenate(ys), 'pred_s2': np.concatenate(preds)})
    d.to_csv(os.path.join('outputs', f'stridetta_{tag}_{split}.csv'), index=False)
    if split == 'Testing':
        sys.exit(0)
    lo = d.true_speed < 60
    print(f'{tag} stride-2 RMSE all {np.sqrt(np.mean((d.pred_s2 - d.true_speed) ** 2)):.3f}  below 60 km/h {np.sqrt(np.mean((d.pred_s2[lo] - d.true_speed[lo]) ** 2)):.3f}')
