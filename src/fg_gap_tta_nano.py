# Predict with an FG checkpoint at several frame gaps (+ horizontal flip) and cache the result.
# ConvNeXt-Nano (FGN) version of fg_gap_tta.py. Usage: [FG_GAPS=1,2,3] py -3.10 fg_gap_tta_nano.py TAG SPLIT [FOLD]
# Writes outputs/gaptta_{TAG}_{SPLIT}.csv with set_id, frame_id, true_speed, g{gap}, g{gap}_flip (km/h, already rescaled by 2/gap).
import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import timm
from frontgray_tta import FrontGrayGapDataset

GAPS = [int(g) for g in os.environ.get('FG_GAPS', '1,2,3').split(',')]
SUFFIX = '' if GAPS == [1, 2, 3] else '_g' + ''.join(map(str, GAPS))
OUT = 'outputs'

if __name__ == '__main__':
    tag, split = sys.argv[1], sys.argv[2]
    fold = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    path = os.path.join(OUT, f'gaptta_{tag}_{split}{SUFFIX}.csv')
    model = timm.create_model('convnext_nano.in12k_ft_in1k', pretrained=False, in_chans=10, num_classes=1).cuda()
    model.load_state_dict(torch.load(os.path.join(OUT, f'ckpt_{tag}_latest.pth'), map_location='cuda')['model'])
    model.eval()
    d = None
    for g in GAPS:
        ds = FrontGrayGapDataset(os.path.join('processed_data', 'master_index_cv.csv'), os.path.join('processed_data', 'fullimage_memmap.dat'),
                                 split=split, cv_fold=fold, eval_gap=g)
        loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
        p, pf, sids, fids, ys = [], [], [], [], []
        with torch.no_grad():
            for x, y, s, f in loader:
                x = x.cuda(non_blocking=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    p.append((model(x).float() * 100.0 * 2 / g).cpu().numpy().ravel())
                    pf.append((model(torch.flip(x, dims=[3])).float() * 100.0 * 2 / g).cpu().numpy().ravel())
                sids += list(s); fids += f.tolist(); ys.append(y.numpy().ravel() * 100.0)
        cur = pd.DataFrame({'set_id': sids, 'frame_id': fids, 'true_speed': np.concatenate(ys),
                            f'g{g}': np.concatenate(p), f'g{g}_flip': np.concatenate(pf)})
        d = cur if d is None else d.merge(cur.drop(columns='true_speed'), on=['set_id', 'frame_id'], validate='one_to_one')
        if split != 'Testing':
            print(f'{tag} gap {g}: RMSE {np.sqrt(np.mean(((cur[f"g{g}"] + cur[f"g{g}_flip"]) / 2 - cur.true_speed) ** 2)):.3f}', flush=True)
    d.to_csv(path, index=False)
