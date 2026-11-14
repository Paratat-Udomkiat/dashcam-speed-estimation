# Final test evaluation (= test #12, the submitted result). Rebuilds the submitted prediction from each model's test
# predictions. Every setting was fixed from nested cross-validation on the 68 train+val drives before the test run.
#   top-down  td1 = 0.5 x mean(6 all-data models) + 0.5 x (0.5 x mean(3 S fold models) + 0.5 x mean(3 SZ fold models))
#             td2 = the same average for stride-2 inputs (output / 2); td = (td1 + td2) / 2 where td1 < 55 km/h, else td1
#   front     fg  = 0.5 x mean(2 all-data FGN15 seeds) + 0.5 x mean(3 FGN15 fold models), each with gap + flip TTA
#   blend     mix = 0.5 td + 0.5 fg;  w = 1 / (1 + exp(-0.2 (fg - 16)));  raw = (1 - w) fg + w mix
#   then      per-drive centred rolling median and mean over 41 frames, minus the out-of-fold bias 0.586 km/h
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from topdown_utils import TopDownDataset, get_convnext_tiny_topdown

OUT = 'outputs'
CSV = os.path.join('processed_data', 'master_index_cv.csv')
TD_MEMMAP = os.path.join('processed_data', 'topdown_memmap.dat')
KEY = ['set_id', 'frame_id']
TD_FULL = [f'Full_ConvNeXt_TopDown17{m}{s}' for m in ('S', 'SZ') for s in ('', '_s142', '_s242')]
TD_S_FOLDS = [f'CV_ConvNeXt_TopDown17S_fold{k}' for k in range(3)]
TD_SZ_FOLDS = [f'CV_ConvNeXt_TopDown17SZ_fold{k}' for k in range(3)]
FG_FULL = [f'Full_FGN15_s{s}' for s in (42, 142)]
FG_FOLDS = [f'CV_FGN15_fold{k}' for k in range(3)]
TD_STRIDE_BELOW, FG_GAP3_BELOW = 55, 60
X0, K, WFG, WINDOW, OFFSET = 16, 0.2, 0.5, 41, 0.586


def td_stride1(tag, loader):
    # stride-1 test prediction of one top-down model, cached per model so a rerun never needs the GPU again
    cache = os.path.join(OUT, f'test_preds_{tag}.csv')
    if not os.path.exists(cache):
        model = get_convnext_tiny_topdown(num_frames=17).cuda()
        model.load_state_dict(torch.load(os.path.join(OUT, f'ckpt_{tag}_latest.pth'), map_location='cuda')['model_state_dict'])
        model.eval()
        preds, sids, fids = [], [], []
        with torch.no_grad():
            for x, y, s, f in loader:
                with torch.amp.autocast('cuda'):
                    preds.append((model(x.cuda()) * 100.0).float().cpu().numpy().ravel())
                sids += list(s); fids += f.tolist()
        pd.DataFrame({'set_id': sids, 'frame_id': fids, 'pred': np.concatenate(preds)}).to_csv(cache, index=False)
    return pd.read_csv(cache).set_index(KEY)['pred']


def td_stride2(tag):
    # stride-2 test prediction (window t-16..t+16 step 2, output / 2) written by td_stride_tta.py
    return pd.read_csv(os.path.join(OUT, f'stridetta_{tag}_Testing.csv')).set_index(KEY)['pred_s2']


def fg_gap_tta(tag):
    # front-view temporal TTA written by fg_gap_tta_nano.py: gaps 1+2, plus gap 3 below 60 km/h, each flip-averaged
    d = pd.read_csv(os.path.join(OUT, f'gaptta_{tag}_Testing.csv')).set_index(KEY)
    g = {k: (d[f'g{k}'] + d[f'g{k}_flip']) / 2 for k in (1, 2, 3)}
    return pd.Series(np.where(g[2] < FG_GAP3_BELOW, (g[1] + g[2] + g[3]) / 3, (g[1] + g[2]) / 2), index=d.index)


if __name__ == '__main__':
    idx = pd.read_csv(CSV).query("split == 'Testing'").sort_values(KEY)
    d = idx[KEY + ['speed']].rename(columns={'speed': 'true_speed'}).reset_index(drop=True)
    ds = TopDownDataset(CSV, TD_MEMMAP, split='Testing', offsets=list(range(-8, 9)), is_eval=True, use_aug=False)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    at = lambda s: s.reindex(pd.MultiIndex.from_frame(d[KEY])).values
    mean = lambda f, tags: np.mean([at(f(t)) for t in tags], axis=0)

    s1 = lambda t: td_stride1(t, loader)
    td1 = 0.5 * mean(s1, TD_FULL) + 0.5 * (0.5 * mean(s1, TD_S_FOLDS) + 0.5 * mean(s1, TD_SZ_FOLDS))
    td2 = 0.5 * mean(td_stride2, TD_FULL) + 0.5 * (0.5 * mean(td_stride2, TD_S_FOLDS) + 0.5 * mean(td_stride2, TD_SZ_FOLDS))
    td = np.where(td1 < TD_STRIDE_BELOW, (td1 + td2) / 2, td1)
    fg = 0.5 * mean(fg_gap_tta, FG_FULL) + 0.5 * mean(fg_gap_tta, FG_FOLDS)
    assert not (np.isnan(td).any() or np.isnan(fg).any())

    mix = (1 - WFG) * td + WFG * fg
    w = 1 / (1 + np.exp(-K * (fg - X0)))       # weight of the top-down/front-view mix (front view alone at low speed)
    raw = (1 - w) * fg + w * mix
    sm = pd.Series(raw).groupby(d.set_id.values).transform(
        lambda x: x.rolling(WINDOW, center=True, min_periods=1).median().rolling(WINDOW, center=True, min_periods=1).mean())
    d['pred_td'], d['pred_fg'], d['pred_final'] = td, fg, sm.values - OFFSET

    err = d.pred_final - d.true_speed
    per = err.groupby(d.set_id).apply(lambda e: np.sqrt(np.mean(e ** 2)))
    print(f'FINAL TEST  overall RMSE {np.sqrt(np.mean(err ** 2)):.3f} km/h  average of the 8 drive RMSEs {per.mean():.3f} km/h  '
          f'MAE {err.abs().mean():.3f} km/h')
    print(pd.DataFrame({'rmse': per, 'bias': err.groupby(d.set_id).mean()}).round(2).to_string())
    d.to_csv(os.path.join(OUT, 'test_predictions_final.csv'), index=False)

    locked = os.path.join(OUT, 'test_predictions_v12.csv')   # output of the locked test #12 script
    if os.path.exists(locked):
        v12 = d[KEY].merge(pd.read_csv(locked)[KEY + ['pred_final_v12']], on=KEY, how='left')
        print(f'max |difference| to the locked test #12 predictions: {np.abs(v12.pred_final_v12.values - d.pred_final.values).max():.4f} km/h')
