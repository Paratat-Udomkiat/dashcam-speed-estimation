# Out-of-fold (OOF) inputs for the ensemble selection, one file per model type and fold, built from the CV models'
# predictions on their held-out fold (68 train+val drives in total):
# - top-down S / SZ with stride-2 TTA: where the model's own prediction is below 55 km/h it is averaged with the
#   stride-2 prediction (td_stride_tta.py, output / 2) -> oof_preds_CV_TD17Smix55 / CV_TD17SZmix55
#   (at test time the same rule is applied to the ensemble prediction instead of each model)
# - front view FGN15 with temporal TTA: gaps 1+2, plus gap 3 where the gap-2 prediction is below 60 km/h, kept with and
#   without flip -> oof_preds_CV_FGN15g123
# Existing files are checked against the rebuild and only written when missing.
import os
import numpy as np
import pandas as pd

OUT = 'outputs'
KEY = ['set_id', 'frame_id']


def td_mix55(variant, k):
    s1 = pd.read_csv(f'{OUT}/oof_preds_CV_ConvNeXt_TopDown17{variant}_fold{k}.csv')
    s2 = pd.read_csv(f'{OUT}/stridetta_CV_ConvNeXt_TopDown17{variant}_fold{k}_Validation.csv')
    d = s1.merge(s2[KEY + ['pred_s2']], on=KEY, validate='one_to_one')
    d['pred_speed'] = np.where(d.pred_speed < 55, (d.pred_speed + d.pred_s2) / 2, d.pred_speed)
    return d[KEY + ['true_speed', 'pred_speed']]


def fg_g123(tag, k):
    d = pd.read_csv(f'{OUT}/gaptta_{tag}_fold{k}_Validation.csv')
    g = {i: (d[f'g{i}'] + d[f'g{i}_flip']) / 2 for i in (1, 2, 3)}
    lo = g[2] < 60
    d['pred_speed'] = np.where(lo, (d.g1 + d.g2 + d.g3) / 3, (d.g1 + d.g2) / 2)
    d['pred_flip'] = np.where(lo, (d.g1_flip + d.g2_flip + d.g3_flip) / 3, (d.g1_flip + d.g2_flip) / 2)
    return d[KEY + ['true_speed', 'pred_speed', 'pred_flip']]


def save_or_check(d, name):
    path = f'{OUT}/oof_preds_{name}.csv'
    if os.path.exists(path):
        old = pd.read_csv(path).merge(d, on=KEY, suffixes=('_old', ''), validate='one_to_one')
        assert len(old) == len(d)
        diff = max(float(np.abs(old[f'{c}_old'] - old[c]).max()) for c in d.columns if c.startswith('pred'))
        print(f'{name}: {len(d)} rows, matches the existing file (max diff {diff:.1e})')
        assert diff < 1e-6
    else:
        d.to_csv(path, index=False)
        print(f'{name}: written, {len(d)} rows')


if __name__ == '__main__':
    for k in range(3):
        for v in ('S', 'SZ'):
            save_or_check(td_mix55(v, k), f'CV_TD17{v}mix55_fold{k}')
        save_or_check(fg_g123('CV_FGN15', k), f'CV_FGN15g123_fold{k}')
