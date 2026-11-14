# Nested cross-fit selection (v3): how to handle low speed on top of the top-down (S+SZ) average.
# Options: hard flow gate, hard front-view (FG) gate, logistic soft blend driven by the FG prediction,
# plus the smoothing window and whether to subtract the mean out-of-fold bias. Everything is chosen on OOF only.
import itertools
import os
import numpy as np
import pandas as pd

OUT = 'outputs'
FG_TAG = os.environ.get('FG_TAG', 'CV_FG')

def load(tag):
    d = pd.concat([pd.read_csv(f'{OUT}/oof_preds_{tag}_fold{k}.csv').assign(fold=k) for k in range(3)])
    return d.sort_values(['set_id', 'frame_id']).reset_index(drop=True)

base = load('CV_FlowOnly_Stacked')
S, Z, G = load(os.environ.get('S_TAG', 'CV_ConvNeXt_TopDown17S')), load(os.environ.get('Z_TAG', 'CV_ConvNeXt_TopDown17SZ')), load(FG_TAG)
for d in (S, Z, G):
    assert (d.set_id.values == base.set_id.values).all() and (d.frame_id.values == base.frame_id.values).all()
t, fold, sets = base.true_speed.values, base.fold.values, base.set_id.values
FLOW, TD0 = base.pred_speed.values, (S.pred_speed.values + Z.pred_speed.values) / 2
FG = (G.pred_speed.values + G.pred_flip.values) / 2

def smooth(p, w):
    s = pd.Series(p).groupby(sets).transform(lambda x: x.rolling(w, center=True, min_periods=1).median())
    return s.groupby(sets).transform(lambda x: x.rolling(w, center=True, min_periods=1).mean()).values

LOW = ([('none',)] + [('flow', g) for g in (8, 12)] + [('fg', g) for g in (12, 20, 30)]
       + [('soft', x0, k) for x0 in (12, 16, 20) for k in (0.2, 0.5, 1.0)])
WINDOWS = [41, 61, 81]
# share of FG mixed into the top-down prediction at all speeds, or ('spd', x0, k): speed-dependent logistic mix where
# the top-down weight is 1/(1+exp(-k(FG - x0))), i.e. mostly front view below x0 and mostly top-down above it
WFG = [0.0, 0.15, 0.25, 0.35, 0.5]
cache = {}
def predict(cfg):
    low, wf, w = cfg
    if cfg not in cache:
        if isinstance(wf, tuple):
            wt = 1 / (1 + np.exp(-wf[2] * (FG - wf[1])))
            TD = (1 - wt) * FG + wt * TD0
        else:
            TD = (1 - wf) * TD0 + wf * FG
        if low[0] == 'flow':
            a = np.where(FLOW < low[1], FLOW, TD)
        elif low[0] == 'fg':
            a = np.where(FG < low[1], FG, TD)
        elif low[0] == 'soft':
            wt = 1 / (1 + np.exp(-low[2] * (FG - low[1])))      # weight of the top-down model
            a = (1 - wt) * FG + wt * TD
        else:
            a = TD
        cache[cfg] = smooth(a, w)
    return cache[cfg]

rmse = lambda p, m: float(np.sqrt(np.mean((t[m] - p[m]) ** 2)))
configs = list(itertools.product(LOW, WFG, WINDOWS))
def corrected(cfg, offset, fit_m):
    p = predict(cfg)
    return p - (p[fit_m] - t[fit_m]).mean() if offset else p

nested = np.zeros_like(t)
for k in range(3):
    tr, te = fold != k, fold == k
    def inner(cfg, off):
        if not off:
            return rmse(predict(cfg), tr)
        a, b = [f for f in range(3) if f != k]
        e = [corrected(cfg, True, fold == x)[fold == y] - t[fold == y] for x, y in ((a, b), (b, a))]
        return float(np.sqrt(np.mean(np.concatenate(e) ** 2)))
    best = min(((c, o) for c in configs for o in (False, True)), key=lambda co: inner(*co))
    nested[te] = corrected(best[0], best[1], tr)[te]
    print(f'fold {k}: chosen = {best} -> held-out RMSE {rmse(nested, te):.3f}')
allm = np.ones_like(t, bool)
print(f'NESTED pooled OOF RMSE: {rmse(nested, allm):.3f}')

def cv_off(cfg):
    p = np.zeros_like(t)
    for k in range(3):
        p[fold == k] = corrected(cfg, True, fold != k)[fold == k]
    return rmse(p, allm)
print('Reference (current test config, flow gate 12, w61):', f'{rmse(predict((("flow", 12), 0.0, 61)), allm):.3f}',
      f'cross-fitted offset {cv_off((("flow", 12), 0.0, 61)):.3f}')
print('Top 8 configs on all OOF (no offset / cross-fitted offset / per fold / mean bias):')
for c in sorted(configs, key=lambda c: cv_off(c))[:8]:
    p = predict(c)
    print('  ', c, f'{rmse(p, allm):.3f}', f'{cv_off(c):.3f}', [round(rmse(p, fold == k), 2) for k in range(3)], f'{np.mean(p - t):+.3f}')
