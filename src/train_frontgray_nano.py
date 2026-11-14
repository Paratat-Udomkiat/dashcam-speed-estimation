# Front-view grayscale model with a timm ConvNeXt-Nano backbone (FGN). Usage: py -3.10 train_frontgray_nano.py FOLD SEED [TAG]
# FOLD 0/1/2 = train on the other two CV folds and write out-of-fold predictions; FOLD -99 = train on all 68 drives.
import math
import os
import random
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from comma_utils import seed_worker
import timm
from frontgray_utils import FrontGrayDataset, bin_weights

def get_convnext_nano_frontgray(in_ch=10, drop_path=0.1):
    # ImageNet-12k -> 1k ConvNeXt-Nano; timm adapts the RGB stem to in_ch channels
    return timm.create_model('convnext_nano.in12k_ft_in1k', pretrained=True, in_chans=in_ch, num_classes=1, drop_path_rate=drop_path)

FOLD, SEED = int(sys.argv[1]), int(sys.argv[2])
TAG = sys.argv[3] if len(sys.argv) > 3 else 'FGN'
EPOCHS, BATCH, LR, WD, WARMUP = int(os.environ.get('FG_EPOCHS', 8)), 64, float(os.environ.get('FG_LR', 4e-4)), 0.05, 300
CSV = os.path.join('processed_data', 'master_index_cv.csv')
MEM = os.path.join('processed_data', 'fullimage_memmap.dat')
OUT = 'outputs'
name = f'CV_{TAG}_fold{FOLD}' if FOLD >= 0 else f'Full_{TAG}_s{SEED}'
ckpt_path = os.path.join(OUT, f'ckpt_{name}_latest.pth')

def predict(model, loader, flip=False):
    model.eval()
    rows = []
    with torch.no_grad():
        for x, y, s, f in loader:
            x = x.cuda(non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                p = model(torch.flip(x, dims=[3]) if flip else x).float() * 100.0
            rows.append(pd.DataFrame({'set_id': list(s), 'frame_id': f.numpy(), 'true_speed': y.numpy().ravel() * 100.0,
                                      'pred_speed': p.cpu().numpy().ravel()}))
    return pd.concat(rows, ignore_index=True)

if __name__ == '__main__':
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True
    g = torch.Generator(); g.manual_seed(SEED)
    train_ds = FrontGrayDataset(CSV, MEM, split='Training', cv_fold=FOLD)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=6, pin_memory=True, drop_last=True,
                              worker_init_fn=seed_worker, generator=g, persistent_workers=True)
    weights = bin_weights(train_ds.speeds).cuda()

    model = get_convnext_nano_frontgray().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    total = EPOCHS * len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARMUP) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    start = 0
    if os.path.exists(ckpt_path):
        c = torch.load(ckpt_path, map_location='cuda')
        model.load_state_dict(c['model']); opt.load_state_dict(c['opt']); sched.load_state_dict(c['sched']); start = c['epoch']
        print(f'resumed {name} at epoch {start}')
    print(f'{name}: {len(train_ds)} training frames, {len(train_loader)} steps/epoch', flush=True)

    for epoch in range(start, EPOCHS):
        t0, model_loss, n = time.time(), 0.0, 0
        model.train()
        for x, y, _, _ in train_loader:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            w = weights[(y * 100.0 // 10).long().clamp(0, 19)]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model(x).float()
            loss = (w * (out - y) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            model_loss += ((out - y) ** 2).sum().item(); n += len(y)
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'epoch': epoch + 1}, ckpt_path)
        print(f'epoch {epoch + 1}/{EPOCHS} {time.time() - t0:.0f}s train RMSE {math.sqrt(model_loss / n) * 100:.2f}', flush=True)

    if FOLD >= 0:
        val_ds = FrontGrayDataset(CSV, MEM, split='Validation', cv_fold=FOLD, is_eval=True)
        val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
        d = predict(model, val_loader)
        d['pred_flip'] = predict(model, val_loader, flip=True).pred_speed.values
        d.to_csv(os.path.join(OUT, f'oof_preds_{name}.csv'), index=False)
        print(f'{name} OOF raw RMSE {np.sqrt(np.mean((d.pred_speed - d.true_speed) ** 2)):.3f}  '
              f'with flip TTA {np.sqrt(np.mean(((d.pred_speed + d.pred_flip) / 2 - d.true_speed) ** 2)):.3f}')
