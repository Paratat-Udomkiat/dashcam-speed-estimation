# Top-down ConvNeXt-Tiny on 17 consecutive bird's-eye frames (t-8..t+8) with speed augmentation (p 0.5).
# Usage: python src/train_topdown.py FOLD SEED VARIANT
#   FOLD 0..2 = 3-fold CV model (writes out-of-fold predictions), FOLD -1 = trained on all 68 train+val drives
#   VARIANT S = speed augmentation, SZ = speed augmentation + random zoom 0.15
# The 12 top-down models were trained with one-cell notebooks (Triage_TD17S_fold0, CV_TD17S_fold1/2, CV_TD17SZ_fold0-2,
# Full_TD17S, Full_TD17S_s142/_s242, Full_TD17SZ, Full_TD17SZ_s142/_s242) that differ only in FOLD, SEED, CONFIG_TAG,
# the training fold (-99 = all drives) and zoom_range. This script is that cell with those settings as arguments.
import os
import sys
import math
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from comma_utils import seed_worker
from topdown_utils import TopDownDataset, get_convnext_tiny_topdown

BASE_DIR = os.path.abspath('.')
DATA_DIR = os.path.join(BASE_DIR, 'processed_data')
OUT_DIR = os.path.join(BASE_DIR, 'outputs')
CSV_CV = os.path.join(DATA_DIR, 'master_index_cv.csv')
TOPDOWN_MEMMAP_PATH = os.path.join(DATA_DIR, 'topdown_memmap.dat')

OFFSETS = list(range(-8, 9))  # 17 consecutive top-down frames (t-8..t+8)
EPOCHS = 15
BATCH_SIZE = 64
LR = 1e-4
WEIGHT_DECAY = 1e-2

FOLD_ARG, SEED, VARIANT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
FULL = FOLD_ARG < 0
FOLD = 0 if FULL else FOLD_ARG  # full-data runs report RMSE on the fold-0 drives (also in training: monitoring only)
ZOOM = 0.15 if VARIANT == 'SZ' else 0.0
CONFIG_TAG = (f"Full_ConvNeXt_TopDown17{VARIANT}" + ('' if SEED == 42 else f"_s{SEED}")) if FULL \
    else f"CV_ConvNeXt_TopDown17{VARIANT}_fold{FOLD}"
fold_ckpt = os.path.join(OUT_DIR, f"ckpt_{CONFIG_TAG}_latest.pth")
oof_path = os.path.join(OUT_DIR, f"oof_preds_{CONFIG_TAG}.csv")


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    set_seed(SEED + FOLD)
    g_seed = torch.Generator()
    g_seed.manual_seed(SEED + FOLD)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | {CONFIG_TAG}")

    train_dataset = TopDownDataset(CSV_CV, TOPDOWN_MEMMAP_PATH, split="Training", cv_fold=-99 if FULL else FOLD,  # -99 = all 68 drives
                                   speed_aug_prob=0.5, zoom_range=ZOOM, offsets=OFFSETS, is_eval=False, use_aug=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g_seed, persistent_workers=False)

    val_dataset = TopDownDataset(CSV_CV, TOPDOWN_MEMMAP_PATH, split="Validation", cv_fold=FOLD, offsets=OFFSETS, is_eval=True, use_aug=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=False)

    model = get_convnext_tiny_topdown(num_frames=len(OFFSETS)).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=True)

    start_epoch = 0
    if os.path.exists(fold_ckpt):
        print(f"Resuming from {fold_ckpt}...")
        checkpoint = torch.load(fold_ckpt, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']

    for epoch in range(start_epoch, EPOCHS):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0

        for inputs, targets, _, _ in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=True):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * inputs.size(0)

        scheduler.step()
        train_rmse = math.sqrt(train_loss / len(train_dataset)) * 100.0

        model.eval()
        val_mse_loss = 0.0
        with torch.no_grad():
            for inputs, targets, _, _ in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.amp.autocast('cuda', enabled=True):
                    outputs = model(inputs)
                val_mse_loss += torch.nn.functional.mse_loss(outputs * 100.0, targets * 100.0, reduction='sum').item()

        val_rmse = math.sqrt(val_mse_loss / len(val_dataset))
        epoch_time = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch+1}/{EPOCHS} | Time: {epoch_time:.1f}s | LR: {current_lr:.1e} | Train RMSE: {train_rmse:.2f} | Val RMSE: {val_rmse:.2f}")

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict()
        }, fold_ckpt)
        if FULL and epoch + 1 >= 13:  # keep the last 3 epochs' weights for an optional epoch ensemble
            torch.save(model.state_dict(), os.path.join(OUT_DIR, f"weights_{CONFIG_TAG}_epoch{epoch+1}.pth"))

    print(f"\nGenerating fold {FOLD} OOF predictions from FINAL epoch weights...")
    model.eval()
    results = []
    with torch.no_grad():
        for inputs, targets, set_ids, frame_ids in val_loader:
            inputs = inputs.to(device)
            with torch.amp.autocast('cuda', enabled=True):
                outputs = model(inputs) * 100.0
            targets_np = targets.cpu().numpy().flatten() * 100.0
            outputs_np = outputs.cpu().numpy().flatten()

            for j in range(len(outputs_np)):
                results.append({"set_id": set_ids[j], "frame_id": frame_ids[j].item(), "true_speed": targets_np[j], "pred_speed": outputs_np[j]})

    df_oof = pd.DataFrame(results)
    df_oof.to_csv(oof_path, index=False)
    print(f"{CONFIG_TAG} fold {FOLD} RMSE: {math.sqrt(((df_oof['true_speed'] - df_oof['pred_speed']) ** 2).mean()):.2f} km/h")


if __name__ == '__main__':
    main()
