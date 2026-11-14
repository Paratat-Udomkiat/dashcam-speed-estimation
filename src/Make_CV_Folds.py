# --- Fold Generation Script (Run Once) ---
import pandas as pd
from sklearn.model_selection import StratifiedKFold

CSV_INDEX = 'processed_data/master_index_memmap.csv'
OUT_CSV = 'processed_data/master_index_cv.csv'
NUM_FOLDS = 3

df = pd.read_csv(CSV_INDEX)
df_trainval = df[df['split'].isin(['Training', 'Validation'])]
stats = df_trainval.groupby('set_id')['speed'].mean().reset_index()

# Use quantiles (q=3) to guarantee perfectly equal bins, preventing StratifiedKFold from crashing
stats['speed_cat'] = pd.qcut(stats['speed'], q=3, labels=['Low', 'Medium', 'High'])

skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
stats['cv_fold'] = -1

for fold, (_, val_idx) in enumerate(skf.split(stats['set_id'], stats['speed_cat'])):
    stats.loc[val_idx, 'cv_fold'] = fold

fold_map = dict(zip(stats['set_id'], stats['cv_fold']))
df['cv_fold'] = df['set_id'].map(fold_map).fillna(-1).astype(int)

df.to_csv(OUT_CSV, index=False)
print(f"Stratified CV splits saved to {OUT_CSV}")
print("\nDrive Counts per Fold & Speed Tier:")
print(stats.groupby(['cv_fold', 'speed_cat'], observed=False).size())
