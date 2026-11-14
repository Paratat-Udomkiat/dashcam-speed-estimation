# Row index of every frame in the image caches (memmap_idx). From RunB_Local.ipynb (cells 1-2); the cell also builds the
# 128 x 384 RGB memmap of the first ResNet-18 runs. The final models only use master_index_memmap.csv.
import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

BASE_DIR = os.path.abspath('.')
DATA_DIR = os.path.join(BASE_DIR, 'processed_data')
CSV_INDEX = os.path.join(DATA_DIR, 'master_index.csv')
MEMMAP_PATH = os.path.join(DATA_DIR, 'dataset_memmap.dat')
MEMMAP_CSV = os.path.join(DATA_DIR, 'master_index_memmap.csv')

# --- Cell 2: Builds the uint8 memmap using PIL if missing or invalid ---
df = pd.read_csv(CSV_INDEX)
N = len(df)
expected_shape = (N, 128, 384, 3)
expected_bytes = N * 128 * 384 * 3

needs_build = True
if os.path.exists(MEMMAP_PATH) and os.path.exists(MEMMAP_CSV):
    if os.path.getsize(MEMMAP_PATH) == expected_bytes:
        print("Valid memmap found. Skipping generation.")
        needs_build = False

if needs_build:
    print(f"Allocating memmap of shape {expected_shape} (~{expected_bytes/1e9:.2f} GB)...")
    mm = np.memmap(MEMMAP_PATH, dtype='uint8', mode='w+', shape=expected_shape)

    for i, row in tqdm(df.iterrows(), total=N, desc="Decoding JPEGs (PIL)"):
        img_path = os.path.join(DATA_DIR, row['filename'])
        img = Image.open(img_path).convert('RGB')
        mm[i] = np.array(img, dtype='uint8')

    mm.flush()
    df['memmap_idx'] = df.index
    df.to_csv(MEMMAP_CSV, index=False)
    print("Memmap successfully created.")
