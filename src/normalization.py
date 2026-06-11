import numpy as np
import soundfile as sf
import os
from pathlib import Path
from . import audiotools as at
from tqdm.notebook import tqdm

def peak_normalization(x, peak=0.98):
  """
  TAKEN FROM THE COURSE NOTEBOOK

  Peak normalization:
  - float32
  - replace NaN/Inf
  - peak-normalize
  - clip to [-1, 1]
  """
  x = np.asarray(x, dtype=np.float32)
  x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

  mx = float(np.max(np.abs(x))) + 1e-12
  g = min(1.0, peak / mx)
  x = x * g

  x = np.clip(x, -1.0, 1.0)
  return x

def rms_normalization(x, target_rms):
  """
  RMS normalization

  Averages loudness across the sample instead of just scaling down relatively to peaks
  """
  x = np.asarray(x, dtype=np.float32)
  x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

  current_rms = np.sqrt(np.mean(x**2)) + 1e-12

  x = x * (target_rms / current_rms)

  return x

def normalize_sample(sample, target_rms):
  y = at.load_audio(sample)
  y = rms_normalization(y, target_rms)
  y = peak_normalization(y)

  if not os.path.exists(sample['path_norm']):
    os.makedirs(Path(sample['path_norm']).parent, exist_ok=True)

  sf.write(sample['path_norm'], y, sample['sr'])

def normalize_dataset(df, samples_dir, progress_bar_desc):
  target_rms = df['rms'].mean()[0]
  total = len(df)

  for (_, row) in tqdm(df.iterrows(), total=total, desc=progress_bar_desc, ascii="░▒█"):
    row['path_norm'] = f"{samples_dir}/{row['filename']}"
    normalize_sample(row, target_rms)

  return df
