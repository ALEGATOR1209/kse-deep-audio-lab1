import numpy as np
import os
import warnings
from . import audiotools as at
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.preprocessing import LabelEncoder
from tqdm.notebook import tqdm

sexLe = LabelEncoder()
emotionLe = LabelEncoder()

def _process_sample(args):
  i, sample = args

  amps = at.load_audio(sample)
  sp = at.spectrogram(amps)

  mfcc_value = at.mfcc(amps, sample['sr']).mean(axis=1)
  mfcc_std = at.mfcc(amps, sample['sr']).std(axis=1)
  (f0_value, _, _) = at.f0(amps, sample['sr'])
  f0_value = np.nanmean(f0_value)
  rms_value = at.rms(sp).mean(axis=1)
  zcr_value = at.zcr(amps).mean(axis=1)

  return i, {
      'mfcc': mfcc_value,
      'mfcc_std': mfcc_std,
      'f0': f0_value,
      'rms': rms_value,
      'zcr': zcr_value,
  }

def extract_features(df, progress_bar_desc):
  total = len(df)
  args = [(i, row) for i, row in df.iterrows()]

  results = [None] * total

  with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    with ProcessPoolExecutor(max_workers=max(os.cpu_count() - 2, 1)) as executor:
        futures = {executor.submit(_process_sample, arg): arg[0] for arg in args}

        for _, future in tqdm(enumerate(as_completed(futures), start=1), total=total, desc=progress_bar_desc, ascii="░▒█"):
          i, feat = future.result()
          results[i] = feat

  features = results

  df['mfcc'] = [f['mfcc'] for f in features]
  df['mfcc_std'] = [f['mfcc_std'] for f in features]
  df['f0'] = np.nan_to_num([f['f0'] for f in features], nan=0.0, posinf=0.0, neginf=0.0)
  df['rms'] = [f['rms'] for f in features]
  df['zcr'] = [f['zcr'] for f in features]

  return df

def to_xy(df, label_col='emotion'):
  sexLe.fit(df['sex'])
  emotionLe.fit(df[label_col])

  X = np.array([
    np.concatenate([
      [sexLe.transform([row.sex])[0]],
      [row.f0],
      row.mfcc,
      row.mfcc_std,
      row.rms,
      row.zcr,
    ])
    for row in df.itertuples()
  ])

  y = emotionLe.transform(df[label_col])

  return X, y
