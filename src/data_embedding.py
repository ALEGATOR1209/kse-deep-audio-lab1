import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
import os
import pandas as pd
from .audiotools import load_audio
from .normalization import rms_normalization, peak_normalization


def _save_and_record(y, sr, path, row, suffix, records):
  sf.write(path, y, sr)
  new_row = row.to_dict()
  new_row['filename'] = os.path.basename(path)
  new_row['path'] = path
  new_row['augmentation'] = suffix
  records.append(new_row)


def noise(df, samples_dir, target_rms, progress_bar_desc, records, noise_rate=0.035):
  for _, row in tqdm(df.iterrows(), total=len(df), desc=progress_bar_desc, ascii="░▒█"):
    y = load_audio(row)
    y = y + np.random.randn(len(y)) * noise_rate * np.max(np.abs(y))
    y = rms_normalization(y, target_rms)
    y = peak_normalization(y)
    path = f"{samples_dir}/{row['filename'].split('.')[0]}_noised.wav"
    _save_and_record(y, row['sr'], path, row, 'noised', records)


def pitch_shift(df, samples_dir, target_rms, progress_bar_desc, records, range=range(-3, 3)):
  for _, row in tqdm(df.iterrows(), total=len(df), desc=progress_bar_desc, ascii="░▒█"):
    y = load_audio(row)
    y = librosa.effects.pitch_shift(y, sr=row['sr'], n_steps=np.random.randint(range.start, range.stop))
    y = rms_normalization(y, target_rms)
    y = peak_normalization(y)
    path = f"{samples_dir}/{row['filename'].split('.')[0]}_pitched.wav"
    _save_and_record(y, row['sr'], path, row, 'pitched', records)


def time_stretch(df, samples_dir, target_rms, progress_bar_desc, records, range_min=1.1, range_max=1.5):
  for _, row in tqdm(df.iterrows(), total=len(df), desc=progress_bar_desc, ascii="░▒█"):
    y = load_audio(row)
    y = librosa.effects.time_stretch(y, rate=np.random.uniform(range_min, range_max))
    y = rms_normalization(y, target_rms)
    y = peak_normalization(y)
    path = f"{samples_dir}/{row['filename'].split('.')[0]}_stretched.wav"
    _save_and_record(y, row['sr'], path, row, 'stretched', records)

def time_shift(df, samples_dir, target_rms, progress_bar_desc, records, rangeMin=-0.1, rangeMax=0.1):
    for _, row in tqdm(df.iterrows(), total=len(df), desc=progress_bar_desc, ascii="░▒█"):
        y = load_audio(row)
        shift = int(np.random.uniform(rangeMin, rangeMax) * len(y))

        if shift > 0:
            y = np.concatenate([np.zeros(shift), y[:-shift]])
        elif shift < 0:
            y = np.concatenate([y[-shift:], np.zeros(-shift)])

        y = rms_normalization(y, target_rms)
        y = peak_normalization(y)
        path = f"{samples_dir}/{row['filename'].split('.')[0]}_shifted.wav"
        _save_and_record(y, row['sr'], path, row, 'shifted', records)

def lower_speed(df, samples_dir, target_rms, progress_bar_desc, records, rate=0.75):
  for _, row in tqdm(df.iterrows(), total=len(df), desc=progress_bar_desc, ascii="░▒█"):
    y = load_audio(row)
    y = librosa.effects.time_stretch(y, rate=rate)
    y = rms_normalization(y, target_rms)
    y = peak_normalization(y)
    path = f"{samples_dir}/{row['filename'].split('.')[0]}_slower.wav"
    _save_and_record(y, row['sr'], path, row, 'slower', records)


def higher_speed(df, samples_dir, target_rms, progress_bar_desc, records, rate=1.25):
  for _, row in tqdm(df.iterrows(), total=len(df), desc=progress_bar_desc, ascii="░▒█"):
    y = load_audio(row)
    y = librosa.effects.time_stretch(y, rate=rate)
    y = rms_normalization(y, target_rms)
    y = peak_normalization(y)
    path = f"{samples_dir}/{row['filename'].split('.')[0]}_faster.wav"
    _save_and_record(y, row['sr'], path, row, 'faster', records)


def copy_orig(df, samples_dir, progress_bar_desc, records):
  for _, row in tqdm(df.iterrows(), total=len(df), desc=progress_bar_desc, ascii="░▒█"):
    y = load_audio(row)
    path = f"{samples_dir}/{row['filename'].split('.')[0]}.wav"
    _save_and_record(y, row['sr'], path, row, 'original', records)


def generate_embeddings(df, samples_dir):
  target_rms = df['rms'].mean()[0]
  os.makedirs(samples_dir, exist_ok=True)

  records = []
  copy_orig(df, samples_dir,                "Copying original files ", records)
  noise(df, samples_dir, target_rms,        "Augmenting: noise      ", records)
  pitch_shift(df, samples_dir, target_rms,  "Augmenting: pitch shift", records)
  time_stretch(df, samples_dir, target_rms, "Augmenting: time stretch", records)
  time_shift(df, samples_dir, target_rms,   "Augmenting: time shift ", records)
  lower_speed(df, samples_dir, target_rms,  "Augmenting: lower speed", records)
  higher_speed(df, samples_dir, target_rms, "Augmenting: higher speed", records)

  aug_df = pd.DataFrame(records).reset_index(drop=True)
  return aug_df
