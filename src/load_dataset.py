import pandas as pd
from pathlib import Path

def load_dataset(dir, clips_dir, csv_file, sr=None):
  df = pd.read_csv(Path(dir) / csv_file)
  df["sex"] = df["filename"].apply(lambda x: 'female' if int(x.split(".")[0].split("_")[-1]) else 'male')
  df["nwords"] = df["text"].apply(lambda x: len(x.split(' ')))
  df["path"] = df["filename"].apply(lambda x: Path(dir) / clips_dir / x)
  df["sr"] = sr

  return df
