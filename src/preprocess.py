import librosa
import numpy as np

def load_audio(dir, sample, sr=None):
  path = dir/sample['filename']
  y, sample_sr = librosa.load(path, sr=sr)

  if sr:
    assert sr == sample_sr

  return y

def spectrogram(amps):
    return np.abs(librosa.stft(amps))

def db_spectrogram(s):
  db = librosa.amplitude_to_db(s, ref=np.max)

  return db

def mel_spectrogram(amps, sr):
  mel = librosa.feature.melspectrogram(y=amps, sr=sr)
  db = librosa.amplitude_to_db(mel, ref=np.max)

  return db

def mfcc(amps, sr):
  return librosa.feature.mfcc(y=amps, sr=sr)

def f0(amps, sr):
  return librosa.pyin(
    y=amps,
    sr=sr,
    fmin=librosa.note_to_hz('C2'),
    fmax=librosa.note_to_hz('C7'),
  )

def rms(s):
  return librosa.feature.rms(S=s)

def zcr(amps):
  return librosa.feature.zero_crossing_rate(amps)
