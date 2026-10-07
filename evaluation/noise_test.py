import glob
import os
import random
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))
from emotion_predictor import predict_emotion  # noqa: E402

SR = 16000
root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("emotion_test")
labels = pd.read_csv(root / "labels.csv").groupby("emotion").head(5)  # 5 clips per emotion


def floor_rms(y):
    fr = librosa.feature.rms(y=y, frame_length=1024, hop_length=512)[0]
    return float(np.percentile(fr, 10))


def db(x):
    return 20 * np.log10(x + 1e-12)


wavs = glob.glob(os.path.join(BASE, "data", "**", "*.wav"), recursive=True)
random.seed(0)
sample = random.sample(wavs, min(40, len(wavs)))
cremad_floor = float(np.median([floor_rms(librosa.load(p, sr=SR)[0]) for p in sample]))

clips = []
for _, r in labels.iterrows():
    f = root.parent / Path(str(r["filename"]).replace("\\", "/"))
    y, _ = librosa.load(str(f), sr=SR, mono=True)
    clips.append((r["emotion"], y))

tts_floor = float(np.median([floor_rms(y) for _, y in clips]))
print(f"CREMA-D noise floor: {cremad_floor:.5f} ({db(cremad_floor):.1f} dBFS)")
print(f"TTS noise floor:     {tts_floor:.5f} ({db(tts_floor):.1f} dBFS)")
print(f"Testing {len(clips)} clips per setting (this takes a few minutes)...\n")

rng = np.random.default_rng(0)
for mult in (1, 2):
    preds = []
    for emo, y in clips:
        z = np.clip(y + rng.normal(0, cremad_floor * mult, len(y)), -1, 1)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as t:
            path = t.name
        sf.write(path, z, SR)
        try:
            preds.append(predict_emotion(path)["final_emotion"].lower())
        finally:
            os.remove(path)
    acc = np.mean([p == e for (e, _), p in zip(clips, preds)]) * 100
    print(f"Noise floor x{mult}: accuracy {acc:.1f}%  predictions: {dict(pd.Series(preds).value_counts())}")