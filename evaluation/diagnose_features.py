import os
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))
import emotion_predictor as ep  # noqa: E402

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("emotion_test")
cols = list(joblib.load(os.path.join(BASE, "models", "feature_columns.pkl")))

csv = next((p for p in [os.path.join(BASE, "data", "cremad_features_v3.csv"),
                        os.path.join(BASE, "cremad_features_v3.csv")] if os.path.exists(p)), None)
if csv is None:
    sys.exit("cremad_features_v3.csv not found in data\\ or the project root.")
train = pd.read_csv(csv)[cols]
mu, sd = train.mean(), train.std().replace(0, np.nan)

names = [n for n in dir(ep) if "extract" in n.lower() and callable(getattr(ep, n))]
names.sort(key=lambda n: ("v3" not in n.lower(), n))
if not names:
    print("No extractor function found in emotion_predictor.py. Callables there:")
    print([n for n in dir(ep) if callable(getattr(ep, n)) and not n.startswith("_")])
    sys.exit(1)
fn = getattr(ep, names[0])
print("Using extractor:", names[0])


def get_features(path):
    try:
        out = fn(path)
    except Exception:
        import librosa
        y, sr = librosa.load(path, sr=16000, mono=True)
        out = fn(y, sr)
    if isinstance(out, dict):
        s = pd.Series(out)
    elif isinstance(out, pd.Series):
        s = out
    elif isinstance(out, pd.DataFrame):
        s = out.iloc[0]
    else:
        arr = np.asarray(out, dtype=float).ravel()
        if len(arr) != len(cols):
            raise ValueError(f"extractor returned {len(arr)} values, expected {len(cols)}")
        s = pd.Series(arr, index=cols)
    return s.reindex(cols).astype(float)


rows = []
for f in sorted(root.glob("*/*.mp3")):
    try:
        rows.append(get_features(str(f)))
    except Exception as e:
        print("FAILED", f.name, "->", e)
if not rows:
    sys.exit("No clips could be processed.")

tts = pd.DataFrame(rows).fillna(train.median())
z_tts = (tts - mu) / sd
z_ref = (train.sample(min(60, len(train)), random_state=0) - mu) / sd

fams = pd.Series({c: (re.split(r"[_\d]", str(c))[0].lower() or str(c)) for c in cols})


def pct_beyond(z):
    return (z.abs() > 4).mean().groupby(fams).mean() * 100


table = pd.DataFrame({
    "features": fams.value_counts(),
    "TTS_%>4sd": pct_beyond(z_tts).round(1),
    "CREMA-D_%>4sd": pct_beyond(z_ref).round(1),
}).sort_values("TTS_%>4sd", ascending=False)

print(f"\nClips processed: {len(tts)}")
print("\nShare of features more than 4 std devs from the training mean, by family:")
print(table.to_string())
print("\nOverall: TTS {:.1f}%  vs  CREMA-D sample {:.1f}%".format(
    (z_tts.abs() > 4).values.mean() * 100, (z_ref.abs() > 4).values.mean() * 100))

zm = z_tts.mean()
top = zm.abs().sort_values(ascending=False).head(15).index
print("\n15 most extreme features (average z-score over the TTS clips):")
print(pd.DataFrame({"train_mean": mu[top].round(4), "tts_mean": tts[top].mean().round(4),
                    "z": zm[top].round(1)}).to_string())