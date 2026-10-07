import glob
import os
import random
import sys
import tempfile
from pathlib import Path

import librosa
import pandas as pd
import soundfile as sf
from funasr import AutoModel

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 16000
SIX = ["angry", "disgust", "fearful", "happy", "neutral", "sad"]
CODE = {"ANG": "angry", "DIS": "disgust", "FEA": "fearful",
        "HAP": "happy", "NEU": "neutral", "SAD": "sad"}

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("emotion_test")
variant = sys.argv[2] if len(sys.argv) > 2 else "large"  # "large" or "base"


def load_model():
    try:
        return AutoModel(model=f"iic/emotion2vec_plus_{variant}")
    except Exception as e:
        print("ModelScope download failed, trying Hugging Face:", e)
        return AutoModel(model=f"emotion2vec/emotion2vec_plus_{variant}", hub="hf")


def predict(model, y):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as t:
        path = t.name
    sf.write(path, y, SR)
    try:
        r = model.generate(path, granularity="utterance", extract_embedding=False)[0]
    finally:
        os.remove(path)
    scores = {}
    for lab, s in zip(r["labels"], r["scores"]):
        k = lab.split("/")[-1].strip().lower()
        scores[{"disgusted": "disgust"}.get(k, k)] = float(s)
    six = {k: scores.get(k, 0.0) for k in SIX}
    return max(six, key=six.get), max(scores, key=scores.get)


def evaluate(title, items, model):
    rows = []
    for i, (path, true) in enumerate(items, 1):
        y, _ = librosa.load(path, sr=SR, mono=True)
        pred, top = predict(model, y)
        rows.append({"file": os.path.basename(path), "true": true, "pred": pred, "top_any": top})
        if i % 20 == 0:
            print(f"  {title}: {i}/{len(items)}")
    df = pd.DataFrame(rows)
    print(f"\n=== {title} ===")
    print("Accuracy (six emotions only): {:.1f}%".format((df.true == df.pred).mean() * 100))
    print(pd.crosstab(df.true, df.pred))
    print("Clips where the model's overall top label was outside the six:",
          int((~df.top_any.isin(SIX)).sum()))
    return df


model = load_model()

labels = pd.read_csv(root / "labels.csv")
tts = [(str(root.parent / Path(str(r["filename"]).replace("\\", "/"))), r["emotion"])
       for _, r in labels.iterrows()]
df1 = evaluate("TTS clips", tts, model)
df1.to_csv("e2v_tts_results.csv", index=False)

random.seed(0)
by_emo = {}
for p in glob.glob(os.path.join(BASE, "data", "**", "*.wav"), recursive=True):
    code = os.path.basename(p).split("_")[2] if p.count("_") >= 3 else ""
    if code in CODE:
        by_emo.setdefault(CODE[code], []).append(p)
crema = [(p, e) for e, ps in by_emo.items() for p in random.sample(ps, min(20, len(ps)))]
if crema:
    df2 = evaluate("CREMA-D sample", crema, model)
    df2.to_csv("e2v_cremad_results.csv", index=False)
    print("\nNote: emotion2vec+ may have seen CREMA-D during its own training, so treat this as a sanity check only.")