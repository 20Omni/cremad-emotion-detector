import os
import sys
from pathlib import Path

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))
from emotion_predictor import predict_emotion  # noqa: E402

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("emotion_test")
labels = pd.read_csv(root / "labels.csv")

rows = []
for _, r in labels.iterrows():
    f = root.parent / Path(str(r["filename"]).replace("\\", "/"))
    try:
        res = predict_emotion(str(f))
        rows.append({
            "file": f.name,
            "true": r["emotion"],
            "pred": res["final_emotion"].lower(),
            "conf": round(float(res["confidence"]), 1),
        })
    except Exception as e:
        print("FAILED", f.name, "->", e)

df = pd.DataFrame(rows)
print("\nClips scored:", len(df))
print("Overall accuracy: {:.1f}%".format((df.true == df.pred).mean() * 100))
print("\nConfusion matrix (rows = true, columns = predicted):")
print(pd.crosstab(df.true, df.pred))
print("\nRecall per emotion:")
print(df.assign(ok=df.true == df.pred).groupby("true").ok.mean().round(2))
print("\nAverage confidence per predicted emotion:")
print(df.groupby("pred").conf.mean().round(1))
df.to_csv("tts_results.csv", index=False)
print("\nSaved tts_results.csv")