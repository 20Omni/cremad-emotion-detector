import math
import os
import sys
import tempfile

import librosa
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "src"))
from emotion_predictor import predict_emotion  # noqa: E402

SR = 16000
SIX = ["angry", "disgust", "fearful", "happy", "neutral", "sad"]
VARIANT = os.getenv("E2V_VARIANT", "large")  # set E2V_VARIANT=base for the smaller model
_e2v = None


def e2v_model():
    global _e2v
    if _e2v is None:
        from funasr import AutoModel
        try:
            _e2v = AutoModel(model=f"iic/emotion2vec_plus_{VARIANT}")
        except Exception:
            _e2v = AutoModel(model=f"emotion2vec/emotion2vec_plus_{VARIANT}", hub="hf")
    return _e2v


print(f"Loading emotion2vec+ ({VARIANT}) ...")
try:
    e2v_model()
    print("emotion2vec+ ready.")
except Exception as e:
    print("Could not load emotion2vec+:", e)

app = FastAPI()


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def run_e2v(path):
    r = e2v_model().generate(path, granularity="utterance", extract_embedding=False)[0]
    scores = {}
    for lab, s in zip(r["labels"], r["scores"]):
        k = lab.split("/")[-1].strip().lower()
        scores[{"disgusted": "disgust"}.get(k, k)] = float(s)
    six = {k: scores.get(k, 0.0) for k in SIX}
    total = sum(six.values()) or 1.0
    probs = {k: v / total * 100 for k, v in six.items()}
    top = max(probs, key=probs.get)
    return {"emotion": top, "confidence": probs[top], "probabilities": probs,
            "top_any": max(scores, key=scores.get)}


def quality(y):
    fr = librosa.feature.rms(y=y, frame_length=1024, hop_length=512)[0]
    floor = float(np.percentile(fr, 10))
    return {"duration": len(y) / SR, "noise_floor_db": 20 * math.log10(floor + 1e-9),
            "peak": float(np.abs(y).max()), "clipped": float((np.abs(y) > 0.99).mean())}


def analyze(data):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(data)
        path = f.name
    try:
        main = run_e2v(path)
        try:
            base = predict_emotion(path)
        except Exception as e:
            print("Baseline failed:", e)
            base = None
        y, _ = librosa.load(path, sr=SR, mono=True)
        q = quality(y)
    finally:
        os.remove(path)

    w = []
    if q["duration"] < 1.0:
        w.append("Very short clip (under 1 second). Results are less reliable.")
    if q["peak"] < 0.05:
        w.append("The recording is very quiet. Try speaking closer to the microphone.")
    if q["clipped"] > 0.01:
        w.append("The audio is clipping (distorted), which can confuse the models.")
    if main["confidence"] < 50:
        w.append("Low confidence: several emotions are close, so treat this as uncertain.")
    if main["top_any"] not in SIX:
        w.append(f"The pretrained model's closest label was '{main['top_any']}', which is outside the six emotions.")
    if base is not None:
        if base["final_emotion"].lower() != main["emotion"]:
            w.append("The pretrained model and the baseline ensemble disagree.")
        if q["noise_floor_db"] < -54:  # heuristic: CREMA-D is about -47 dBFS, clean TTS about -58
            w.append("This audio is much cleaner than the baseline's training data, so the baseline result may be unreliable.")
    return {"main": main, "baseline": base, "quality": q, "warnings": w}


@app.get("/")
def index():
    return FileResponse(os.path.join(BASE, "web", "index.html"))


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    try:
        return JSONResponse(clean(await run_in_threadpool(analyze, data)))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)