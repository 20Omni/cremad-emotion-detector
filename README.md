# Voice Emotion Detector

Predicts one of six emotions (angry, disgust, fearful, happy, neutral, sad) from a short voice clip.
Two models run side by side: a **pretrained emotion2vec+ model** (main result) and my own
**handcrafted-feature ensemble** (baseline). A small FastAPI + HTML website serves both.

## Results (honest summary)

| Model / test | Result | Notes |
|---|---|---|
| Baseline ensemble (721 acoustic features, XGBoost 40% + SVM 40% + LightGBM 20%) on CREMA-D | **60.24%** | 19 unseen speakers, 1,557 clips, speaker-independent split |
| Baseline on 60 clean ElevenLabs TTS clips | 16.7% | Predicted "angry" for every clip |
| emotion2vec+ (pretrained, no training) on the same 60 TTS clips | 88.3% | One synthetic voice; the words themselves carried emotion |
| emotion2vec+ on 120 TTS clips, 2 voices, same neutral sentences for all emotions | 75.0% | Harder, fairer test; fearful is the weakest class |
| emotion2vec+ on 120 random CREMA-D clips | 84.2% | May overlap its own training data, sanity check only |

60.24% measures agreement with CREMA-D labels on unseen speakers, not with human perception.

## What I found

The baseline collapsed to one class on clean audio. A feature-distribution check
(`diagnose_features.py`) showed 6.7% of its 721 features were more than 4 standard deviations from the
training mean, against 0.2% for real CREMA-D clips. The most extreme were low mel-band variation features:
CREMA-D has a steady noise floor (about -47 dBFS) that clean TTS (about -58 dBFS) lacks.
Adding CREMA-D-level noise made the predictions spread across classes (`noise_test.py`).
Handcrafted dB-scale features trained on one corpus do not transfer; a pretrained speech model does much better.

## Run

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
uvicorn server:app --port 8000      # then open http://localhost:8000
```

Needs the `models\` folder (included) and, for the diagnostics, `data\cremad_features_v3.csv` (not in git).
First start downloads the emotion2vec+ model (about 1 GB). Set `E2V_VARIANT=base` for the smaller one.

## Layout

- `server.py`, `web\index.html` : website and API
- `src\emotion_predictor.py`, `models\` : baseline ensemble (frozen)
- `src\feature_extraction_v3.py`, `src\ensemble_v3_three_models.py` : how the baseline was built
- `emotion2vec_test.py`, `eval_tts.py`, `diagnose_features.py`, `noise_test.py` : evaluation and diagnosis
- `tools\generate_tts_testset.py` : makes the multi-voice TTS test set (needs your own ElevenLabs key in `.env`)

## Limits

Trained on acted English speech. Microphone, noise, accent and subtle emotion lower accuracy, and the
pretrained model's scores are overconfident. TTS clips are one kind of synthetic voice, not real people.

## Next

Collect a small real-voice test set from many speakers, measure honest accuracy, calibrate the confidence,
and test on further datasets (RAVDESS, SAVEE, ESD).
