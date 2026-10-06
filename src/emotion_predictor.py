import os
import pickle
import joblib
import numpy as np
import pandas as pd
import librosa


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")


# ============================================================
# LOAD SAVED MODELS
# ============================================================

def load_pickle(filename):
    """Load saved model/artifact using joblib, with pickle fallback."""
    path = os.path.join(MODEL_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required model file not found:\n{path}"
        )

    try:
        return joblib.load(path)
    except Exception as joblib_error:
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as pickle_error:
            raise RuntimeError(
                f"Could not load '{filename}'.\n"
                f"Path: {path}\n\n"
                f"joblib error: {joblib_error}\n\n"
                f"pickle error: {pickle_error}"
            ) from pickle_error


xgb_model = load_pickle("xgboost_v3.pkl")
svm_model = load_pickle("svm_v3.pkl")
lgb_model = load_pickle("lightgbm_v3.pkl")

label_encoder = load_pickle("label_encoder.pkl")
feature_columns = load_pickle("feature_columns.pkl")
feature_medians = load_pickle("feature_medians.pkl")
ensemble_config = load_pickle("ensemble_config.pkl")


# ============================================================
# FEATURE STATISTICS
# ============================================================

def extract_statistics(values, prefix):

    values = np.asarray(values)

    return {
        f"{prefix}_mean": np.mean(values),
        f"{prefix}_std": np.std(values),
        f"{prefix}_min": np.min(values),
        f"{prefix}_max": np.max(values),
        f"{prefix}_median": np.median(values),
        f"{prefix}_range": np.max(values) - np.min(values),
    }


# ============================================================
# V3 FEATURE EXTRACTION
# ============================================================

def extract_features(file_path):

    y, sr = librosa.load(
        file_path,
        sr=16000,
        mono=True
    )

    if len(y) == 0:
        raise ValueError("Audio file is empty.")

    features = {}

    # ========================================================
    # 1. MFCC
    # ========================================================

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=20,
        n_fft=512,
        hop_length=256
    )

    for i in range(20):

        features.update(
            extract_statistics(
                mfcc[i],
                f"mfcc_{i + 1}"
            )
        )

    # ========================================================
    # 2. DELTA MFCC
    # ========================================================

    delta = librosa.feature.delta(mfcc)

    for i in range(20):

        features.update(
            extract_statistics(
                delta[i],
                f"delta_{i + 1}"
            )
        )

    # ========================================================
    # 3. DELTA-DELTA MFCC
    # ========================================================

    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    for i in range(20):

        features.update(
            extract_statistics(
                delta2[i],
                f"delta2_{i + 1}"
            )
        )

    # ========================================================
    # 4. CHROMA
    # ========================================================

    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr,
        n_fft=512,
        hop_length=256
    )

    for i in range(12):

        features.update(
            extract_statistics(
                chroma[i],
                f"chroma_{i + 1}"
            )
        )

    # ========================================================
    # 5. MEL SPECTROGRAM
    # ========================================================

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=40,
        n_fft=512,
        hop_length=256
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    for i in range(40):

        features.update(
            extract_statistics(
                mel_db[i],
                f"mel_{i + 1}"
            )
        )

    # ========================================================
    # 6. SPECTRAL CENTROID
    # ========================================================

    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr,
        n_fft=512,
        hop_length=256
    )[0]

    features.update(
        extract_statistics(
            centroid,
            "spectral_centroid"
        )
    )

    # ========================================================
    # 7. SPECTRAL BANDWIDTH
    # ========================================================

    bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr,
        n_fft=512,
        hop_length=256
    )[0]

    features.update(
        extract_statistics(
            bandwidth,
            "spectral_bandwidth"
        )
    )

    # ========================================================
    # 8. SPECTRAL ROLLOFF
    # ========================================================

    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr,
        n_fft=512,
        hop_length=256
    )[0]

    features.update(
        extract_statistics(
            rolloff,
            "spectral_rolloff"
        )
    )

    # ========================================================
    # 9. ZERO CROSSING RATE
    # ========================================================

    zcr = librosa.feature.zero_crossing_rate(
        y,
        frame_length=512,
        hop_length=256
    )[0]

    features.update(
        extract_statistics(
            zcr,
            "zero_crossing"
        )
    )

    # ========================================================
    # 10. RMS ENERGY
    # ========================================================

    rms = librosa.feature.rms(
        y=y,
        frame_length=512,
        hop_length=256
    )[0]

    features.update(
        extract_statistics(
            rms,
            "rms"
        )
    )

    # ========================================================
    # 11. PITCH / F0
    # ========================================================

    try:

        f0, voiced_flag, voiced_prob = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
            frame_length=1024,
            hop_length=256
        )

        valid_f0 = f0[~np.isnan(f0)]

        if len(valid_f0) > 0:

            features.update(
                extract_statistics(
                    valid_f0,
                    "pitch"
                )
            )

            features["pitch_voiced_ratio"] = (
                len(valid_f0) / len(f0)
            )

        else:

            for name in [
                "mean",
                "std",
                "min",
                "max",
                "median",
                "range"
            ]:
                features[f"pitch_{name}"] = 0.0

            features["pitch_voiced_ratio"] = 0.0

    except Exception:

        for name in [
            "mean",
            "std",
            "min",
            "max",
            "median",
            "range"
        ]:
            features[f"pitch_{name}"] = 0.0

        features["pitch_voiced_ratio"] = 0.0

    # ========================================================
    # 12. HARMONIC FEATURES
    # ========================================================

    harmonic = librosa.effects.harmonic(y)

    harmonic_rms = librosa.feature.rms(
        y=harmonic,
        frame_length=512,
        hop_length=256
    )[0]

    features.update(
        extract_statistics(
            harmonic_rms,
            "harmonic_rms"
        )
    )

    # ========================================================
    # 13. TEMPO
    # ========================================================

    try:

        tempo, _ = librosa.beat.beat_track(
            y=y,
            sr=sr,
            hop_length=256
        )

        features["tempo"] = float(
            np.asarray(tempo).flatten()[0]
        )

    except Exception:

        features["tempo"] = 0.0

    # ========================================================
    # 14. GLOBAL AUDIO FEATURES
    # ========================================================

    features["duration"] = len(y) / sr

    features["audio_mean"] = np.mean(y)

    features["audio_std"] = np.std(y)

    features["audio_max"] = np.max(
        np.abs(y)
    )

    features["audio_energy"] = np.mean(
        y ** 2
    )

    return features


# ============================================================
# PREDICT EMOTION
# ============================================================

def predict_emotion(audio_path):

    # --------------------------------------------------------
    # Extract V3 features
    # --------------------------------------------------------

    raw_features = extract_features(audio_path)

    feature_df = pd.DataFrame([raw_features])

    # --------------------------------------------------------
    # Force exact training feature order
    # --------------------------------------------------------

    feature_df = feature_df.reindex(
        columns=feature_columns
    )

    # --------------------------------------------------------
    # Fill missing values using saved training medians
    # --------------------------------------------------------

    if isinstance(feature_medians, pd.Series):

        feature_df = feature_df.fillna(
            feature_medians
        )

    elif isinstance(feature_medians, dict):

        feature_df = feature_df.fillna(
            pd.Series(feature_medians)
        )

    else:

        feature_df = feature_df.fillna(0)

    # Final safety
    feature_df = feature_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    feature_df = feature_df.fillna(0)

    # --------------------------------------------------------
    # Verify dimensions
    # --------------------------------------------------------

    if feature_df.shape[1] != 721:

        raise ValueError(
            f"Expected 721 features, "
            f"but received {feature_df.shape[1]}"
        )

    # --------------------------------------------------------
    # Individual model probabilities
    # --------------------------------------------------------

    xgb_prob = xgb_model.predict_proba(
        feature_df
    )[0]

    svm_prob = svm_model.predict_proba(
        feature_df
    )[0]

    lgb_prob = lgb_model.predict_proba(
        feature_df
    )[0]

    # --------------------------------------------------------
    # Official ensemble
    # XGB = 40%
    # SVM = 40%
    # LGB = 20%
    # --------------------------------------------------------

    ensemble_prob = (
        0.40 * xgb_prob
        + 0.40 * svm_prob
        + 0.20 * lgb_prob
    )

    # --------------------------------------------------------
    # Final prediction
    # --------------------------------------------------------

    predicted_index = np.argmax(
        ensemble_prob
    )

    predicted_label = label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    # --------------------------------------------------------
    # Emotion probabilities
    # --------------------------------------------------------

    emotions = {}

    for i, emotion in enumerate(
        label_encoder.classes_
    ):

        emotions[str(emotion)] = float(
            ensemble_prob[i] * 100
        )

    # --------------------------------------------------------
    # Individual model predictions
    # --------------------------------------------------------

    xgb_index = np.argmax(xgb_prob)
    svm_index = np.argmax(svm_prob)
    lgb_index = np.argmax(lgb_prob)

    xgb_label = label_encoder.inverse_transform(
        [xgb_index]
    )[0]

    svm_label = label_encoder.inverse_transform(
        [svm_index]
    )[0]

    lgb_label = label_encoder.inverse_transform(
        [lgb_index]
    )[0]

    # --------------------------------------------------------
    # Acoustic information
    # --------------------------------------------------------

    def get_value(name):

        value = raw_features.get(name, 0)

        try:
            return float(value)
        except Exception:
            return 0.0

    audio_info = {

        "duration": get_value(
            "duration"
        ),

        "sample_rate": 16000,

        "pitch_mean": get_value(
            "pitch_mean"
        ),

        "pitch_std": get_value(
            "pitch_std"
        ),

        "pitch_min": get_value(
            "pitch_min"
        ),

        "pitch_max": get_value(
            "pitch_max"
        ),

        "pitch_range": get_value(
            "pitch_range"
        ),

        "pitch_voiced_ratio": get_value(
            "pitch_voiced_ratio"
        ),

        "rms_mean": get_value(
            "rms_mean"
        ),

        "rms_std": get_value(
            "rms_std"
        ),

        "zero_crossing_mean": get_value(
            "zero_crossing_mean"
        ),

        "spectral_centroid_mean": get_value(
            "spectral_centroid_mean"
        ),

        "spectral_bandwidth_mean": get_value(
            "spectral_bandwidth_mean"
        ),

        "spectral_rolloff_mean": get_value(
            "spectral_rolloff_mean"
        ),

        "harmonic_rms_mean": get_value(
            "harmonic_rms_mean"
        ),

        "tempo": get_value(
            "tempo"
        ),

        "audio_energy": get_value(
            "audio_energy"
        )
    }

    return {

        "final_emotion": str(
            predicted_label
        ),

        "confidence": float(
            ensemble_prob[predicted_index] * 100
        ),

        "probabilities": emotions,

        "models": {

            "XGBoost": {
                "prediction": str(xgb_label),
                "confidence": float(
                    np.max(xgb_prob) * 100
                )
            },

            "SVM": {
                "prediction": str(svm_label),
                "confidence": float(
                    np.max(svm_prob) * 100
                )
            },

            "LightGBM": {
                "prediction": str(lgb_label),
                "confidence": float(
                    np.max(lgb_prob) * 100
                )
            }
        },

        "audio": audio_info,

        "feature_count": feature_df.shape[1]
    }