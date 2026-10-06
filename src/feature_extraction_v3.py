import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm


# ==========================================
# PATHS
# ==========================================

DATASET_PATH = DATASET_PATH = "../data"
OUTPUT_PATH = "../data/cremad_features_v3.csv"


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def extract_statistics(values, prefix):
    """
    Extract statistical information from a feature.
    """
    values = np.asarray(values)

    return {
        f"{prefix}_mean": np.mean(values),
        f"{prefix}_std": np.std(values),
        f"{prefix}_min": np.min(values),
        f"{prefix}_max": np.max(values),
        f"{prefix}_median": np.median(values),
        f"{prefix}_range": np.max(values) - np.min(values),
    }


def extract_features(file_path):
    """
    Extract acoustic, spectral and prosodic features.
    """

    try:

        # ------------------------------------------
        # Load audio
        # ------------------------------------------

        y, sr = librosa.load(
            file_path,
            sr=16000,
            mono=True
        )

        if len(y) == 0:
            return None

        features = {}


        # ==========================================
        # 1. MFCC FEATURES
        # ==========================================

        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=20,
            n_fft=512,
            hop_length=256
        )

        for i in range(20):

            stats = extract_statistics(
                mfcc[i],
                f"mfcc_{i+1}"
            )

            features.update(stats)


        # ==========================================
        # 2. DELTA MFCC
        # ==========================================

        delta = librosa.feature.delta(mfcc)

        for i in range(20):

            stats = extract_statistics(
                delta[i],
                f"delta_{i+1}"
            )

            features.update(stats)


        # ==========================================
        # 3. DELTA-DELTA MFCC
        # ==========================================

        delta2 = librosa.feature.delta(
            mfcc,
            order=2
        )

        for i in range(20):

            stats = extract_statistics(
                delta2[i],
                f"delta2_{i+1}"
            )

            features.update(stats)


        # ==========================================
        # 4. CHROMA
        # ==========================================

        chroma = librosa.feature.chroma_stft(
            y=y,
            sr=sr,
            n_fft=512,
            hop_length=256
        )

        for i in range(12):

            stats = extract_statistics(
                chroma[i],
                f"chroma_{i+1}"
            )

            features.update(stats)


        # ==========================================
        # 5. MEL SPECTROGRAM
        # ==========================================

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

            stats = extract_statistics(
                mel_db[i],
                f"mel_{i+1}"
            )

            features.update(stats)


        # ==========================================
        # 6. SPECTRAL CENTROID
        # ==========================================

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


        # ==========================================
        # 7. SPECTRAL BANDWIDTH
        # ==========================================

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


        # ==========================================
        # 8. SPECTRAL ROLLOFF
        # ==========================================

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


        # ==========================================
        # 9. ZERO CROSSING RATE
        # ==========================================

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


        # ==========================================
        # 10. RMS ENERGY
        # ==========================================

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


        # ==========================================
        # 11. PITCH / F0
        # ==========================================

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


        # ==========================================
        # 12. HARMONIC FEATURES
        # ==========================================

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


        # ==========================================
        # 13. TEMPO
        # ==========================================

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


        # ==========================================
        # 14. GLOBAL AUDIO FEATURES
        # ==========================================

        features["duration"] = len(y) / sr

        features["audio_mean"] = np.mean(y)

        features["audio_std"] = np.std(y)

        features["audio_max"] = np.max(np.abs(y))

        features["audio_energy"] = np.mean(y ** 2)


        return features


    except Exception as e:

        print(
            f"\nError processing {file_path}: {e}"
        )

        return None


# ==========================================
# MAIN
# ==========================================

def main():

    print("=" * 60)
    print("CREMA-D V3 FEATURE EXTRACTION")
    print("=" * 60)

    # ------------------------------------------
    # Find WAV files
    # ------------------------------------------

    wav_files = []

    for root, dirs, files in os.walk(DATASET_PATH):

        for file in files:

            if file.lower().endswith(".wav"):

                wav_files.append(
                    os.path.join(root, file)
                )

    wav_files.sort()

    print(f"\nFound {len(wav_files)} WAV files.")

    if len(wav_files) == 0:

        print("ERROR: No WAV files found!")

        return


    # ------------------------------------------
    # Extract features
    # ------------------------------------------

    all_features = []

    failed = 0

    print("\nExtracting V3 features...\n")

    for file_path in tqdm(
        wav_files,
        desc="Processing"
    ):

        filename = os.path.basename(file_path)

        # CREMA-D format:
        # 1001_DFA_ANG_XX.wav

        parts = filename.replace(
            ".wav", ""
        ).split("_")

        if len(parts) != 4:

            print(
                f"Skipping invalid filename: {filename}"
            )

            failed += 1

            continue


        speaker = parts[0]

        sentence = parts[1]

        emotion_code = parts[2]

        level = parts[3]


        # ------------------------------------------
        # Emotion mapping
        # ------------------------------------------

        emotion_map = {

            "ANG": "angry",

            "DIS": "disgust",

            "FEA": "fearful",

            "HAP": "happy",

            "NEU": "neutral",

            "SAD": "sad"

        }


        if emotion_code not in emotion_map:

            failed += 1

            continue


        emotion = emotion_map[emotion_code]


        # ------------------------------------------
        # Extract audio features
        # ------------------------------------------

        features = extract_features(
            file_path
        )


        if features is None:

            failed += 1

            continue


        # ------------------------------------------
        # Add metadata
        # ------------------------------------------

        features["speaker"] = speaker

        features["sentence"] = sentence

        features["level"] = level

        features["emotion"] = emotion

        features["filename"] = filename


        all_features.append(features)


    # ==========================================
    # CREATE DATAFRAME
    # ==========================================

    df = pd.DataFrame(all_features)

    print("\n" + "=" * 60)

    print("EXTRACTION COMPLETE")

    print("=" * 60)

    print(
        f"Successfully processed: {len(df)}"
    )

    print(
        f"Failed/skipped: {failed}"
    )

    print(
        f"Total files: {len(wav_files)}"
    )

    print(
        f"Total columns: {len(df.columns)}"
    )


    # ==========================================
    # HANDLE MISSING VALUES
    # ==========================================

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    df[numeric_columns] = df[
        numeric_columns
    ].replace(
        [np.inf, -np.inf],
        np.nan
    )

    df[numeric_columns] = df[
        numeric_columns
    ].fillna(0)


    # ==========================================
    # SAVE
    # ==========================================

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"\nFeatures saved to:"
    )

    print(OUTPUT_PATH)

    print(
        f"\nFinal shape: {df.shape}"
    )

    print(
        f"Number of ML features: "
        f"{len(df.columns) - 5}"
    )

    print("\nEmotion distribution:")

    print(
        df["emotion"].value_counts()
    )

    print(
        "\nUnique speakers:",
        df["speaker"].nunique()
    )

    print("\nDone!")


# ==========================================
# RUN
# ==========================================

if __name__ == "__main__":

    main()