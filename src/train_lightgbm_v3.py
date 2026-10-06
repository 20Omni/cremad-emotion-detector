import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from lightgbm import LGBMClassifier


print("=" * 60)
print("CREMA-D V3 LIGHTGBM")
print("=" * 60)


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_FILE = "../data/cremad_features_v3.csv"

df = pd.read_csv(DATA_FILE)

print("\nDataset shape:", df.shape)


# ============================================================
# 2. PREPARE FEATURES
# ============================================================

metadata_columns = [
    "speaker",
    "sentence",
    "level",
    "emotion",
    "filename"
]

X = df.drop(columns=metadata_columns)
y = df["emotion"]

X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

print("Number of ML features:", X.shape[1])


# ============================================================
# 3. ENCODE LABELS
# ============================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nEmotion mapping:")

for i, label in enumerate(label_encoder.classes_):
    print(i, "->", label)


# ============================================================
# 4. SPEAKER-INDEPENDENT SPLIT
# ============================================================

np.random.seed(42)

speakers = df["speaker"].unique()

np.random.shuffle(speakers)

split_index = int(len(speakers) * 0.80)

train_speakers = speakers[:split_index]
test_speakers = speakers[split_index:]

train_mask = df["speaker"].isin(train_speakers)
test_mask = df["speaker"].isin(test_speakers)

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y_encoded[train_mask]
y_test = y_encoded[test_mask]


print("\n" + "=" * 60)
print("SPEAKER-INDEPENDENT SPLIT")
print("=" * 60)

print("Training speakers:", len(train_speakers))
print("Testing speakers:", len(test_speakers))

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print(
    "Speaker overlap:",
    len(set(train_speakers) & set(test_speakers))
)


# ============================================================
# 5. LIGHTGBM MODEL
# ============================================================

print("\n" + "=" * 60)
print("TRAINING LIGHTGBM")
print("=" * 60)

model = LGBMClassifier(
    objective="multiclass",
    num_class=len(label_encoder.classes_),

    n_estimators=400,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=31,

    min_child_samples=20,

    subsample=0.8,
    colsample_bytree=0.8,

    reg_alpha=0.0,
    reg_lambda=0.1,

    random_state=42,

    n_jobs=2,

    verbosity=-1
)


# ============================================================
# 6. TRAIN
# ============================================================

print("\nStarting training...")
print("CPU threads limited to 2.\n")

model.fit(
    X_train,
    y_train
)

print("\nTraining completed!")


# ============================================================
# 7. PREDICTION
# ============================================================

print("\nGenerating predictions...")

y_pred = model.predict(X_test)

# LightGBM can return shape (n_samples, 1)
# depending on configuration, so flatten it.
y_pred = np.asarray(y_pred).astype(int).flatten()


# ============================================================
# 8. ACCURACY
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(f"\nLightGBM V3 Accuracy: {accuracy * 100:.2f}%")


# ============================================================
# 9. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)


# ============================================================
# 10. COMPARISON
# ============================================================

print("=" * 60)
print("COMPARISON")
print("=" * 60)

print("XGBoost V3              : 58.12%")
print("XGBoost + SVM Ensemble  : 59.92%")
print(f"LightGBM V3             : {accuracy * 100:.2f}%")

print(
    f"\nDifference vs Ensemble: "
    f"{(accuracy * 100) - 59.92:+.2f}%"
)

print("\nDone!")