import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier


print("=" * 60)
print("CREMA-D V3 XGBOOST TRAINING")
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

# Columns that are NOT ML features
metadata_columns = [
    "speaker",
    "sentence",
    "level",
    "emotion",
    "filename"
]

X = df.drop(columns=metadata_columns)
y = df["emotion"]

print("Number of ML features:", X.shape[1])
print("Number of samples:", len(X))
print("Unique speakers:", df["speaker"].nunique())


# ============================================================
# 3. HANDLE MISSING / INFINITE VALUES
# ============================================================

X = X.replace([np.inf, -np.inf], np.nan)

X = X.fillna(X.median())

print("\nMissing values after cleaning:", X.isna().sum().sum())


# ============================================================
# 4. ENCODE EMOTIONS
# ============================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nEmotion mapping:")

for i, label in enumerate(label_encoder.classes_):
    print(i, "->", label)


# ============================================================
# 5. SPEAKER-INDEPENDENT SPLIT
# ============================================================
#
# IMPORTANT:
# We split SPEAKERS, not individual audio files.
#
# This prevents the same speaker from appearing in both
# training and testing data.
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
# 6. XGBOOST V3 MODEL
# ============================================================

print("\n" + "=" * 60)
print("TRAINING XGBOOST V3")
print("=" * 60)

model = XGBClassifier(
    n_estimators=400,
    max_depth=7,
    learning_rate=0.05,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,

    objective="multi:softmax",
    num_class=len(label_encoder.classes_),

    eval_metric="mlogloss",

    tree_method="hist",
    n_jobs=2,

    random_state=42
)


# ============================================================
# 7. TRAIN
# ============================================================

print("\nStarting training...")
print("Please do not run another heavy process at the same time.\n")

model.fit(
    X_train,
    y_train
)

print("\nTraining completed!")


# ============================================================
# 8. PREDICTION
# ============================================================

print("\nGenerating predictions...")

y_pred = model.predict(X_test)


# ============================================================
# 9. ACCURACY
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(f"\nXGBoost V3 Accuracy: {accuracy * 100:.2f}%")


# ============================================================
# 10. CLASSIFICATION REPORT
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
# 11. COMPARISON
# ============================================================

print("=" * 60)
print("COMPARISON")
print("=" * 60)

print("Previous best XGBoost V2 : 55.30%")
print(f"Current XGBoost V3       : {accuracy * 100:.2f}%")

improvement = (accuracy * 100) - 55.30

print(f"Improvement              : {improvement:+.2f}%")

print("\nDone!")