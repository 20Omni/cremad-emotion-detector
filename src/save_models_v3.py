import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ============================================================
# CREMA-D V3 - SAVE FINAL 60.24% ENSEMBLE MODELS
# ============================================================
#
# This script reproduces the FINAL TRAINING stage of the
# 60.24% XGBoost + SVM + LightGBM ensemble.
#
# It does NOT run the internal validation or the final test.
# It simply recreates the same 72-speaker training split and
# saves the three trained models for the frontend.
#
# Expected saved files:
#   models/
#       xgboost_v3.pkl
#       svm_v3.pkl
#       lightgbm_v3.pkl
#       label_encoder.pkl
#       feature_columns.pkl
#       feature_medians.pkl
#       ensemble_config.pkl
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(
    BASE_DIR,
    "..",
    "data",
    "cremad_features_v3.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "..",
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)


print("=" * 70)
print("CREMA-D V3 - SAVE FINAL ENSEMBLE MODELS")
print("=" * 70)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)


# ============================================================
# 3. PREPARE FEATURES
# ============================================================

metadata_columns = [
    "speaker",
    "sentence",
    "level",
    "emotion",
    "filename"
]

X = df.drop(columns=metadata_columns).copy()
y = df["emotion"].copy()

# EXACTLY as used in the 60.24% training script
X = X.replace([np.inf, -np.inf], np.nan)
feature_medians = X.median()
X = X.fillna(feature_medians)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

feature_columns = X.columns.tolist()

print("ML features:", len(feature_columns))
print("Speakers:", df["speaker"].nunique())

print("\nEmotion mapping:")
for i, emotion in enumerate(label_encoder.classes_):
    print(f"{i} -> {emotion}")


# ============================================================
# 4. RECREATE THE EXACT SPEAKER-INDEPENDENT SPLIT
# ============================================================
#
# This matches:
#
# np.random.seed(42)
# speakers = df["speaker"].unique()
# np.random.shuffle(speakers)
# split_index = int(len(speakers) * 0.80)
#
# Therefore:
# 72 training speakers
# 19 unseen test speakers
#
# We train only on the 72 training speakers.
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
y_train = y_encoded[train_mask]

X_test = X.loc[test_mask]
y_test = y_encoded[test_mask]

print("\n" + "=" * 70)
print("SPEAKER-INDEPENDENT SPLIT")
print("=" * 70)

print("Training speakers:", len(train_speakers))
print("Testing speakers :", len(test_speakers))
print("Training samples :", len(X_train))
print("Testing samples  :", len(X_test))

overlap = set(train_speakers) & set(test_speakers)

print("Speaker overlap   :", len(overlap))


# ============================================================
# 5. TRAIN XGBOOST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING XGBOOST")
print("=" * 70)

xgb_model = XGBClassifier(
    n_estimators=400,
    max_depth=7,
    learning_rate=0.05,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,

    objective="multi:softprob",
    num_class=len(label_encoder.classes_),
    eval_metric="mlogloss",

    tree_method="hist",
    n_jobs=2,

    random_state=42
)

xgb_model.fit(X_train, y_train)

print("XGBoost training completed.")


# ============================================================
# 6. TRAIN SVM
# ============================================================

print("\n" + "=" * 70)
print("TRAINING SVM")
print("=" * 70)

svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        C=10,
        gamma=0.001,
        kernel="rbf",
        probability=True,
        random_state=42
    ))
])

svm_model.fit(X_train, y_train)

print("SVM training completed.")


# ============================================================
# 7. TRAIN LIGHTGBM
# ============================================================

print("\n" + "=" * 70)
print("TRAINING LIGHTGBM")
print("=" * 70)

lgb_model = LGBMClassifier(
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

lgb_model.fit(X_train, y_train)

print("LightGBM training completed.")


# ============================================================
# 8. SAVE MODELS
# ============================================================

print("\n" + "=" * 70)
print("SAVING MODELS")
print("=" * 70)

xgb_path = os.path.join(
    MODEL_DIR,
    "xgboost_v3.pkl"
)

svm_path = os.path.join(
    MODEL_DIR,
    "svm_v3.pkl"
)

lgb_path = os.path.join(
    MODEL_DIR,
    "lightgbm_v3.pkl"
)

encoder_path = os.path.join(
    MODEL_DIR,
    "label_encoder.pkl"
)

columns_path = os.path.join(
    MODEL_DIR,
    "feature_columns.pkl"
)

medians_path = os.path.join(
    MODEL_DIR,
    "feature_medians.pkl"
)

config_path = os.path.join(
    MODEL_DIR,
    "ensemble_config.pkl"
)


joblib.dump(xgb_model, xgb_path)
joblib.dump(svm_model, svm_path)
joblib.dump(lgb_model, lgb_path)

joblib.dump(label_encoder, encoder_path)
joblib.dump(feature_columns, columns_path)
joblib.dump(feature_medians, medians_path)

ensemble_config = {
    "xgb_weight": 0.4,
    "svm_weight": 0.4,
    "lgb_weight": 0.2,
    "feature_count": len(feature_columns),
    "training_speakers": len(train_speakers),
    "test_speakers": len(test_speakers),
    "random_state": 42
}

joblib.dump(
    ensemble_config,
    config_path
)


# ============================================================
# 9. VERIFY FILES
# ============================================================

print("\nSaved files:")

for path in [
    xgb_path,
    svm_path,
    lgb_path,
    encoder_path,
    columns_path,
    medians_path,
    config_path
]:
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(
        f"  {os.path.basename(path):25s}"
        f" {size_mb:.2f} MB"
    )


# ============================================================
# 10. VERIFY MODEL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("VERIFYING SAVED MODELS")
print("=" * 70)

xgb_prob = xgb_model.predict_proba(X_test)
svm_prob = svm_model.predict_proba(X_test)
lgb_prob = lgb_model.predict_proba(X_test)

final_probability = (
    0.4 * xgb_prob
    + 0.4 * svm_prob
    + 0.2 * lgb_prob
)

final_prediction = np.argmax(
    final_probability,
    axis=1
)

accuracy = np.mean(
    final_prediction == y_test
)

print(f"\nVerification ensemble accuracy: {accuracy * 100:.2f}%")

print("\nExpected official result: approximately 60.24%")

print("\n" + "=" * 70)
print("MODEL EXPORT COMPLETE")
print("=" * 70)

print("\nYour frontend can now load:")
print("  XGBoost  -> 40%")
print("  SVM      -> 40%")
print("  LightGBM -> 20%")

print("\nNo model training is required when the frontend runs.")
