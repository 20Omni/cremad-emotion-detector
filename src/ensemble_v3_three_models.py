import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


print("=" * 60)
print("CREMA-D V3 - THREE MODEL ENSEMBLE")
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

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("ML features:", X.shape[1])
print("Speakers:", df["speaker"].nunique())


# ============================================================
# 3. SPEAKER-INDEPENDENT SPLIT
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
print("Testing speakers :", len(test_speakers))
print("Training samples :", len(X_train))
print("Testing samples  :", len(X_test))

overlap = set(train_speakers) & set(test_speakers)

print("Speaker overlap   :", len(overlap))


# ============================================================
# 4. CREATE A VALIDATION SPLIT FROM TRAINING SPEAKERS
# ============================================================
#
# IMPORTANT:
# The final 19 test speakers are NOT used to choose weights.
#
# We split the 72 training speakers into:
#
#   80% model-training speakers
#   20% ensemble-validation speakers
#
# ============================================================

np.random.seed(42)

internal_speakers = train_speakers.copy()
np.random.shuffle(internal_speakers)

internal_split = int(len(internal_speakers) * 0.80)

model_speakers = internal_speakers[:internal_split]
validation_speakers = internal_speakers[internal_split:]

model_mask = df["speaker"].isin(model_speakers)
validation_mask = df["speaker"].isin(validation_speakers)

X_model = X.loc[model_mask]
X_validation = X.loc[validation_mask]

y_model = y_encoded[model_mask]
y_validation = y_encoded[validation_mask]

print("\n" + "=" * 60)
print("INTERNAL VALIDATION")
print("=" * 60)

print("Model-training speakers:", len(model_speakers))
print("Validation speakers    :", len(validation_speakers))

print("Model-training samples:", len(X_model))
print("Validation samples    :", len(X_validation))


# ============================================================
# 5. XGBOOST
# ============================================================

print("\n" + "=" * 60)
print("TRAINING XGBOOST")
print("=" * 60)

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

xgb_model.fit(X_model, y_model)

print("XGBoost completed.")


# ============================================================
# 6. SVM
# ============================================================

print("\n" + "=" * 60)
print("TRAINING SVM")
print("=" * 60)

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

svm_model.fit(X_model, y_model)

print("SVM completed.")


# ============================================================
# 7. LIGHTGBM
# ============================================================

print("\n" + "=" * 60)
print("TRAINING LIGHTGBM")
print("=" * 60)

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

lgb_model.fit(X_model, y_model)

print("LightGBM completed.")


# ============================================================
# 8. VALIDATION PROBABILITIES
# ============================================================

print("\nGenerating validation probabilities...")

xgb_val_prob = xgb_model.predict_proba(X_validation)
svm_val_prob = svm_model.predict_proba(X_validation)
lgb_val_prob = lgb_model.predict_proba(X_validation)


# ============================================================
# 9. TEST SELECTED WEIGHTS ON VALIDATION SET
# ============================================================
#
# We deliberately keep the search small.
#
# The final test set is NOT touched here.
# ============================================================

print("\n" + "=" * 60)
print("SELECTING ENSEMBLE WEIGHTS")
print("=" * 60)

weight_combinations = [
    (0.6, 0.3, 0.1),
    (0.6, 0.2, 0.2),
    (0.5, 0.4, 0.1),
    (0.5, 0.3, 0.2),
    (0.5, 0.2, 0.3),
    (0.4, 0.4, 0.2),
    (0.4, 0.3, 0.3),
    (0.4, 0.2, 0.4),
    (0.3, 0.4, 0.3),
    (0.3, 0.3, 0.4),
]

best_val_accuracy = 0
best_weights = None

for xgb_weight, svm_weight, lgb_weight in weight_combinations:

    combined_probability = (
        xgb_weight * xgb_val_prob
        + svm_weight * svm_val_prob
        + lgb_weight * lgb_val_prob
    )

    validation_prediction = np.argmax(
        combined_probability,
        axis=1
    )

    accuracy = accuracy_score(
        y_validation,
        validation_prediction
    )

    print(
        f"XGB={xgb_weight:.1f} | "
        f"SVM={svm_weight:.1f} | "
        f"LGB={lgb_weight:.1f} | "
        f"Accuracy={accuracy * 100:.2f}%"
    )

    if accuracy > best_val_accuracy:
        best_val_accuracy = accuracy
        best_weights = (
            xgb_weight,
            svm_weight,
            lgb_weight
        )


print("\nBest validation accuracy:",
      f"{best_val_accuracy * 100:.2f}%")

print(
    "Selected weights:",
    f"XGB={best_weights[0]:.1f},",
    f"SVM={best_weights[1]:.1f},",
    f"LGB={best_weights[2]:.1f}"
)


# ============================================================
# 10. RETRAIN ALL THREE MODELS ON ALL 72 TRAINING SPEAKERS
# ============================================================
#
# Now that the weights are fixed using ONLY the internal
# validation speakers, we can use all 72 training speakers.
# ============================================================

print("\n" + "=" * 60)
print("RETRAINING ON ALL TRAINING SPEAKERS")
print("=" * 60)

xgb_final = XGBClassifier(
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

svm_final = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        C=10,
        gamma=0.001,
        kernel="rbf",
        probability=True,
        random_state=42
    ))
])

lgb_final = LGBMClassifier(
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

print("\nTraining XGBoost...")
xgb_final.fit(X_train, y_train)

print("Training SVM...")
svm_final.fit(X_train, y_train)

print("Training LightGBM...")
lgb_final.fit(X_train, y_train)

print("\nAll final models trained.")


# ============================================================
# 11. FINAL TEST SET
# ============================================================
#
# THIS IS THE FIRST TIME WE USE THE 19 TEST SPEAKERS.
# ============================================================

print("\n" + "=" * 60)
print("FINAL UNSEEN-SPEAKER TEST")
print("=" * 60)

xgb_test_prob = xgb_final.predict_proba(X_test)
svm_test_prob = svm_final.predict_proba(X_test)
lgb_test_prob = lgb_final.predict_proba(X_test)


# ============================================================
# 12. INDIVIDUAL MODEL RESULTS
# ============================================================

xgb_prediction = np.argmax(xgb_test_prob, axis=1)
svm_prediction = np.argmax(svm_test_prob, axis=1)
lgb_prediction = np.argmax(lgb_test_prob, axis=1)

xgb_accuracy = accuracy_score(y_test, xgb_prediction)
svm_accuracy = accuracy_score(y_test, svm_prediction)
lgb_accuracy = accuracy_score(y_test, lgb_prediction)


# ============================================================
# 13. THREE MODEL ENSEMBLE
# ============================================================

final_probability = (
    best_weights[0] * xgb_test_prob
    + best_weights[1] * svm_test_prob
    + best_weights[2] * lgb_test_prob
)

final_prediction = np.argmax(
    final_probability,
    axis=1
)

ensemble_accuracy = accuracy_score(
    y_test,
    final_prediction
)


# ============================================================
# 14. FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(f"\nXGBoost V3             : {xgb_accuracy * 100:.2f}%")
print(f"SVM V3                : {svm_accuracy * 100:.2f}%")
print(f"LightGBM V3           : {lgb_accuracy * 100:.2f}%")
print(
    f"Three-Model Ensemble  : "
    f"{ensemble_accuracy * 100:.2f}%"
)

print(
    "\nSelected weights:",
    f"XGB={best_weights[0]:.1f},",
    f"SVM={best_weights[1]:.1f},",
    f"LGB={best_weights[2]:.1f}"
)

print(
    f"\nImprovement over previous "
    f"59.92% ensemble: "
    f"{(ensemble_accuracy * 100) - 59.92:+.2f}%"
)


# ============================================================
# 15. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("THREE-MODEL ENSEMBLE REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        final_prediction,
        target_names=label_encoder.classes_
    )
)


print("=" * 60)
print("DONE")
print("=" * 60)