from pathlib import Path
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# 1) File paths
# -----------------------------
base_dir = Path(".")
names_file = base_dir / "audiology.standardized.names"
data_file = base_dir / "audiology.standardized.data"
test_file = base_dir / "audiology.standardized.test"

# -----------------------------
# 2) Read the raw files
# -----------------------------
names_text = names_file.read_text(errors="ignore")
data_lines = [line.strip() for line in data_file.read_text(errors="ignore").splitlines() if line.strip()]
test_lines = [line.strip() for line in test_file.read_text(errors="ignore").splitlines() if line.strip()]

print("Loaded files:")
print("Names file characters:", len(names_text))
print("Data rows:", len(data_lines))
print("Test rows:", len(test_lines))

# -----------------------------
# 3) Column names
# -----------------------------
columns = [
    "identifier",
    "age_gt_60",
    "air",
    "airBoneGap",
    "ar_c",
    "ar_u",
    "bone",
    "boneAbnormal",
    "bser",
    "history_buzzing",
    "history_dizziness",
    "history_fluctuating",
    "history_fullness",
    "history_heredity",
    "history_nausea",
    "history_noise",
    "history_recruitment",
    "history_ringing",
    "history_roaring",
    "history_vomiting",
    "late_wave_poor",
    "m_at_2k",
    "m_cond_lt_1k",
    "m_gt_1k",
    "m_m_gt_2k",
    "m_m_sn",
    "m_m_sn_gt_1k",
    "m_m_sn_gt_2k",
    "m_m_sn_gt_500",
    "m_p_sn_gt_2k",
    "m_s_gt_500",
    "m_s_sn",
    "m_s_sn_gt_1k",
    "m_s_sn_gt_2k",
    "m_s_sn_gt_3k",
    "m_s_sn_gt_4k",
    "m_sn_2_3k",
    "m_sn_gt_1k",
    "m_sn_gt_2k",
    "m_sn_gt_3k",
    "m_sn_gt_4k",
    "m_sn_gt_500",
    "m_sn_gt_6k",
    "m_sn_lt_1k",
    "m_sn_lt_2k",
    "m_sn_lt_3k",
    "middle_wave_poor",
    "mod_gt_4k",
    "mod_mixed",
    "mod_s_mixed",
    "mod_s_sn_gt_500",
    "mod_sn",
    "mod_sn_gt_1k",
    "mod_sn_gt_2k",
    "mod_sn_gt_3k",
    "mod_sn_gt_4k",
    "mod_sn_gt_500",
    "notch_4k",
    "notch_at_4k",
    "o_ar_c",
    "o_ar_u",
    "s_sn_gt_1k",
    "s_sn_gt_2k",
    "s_sn_gt_4k",
    "speech",
    "static_normal",
    "tymp",
    "viith_nerve_signs",
    "wave_V_delayed",
    "waveform_ItoV_prolonged",
    "class"
]

print("Total columns:", len(columns))

# -----------------------------
# 4) Parse rows
# -----------------------------
def parse_rows(lines, col_list):
    rows = []
    for line in lines:
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != len(col_list):
            print("Skipping row with unexpected length:", len(parts), line[:120])
            continue
        rows.append(parts)
    return pd.DataFrame(rows, columns=col_list)

data_df = parse_rows(data_lines, columns)
test_df = parse_rows(test_lines, columns)

print("\nData shape:", data_df.shape)
print("Test shape:", test_df.shape)

print("\nTraining sample:")
print(data_df.head())

print("\nTest sample:")
print(test_df.head())

print("\nClass distribution in training data:")
print(data_df["class"].value_counts())

print("\nMissing values in training data:")
print(data_df.isna().sum().sum())

print("\nMissing values in test data:")
print(test_df.isna().sum().sum())

# -----------------------------
# 5) Save parsed files
# -----------------------------
data_df["split"] = "train"
test_df["split"] = "test"
full_df = pd.concat([data_df, test_df], ignore_index=True)

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

data_df.to_csv(output_dir / "audiology_train_parsed.csv", index=False)
test_df.to_csv(output_dir / "audiology_test_parsed.csv", index=False)
full_df.to_csv(output_dir / "audiology_full_parsed.csv", index=False)

print("\nSaved parsed files to output/")

# -----------------------------
# 6) Prepare data
# -----------------------------
X = full_df.drop(columns=["class", "split"])
y = full_df["class"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

train_mask = full_df["split"] == "train"
X_train = X[train_mask].copy()
X_test = X[~train_mask].copy()
y_train = y_encoded[train_mask]
y_test = y_encoded[~train_mask]

categorical_cols = X_train.columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]
            ),
            categorical_cols
        )
    ]
)

# -----------------------------
# 7) Random forest model
# -----------------------------
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

rf_clf = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", rf_model)
    ]
)

rf_clf.fit(X_train, y_train)
rf_preds = rf_clf.predict(X_test)

rf_acc = accuracy_score(y_test, rf_preds)
rf_cm = confusion_matrix(y_test, rf_preds)

used_labels = sorted(set(y_test) | set(rf_preds))
used_class_names = label_encoder.inverse_transform(used_labels)

rf_report = classification_report(
    y_test,
    rf_preds,
    labels=used_labels,
    target_names=used_class_names,
    zero_division=0
)

print("\nRandom Forest Accuracy:", rf_acc)
print("\nRandom Forest Confusion Matrix:\n", rf_cm)
print("\nRandom Forest Classification Report:\n", rf_report)

with open(output_dir / "random_forest_results.txt", "w") as f:
    f.write(f"Accuracy: {rf_acc}\n\n")
    f.write("Confusion matrix:\n")
    f.write(str(rf_cm))
    f.write("\n\nClassification report:\n")
    f.write(rf_report)