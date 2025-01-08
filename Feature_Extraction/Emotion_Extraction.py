import os
import librosa
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import logging
from sklearn.model_selection import GridSearchCV

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Hugging Face model and feature extractor
MODEL_NAME = "CAiRE/SER-wav2vec2-large-xlsr-53-eng-zho-all-age"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
wav2vec_model = Wav2Vec2Model.from_pretrained(MODEL_NAME)

# Paths to dataset and output
DATASET_PATH = os.path.join(SCRIPT_DIR, "../PROCESS-V1/PROCESS-V1")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "../Feature_Extraction")
CSV_FILE = os.path.join(DATASET_PATH, "dem-info.csv")
FEATURES_FILE = os.path.join(OUTPUT_PATH, "extracted_features.csv")

os.makedirs(OUTPUT_PATH, exist_ok=True)

participant_data = pd.read_csv(CSV_FILE)

# Extract emotion features from multiple layers
def extract_multi_layer_features(audio_file):
    audio, rate = librosa.load(audio_file, sr=16000)
    inputs = feature_extractor(audio, sampling_rate=rate, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = wav2vec_model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    layer_indices = [0, 6, 12]  # Example layers: low, mid, high
    aggregated_features = torch.cat([hidden_states[i].mean(dim=1) for i in layer_indices], dim=-1)
    return aggregated_features.squeeze().tolist()

# Function to clean data
def clean_data(df):
    for col in ["Age", "Converted-MMSE"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna()

# Custom ClippedSVR wrapper
from sklearn.base import BaseEstimator, RegressorMixin

class ClippedSVR(BaseEstimator, RegressorMixin):
    def __init__(self, svr=None, min_val=0, max_val=30):
        if svr is None:
            svr = SVR()  # Default SVR if none provided
        self.svr = svr
        self.min_val = min_val
        self.max_val = max_val

    def fit(self, X, y):
        self.svr.fit(X, y)
        return self

    def predict(self, X):
        predictions = self.svr.predict(X)
        return np.clip(predictions, self.min_val, self.max_val)

    def get_params(self, deep=True):
        params = {"min_val": self.min_val, "max_val": self.max_val}
        if deep:
            params.update({f"svr__{key}": value for key, value in self.svr.get_params(deep=deep).items()})
        return params

    def set_params(self, **params):
        svr_params = {key.split("__", 1)[1]: value for key, value in params.items() if key.startswith("svr__")}
        wrapper_params = {key: value for key, value in params.items() if not key.startswith("svr__")}
        self.svr.set_params(**svr_params)
        for key, value in wrapper_params.items():
            setattr(self, key, value)
        return self

# Check if features file exists
if not os.path.exists(FEATURES_FILE):
    logging.info("Features file not found. Starting feature extraction...")
    all_features = []
    for person_folder in os.listdir(DATASET_PATH):
        person_path = os.path.join(DATASET_PATH, person_folder)
        if not os.path.isdir(person_path):
            logging.warning(f"Skipping {person_folder}, not a directory.")
            continue
        logging.info(f"Processing folder: {person_folder}")
        participant_info = participant_data[participant_data["Record-ID"] == person_folder]
        if participant_info.empty:
            logging.warning(f"No metadata found for {person_folder}. Skipping...")
            continue
        record_id = participant_info["Record-ID"].values[0]
        train_or_dev = participant_info["TrainOrDev"].values[0]
        cls = participant_info["Class"].values[0]
        gender = participant_info["Gender"].values[0]
        age = participant_info["Age"].values[0]
        converted_mmse = participant_info["Converted-MMSE"].values[0]
        for file in os.listdir(person_path):
            if file.endswith(".wav"):
                audio_path = os.path.join(person_path, file)
                logging.info(f"Processing audio file: {audio_path}")
                try:
                    multi_layer_features = extract_multi_layer_features(audio_path)
                    feature_data = {
                        "Record-ID": record_id,
                        "TrainOrDev": train_or_dev,
                        "Class": cls,
                        "Gender": gender,
                        "Age": age,
                        "Converted-MMSE": converted_mmse,
                    }
                    for i, feature in enumerate(multi_layer_features):
                        feature_data[f"feature_{i}"] = feature
                    all_features.append(feature_data)
                except Exception as e:
                    logging.error(f"Error processing {audio_path}: {e}")
    features_df = pd.DataFrame(all_features)
    features_df.to_csv(FEATURES_FILE, index=False)
    logging.info("Feature extraction complete. Saved to extracted_features.csv")
else:
    logging.info("Features file found. Loading data...")
    features_df = pd.read_csv(FEATURES_FILE)

features_df = clean_data(features_df)

logging.info(f"Number of samples after cleaning: {len(features_df)}")

if len(features_df) == 0:
    logging.error("No valid samples available after cleaning. Please check the dataset.")
    exit()

if "Converted-MMSE" not in features_df.columns:
    logging.error("'Converted-MMSE' column not found in features file. Ensure the dataset is complete.")
    exit()

# Separate train and dev sets based on 'TrainOrDev' column
train_data = features_df[features_df["TrainOrDev"] == "train"]
dev_data = features_df[features_df["TrainOrDev"] == "dev"]

X_train = train_data.drop(columns=["Converted-MMSE", "Class", "Record-ID", "TrainOrDev"])
y_train = train_data["Converted-MMSE"]

X_test = dev_data.drop(columns=["Converted-MMSE", "Class", "Record-ID", "TrainOrDev"])
y_test = dev_data["Converted-MMSE"]

if "Gender" in X_train.columns:
    X_train = pd.get_dummies(X_train, columns=["Gender"], drop_first=True)
    X_test = pd.get_dummies(X_test, columns=["Gender"], drop_first=True)

X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

logging.info(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")

# Define SVR pipeline
svr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", ClippedSVR(SVR(), min_val=0, max_val=30))
])

# Define hyperparameter grid for tuning
param_grid = {
    "svr__kernel": ["linear", "rbf", "poly"],
    "svr__C": [0.1, 1, 10],
    "svr__gamma": [0.1, 0.01, 0.001],
    "svr__epsilon": [0.1, 0.2, 0.5],
}

# Apply GridSearchCV
grid_search = GridSearchCV(svr_pipeline, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model after tuning
best_svr_model = grid_search.best_estimator_

# Evaluate the best model on the test data
y_pred = best_svr_model.predict(X_test)

# Evaluate metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

logging.info("Evaluation Metrics (Best Model):")
logging.info(f"Mean Squared Error (MSE): {mse:.2f}")
logging.info(f"Mean Absolute Error (MAE): {mae:.2f}")
logging.info(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
logging.info(f"R-squared (R^2): {r2:.2f}")

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, 21), grid_search.cv_results_['mean_test_score'], marker='o', label='Validation Loss (MSE)')
plt.xlabel("Iteration")
plt.ylabel("Mean Squared Error")
plt.title("Learning Curve with Hyperparameter Tuning")
plt.legend()
plt.grid()
plt.savefig(os.path.join(OUTPUT_PATH, "learning_curve_tuned.png"))
logging.info("Learning curve saved to learning_curve_tuned.png")

# Save predictions
results = pd.DataFrame({"True_MMSE": y_test, "Predicted_MMSE": y_pred})
results.to_csv(os.path.join(OUTPUT_PATH, "svr_predictions_tuned.csv"), index=False)
logging.info("Predictions saved to svr_predictions_tuned.csv")
