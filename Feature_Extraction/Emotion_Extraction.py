import os
import librosa
import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import joblib
import csv

MODEL_NAME = "CAiRE/SER-wav2vec2-large-xlsr-53-eng-zho-all-age"
processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
model = Wav2Vec2Model.from_pretrained(MODEL_NAME, output_hidden_states=True)

MMSE_RANGE = 30

results = []

wav2vec_feature_scaler = StandardScaler()
feature_scaler = StandardScaler()
target_scaler = StandardScaler()
one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

SCRIPT_DIR = os.path.dirname(__file__)
DATASET_PATH = os.path.join(SCRIPT_DIR, "../PROCESS-V1/PROCESS-V1")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "../Feature_Extraction")
FEATURES_FILE = os.path.join(OUTPUT_PATH, "middle_level_features.csv")
CSV_FILE = os.path.join(DATASET_PATH, "dem-info.csv")
PLOT_PNG = os.path.join(OUTPUT_PATH, "plot_middle_level_metrics.png")
SCALER_FEATURES = os.path.join(OUTPUT_PATH, "feature_scaler.joblib")
SCALER_TARGET = os.path.join(OUTPUT_PATH, "target_scaler.joblib")

os.makedirs(OUTPUT_PATH, exist_ok=True)

print("Loading participant metadata from CSV:")
participant_data = pd.read_csv(CSV_FILE)
print(f"Loaded {len(participant_data)} participant records from {CSV_FILE}.")


def calculate_std(y_true, y_pred):
    # Step 1: Calculate the mean of the predictions
    mean_pred = sum(y_pred) / len(y_pred)

    # Step 2: Calculate the squared differences from the mean
    squared_diff = [(y_p - mean_pred) ** 2 for y_p in y_pred]

    # Step 3: Calculate the mean of squared differences
    mean_squared_diff = sum(squared_diff) / len(squared_diff)

    # Step 4: Take the square root of the mean squared differences
    std = mean_squared_diff ** 0.5

    return std

def calculate_mae(y_true, y_pred):
    error_sum = sum(abs(y_t - y_p) for y_t, y_p in zip(y_true, y_pred))
    return error_sum / len(y_true)

def calculate_mse(y_true, y_pred):
    error_sum = sum((y_t - y_p) ** 2 for y_t, y_p in zip(y_true, y_pred))
    return error_sum / len(y_true)

def plot_metrics(mse_values, mae_values, plot_path):
    plt.figure(figsize=(14,7))
    plt.plot(range(1, len(mse_values) + 1), mse_values, label="MSE", marker='o')
    plt.plot(range(1, len(mae_values) + 1), mae_values, label="MAE", marker='x')
    plt.title("MSE and MAE")
    plt.xlabel("Fold")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(plot_path, format='png')
    plt.close()

def scale_features_and_encode(df):
    # Scale numerical columns (Age and MMSE)
    df["Age"] = feature_scaler.fit_transform(df[["Age"]])
    df["Converted-MMSE"] = target_scaler.fit_transform(df[["Converted-MMSE"]])

    # Encode categorical columns (Gender and Class)
    categorical_data = df[["Gender", "Class"]]
    encoded_data = one_hot_encoder.fit_transform(categorical_data)

    # Combine scaled features, encoded categories, and audio features
    feature_columns = [col for col in df.columns if col.startswith("feature_")]
    scaled_features = wav2vec_feature_scaler.fit_transform(df[feature_columns])
    return np.hstack((scaled_features, encoded_data, df[["Age"]]))

def inverse_scale_target(y_scaled, scaler_path):
    target_scaler = joblib.load(scaler_path)
    return target_scaler.inverse_transform(y_scaled)

def extract_features(audio_file):
    try:
        audio, rate = librosa.load(audio_file, sr=16000)
        inputs = processor(audio, sampling_rate=rate, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states

        level_states = hidden_states[10:13]
        pooled_features = [torch.mean(layer, dim=1).squeeze(0) for layer in level_states]
        final_features = torch.cat(pooled_features, dim=0).numpy()

        return final_features

    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None

def process_person_folder(person_folder):
    person_path = os.path.join(DATASET_PATH, person_folder)
    if not os.path.isdir(person_path) or person_folder == ".DS_Store":
        return []

    participant_info = participant_data[participant_data["Record-ID"] == person_folder]
    if participant_info.empty:
        print(f"No metadata found for {person_folder}. Skipping...")
        return []

    record_id = participant_info["Record-ID"].values[0]
    train_or_dev = participant_info["TrainOrDev"].values[0]
    classification = participant_info["Class"].values[0]
    gender = participant_info["Gender"].values[0]
    age = participant_info["Age"].values[0]
    converted_mmse = participant_info["Converted-MMSE"].values[0]
    all_features = []

    print(f"Processing folder: {person_folder}")
    for file in os.listdir(person_path):
        if file.endswith(".wav"):
            audio_path = os.path.join(person_path, file)
            level_features = extract_features(audio_path)
            if level_features is not None:
                feature_data = {
                    "Record-ID": record_id,
                    "TrainOrDev": train_or_dev,
                    "Class": classification,
                    "Gender": gender,
                    "Age": age,
                    "Converted-MMSE": converted_mmse,
                }
                for i, value in enumerate(level_features):
                    feature_data[f"feature_{i}"] = value
                all_features.append(feature_data)

    print(f"Processed {len(all_features)} files in {person_folder}")
    return all_features

def gather_features():
    all_features = []
    print("Starting feature extraction...")
    for person_folder in os.listdir(DATASET_PATH):
        features = process_person_folder(person_folder)
        all_features.extend(features)
    print(f"Feature extraction completed. Total features extracted: {len(all_features)}.")
    return all_features

if __name__ == "__main__":
# Load features if the file exists
    if not os.path.exists(FEATURES_FILE):
        all_features = gather_features()
        if all_features:
            feature_df = pd.DataFrame(all_features)
            feature_df.to_csv(FEATURES_FILE, index=False)
            print("Feature extraction complete. Results saved in:", FEATURES_FILE)
        else:
            print("No features extracted!")
    else:
        print(f"Features file already exists: {FEATURES_FILE}. Loading features:")
        feature_df = pd.read_csv(FEATURES_FILE)

        print("Data for 5-Fold Cross-Validation:")
        X = scale_features_and_encode(feature_df)
        y = feature_df["Converted-MMSE"].values.reshape(-1, 1)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        mse_values = []
        mae_values = []
        all_preds = []
        all_true = []

        print("5-Fold Cross-Validation:")
        for fold, (train_index, val_index) in enumerate(kf.split(X), start=1):
            print(f"\nFold {fold}:")
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            val_ids = feature_df.iloc[val_index]["Record-ID"].values

            svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
            svr.fit(X_train, y_train.ravel())

            y_pred_scaled = svr.predict(X_val)
            y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
            y_val_original = target_scaler.inverse_transform(y_val)

            y_pred = np.clip(y_pred, 0, MMSE_RANGE)

            for true_value, pred_value, record_id in zip(y_val, y_pred, val_ids):
                results.append({
                    "Record-ID": record_id,
                    "True-MMSE": true_value,
                    "Predicted-MMSE": pred_value
                })

            mae = calculate_mae(y_val_original, y_pred)
            mse = calculate_mse(y_val_original, y_pred)
            std = calculate_std(y_val_original, y_pred)

            output_csv = os.path.join(OUTPUT_PATH, "true_and_predicted_mmse.csv")

            # Write to CSV
            with open(output_csv, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=["Record-ID", "True-MMSE", "Predicted-MMSE"])
                writer.writeheader()
                writer.writerows(results)

            print(f"True and Predicted values saved to {output_csv}")

            mse_values.append(mse)
            mae_values.append(mae)

            print(f"Fold {fold} MSE: {mse:.4f}, MAE: {mae:.4f}")

        plot_metrics(mse_values, mae_values, plot_path=PLOT_PNG)

        print("Training SVR model.")
        final_svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        final_svr.fit(X, y_pred_scaled.ravel())

        model_path = os.path.join(OUTPUT_PATH, "svr_mmse.joblib")
        joblib.dump(final_svr, model_path)

        print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, STD: {std:.4f}")
