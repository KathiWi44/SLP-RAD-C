import os
import librosa
import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import numpy as np
import joblib

MODEL_NAME = "CAiRE/SER-wav2vec2-large-xlsr-53-eng-zho-all-age"
processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
model = Wav2Vec2Model.from_pretrained(MODEL_NAME, output_hidden_states=True)

MMSE_RANGE = 30

SCRIPT_DIR = os.path.dirname(__file__)
DATASET_PATH = os.path.join(SCRIPT_DIR, "../PROCESS-V1/PROCESS-V1")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "../Feature_Extraction")
FEATURES_FILE = os.path.join(OUTPUT_PATH, "high_level_features.csv")
CSV_FILE = os.path.join(DATASET_PATH, "dem-info.csv")
PLOT_PNG = os.path.join(OUTPUT_PATH, "plot_metrics.png")

os.makedirs(OUTPUT_PATH, exist_ok=True)

print("Loading participant metadata from CSV:")
participant_data = pd.read_csv(CSV_FILE)
print(f"Loaded {len(participant_data)} participant records from {CSV_FILE}.")

def plot_metrics(accuracies, losses, mae_values, plot_path):
    # Accuracy plot
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o')
    plt.title("Accuracy")
    plt.xlabel("Fold")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)

    # Loss and MAE plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(losses) + 1), losses, label="MSE", marker='o')
    plt.plot(range(1, len(mae_values) + 1), mae_values, label="MAE", marker='x')
    plt.title("MSE and MAE")
    plt.xlabel("Fold")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(plot_path, format='png')
    plt.close()

def extract_features(audio_file):
    try:
        audio, rate = librosa.load(audio_file, sr=16000)
        inputs = processor(audio, sampling_rate=rate, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states

        high_level_states = hidden_states[10:13]
        pooled_features = [torch.mean(layer, dim=1).squeeze(0) for layer in high_level_states]
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
    converted_mmse = participant_info["Converted-MMSE"].values[0]
    all_features = []

    print(f"Processing folder: {person_folder}")
    for file in os.listdir(person_path):
        if file.endswith(".wav"):
            audio_path = os.path.join(person_path, file)
            high_level_features = extract_features(audio_path)
            if high_level_features is not None:
                feature_data = {
                    "Record-ID": record_id,
                    "Audio-File": file,
                    "Converted-MMSE": converted_mmse,
                }
                for i, value in enumerate(high_level_features):
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
        feature_columns = [col for col in feature_df.columns if col.startswith("feature_")]
        X = feature_df[feature_columns].values
        y = feature_df["Converted-MMSE"].values

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        accuracy_scores = []
        losses = []
        mae_values = []
        all_preds = []
        all_true = []

        print("5-Fold Cross-Validation:")
        for fold, (train_index, val_index) in enumerate(kf.split(X), start=1):
            print(f"\nFold {fold}:")
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
            svr.fit(X_train, y_train)
            y_pred = svr.predict(X_val)
            y_pred = np.clip(y_pred, 0, MMSE_RANGE)

            mae = mean_absolute_error(y_val, y_pred)
            mse = np.mean((y_val - y_pred) ** 2)
            accuracy = (1 - mae / MMSE_RANGE) * 100

            accuracy_scores.append(accuracy)
            losses.append(mse)
            mae_values.append(mae)

            print(f"Fold {fold} Accuracy: {accuracy:.2f}%, MSE: {mse:.4f}, MAE: {mae:.4f}")

        plot_metrics(accuracy_scores, losses, mae_values, plot_path=PLOT_PNG)

        mean_accuracy = np.mean(accuracy_scores)
        std_accuracy = np.std(accuracy_scores)

        print("Training SVR model.")
        final_svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        final_svr.fit(X, y)

        model_path = os.path.join(OUTPUT_PATH, "svr_mmse.joblib")
        joblib.dump(final_svr, model_path)

        print(f"Accuracy: {mean_accuracy:.2f}% (std: {std_accuracy:.2f})")
