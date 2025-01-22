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
from collections import defaultdict

MODEL_NAME = "CAiRE/SER-wav2vec2-large-xlsr-53-eng-zho-all-age"
processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
model = Wav2Vec2Model.from_pretrained(MODEL_NAME, output_hidden_states=True)

MMSE_RANGE = 30

results = []

SCRIPT_DIR = os.path.dirname(__file__)
DATASET_PATH = os.path.join(SCRIPT_DIR, "../PROCESS-V1/PROCESS-V1")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "../Feature_Extraction")
FEATURES_FILE = os.path.join(OUTPUT_PATH, "high_level_features.csv")
CSV_FILE = os.path.join(DATASET_PATH, "dem-info.csv")
PLOT_PNG = os.path.join(OUTPUT_PATH, "plot_high_level_metrics.png")
SCALER_FEATURES = os.path.join(OUTPUT_PATH, "feature_scaler.joblib")
SCALER_TARGET = os.path.join(OUTPUT_PATH, "target_scaler.joblib")

os.makedirs(OUTPUT_PATH, exist_ok=True)

print("Loading participant metadata from CSV:")
participant_data = pd.read_csv(CSV_FILE)
print(f"Loaded {len(participant_data)} participant records from {CSV_FILE}.")


def calculate_std(y_true, y_pred):
    # Step 1: Calculate the mean of the prediction errors
    n = len(y_pred)
    mean_pred_error = sum((y_t - y_p) for y_t, y_p in zip(y_true, y_pred)) / n

    # Step 2: Calculate the squared differences from the mean error
    squared_diffs = [(y_t - y_p - mean_pred_error) ** 2 for y_t, y_p in zip(y_true, y_pred)]

    # Step 3: Calculate the variance (mean of squared differences)
    variance = sum(squared_diffs) / n
    std = variance ** 0.5

    return std

def calculate_mae(y_true, y_pred):
    n = len(y_true)  # Number of samples
    absolute_errors = [abs(y_t - y_p) for y_t, y_p in zip(y_true, y_pred)]
    mae = sum(absolute_errors) / n  # Mean of absolute errors
    return mae

def calculate_mse(y_true, y_pred):
    n = len(y_true)  # Number of samples
    squared_errors = [(y_t - y_p) ** 2 for y_t, y_p in zip(y_true, y_pred)]
    mse = sum(squared_errors) / n  # Mean of squared errors
    return mse

def plot_metrics(mse_values, mae_values, plot_path):
    plt.figure(figsize=(14,7))
    plt.plot(range(1, len(mse_values) + 1), mse_values, label="MSE", marker='o')
    plt.plot(range(1, len(mae_values) + 1), mae_values, label="MAE", marker='x')
    plt.title("MSE and MAE of high level feature extraction")
    plt.xlabel("Fold")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(plot_path, format='png')
    plt.close()

def scale_features_and_encode(df, scaler_dict=None, encoder=None, fit=True):
    if scaler_dict is None:
        scaler_dict = {
            "wav2vec_feature_scaler": StandardScaler(),
            "feature_scaler": StandardScaler(),
            "target_scaler": StandardScaler()
        }
    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    # Scale numerical columns Age and MMSE
    if fit:
        df["Age"] = scaler_dict["feature_scaler"].fit_transform(df[["Age"]])
        df["Converted-MMSE"] = scaler_dict["target_scaler"].fit_transform(df[["Converted-MMSE"]])
    else:
        df["Age"] = scaler_dict["feature_scaler"].transform(df[["Age"]])
        df["Converted-MMSE"] = scaler_dict["target_scaler"].transform(df[["Converted-MMSE"]])

    # Encode categorical columns Gender and Clas
    categorical_data = df[["Gender"]]
    if fit:
        encoded_data = encoder.fit_transform(categorical_data)
    else:
        encoded_data = encoder.transform(categorical_data)

    # Scale wav2vec features
    feature_columns = [col for col in df.columns if col.startswith("feature_")]
    if fit:
        scaled_features = scaler_dict["wav2vec_feature_scaler"].fit_transform(df[feature_columns])
    else:
        scaled_features = scaler_dict["wav2vec_feature_scaler"].transform(df[feature_columns])

    # Combine scaled features, encoded categories, and numerical features
    scaled_numerical = df[["Age"]].values
    combined_features = np.hstack((scaled_features, encoded_data, scaled_numerical))
    return combined_features, scaler_dict, encoder

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

        level_states = hidden_states[1:4] #[10:13] for middle layers and [-3:] for high layers
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

def perform_k_fold_cv(X, y, feature_df, n_splits=5, random_state=42):#
    results = defaultdict(list)
    missing_mmse_path = os.path.join(DATASET_PATH, "dem-info-original.csv")
    original_df = pd.read_csv(missing_mmse_path)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    mse_values = []
    mae_values = []

    print("Starting 5-Fold Cross-Validation...")
    for fold, (train_index, val_index) in enumerate(kf.split(X), start=1):
        print(f"\nFold {fold}:")

        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Initialize scalers and encoder
        scaler_dict = {
            "wav2vec_feature_scaler": StandardScaler(),
            "feature_scaler": StandardScaler(),
            "target_scaler": StandardScaler()
        }
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        # Create temporary DataFrame for scaling and encoding
        temp_train_df = feature_df.iloc[train_index].drop(columns=["Class"]).copy()  # Drop "Class"
        temp_val_df = feature_df.iloc[val_index].drop(columns=["Class"]).copy()

        # Scale and encode
        X_train_scaled, scaler_dict, encoder = scale_features_and_encode(temp_train_df, scaler_dict, encoder, fit=True)
        X_val_scaled, _, _ = scale_features_and_encode(temp_val_df, scaler_dict, encoder, fit=False)

        # Scale target
        y_train_scaled = scaler_dict["target_scaler"].fit_transform(y_train)
        y_val_scaled = scaler_dict["target_scaler"].transform(y_val)

        # Train SVR
        svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        svr.fit(X_train_scaled, y_train_scaled.ravel())

        y_pred_scaled = svr.predict(X_val_scaled)
        y_pred = scaler_dict["target_scaler"].inverse_transform(y_pred_scaled.reshape(-1, 1))
        y_val_original = scaler_dict["target_scaler"].inverse_transform(y_val_scaled)

        # Clip predictions from 0-30
        y_pred = np.clip(y_pred, 0, MMSE_RANGE)

        val_ids = feature_df.iloc[val_index]["Record-ID"].values
        valid_rows = original_df["Record-ID"].isin(val_ids) & original_df["Converted-MMSE"].notna()
        valid_val_ids = original_df.loc[valid_rows, "Record-ID"].values
        valid_true = original_df.loc[valid_rows, "Converted-MMSE"].values

        for record_id, true_val, pred_val in zip(val_ids, y_val_original.ravel(), y_pred.ravel()):
            if record_id in valid_val_ids:
                results[record_id].append({
                    "True-MMSE": true_val,
                    "Predicted-MMSE": pred_val
                })

        # Extract valid predictions and true values
        valid_pred = [y for record_id, y in zip(val_ids, y_pred.ravel()) if
                      record_id in original_df.loc[valid_rows, "Record-ID"].values]

        # Calculate metrics
        mse = calculate_mse(valid_true, valid_pred)
        mae = calculate_mae(valid_true, valid_pred)
        std = calculate_std(valid_true, valid_pred)

        mse_values.append(mse)
        mae_values.append(mae)

        print(f"Fold {fold} MSE: {mse:.4f}, MAE: {mae:.4f}, std: {std:.4f}")

        # Aggregate predictions at the participant level
        participant_results = []
        for record_id, predictions in results.items():
            true_values = [pred["True-MMSE"] for pred in predictions]
            predicted_values = [pred["Predicted-MMSE"] for pred in predictions]

            # Compute mean of predictions and single true value
            participant_mean_prediction = np.mean(predicted_values)
            participant_true_value = np.mean(true_values)  # Assumes true values are consistent per participant

            participant_results.append({
                "Record-ID": record_id,
                "True-MMSE": participant_true_value,
                "Mean-Predicted-MMSE": participant_mean_prediction
            })

    # Save flattened results to CSV
    output_csv = os.path.join(OUTPUT_PATH, "true_and_predicted_mmse.csv")
    with open(output_csv, mode="w", newline="") as csvfile:
        fieldnames = ["Record-ID", "True-MMSE", "Mean-Predicted-MMSE"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(participant_results)

    all_true = [entry["True-MMSE"] for entry in participant_results]
    all_pred = [entry["Mean-Predicted-MMSE"] for entry in participant_results]

    # Plot metrics
    plot_metrics(mse_values, mae_values, plot_path=PLOT_PNG)

    # Return average metrics
    avg_mse = calculate_mse(all_true, all_pred)
    avg_mae = calculate_mae(all_true, all_pred)
    avg_std = calculate_std(all_true, all_pred)
    print(f"\n5-fold CV:\nAverage MSE: {avg_mse:.4f}, Average MAE: {avg_mae:.4f}, Average STD: {avg_std:.4f}")
    return avg_mse, avg_mae

def save_predictions(results, output_csv):
    print(f"Saving predictions to {output_csv}...")
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["Record-ID", "True-MMSE", "Predicted-MMSE"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Predictions saved to {output_csv}")

def train_svr_model(feature_df, feature_columns, numerical_columns, categorical_columns, scaler_dict, encoder):
    print("\nTraining final SVR model on the entire dataset...")

    # Scale and encode dataset
    X_scaled, _, _ = scale_features_and_encode(feature_df.copy(), scaler_dict, encoder, fit=True)
    y = feature_df["Converted-MMSE"].values.reshape(-1, 1)
    y_scaled = scaler_dict["target_scaler"].fit_transform(y)

    # Train SVR
    final_svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    final_svr.fit(X_scaled, y_scaled.ravel())

    # Save the final model
    model_path = os.path.join(OUTPUT_PATH, "svr_mmse.joblib")
    joblib.dump(final_svr, model_path)
    print(f"Final SVR model saved to {model_path}")

    # Optionally, save scalers and encoders
    scaler_path = os.path.join(OUTPUT_PATH, "scaler_dict.joblib")
    encoder_path = os.path.join(OUTPUT_PATH, "one_hot_encoder.joblib")
    joblib.dump(scaler_dict, scaler_path)
    joblib.dump(encoder, encoder_path)
    print(f"Scalers saved to {scaler_path}")
    print(f"Encoder saved to {encoder_path}")

    return final_svr

def main():
    # Step 1: Feature Extraction
    if not os.path.exists(FEATURES_FILE):
        all_features = gather_features()
        if all_features:
            feature_df = pd.DataFrame(all_features)
            feature_df.to_csv(FEATURES_FILE, index=False)
            print("Feature extraction complete. Results saved in:", FEATURES_FILE)
        else:
            print("No features extracted!")
            return
    else:
        print(f"Features file already exists: {FEATURES_FILE}. Loading features:")
        feature_df = pd.read_csv(FEATURES_FILE)

    # Define feature columns
    feature_columns = [col for col in feature_df.columns if col.startswith("feature_")]
    numerical_columns = ["Age"]
    categorical_columns = ["Gender"]

    # Prepare data for Cross-Validation
    X = feature_df[feature_columns + numerical_columns + categorical_columns].values
    y = feature_df["Converted-MMSE"].values.reshape(-1, 1)

    # Perform 5-Fold Cross-Validation
    avg_mse, avg_mae = perform_k_fold_cv(X, y, feature_df, n_splits=5, random_state=42)

    # Train SVR Model on Entire Dataset
    # Initialize scalers and encoder for final training
    scaler_dict = {
        "wav2vec_feature_scaler": StandardScaler(),
        "feature_scaler": StandardScaler(),
        "target_scaler": StandardScaler()
    }
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    final_model = train_svr_model(feature_df.drop(columns=["Class"]), feature_columns, numerical_columns, categorical_columns, scaler_dict, encoder)

if __name__ == "__main__":
    main()
