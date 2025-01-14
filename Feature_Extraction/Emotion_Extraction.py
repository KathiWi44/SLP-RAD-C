import os
import librosa
import pandas as pd
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import joblib

MODEL_NAME = "CAiRE/SER-wav2vec2-large-xlsr-53-eng-zho-all-age"
processor =  Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
model = Wav2Vec2Model.from_pretrained(MODEL_NAME, output_hidden_states=True)

cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")
if os.path.exists(cache_dir):
    print("Clearing Hugging Face cache...")
    for file in os.listdir(cache_dir):
        file_path = os.path.join(cache_dir, file)
        try:
            if os.path.isdir(file_path):
                os.rmdir(file_path)  # Remove directories recursively
            else:
                os.remove(file_path)  # Remove files
        except Exception as e:
            print(f"Error clearing cache: {e}")

SCRIPT_DIR = os.path.dirname(__file__)
DATASET_PATH = os.path.join(SCRIPT_DIR, "../PROCESS-V1/PROCESS-V1")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "../Feature_Extraction")
FEATURES_FILE = os.path.join(OUTPUT_PATH, "high_level_features.csv")
CSV_FILE = os.path.join(DATASET_PATH, "dem-info.csv")

os.makedirs(OUTPUT_PATH, exist_ok=True)

print("Loading participant metadata from CSV...")
participant_data = pd.read_csv(CSV_FILE)
print(f"Loaded {len(participant_data)} participant records from {CSV_FILE}.")

# Function to extract features from high-level layers of Wav2Vec2
def extract_high_level_features(audio_file):
    try:
        # Load audio file
        audio, rate = librosa.load(audio_file, sr=16000)
        inputs = processor(audio, sampling_rate=rate, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states  # List of hidden states

        # Extract features from high-level layers (e.g., last 3 layers)
        high_level_states = hidden_states[-3:]  # Layers 22, 23, 24

        # Mean pooling across the sequence dimension for fixed-size representation
        pooled_features = [torch.mean(layer, dim=1).squeeze(0) for layer in high_level_states]

        # Concatenate pooled features from all selected layers
        final_features = torch.cat(pooled_features, dim=0).numpy()

        return final_features

    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None

# Process person folder sequentially (one folder at a time)
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
    cls = participant_info["Class"].values[0]
    gender = participant_info["Gender"].values[0]
    age = participant_info["Age"].values[0]
    converted_mmse = participant_info["Converted-MMSE"].values[0]

    all_features = []

    print(f"Processing folder: {person_folder}")
    for file in os.listdir(person_path):
        if file.endswith(".wav"):
            audio_path = os.path.join(person_path, file)

            high_level_features = extract_high_level_features(audio_path)

            if high_level_features is not None:
                feature_data = {
                    "Record-ID": record_id,
                    "Audio-File": file,
                    "TrainOrDev": train_or_dev,
                    "Class": cls,
                    "Gender": gender,
                    "Age": age,
                    "Converted-MMSE": converted_mmse,
                }

                # Add feature dimensions to the feature_data dictionary
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
    if not os.path.exists(FEATURES_FILE):
        # Step 1: Extract features sequentially
        all_features = gather_features()

        if all_features:
            print("Saving extracted features to CSV...")
            feature_df = pd.DataFrame(all_features)
            feature_df.to_csv(FEATURES_FILE, index=False)
            print("Feature extraction complete. Results saved in:", FEATURES_FILE)
        else:
            print("No features extracted. Check dataset structure and file paths.")
    else:
        print(f"Features file already exists: {FEATURES_FILE}. Loading features...")
        feature_df = pd.read_csv(FEATURES_FILE)

    # Step 2: Train SVR model
    print("Preparing data for SVR model...")
    feature_columns = [col for col in feature_df.columns if col.startswith("feature_")]
    X_train = feature_df[feature_df["TrainOrDev"] == "train"][feature_columns].values
    y_train = feature_df[feature_df["TrainOrDev"] == "train"]["Converted-MMSE"].values

    X_dev = feature_df[feature_df["TrainOrDev"] == "dev"][feature_columns].values
    y_dev = feature_df[feature_df["TrainOrDev"] == "dev"]["Converted-MMSE"].values

    print("Training SVR model...")
    svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    svr.fit(X_train, y_train)

    print("Evaluating SVR model...")
    y_pred = svr.predict(X_dev)

    # Clip predictions to range [0, 30]
    y_pred = np.clip(y_pred, 0, 30)

    mse = mean_squared_error(y_dev, y_pred)
    mae = mean_absolute_error(y_dev, y_pred)
    print(f"Mean Squared Error on dev set: {mse}")
    print(f"Mean Absolute Error on dev set: {mae}")

    # Save the trained model
    model_path = os.path.join(OUTPUT_PATH, "svr_model.joblib")
    joblib.dump(svr, model_path)
    print(f"SVR model saved to {model_path}")
