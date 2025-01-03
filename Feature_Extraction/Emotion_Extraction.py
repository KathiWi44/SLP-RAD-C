import os
import librosa
import pandas as pd
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define SCRIPT_DIR for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize Hugging Face model and feature extractor
MODEL_NAME = "CAiRE/SER-wav2vec2-large-xlsr-53-eng-zho-all-age"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME)

# Paths to dataset and output
DATASET_PATH = os.path.join(SCRIPT_DIR, "../PROCESS-V1/PROCESS-V1")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "../Feature_Extraction")
CSV_FILE = os.path.join(DATASET_PATH, "dem-info.csv")
FEATURES_FILE = os.path.join(OUTPUT_PATH, "extracted_features.csv")

# Ensure the output directory exists
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Load participant metadata
participant_data = pd.read_csv(CSV_FILE)

# Function to extract emotion features
def extract_emotion_features(audio_file):
    audio, rate = librosa.load(audio_file, sr=16000)
    inputs = feature_extractor(audio, sampling_rate=rate, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    return logits.squeeze().tolist()

# Function to clean data
def clean_data(df):
    # Ensure numeric data is properly converted
    for col in ["Age", "Converted-MMSE"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Drop rows with missing or invalid data
    return df.dropna()

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
        cls = participant_info["Class"].values[0]  # Target variable
        gender = participant_info["Gender"].values[0]
        age = participant_info["Age"].values[0]
        converted_mmse = participant_info["Converted-MMSE"].values[0]
        for file in os.listdir(person_path):
            if file.endswith(".wav"):
                audio_path = os.path.join(person_path, file)
                logging.info(f"Processing audio file: {audio_path}")
                try:
                    emotion_scores = extract_emotion_features(audio_path)
                    feature_data = {
                        "Record-ID": record_id,
                        "TrainOrDev": train_or_dev,
                        "Class": cls,
                        "Gender": gender,
                        "Age": age,
                        "Converted-MMSE": converted_mmse,
                    }
                    for i, score in enumerate(emotion_scores):
                        feature_data[f"emotion_score_{i}"] = score
                    all_features.append(feature_data)
                except Exception as e:
                    logging.error(f"Error processing {audio_path}: {e}")
    features_df = pd.DataFrame(all_features)
    features_df.to_csv(FEATURES_FILE, index=False)
    logging.info("Feature extraction complete. Saved to extracted_features.csv")
else:
    logging.info("Features file found. Loading data...")
    features_df = pd.read_csv(FEATURES_FILE)

# Clean the data
features_df = clean_data(features_df)

# Log the number of samples after cleaning
logging.info(f"Number of samples after cleaning: {len(features_df)}")

# Check if there are enough samples
if len(features_df) == 0:
    logging.error("No valid samples available after cleaning. Please check the dataset.")
    exit()

# Check if the target column exists
if "Converted-MMSE" not in features_df.columns:
    logging.error("'Converted-MMSE' column not found in features file. Ensure the dataset is complete.")
    exit()

# Split features and labels
X = features_df.drop(columns=["Converted-MMSE", "Class", "Record-ID", "TrainOrDev"])
y = features_df["Converted-MMSE"]

# Log feature and target info
logging.info(f"Number of features: {X.shape[1]}, Number of target samples: {y.shape[0]}")

# One-hot encode categorical variables
if "Gender" in X.columns:
    X = pd.get_dummies(X, columns=["Gender"], drop_first=True)

# Train/test split
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
except ValueError as e:
    logging.error(f"Error during train-test split: {e}")
    exit()

# SVM Regression Pipeline
scaler = StandardScaler()
svr_pipeline = Pipeline([
    ("scaler", scaler),
    ("svr", SVR(kernel="linear"))
])

logging.info("Training SVR model...")
svr_pipeline.fit(X_train, y_train)

# Evaluate regression model
y_pred = svr_pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

logging.info("Evaluation Metrics:")
logging.info(f"Mean Squared Error: {mse:.2f}")
logging.info(f"Mean Absolute Error: {mae:.2f}")

# Save predictions
results = pd.DataFrame({"True_MMSE": y_test, "Predicted_MMSE": y_pred})
results.to_csv(os.path.join(OUTPUT_PATH, "svr_predictions.csv"), index=False)
logging.info("Predictions saved to svr_predictions.csv")
