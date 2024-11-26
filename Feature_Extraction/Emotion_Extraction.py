import os
import librosa
import pandas as pd
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

MODEL_NAME = "superb/wav2vec2-large-superb-er"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME)

SCRIPT_DIR = os.path.dirname(__file__)

DATASET_PATH = os.path.join(SCRIPT_DIR, "../PROCESS-V1/PROCESS-V1")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "../Feature_Extraction")
CSV_FILE = os.path.join(DATASET_PATH, "dem-info.csv")

os.makedirs(OUTPUT_PATH, exist_ok=True)

print("Loading participant metadata from CSV...")
participant_data = pd.read_csv(CSV_FILE)
print(f"Loaded {len(participant_data)} participant records from {CSV_FILE}.")


# Function to extract emotion features for a given file
def extract_emotion_features(audio_file):
    try:
        audio, rate = librosa.load(audio_file, sr=16000)

        inputs = feature_extractor(audio, sampling_rate=rate, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits

        # Get predicted emotion and raw scores
        predicted_emotion = torch.argmax(logits, dim=-1).item()
        emotion_scores = logits.squeeze().tolist()

        return predicted_emotion, emotion_scores
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None, None


# Process person folder
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

            predicted_emotion, emotion_scores = extract_emotion_features(audio_path)

            if predicted_emotion is not None:
                feature_data = {
                    "Record-ID": record_id,
                    "Audio-File": file,
                    "TrainOrDev": train_or_dev,
                    "Class": cls,
                    "Gender": gender,
                    "Age": age,
                    "Converted-MMSE": converted_mmse,
                    "Predicted-Emotion": predicted_emotion,
                }

                for i, score in enumerate(emotion_scores):
                    feature_data[f"emotion_score_{i}"] = score

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
    all_features = gather_features()

    if all_features:
        print("Saving extracted features to CSV...")
        feature_df = pd.DataFrame(all_features)
        feature_df.to_csv(os.path.join(OUTPUT_PATH, "emotion_features.csv"), index=False)
        print("Feature extraction complete. Results saved in:", OUTPUT_PATH)
    else:
        print("No features extracted. Check dataset structure and file paths.")