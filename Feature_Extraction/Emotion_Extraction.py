import os
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

def plot_metrics(accuracies, losses, mae_values, plot_path):
    # Accuracy plot
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o')
    plt.title("Fold-wise Accuracy")
    plt.xlabel("Fold")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)

    # Loss and MAE plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(losses) + 1), losses, label="Loss (MSE)", marker='o')
    plt.plot(range(1, len(mae_values) + 1), mae_values, label="MAE", marker='x')
    plt.title("Loss and MAE over Folds")
    plt.xlabel("Fold")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(plot_path, format='png')
    plt.close()

# Load features if the file exists
if not os.path.exists(FEATURES_FILE):
    print("Features file not found!")
else:
    print(f"Features file found: {FEATURES_FILE}. Loading features.")
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

        print(f"Fold {fold} Accuracy: {accuracy:.2f}%, Loss (MSE): {mse:.4f}, MAE: {mae:.4f}")

    plot_metrics(accuracy_scores, losses, mae_values, plot_path=PLOT_PNG)


    mean_accuracy = np.mean(accuracy_scores)
    std_accuracy = np.std(accuracy_scores)
    print(f"\nFinal Results:")
    print(f"Accuracy: {mean_accuracy:.2f}% (std: {std_accuracy:.2f})")

    print("Training model:")
    final_svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    final_svr.fit(X, y)

    model_path = os.path.join(OUTPUT_PATH, "svr_mmse.joblib")
    joblib.dump(final_svr, model_path)
