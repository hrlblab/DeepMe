import numpy as np
import pandas as pd
import sklearn.metrics
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import TimeSeriesBERT
import joblib
import matplotlib.pyplot as plt
import os
from datetime import datetime
from config import *
from sklearn.metrics import r2_score



torch.manual_seed(SEED)
np.random.seed(SEED)


class TimeSeriesDataset(Dataset):
    """dataset class"""

    def __init__(self, sequences, targets):

        self.sequences = sequences.astype(np.float32)
        self.targets = targets.astype(np.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):

        sequence = torch.as_tensor(self.sequences[idx])
        target = torch.as_tensor(self.targets[idx])
        return sequence, target


def create_sequences_with_zeros(features, targets, seq_length=168):


    sequences = []
    labels = []
    time_indices = []


    for i in range(seq_length, len(features)):

        seq = features[i - seq_length:i]

        label = targets[i]

        sequences.append(seq)
        labels.append(label)
        time_indices.append(i)

    return np.array(sequences), np.array(labels), np.array(time_indices)


def load_and_preprocess_test_data(file_path, feature_cols, train_stats, target_col, id_col):

    print(f"Loading test data: {file_path}")
    df = pd.read_csv(file_path)


    df = df.sort_values(id_col).reset_index(drop=True)


    features = df[feature_cols].copy()
    target = df[target_col].copy()


    print("Standardizing test data using training statistics...")


    standardized_features = pd.DataFrame(index=features.index, columns=features.columns)


    for col in feature_cols:
        if col not in train_stats:
            print(f"Warning: Column {col} not found in training statistics. Using default values.")
            mean_val = 0.0
            std_val = 1.0
        else:
            mean_val = train_stats[col]['mean']
            std_val = train_stats[col]['std']


        col_data = features[col].copy()


        non_zero_mask = (col_data != 0)
        non_zero_values = col_data[non_zero_mask]

        if len(non_zero_values) > 0:

            standardized_values = (non_zero_values - mean_val) / std_val


            standardized_features.loc[non_zero_mask, col] = standardized_values


            standardized_features.loc[~non_zero_mask, col] = 0.0
        else:
            standardized_features[col] = 0.0


    print("Creating test sequences (including zero labels)...")
    sequences, labels, time_indices = create_sequences_with_zeros(
        standardized_features.values,
        target.values,
        seq_length=SEQ_LENGTH
    )


    sequences = sequences.astype(np.float32)
    labels = labels.astype(np.float32)

    print(f"Total test samples: {len(sequences)}")
    print(f"Sequence shape: {sequences.shape}")
    print(f"Sequence dtype: {sequences.dtype}, Labels dtype: {labels.dtype}")


    return sequences, labels, time_indices, df[id_col].values, standardized_features


def evaluate_model(model, test_loader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    model.eval()


    all_predictions = []
    all_targets = []
    all_indices = []

    with torch.no_grad():
        for batch_idx, (sequences, targets) in enumerate(test_loader):
            sequences = sequences.to(device)
            outputs = model(sequences).cpu().numpy()


            all_predictions.extend(outputs)
            all_targets.extend(targets.numpy())


            batch_indices = [batch_idx * test_loader.batch_size + i for i in range(len(outputs))]
            all_indices.extend(batch_indices)

    return np.array(all_predictions), np.array(all_targets), np.array(all_indices)


def calculate_metrics(predictions, targets):

    non_zero_mask = (targets != 0)
    valid_targets = targets[non_zero_mask]
    valid_predictions = predictions[non_zero_mask]

    if len(valid_targets) == 0:
        print("Warning: No non-zero targets found for metric calculation!")
        return float('nan'), float('nan'), float('nan'), 0


    mse = np.mean((valid_predictions - valid_targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(valid_predictions - valid_targets))
    r2 = r2_score(valid_targets,valid_predictions)
    r = np.corrcoef(valid_targets,valid_predictions)[0, 1]

    return mse, rmse, mae, len(valid_targets), r2, r


def visualize_results(predictions, targets, time_indices, time_ids, non_zero_mask):

    os.makedirs('eval_results', exist_ok=True)


    valid_time_indices = time_indices[non_zero_mask]
    valid_time_ids = time_ids[valid_time_indices]
    valid_targets = targets[non_zero_mask]
    valid_predictions = predictions[non_zero_mask]


    sorted_indices = np.argsort(valid_time_ids)
    sorted_time_ids = valid_time_ids[sorted_indices]
    sorted_targets = valid_targets[sorted_indices]
    sorted_predictions = valid_predictions[sorted_indices]


    plt.figure(figsize=(14, 7))
    plt.plot(sorted_time_ids, sorted_targets, label='Actual', alpha=0.7, marker='o', markersize=3)
    plt.plot(sorted_time_ids, sorted_predictions, label='Predicted', alpha=0.7, marker='x', markersize=3)
    plt.title('Actual vs Predicted Values (Non-Zero Targets)')
    plt.xlabel('Time ID')
    plt.ylabel(TARGET_COL)
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('eval_results/test_predictions.png', dpi=300)
    plt.close()


    residuals = sorted_targets - sorted_predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(sorted_predictions, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title('Residuals vs Predicted Values (Non-Zero Targets)')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.grid(True)
    plt.savefig('eval_results/test_residuals.png', dpi=300)
    plt.close()


    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, alpha=0.7)
    plt.title('Error Distribution (Non-Zero Targets)')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig('eval_results/error_distribution.png', dpi=300)
    plt.close()


def save_results(predictions, targets, time_indices, time_ids, mse, rmse, mae, num_valid_samples):

    os.makedirs('eval_results', exist_ok=True)


    non_zero_mask = (targets != 0)
    valid_targets = targets[non_zero_mask]
    valid_predictions = predictions[non_zero_mask]
    valid_time_indices = time_indices[non_zero_mask]
    valid_time_ids = time_ids[valid_time_indices]

    results = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "num_total_samples": len(predictions),
        "num_valid_samples": num_valid_samples,
        "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }




    results_df = pd.DataFrame({
        'TimeIndex': valid_time_indices,
        'TimeID': valid_time_ids,
        'Actual': valid_targets,
        'Predicted': valid_predictions,
        'Error': valid_targets - valid_predictions,
        'AbsoluteError': np.abs(valid_targets - valid_predictions)
    })
    results_df.to_csv('eval_results/detailed_predictions.csv', index=False)


    full_results_df = pd.DataFrame({
        'TimeIndex': time_indices,
        'TimeID': time_ids[time_indices],
        'Actual': targets,
        'Predicted': predictions,
        'IsValidTarget': (targets != 0).astype(int)
    })
    full_results_df.to_csv('eval_results/full_predictions.csv', index=False)


    with open('eval_results/summary_report.txt', 'w') as f:
        f.write("Model Evaluation Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Test File: {TEST_FILE}\n")
        f.write(f"Model Used: {MODEL_PATH}\n")
        f.write(f"Evaluation Date: {results['evaluation_date']}\n\n")

        f.write("Performance Metrics (Non-Zero Targets Only):\n")
        f.write(f"  MSE:  {mse:.6f}\n")
        f.write(f"  RMSE: {rmse:.6f}\n")
        f.write(f"  MAE:  {mae:.6f}\n")
        f.write(f"  Valid samples evaluated: {num_valid_samples}\n")
        f.write(f"  Total samples: {len(predictions)}\n\n")


        abs_errors = np.abs(results_df['Error'])
        f.write("Error Analysis (Non-Zero Targets):\n")
        f.write(f"  Min Error:    {results_df['Error'].min():.4f}\n")
        f.write(f"  Max Error:    {results_df['Error'].max():.4f}\n")
        f.write(f"  Mean Error:   {results_df['Error'].mean():.4f}\n")
        f.write(f"  Std Error:    {results_df['Error'].std():.4f}\n")
        f.write(f"  Median Error: {results_df['Error'].median():.4f}\n")
        f.write(f"  MAE:          {abs_errors.mean():.4f}\n")
        f.write(f"  RMSE:         {np.sqrt((results_df['Error'] ** 2).mean()):.4f}\n")


        if num_valid_samples > 0:
            worst_predictions = results_df.nlargest(5, 'AbsoluteError')
            f.write("\nTop 5 Worst Predictions (Non-Zero Targets):\n")
            for i, row in worst_predictions.iterrows():
                f.write(
                    f"  TimeID {row['TimeID']}: Actual={row['Actual']:.4f}, Predicted={row['Predicted']:.4f}, Error={row['Error']:.4f}\n")


def main():
    print("\n" + "=" * 50)
    print("Starting Model Evaluation")
    print("=" * 50)
    print(f"Test configuration:")
    print(f"  Data file: {TEST_FILE}")
    print(f"  Target column: {TARGET_COL}")
    print(f"  ID column: {ID_COL}")
    print(f"  Sequence length: {SEQ_LENGTH} hours (7 days)")
    print(f"  Model: {MODEL_PATH}")


    feature_cols = joblib.load(FEATURE_COLS_PATH)
    train_stats = joblib.load(TRAIN_STATS_PATH)
    print(f"Loaded {len(feature_cols)} feature columns and training statistics")


    model = TimeSeriesBERT(
        num_features=len(feature_cols),
        seq_length=SEQ_LENGTH,
        hidden_size=192,
        num_layers=6
    )
    model.load_state_dict(torch.load(MODEL_PATH))
    print(f"Model loaded from {MODEL_PATH}")


    sequences, labels, time_indices, time_ids, _ = load_and_preprocess_test_data(
        file_path=TEST_FILE,
        feature_cols=feature_cols,
        train_stats=train_stats,
        target_col=TARGET_COL,
        id_col=ID_COL
    )


    test_dataset = TimeSeriesDataset(sequences, labels)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE * 4,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )


    print("Running inference on all data points...")
    predictions, targets, batch_indices = evaluate_model(model, test_loader)


    non_zero_mask = (targets != 0)
    print(f"Total predictions: {len(predictions)}")
    print(f"Non-zero targets: {np.sum(non_zero_mask)}")

    mse, rmse, mae, num_valid_samples, r2, r = calculate_metrics(predictions, targets)

    print("\n" + "=" * 50)
    print(f"Test Results (Non-Zero Targets Only):")
    print(f"  MSE:  {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  R2:  {r2:.6f}")
    print(f"  R:  {r:.6f}")
    print(f"  Valid samples evaluated: {num_valid_samples}")
    print("=" * 50)


    all_time_indices = time_indices[batch_indices]


    visualize_results(predictions, targets, all_time_indices, time_ids, non_zero_mask)
    save_results(predictions, targets, all_time_indices, time_ids, mse, rmse, mae, num_valid_samples)

    print("\nEvaluation completed!")
    print(f"Results saved to 'eval_results' directory")
    print(f"Final Metrics (Non-Zero Targets): MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}, R={r:.4f}")


if __name__ == "__main__":
    main()