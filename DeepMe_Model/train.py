import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import TimeSeriesBERT
import matplotlib.pyplot as plt
import os
import joblib
from config import *
from sklearn.model_selection import KFold

torch.manual_seed(SEED)
np.random.seed(SEED)


class TimeSeriesDataset(Dataset):
    """Time series dataset class"""

    def __init__(self, sequences, targets):

        self.sequences = sequences.astype(np.float32)
        self.targets = targets.astype(np.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):

        sequence = torch.as_tensor(self.sequences[idx])
        target = torch.as_tensor(self.targets[idx])
        return sequence, target


def load_and_preprocess_data(file_path):

    print(f"Loading data: {file_path}")
    df = pd.read_csv(file_path)


    feature_cols = INPUT_COLS
    print(f"Features: {len(feature_cols)} columns")


    df = df.sort_values(ID_COL).reset_index(drop=True)


    features = df[feature_cols].copy()
    target = df[TARGET_COL].copy()


    print("Standardizing...")


    standardized_features = pd.DataFrame(index=features.index, columns=features.columns)


    train_stats = {}


    for col in feature_cols:

        col_data = features[col].copy()


        non_zero_mask = (col_data != 0)
        non_zero_values = col_data[non_zero_mask]

        if len(non_zero_values) > 0:

            mean_val = non_zero_values.mean()
            std_val = non_zero_values.std()


            if std_val < 1e-6:
                std_val = 1.0


            train_stats[col] = {
                'mean': mean_val,
                'std': std_val,
                'non_zero_count': len(non_zero_values)
            }


            standardized_values = (non_zero_values - mean_val) / std_val


            standardized_features.loc[non_zero_mask, col] = standardized_values


            standardized_features.loc[~non_zero_mask, col] = 0.0
        else:

            standardized_features[col] = 0.0
            train_stats[col] = {
                'mean': 0.0,
                'std': 1.0,
                'non_zero_count': 0
            }


    print("Creating training sequences...")
    sequences, labels = TimeSeriesBERT.create_sequences(
        standardized_features.values,
        target.values,
        seq_length=SEQ_LENGTH
    )


    sequences = sequences.astype(np.float32)
    labels = labels.astype(np.float32)

    print(f"Total samples: {len(sequences)}")
    print(f"Sequence shape: {sequences.shape}")
    print(f"Sequence dtype: {sequences.dtype}, Labels dtype: {labels.dtype}")

    return sequences, labels, feature_cols, train_stats


def train_model(model, train_loader, val_loader, epochs, patience):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    criterion = nn.MSELoss(reduction='none')
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5, factor=0.5
    )

    best_val_loss = float('inf')
    early_stop_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):

        model.train()
        train_loss = 0.0
        total_train_samples = 0

        for sequences, targets in train_loader:
            sequences, targets = sequences.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(sequences)


            non_zero_mask = (targets != 0).float()
            num_non_zero = non_zero_mask.sum().item()

            if num_non_zero > 0:

                losses = criterion(outputs, targets)
                masked_losses = losses * non_zero_mask
                loss = masked_losses.sum() / num_non_zero

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item() * num_non_zero
                total_train_samples += num_non_zero


        if total_train_samples > 0:
            avg_train_loss = train_loss / total_train_samples
        else:
            avg_train_loss = float('nan')
        history['train_loss'].append(avg_train_loss)


        model.eval()
        val_loss = 0.0
        total_val_samples = 0

        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences, targets = sequences.to(device), targets.to(device)
                outputs = model(sequences)


                non_zero_mask = (targets != 0).float()
                num_non_zero = non_zero_mask.sum().item()

                if num_non_zero > 0:

                    losses = criterion(outputs, targets)
                    masked_losses = losses * non_zero_mask
                    loss = masked_losses.sum() / num_non_zero

                    val_loss += loss.item() * num_non_zero
                    total_val_samples += num_non_zero


        if total_val_samples > 0:
            avg_val_loss = val_loss / total_val_samples
        else:
            avg_val_loss = float('nan')
        history['val_loss'].append(avg_val_loss)


        if not np.isnan(avg_val_loss):
            scheduler.step(avg_val_loss)


        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {avg_val_loss:.6f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")


        if not np.isnan(avg_val_loss) and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0

            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved best model (Val Loss: {best_val_loss:.6f})")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered! Best val loss: {best_val_loss:.6f}")
                break


    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_loss.png')
    plt.close()

    print("Training completed!")
    return history


def main():

    print(f"Training configuration:")
    print(f"  Data file: {DATA_FILE}")
    print(f"  Target column: {TARGET_COL}")
    print(f"  ID column: {ID_COL}")
    print(f"  Sequence length: {SEQ_LENGTH} hours (7 days)")


    sequences, labels, feature_cols, train_stats = load_and_preprocess_data(DATA_FILE)

    joblib.dump(feature_cols, 'feature_columns.pkl')
    joblib.dump(train_stats, 'train_stats.pkl')
    print("Saved feature columns and standardization statistics")


    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_id = 1

    all_fold_val_losses = []

    for train_index, val_index in kf.split(sequences):
        print("\n" + "="*60)
        print(f"Starting Fold {fold_id}/5")
        print("="*60)

        X_train, X_val = sequences[train_index], sequences[val_index]
        y_train, y_val = labels[train_index], labels[val_index]

        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )


        num_features = len(feature_cols)
        model = TimeSeriesBERT(
            num_features=num_features,
            seq_length=SEQ_LENGTH,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS
        )

        print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


        history = train_model(
            model,
            train_loader,
            val_loader,
            epochs=EPOCHS,
            patience=PATIENCE
        )


        fold_best_loss = min(history["val_loss"])
        all_fold_val_losses.append(fold_best_loss)


        fold_model_path = f"best_model_fold{fold_id}.pth"
        torch.save(model.state_dict(), fold_model_path)
        print(f"[Fold {fold_id}] Best Val Loss = {fold_best_loss:.6f}, Saved to {fold_model_path}")

        fold_id += 1


    print("\n" + "="*60)
    print("5-Fold Cross-Validation completed!")
    print("Fold Validation Losses:")
    for i, loss in enumerate(all_fold_val_losses, 1):
        print(f"  Fold {i}: {loss:.6f}")

    print(f"\nAverage Val Loss: {np.mean(all_fold_val_losses):.6f}")
    print("="*60)


if __name__ == "__main__":
    main()