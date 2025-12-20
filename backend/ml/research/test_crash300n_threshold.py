"""
Testa diferentes thresholds de decisao para balancear Precision/Recall
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class CRASH300Dataset(Dataset):
    def __init__(self, df, lookback=50, feature_cols=None, target_col='crashed_in_next_10'):
        self.lookback = lookback
        self.feature_cols = feature_cols or ['open', 'high', 'low', 'close', 'rolling_volatility']
        self.target_col = target_col

        df_features = df[self.feature_cols + [self.target_col]].copy()
        df_features = df_features.dropna()

        self.data = df_features[self.feature_cols].values.astype(np.float32)
        self.labels = df_features[self.target_col].values.astype(np.int64)

    def __len__(self):
        return len(self.data) - self.lookback

    def __getitem__(self, idx):
        X = self.data[idx:idx+self.lookback]
        window_min = X.min(axis=0)
        window_max = X.max(axis=0)
        window_range = window_max - window_min + 1e-8
        X_norm = (X - window_min) / window_range
        y = self.labels[idx + self.lookback - 1]
        return torch.FloatTensor(X_norm), torch.LongTensor([y])


class LSTMBinaryClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64, dropout=0.3):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim2, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


print("="*70)
print("CRASH300N - THRESHOLD SEARCH")
print("="*70)

device = torch.device('cpu')

# Load test set
data_dir = Path(__file__).parent / "data"
df = pd.read_csv(data_dir / "CRASH300_M1_survival_labeled.csv")
train_size = int(0.70 * len(df))
val_size = int(0.15 * len(df))
df_test = df.iloc[train_size+val_size:].copy()

feature_cols = ['open', 'high', 'low', 'close', 'rolling_volatility',
                'ticks_since_crash', 'crash_size_lag1', 'tick_velocity',
                'acceleration', 'price_deviation', 'momentum']

test_dataset = CRASH300Dataset(df_test, lookback=50, feature_cols=feature_cols)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load model
model_path = Path(__file__).parent / "models" / "crash300n_survival_lstm.pth"
model = LSTMBinaryClassifier(input_dim=len(feature_cols)).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Get probabilities
test_probs = []
test_labels = []

with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device).squeeze()
        outputs = model(X)
        probs = torch.softmax(outputs, dim=1)
        test_probs.extend(probs[:, 1].cpu().numpy())  # P(CRASH)
        test_labels.extend(y.cpu().numpy())

test_probs = np.array(test_probs)
test_labels = np.array(test_labels)

print(f"\n[DATA] Test samples: {len(test_labels):,}")
print(f"\n[PROBABILITIES]")
print(f"  Min P(CRASH): {test_probs.min():.4f}")
print(f"  Max P(CRASH): {test_probs.max():.4f}")
print(f"  Mean P(CRASH): {test_probs.mean():.4f}")
print(f"  Median P(CRASH): {np.median(test_probs):.4f}")

# Test different thresholds
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

print(f"\n{'='*70}")
print("THRESHOLD SEARCH")
print(f"{'='*70}")
print(f"\n{'Threshold':<12} | {'Accuracy':<10} | {'Precision':<12} | {'Recall':<10} | {'F1-Score':<10}")
print(f"{'-'*12}-+-{'-'*10}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}")

best_f1 = 0
best_threshold = 0.5

for threshold in thresholds:
    # Predictions with threshold
    preds = (test_probs >= threshold).astype(int)

    acc = accuracy_score(test_labels, preds)
    prec = precision_score(test_labels, preds, zero_division=0)
    rec = recall_score(test_labels, preds, zero_division=0)
    f1 = f1_score(test_labels, preds, zero_division=0)

    print(f"{threshold:<12.2f} | {acc*100:<10.2f} | {prec*100:<12.2f} | {rec*100:<10.2f} | {f1*100:<10.2f}")

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"\n{'='*70}")
print(f"BEST THRESHOLD: {best_threshold:.2f} (F1={best_f1*100:.2f}%)")
print(f"{'='*70}\n")
