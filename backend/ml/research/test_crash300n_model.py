"""
Testa o modelo CRASH300N salvo no test set
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class CRASH300Dataset(Dataset):
    def __init__(self, df, lookback=50, feature_cols=None, target_col='crashed_in_next_10'):
        self.lookback = lookback
        self.feature_cols = feature_cols or ['open', 'high', 'low', 'close', 'rolling_volatility']
        self.target_col = target_col

        # Selecionar apenas features + target
        df_features = df[self.feature_cols + [self.target_col]].copy()
        df_features = df_features.dropna()

        self.data = df_features[self.feature_cols].values.astype(np.float32)
        self.labels = df_features[self.target_col].values.astype(np.int64)

    def __len__(self):
        return len(self.data) - self.lookback

    def __getitem__(self, idx):
        X = self.data[idx:idx+self.lookback]

        # Normalizar por janela
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
print("CRASH300N - TEST MODEL")
print("="*70)

device = torch.device('cpu')

# Load dataset
data_dir = Path(__file__).parent / "data"
df = pd.read_csv(data_dir / "CRASH300_M1_survival_labeled.csv")

# Split (usar mesmo split do treino: 70/15/15)
train_size = int(0.70 * len(df))
val_size = int(0.15 * len(df))
df_test = df.iloc[train_size+val_size:].copy()

print(f"\n[DATA] Test set: {len(df_test):,} candles")

# Features
feature_cols = [
    'open', 'high', 'low', 'close', 'rolling_volatility',
    'ticks_since_crash', 'crash_size_lag1', 'tick_velocity',
    'acceleration', 'price_deviation', 'momentum'
]

# Dataset
test_dataset = CRASH300Dataset(df_test, lookback=50, feature_cols=feature_cols)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"  Test samples: {len(test_dataset):,}")
print(f"  Test batches: {len(test_loader)}")

# Load model
model_path = Path(__file__).parent / "models" / "crash300n_survival_lstm.pth"
model = LSTMBinaryClassifier(input_dim=len(feature_cols), hidden_dim1=128, hidden_dim2=64, dropout=0.3).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print(f"\n[MODEL] Loaded from: {model_path.name}")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Evaluate
test_preds = []
test_labels = []
test_probs = []

with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device).squeeze()
        outputs = model(X)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        test_preds.extend(predicted.cpu().numpy())
        test_labels.extend(y.cpu().numpy())
        test_probs.extend(probs[:, 1].cpu().numpy())  # P(CRASH)

# Métricas
test_acc = accuracy_score(test_labels, test_preds)
test_precision = precision_score(test_labels, test_preds, zero_division=0)
test_recall = recall_score(test_labels, test_preds, zero_division=0)
test_f1 = f1_score(test_labels, test_preds, zero_division=0)

print(f"\n{'='*70}")
print("TEST RESULTS")
print(f"{'='*70}")

print(f"\n[METRICS]")
print(f"  Accuracy:  {test_acc*100:.2f}%")
print(f"  Precision: {test_precision*100:.2f}%")
print(f"  Recall:    {test_recall*100:.2f}%")
print(f"  F1-Score:  {test_f1*100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(test_labels, test_preds)
print(f"\n[CONFUSION MATRIX]")
print(f"                  Predicted SAFE | Predicted CRASH")
print(f"  Actual SAFE:    {cm[0,0]:>14} | {cm[0,1]:>15}")
print(f"  Actual CRASH:   {cm[1,0]:>14} | {cm[1,1]:>15}")

# Class distribution
n_safe_pred = sum(1 for p in test_preds if p == 0)
n_crash_pred = sum(1 for p in test_preds if p == 1)
n_safe_actual = sum(1 for l in test_labels if l == 0)
n_crash_actual = sum(1 for l in test_labels if l == 1)

print(f"\n[PREDICTIONS]")
print(f"  SAFE:  {n_safe_pred:,} ({n_safe_pred/len(test_preds)*100:.1f}%)")
print(f"  CRASH: {n_crash_pred:,} ({n_crash_pred/len(test_preds)*100:.1f}%)")

print(f"\n[ACTUAL LABELS]")
print(f"  SAFE:  {n_safe_actual:,} ({n_safe_actual/len(test_labels)*100:.1f}%)")
print(f"  CRASH: {n_crash_actual:,} ({n_crash_actual/len(test_labels)*100:.1f}%)")

# Diagnóstico
print(f"\n{'='*70}")
print("DIAGNOSTICO")
print(f"{'='*70}")

if test_recall == 0:
    print("\nMODELO COLAPSOU!")
    print("  - Recall = 0 (nunca previu CRASH)")
    print("  - Sempre previu SAFE (classe majoritaria)")
    print("  - Undersampling NAO resolveu o problema")
    print("\nPROXIMOS PASSOS:")
    print("  1. Ajustar class weights no loss")
    print("  2. Usar focal loss")
    print("  3. Aumentar complexidade do modelo")
elif test_recall < 0.3:
    print("\nMODELO COM BAIXO RECALL!")
    print(f"  - Recall = {test_recall*100:.1f}% (detecta poucos crashes)")
    print("  - Tende a prever SAFE na duvida")
elif test_precision < 0.3:
    print("\nMODELO COM BAIXA PRECISION!")
    print(f"  - Precision = {test_precision*100:.1f}% (muitos falsos positivos)")
    print("  - Tende a prever CRASH na duvida")
else:
    print("\nMODELO FUNCIONAL!")
    print(f"  - Accuracy: {test_acc*100:.1f}%")
    print(f"  - Recall: {test_recall*100:.1f}% (detecta crashes)")
    print(f"  - Precision: {test_precision*100:.1f}% (poucos falsos positivos)")
    print("\nPROXIMO PASSO: Backtest realista!")

print(f"\n{'='*70}\n")
