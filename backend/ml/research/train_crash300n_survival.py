"""
CRASH300N - Treinamento LSTM Survival Analysis

Dataset: CRASH300N M1 (7,392 crashes detectados)
Target: crashed_in_next_10 (25% CRASH / 75% SAFE)
Estrategia: Undersampling 50/50 para forcar modelo a aprender

Comparacao com CRASH500:
- CRASH500: 7 crashes em 6 meses (INSUFICIENTE)
- CRASH300N: 7,392 crashes em 6 meses (VIAVEL!)
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Todas as classes necessarias sao definidas neste arquivo


def undersample_train_set(df_train, target_col='crashed_in_next_10'):
    """
    Aplica undersampling no train set para balancear 50/50

    Args:
        df_train: DataFrame de treino desbalanceado
        target_col: Nome da coluna target (crashed_in_next_10)

    Returns:
        df_train_balanced: DataFrame de treino balanceado 50/50
    """
    print(f"\n[UNDERSAMPLING] Balanceando train set...")

    # Separar classes (CRASH=1 vs SAFE=0)
    df_crash = df_train[df_train[target_col] == 1]
    df_safe = df_train[df_train[target_col] == 0]

    n_crash = len(df_crash)
    n_safe = len(df_safe)

    print(f"  Original:")
    print(f"    CRASH: {n_crash:,} ({n_crash/(n_crash+n_safe)*100:.1f}%)")
    print(f"    SAFE: {n_safe:,} ({n_safe/(n_crash+n_safe)*100:.1f}%)")

    # Undersampling da classe majoritaria (SAFE)
    df_safe_sampled = df_safe.sample(n=n_crash, random_state=42)

    # Combinar e embaralhar
    df_balanced = pd.concat([df_crash, df_safe_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)

    n_crash_new = (df_balanced[target_col] == 1).sum()
    n_safe_new = (df_balanced[target_col] == 0).sum()

    print(f"\n  Balanceado:")
    print(f"    CRASH: {n_crash_new:,} ({n_crash_new/len(df_balanced)*100:.1f}%)")
    print(f"    SAFE: {n_safe_new:,} ({n_safe_new/len(df_balanced)*100:.1f}%)")
    print(f"    Total: {len(df_balanced):,} (removidos {len(df_train) - len(df_balanced):,} SAFEs)")

    return df_balanced


def main():
    print("="*70)
    print("CRASH300N M1 - LSTM SURVIVAL ANALYSIS")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[DEVICE] {device}")

    # 1. Carregar dados CRASH300N
    print(f"\n[DATA] Loading CRASH300N dataset...")
    data_dir = Path(__file__).parent / "data"
    df = pd.read_csv(data_dir / "CRASH300_M1_survival_labeled.csv")

    print(f"  Total candles: {len(df):,}")

    # Verificar se coluna target existe
    target_col = 'crashed_in_next_10'
    if target_col not in df.columns:
        raise ValueError(f"Coluna '{target_col}' nao encontrada! Colunas disponiveis: {df.columns.tolist()}")

    # Class distribution (original)
    n_crash = (df[target_col] == 1).sum()
    n_safe = (df[target_col] == 0).sum()
    print(f"\n  Class distribution (original dataset):")
    print(f"    CRASH (1): {n_crash:,} ({n_crash/len(df)*100:.1f}%)")
    print(f"    SAFE (0): {n_safe:,} ({n_safe/len(df)*100:.1f}%)")

    # 2. Split: 70% train, 15% val, 15% test
    train_size = int(0.70 * len(df))
    val_size = int(0.15 * len(df))

    df_train = df.iloc[:train_size].copy()
    df_val = df.iloc[train_size:train_size+val_size].copy()
    df_test = df.iloc[train_size+val_size:].copy()

    print(f"\n  Split (ANTES do undersampling):")
    print(f"    Train: {len(df_train):,} ({len(df_train)/len(df)*100:.0f}%)")
    print(f"    Val:   {len(df_val):,} ({len(df_val)/len(df)*100:.0f}%)")
    print(f"    Test:  {len(df_test):,} ({len(df_test)/len(df)*100:.0f}%)")

    # 3. UNDERSAMPLING no TRAIN set (Val e Test ficam intactos!)
    df_train_balanced = undersample_train_set(df_train, target_col=target_col)

    print(f"\n  Split (DEPOIS do undersampling):")
    print(f"    Train: {len(df_train_balanced):,} (50% CRASH / 50% SAFE)")
    print(f"    Val:   {len(df_val):,} (distribuicao real mantida)")
    print(f"    Test:  {len(df_test):,} (distribuicao real mantida)")

    # 4. Create datasets (usando SurvivalDataset adaptado)
    lookback = 50
    batch_size = 64

    # Features: OHLC + crash-specific (realized_vol nao existe, usar rolling_volatility)
    feature_cols = [
        'open', 'high', 'low', 'close', 'rolling_volatility',
        'ticks_since_crash', 'crash_size_lag1', 'tick_velocity',
        'acceleration', 'price_deviation', 'momentum'
    ]

    # Verificar se todas as features existem
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"\n[ERROR] Colunas faltando: {missing_cols}")
        print(f"[ERROR] Colunas disponiveis: {df.columns.tolist()}")
        raise ValueError(f"Features faltando no dataset: {missing_cols}")

    print(f"\n  Features utilizadas ({len(feature_cols)}): {feature_cols}")

    # Criar datasets com features corretas
    class CRASH300Dataset(Dataset):
        def __init__(self, df, lookback=50, feature_cols=None, target_col='crashed_in_next_10'):
            self.lookback = lookback
            self.feature_cols = feature_cols or ['open', 'high', 'low', 'close', 'rolling_volatility']
            self.target_col = target_col

            # Selecionar apenas features + target
            df_features = df[self.feature_cols + [self.target_col]].copy()

            # Remover NaNs
            df_features = df_features.dropna()

            self.data = df_features[self.feature_cols].values.astype(np.float32)
            self.labels = df_features[self.target_col].values.astype(np.int64)

        def __len__(self):
            return len(self.data) - self.lookback

        def __getitem__(self, idx):
            # Features: janela de lookback candles
            X = self.data[idx:idx+self.lookback]

            # Normalizar por janela (Min-Max)
            window_min = X.min(axis=0)
            window_max = X.max(axis=0)
            window_range = window_max - window_min + 1e-8
            X_norm = (X - window_min) / window_range

            # Label: target do ultimo candle da janela
            y = self.labels[idx + self.lookback - 1]

            return torch.FloatTensor(X_norm), torch.LongTensor([y])

    train_dataset = CRASH300Dataset(df_train_balanced, lookback=lookback, feature_cols=feature_cols, target_col=target_col)
    val_dataset = CRASH300Dataset(df_val, lookback=lookback, feature_cols=feature_cols, target_col=target_col)
    test_dataset = CRASH300Dataset(df_test, lookback=lookback, feature_cols=feature_cols, target_col=target_col)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"\n  Lookback: {lookback} candles")
    print(f"  Batch size: {batch_size}")
    print(f"  Train batches: {len(train_loader)} (balanceado 50/50)")
    print(f"  Val batches: {len(val_loader)} (distribuicao real)")
    print(f"  Test batches: {len(test_loader)} (distribuicao real)")

    # 5. Create model (CLASSIFICACAO BINARIA, nao regressao!)
    model = nn.Sequential(
        nn.LSTM(input_size=len(feature_cols), hidden_size=128, num_layers=2, batch_first=True, dropout=0.3),
    )

    # Adaptar para classificacao binaria
    class LSTMBinaryClassifier(nn.Module):
        def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64, dropout=0.3):
            super().__init__()
            self.lstm1 = nn.LSTM(input_dim, hidden_dim1, batch_first=True)
            self.dropout1 = nn.Dropout(dropout)
            self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)
            self.dropout2 = nn.Dropout(dropout)
            self.fc1 = nn.Linear(hidden_dim2, 32)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(32, 2)  # 2 classes: SAFE(0) vs CRASH(1)

        def forward(self, x):
            # x: (batch, seq_len, input_dim)
            out, _ = self.lstm1(x)
            out = self.dropout1(out)
            out, _ = self.lstm2(out)
            out = self.dropout2(out)
            out = out[:, -1, :]  # Pegar ultimo timestep
            out = self.fc1(out)
            out = self.relu(out)
            out = self.fc2(out)  # Logits para 2 classes
            return out

    model = LSTMBinaryClassifier(
        input_dim=len(feature_cols),
        hidden_dim1=128,
        hidden_dim2=64,
        dropout=0.3
    ).to(device)

    print(f"\n[MODEL] LSTM Binary Classifier")
    print(f"  Input: {len(feature_cols)} features x {lookback} candles")
    print(f"  Architecture: LSTM({len(feature_cols)}->128->64) + FC(64->32->2)")
    print(f"  Output: 2 classes (SAFE=0, CRASH=1)")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 6. Train com CLASS WEIGHTS (forcar modelo a aprender CRASH)
    print(f"\n[TRAINING] Treinando com dataset balanceado (50/50) + CLASS WEIGHTS...")

    # Class weights: SAFE=1.0, CRASH=2.0 (peso intermediario)
    class_weights = torch.FloatTensor([1.0, 2.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"  Class weights: SAFE=1.0, CRASH=2.0")

    best_val_acc = 0
    patience = 15
    patience_counter = 0

    for epoch in range(100):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device).squeeze()

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += y.size(0)
            train_correct += predicted.eq(y).sum().item()

        train_acc = 100. * train_correct / train_total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device).squeeze()
                outputs = model(X)
                _, predicted = outputs.max(1)
                val_total += y.size(0)
                val_correct += predicted.eq(y).sum().item()

        val_acc = 100. * val_correct / val_total

        print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Salvar melhor modelo
            model_path = Path(__file__).parent / "models" / "crash300n_survival_lstm.pth"
            model_path.parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"  >> Best model saved! Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[EARLY STOPPING] Patience reached ({patience} epochs without improvement)")
                break

    # 7. Load best model
    model.load_state_dict(torch.load(model_path, map_location=device))

    # 8. Evaluate on test set
    print(f"\n{'='*70}")
    print(f"TESTING PHASE (Test set com distribuicao REAL)")
    print(f"{'='*70}")

    model.eval()
    test_preds = []
    test_labels = []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device).squeeze()
            outputs = model(X)
            _, predicted = outputs.max(1)
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(y.cpu().numpy())

    # Métricas
    test_acc = accuracy_score(test_labels, test_preds)
    test_precision = precision_score(test_labels, test_preds, zero_division=0)
    test_recall = recall_score(test_labels, test_preds, zero_division=0)
    test_f1 = f1_score(test_labels, test_preds, zero_division=0)

    print(f"\n[TEST RESULTS]")
    print(f"  Accuracy: {test_acc*100:.2f}%")
    print(f"  Precision: {test_precision*100:.2f}%")
    print(f"  Recall: {test_recall*100:.2f}%")
    print(f"  F1-Score: {test_f1*100:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(test_labels, test_preds)
    print(f"\n  Confusion Matrix:")
    print(f"    Predicted SAFE | Predicted CRASH")
    print(f"    {cm[0,0]:<14} | {cm[0,1]:<14}  (Actual SAFE)")
    print(f"    {cm[1,0]:<14} | {cm[1,1]:<14}  (Actual CRASH)")

    # Verificar se modelo esta prevendo ambas as classes
    n_safe_pred = sum(1 for p in test_preds if p == 0)
    n_crash_pred = sum(1 for p in test_preds if p == 1)
    print(f"\n  Predicoes:")
    print(f"    SAFE: {n_safe_pred} ({n_safe_pred/len(test_preds)*100:.2f}%)")
    print(f"    CRASH: {n_crash_pred} ({n_crash_pred/len(test_preds)*100:.2f}%)")

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETED")
    print(f"{'='*70}")
    print(f"\nBest Val Accuracy: {best_val_acc:.2f}%")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Model saved: {model_path}")

    # Comparacao com CRASH500
    print(f"\n{'='*70}")
    print(f"COMPARACAO: CRASH500 vs CRASH300N")
    print(f"{'='*70}")
    print(f"\nCRASH500 (Survival):")
    print(f"  Dataset: 7 crashes (INSUFICIENTE)")
    print(f"  Result: Impossivel treinar ML")
    print(f"\nCRASH300N (Survival):")
    print(f"  Dataset: 7,392 crashes (VIAVEL!)")
    print(f"  Best Val Acc: {best_val_acc:.2f}%")
    print(f"  Test Acc: {test_acc*100:.2f}%")
    print(f"  Precision: {test_precision*100:.2f}%")
    print(f"  Recall: {test_recall*100:.2f}%")

    if test_recall > 0:
        print(f"\n  MODELO FUNCIONAL!")
        print(f"  Proxima etapa: Backtest realista (SL/TP dinamico)")
    else:
        print(f"\n  MODELO COLAPSOU (Recall = 0)")
        print(f"  Sempre preve SAFE. Precisa ajustar arquitetura/hiperparametros")


if __name__ == "__main__":
    main()
