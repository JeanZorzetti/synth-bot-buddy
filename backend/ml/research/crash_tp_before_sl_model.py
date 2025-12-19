"""
LSTM Classifier para TP Before SL em CRASH 500

OBJETIVO: Prever "TP será atingido antes de SL?" (classificação binária)

TARGET: 1 = WIN (TP hit), 0 = LOSS (SL/timeout hit)

DIFERENÇA vs Survival Analysis:
- Survival: Regressão (prever candles_to_crash) → MSELoss
- TP Before SL: Classificação binária (prever WIN/LOSS) → BCELoss

WIN RATE ESPERADO: 60-70% (se modelo aprende os padrões)
"""
import os
# Force CPU only (evita problemas com CUDA)
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class TPBeforeSLDataset(Dataset):
    """
    Dataset para LSTM de classificação binária (prever TP before SL)
    """
    def __init__(self, df, lookback=50):
        self.lookback = lookback

        # Features: OHLC + realized_vol + rsi + atr (7 features)
        feature_cols = ['open', 'high', 'low', 'close', 'realized_vol', 'rsi', 'atr']
        self.features = df[feature_cols].values.astype(np.float32)

        # Label: tp_before_sl (classificação binária: 0 ou 1)
        self.labels = df['tp_before_sl'].values.astype(np.float32)

        # Número de amostras
        self.n_samples = len(self.features) - self.lookback

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Janela deslizante
        window = self.features[idx:idx + self.lookback]

        # Normalização robusta (min-max para evitar NaN)
        window_min = window.min(axis=0)
        window_max = window.max(axis=0)
        window_range = window_max - window_min + 1e-8  # Evitar divisão por zero
        x = (window - window_min) / window_range

        # Label (0 ou 1)
        y = self.labels[idx + self.lookback]

        return torch.FloatTensor(x), torch.FloatTensor([y])


class LSTMTPBeforeSLModel(nn.Module):
    """
    LSTM para classificação binária (prever TP before SL)
    """
    def __init__(self, input_dim=7, hidden_dim1=128, hidden_dim2=64, dropout=0.3):
        super().__init__()

        # LSTM layers
        self.lstm1 = nn.LSTM(input_dim, hidden_dim1, batch_first=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.dropout1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.dropout2 = nn.Dropout(dropout)

        # Classification head
        self.fc1 = nn.Linear(hidden_dim2, 32)
        self.dropout3 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 1)  # Output: single probability (0-1)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]

        # LSTM 1
        out, _ = self.lstm1(x)
        out = out[:, -1, :]  # [batch, hidden_dim1]
        out = self.bn1(out)
        out = self.dropout1(out)

        # Expandir para próximo LSTM
        out = out.unsqueeze(1)  # [batch, 1, hidden_dim1]

        # LSTM 2
        out, _ = self.lstm2(out)
        out = out[:, -1, :]  # [batch, hidden_dim2]
        out = self.bn2(out)
        out = self.dropout2(out)

        # Classification head
        out = torch.relu(self.fc1(out))
        out = self.dropout3(out)
        out = self.fc2(out)

        # Sigmoid para probabilidade [0, 1]
        out = torch.sigmoid(out)

        return out


class TPBeforeSLTrainer:
    """
    Treinador para LSTM TP Before SL (classificação binária)
    """
    def __init__(self, model, device='cpu', class_weights=None):
        self.model = model.to(device)
        self.device = device

        # Loss: BCELoss (classificação binária)
        if class_weights is not None:
            # Peso maior para classe minoritária (WIN)
            pos_weight = torch.tensor([class_weights[1] / class_weights[0]]).to(device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.criterion = nn.BCELoss()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=0.001,
            weight_decay=0.01
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # Maximize accuracy
            factor=0.5,
            patience=5
        )

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)

            # Forward
            pred = self.model(x)
            loss = self.criterion(pred, y)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            all_preds.extend((pred > 0.5).cpu().numpy().flatten())
            all_labels.extend(y.cpu().numpy().flatten())

        # Calculate accuracy
        acc = accuracy_score(all_labels, all_preds)

        return total_loss / len(train_loader), acc

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)

                pred = self.model(x)
                loss = self.criterion(pred, y)

                total_loss += loss.item()
                all_preds.extend((pred > 0.5).cpu().numpy().flatten())
                all_labels.extend(y.cpu().numpy().flatten())

        # Calculate metrics
        acc = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        return total_loss / len(val_loader), acc, precision, recall, f1

    def train(self, train_loader, val_loader, epochs=100, patience=15):
        best_val_acc = 0
        epochs_no_improve = 0

        print(f"\n[TRAINING] Starting training for {epochs} epochs...")
        print(f"  Device: {self.device}")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc, val_precision, val_recall, val_f1 = self.validate(val_loader)

            self.scheduler.step(val_acc)

            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
                  f"Prec: {val_precision:.4f}, Rec: {val_recall:.4f}, F1: {val_f1:.4f}")

            # Early stopping (baseado em accuracy)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0

                # Save model
                model_path = Path(__file__).parent / "models" / "crash_tp_before_sl_lstm.pth"
                model_path.parent.mkdir(exist_ok=True)
                torch.save(self.model.state_dict(), model_path)

                print(f"  [BEST] Model saved! Accuracy: {best_val_acc:.4f}")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"\n[EARLY STOP] No improvement for {patience} epochs")
                break

        print(f"\n[OK] Training completed! Best val accuracy: {best_val_acc:.4f}")
        return best_val_acc


def evaluate_on_test_set(model, test_loader, device, threshold=0.7):
    """
    Avalia modelo no test set e calcula métricas de trading

    Args:
        threshold: Probabilidade mínima para entrar no trade (default 70%)
    """
    print(f"\n[EVALUATION] Evaluating on test set...")
    print(f"  Entry threshold: {threshold*100:.0f}% (só entra se P(WIN) >= {threshold*100:.0f}%)")

    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            prob = model(x)

            all_probs.extend(prob.cpu().numpy().flatten())
            all_preds.extend((prob > 0.5).cpu().numpy().flatten())
            all_labels.extend(y.cpu().numpy().flatten())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Métricas gerais
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print(f"\n[METRICS - All Predictions]")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\n[CONFUSION MATRIX]")
    print(f"  TN: {cm[0,0]:,} | FP: {cm[0,1]:,}")
    print(f"  FN: {cm[1,0]:,} | TP: {cm[1,1]:,}")

    # Trading strategy: só entra se P(WIN) >= threshold
    high_conf_mask = all_probs >= threshold
    n_trades = high_conf_mask.sum()

    if n_trades > 0:
        filtered_preds = all_preds[high_conf_mask]
        filtered_labels = all_labels[high_conf_mask]

        trade_acc = accuracy_score(filtered_labels, filtered_preds)
        wins = (filtered_labels == 1).sum()
        losses = (filtered_labels == 0).sum()
        win_rate = wins / n_trades * 100

        print(f"\n[TRADING STRATEGY - P(WIN) >= {threshold*100:.0f}%]")
        print(f"  Total Trades: {n_trades:,} ({n_trades/len(all_labels)*100:.1f}% dos candles)")
        print(f"  Wins: {wins:,} | Losses: {losses:,}")
        print(f"  Win Rate: {win_rate:.2f}%")
        print(f"  Accuracy: {trade_acc:.4f}")

        if win_rate >= 60:
            print(f"\n  MODELO APROVADO PARA PRODUCAO!")
            print(f"  Win rate >= 60% threshold")
        else:
            print(f"\n  Modelo ainda precisa melhorar")
            print(f"  Win rate < 60% threshold")
    else:
        print(f"\n[WARNING] Nenhum trade com P(WIN) >= {threshold*100:.0f}%")
        print(f"  Modelo não tem confiança suficiente para entrar em trades")

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist(),
        'trading_strategy': {
            'threshold': threshold,
            'n_trades': int(n_trades),
            'win_rate': win_rate if n_trades > 0 else 0,
        }
    }


def main():
    print("="*70)
    print("CRASH 500 - TP BEFORE SL LSTM CLASSIFIER")
    print("="*70)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[DEVICE] {device}")

    # 1. Carregar dados
    print(f"\n[DATA] Loading dataset...")
    data_dir = Path(__file__).parent / "data"
    df = pd.read_csv(data_dir / "CRASH500_5min_tp_before_sl_labeled.csv")

    print(f"  Total candles: {len(df):,}")
    print(f"  Features: {['open', 'high', 'low', 'close', 'realized_vol', 'rsi', 'atr']}")
    print(f"  Label: tp_before_sl (0=LOSS, 1=WIN)")

    # Class distribution
    n_wins = (df['tp_before_sl'] == 1).sum()
    n_losses = (df['tp_before_sl'] == 0).sum()
    print(f"\n  Class distribution:")
    print(f"    LOSS (0): {n_losses:,} ({n_losses/len(df)*100:.1f}%)")
    print(f"    WIN (1): {n_wins:,} ({n_wins/len(df)*100:.1f}%)")

    # 2. Split: 70% train, 15% val, 15% test
    train_size = int(0.70 * len(df))
    val_size = int(0.15 * len(df))

    df_train = df.iloc[:train_size]
    df_val = df.iloc[train_size:train_size+val_size]
    df_test = df.iloc[train_size+val_size:]

    print(f"\n  Train: {len(df_train):,} ({len(df_train)/len(df)*100:.0f}%)")
    print(f"  Val:   {len(df_val):,} ({len(df_val)/len(df)*100:.0f}%)")
    print(f"  Test:  {len(df_test):,} ({len(df_test)/len(df)*100:.0f}%)")

    # 3. Create datasets
    lookback = 50
    batch_size = 64

    train_dataset = TPBeforeSLDataset(df_train, lookback=lookback)
    val_dataset = TPBeforeSLDataset(df_val, lookback=lookback)
    test_dataset = TPBeforeSLDataset(df_test, lookback=lookback)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"\n  Lookback: {lookback} candles")
    print(f"  Batch size: {batch_size}")

    # 4. Create model
    model = LSTMTPBeforeSLModel(
        input_dim=7,       # OHLC + realized_vol + rsi + atr
        hidden_dim1=128,
        hidden_dim2=64,
        dropout=0.3
    )

    print(f"\n[MODEL] LSTM Classifier")
    print(f"  Input: 7 features x {lookback} candles")
    print(f"  Architecture: LSTM(7->128->64) + FC(64->32->1)")
    print(f"  Output: Probability [0, 1]")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 5. Class weights (balancear classes)
    class_weights = [n_losses, n_wins]

    # 6. Train
    trainer = TPBeforeSLTrainer(model, device=device, class_weights=None)
    best_acc = trainer.train(
        train_loader,
        val_loader,
        epochs=100,
        patience=15
    )

    # 7. Load best model
    model_path = Path(__file__).parent / "models" / "crash_tp_before_sl_lstm.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))

    # 8. Evaluate on test set
    results = evaluate_on_test_set(model, test_loader, device, threshold=0.7)

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETED")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
