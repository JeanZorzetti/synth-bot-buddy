"""
LSTM com Features Enriquecidas (24 features)

DIFERENÇA vs LSTM Baseline:
- Baseline: 4 features OHLC → 54.3% win rate (colapso para LONG)
- Rich: 24 features (RSI, MACD, ATR, etc.) → Expectativa: 55-58%

Arquitetura:
Input: [batch, 50 candles, 24 features]
↓
LSTM(128) → Dropout(0.3) → BatchNorm
↓
LSTM(64) → Dropout(0.3) → BatchNorm
↓
Dense(32, ReLU) → Dropout(0.2)
↓
Output(3, Softmax) → [NO_TRADE, LONG, SHORT]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix

# Import labeling
from scalping_labeling import ScalpingLabeler


class RichFeaturesDataset(Dataset):
    """
    Dataset para LSTM com 24 features enriquecidas
    """
    def __init__(self, df, lookback=50):
        self.lookback = lookback

        # Excluir colunas não-feature
        exclude_cols = ['timestamp', 'label', 'label_metadata', 'epoch']
        self.feature_cols = [c for c in df.columns if c not in exclude_cols]

        print(f"[DATASET] Features utilizadas: {len(self.feature_cols)}")
        print(f"   {', '.join(self.feature_cols[:10])}...")

        # Extrair features
        self.features = df[self.feature_cols].values.astype(np.float32)

        # Labels (converter -1 para 2)
        self.labels = df['label'].values
        self.labels = np.where(self.labels == -1, 2, self.labels)

        # Número de amostras
        self.n_samples = len(self.features) - self.lookback

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Janela deslizante
        window = self.features[idx:idx + self.lookback]

        # Normalização Z-Score por janela (preserva tendência)
        mean = window.mean(axis=0)
        std = window.std(axis=0) + 1e-8
        x = (window - mean) / std

        # Label
        y = self.labels[idx + self.lookback]

        return torch.FloatTensor(x), torch.LongTensor([y]).squeeze()


class LSTMRichModel(nn.Module):
    """
    LSTM com 24 features enriquecidas
    """
    def __init__(self, input_dim=24, hidden_dim1=128, hidden_dim2=64, output_dim=3):
        super().__init__()

        # LSTM layers
        self.lstm1 = nn.LSTM(input_dim, hidden_dim1, batch_first=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.dropout1 = nn.Dropout(0.3)

        self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.dropout2 = nn.Dropout(0.3)

        # Dense layers
        self.fc1 = nn.Linear(hidden_dim2, 32)
        self.dropout3 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, output_dim)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]

        # LSTM 1
        out, _ = self.lstm1(x)
        # Pegar última saída
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

        # Dense
        out = F.relu(self.fc1(out))
        out = self.dropout3(out)
        out = self.fc2(out)

        return out


class LSTMTrainer:
    """
    Treinador para LSTM Rich
    """
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device

        # Loss: CrossEntropy com class weighting dinâmico
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=0.001,
            weight_decay=0.01
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)

            # Forward
            logits = self.model(x)
            loss = self.criterion(logits, y)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        return total_loss / len(train_loader), correct / total

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)

                logits = self.model(x)
                loss = self.criterion(logits, y)

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        return total_loss / len(val_loader), correct / total

    def train(self, train_loader, val_loader, epochs=50, patience=10):
        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            self.scheduler.step(val_loss)

            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # Save model
                model_path = Path(__file__).parent / "models" / "lstm_rich_features.pth"
                torch.save(self.model.state_dict(), model_path)
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"\nEarly stopping após {epoch+1} épocas")
                break

        print(f"\n[OK] Treinamento concluído! Best val loss: {best_val_loss:.4f}")


def main():
    print("="*70)
    print("LSTM COM FEATURES ENRIQUECIDAS (24 features)")
    print("="*70)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[DEVICE] {device}")

    # 1. Carregar dataset enriquecido
    print(f"\n[LOAD] Carregando dataset com features enriquecidas...")
    data_dir = Path(__file__).parent / "data"
    df = pd.read_csv(data_dir / "1HZ100V_5min_rich_features.csv")
    print(f"   Dataset: {len(df):,} candles")
    print(f"   Features: {len([c for c in df.columns if c not in ['timestamp', 'label', 'label_metadata', 'epoch']])}")

    # 2. Criar datasets
    print(f"\n[DATASET] Criando datasets...")
    dataset = RichFeaturesDataset(df, lookback=50)

    # Split temporal
    train_size = int(0.70 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
    test_dataset = torch.utils.data.Subset(dataset, range(train_size + val_size, len(dataset)))

    print(f"   Train: {len(train_dataset):,}")
    print(f"   Val: {len(val_dataset):,}")
    print(f"   Test: {len(test_dataset):,}")

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # 3. Criar modelo
    print(f"\n[MODEL] Criando LSTM Rich...")
    n_features = len(dataset.feature_cols)
    print(f"   Input dim: {n_features} features")
    model = LSTMRichModel(input_dim=n_features, hidden_dim1=128, hidden_dim2=64, output_dim=3)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parâmetros: {total_params:,}")

    # 4. Treinar
    print(f"\n[TRAIN] Treinando LSTM Rich...")
    trainer = LSTMTrainer(model, device=device)
    trainer.train(train_loader, val_loader, epochs=50, patience=10)

    # 5. Avaliar
    print(f"\n[EVAL] Avaliando em Test Set...")
    model_path = Path(__file__).parent / "models" / "lstm_rich_features.pth"
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Filtrar apenas trades
    trade_mask = (all_labels == 1) | (all_labels == 2)
    trade_preds = all_preds[trade_mask]
    trade_labels = all_labels[trade_mask]

    if len(trade_labels) > 0:
        long_mask = trade_labels == 1
        short_mask = trade_labels == 2

        long_accuracy = (trade_preds[long_mask] == 1).sum() / long_mask.sum() if long_mask.sum() > 0 else 0
        short_accuracy = (trade_preds[short_mask] == 2).sum() / short_mask.sum() if short_mask.sum() > 0 else 0
        win_rate = (trade_preds == trade_labels).sum() / len(trade_labels)

        print(f"\n{'='*70}")
        print(f"RESULTADOS FINAIS - LSTM RICH")
        print(f"{'='*70}")
        print(f"\n   WIN RATE (LONG+SHORT): {win_rate:.2%}")
        print(f"   LONG Accuracy: {long_accuracy:.2%}")
        print(f"   SHORT Accuracy: {short_accuracy:.2%}")
        print(f"\n   Total trades no test: {len(trade_labels):,}")
        print(f"   LONG: {long_mask.sum():,} ({long_mask.sum()/len(trade_labels):.1%})")
        print(f"   SHORT: {short_mask.sum():,} ({short_mask.sum()/len(trade_labels):.1%})")

        # Confusion matrix
        cm = confusion_matrix(trade_labels, trade_preds, labels=[1, 2])
        print(f"\n   Confusion Matrix (LONG=1, SHORT=2):")
        print(f"   Predicted:  LONG  SHORT")
        print(f"   Real LONG:  {cm[0,0]:4d}  {cm[0,1]:4d}  ({cm[0,0]/(cm[0,0]+cm[0,1]):.1%} recall)")
        print(f"   Real SHORT: {cm[1,0]:4d}  {cm[1,1]:4d}  ({cm[1,1]/(cm[1,0]+cm[1,1]):.1%} recall)")

        # Comparação
        print(f"\n{'='*70}")
        print(f"COMPARACAO: LSTM Baseline vs LSTM Rich")
        print(f"{'='*70}")
        print(f"\n   Metrica          | Baseline | Rich     | Delta")
        print(f"   -----------------+----------+----------+-------")
        print(f"   Features         | 4 OHLC   | 24       | +20")
        print(f"   Win Rate         | 54.3%    | {win_rate:.1%}   | {(win_rate-0.543)*100:+.1f}pp")
        print(f"   LONG Accuracy    | 100.0%   | {long_accuracy:.1%}  | {(long_accuracy-1.0)*100:+.1f}pp")
        print(f"   SHORT Accuracy   | 0.0%     | {short_accuracy:.1%}  | {(short_accuracy-0.0)*100:+.1f}pp")

        # Status
        if win_rate >= 0.58:
            status = "[META ATINGIDA] 58%+"
        elif win_rate >= 0.55:
            status = "[PROMISSOR] 55-58%"
        elif win_rate > 0.543:
            status = "[MELHORIA] > baseline"
        else:
            status = "[FALHOU] <= baseline"

        print(f"\n   STATUS: {status}")

    print(f"\n{'='*70}")

if __name__ == "__main__":
    main()
