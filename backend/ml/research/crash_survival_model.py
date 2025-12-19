"""
LSTM de Regressão para Survival Analysis em CRASH 500

OBJETIVO: Prever "Quantos candles até alta volatilidade?"

ESTRATÉGIA:
- Se modelo prever >= 20 candles: ENTRAR LONG (zona segura)
- Se modelo prever < 20 candles: FICAR FORA (zona de perigo)

DIFERENÇA vs V100:
- V100: Prever direção (LONG/SHORT) = aleatório (50%)
- CRASH 500: Prever risco (safe/danger) = estruturado (88% safe)
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class CrashSurvivalDataset(Dataset):
    """
    Dataset para LSTM de regressão (prever candles_to_crash)
    """
    def __init__(self, df, lookback=50):
        self.lookback = lookback

        # Features: OHLC + volatilidade
        feature_cols = ['open', 'high', 'low', 'close', 'realized_vol']
        self.features = df[feature_cols].values.astype(np.float32)

        # Label: candles_to_crash (regressão)
        self.labels = df['candles_to_crash'].values.astype(np.float32)

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

        # Label (número de candles até crash/alta vol)
        y = self.labels[idx + self.lookback]

        # Clip label (máximo 100 candles para estabilidade)
        y = min(y, 100.0)

        return torch.FloatTensor(x), torch.FloatTensor([y])


class LSTMSurvivalModel(nn.Module):
    """
    LSTM para regressão (prever número de candles)
    """
    def __init__(self, input_dim=5, hidden_dim1=128, hidden_dim2=64):
        super().__init__()

        # LSTM layers
        self.lstm1 = nn.LSTM(input_dim, hidden_dim1, batch_first=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.dropout1 = nn.Dropout(0.3)

        self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.dropout2 = nn.Dropout(0.3)

        # Regression head
        self.fc1 = nn.Linear(hidden_dim2, 32)
        self.dropout3 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 1)  # Output: single value (candles_to_crash)

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

        # Regression
        out = torch.relu(self.fc1(out))
        out = self.dropout3(out)
        out = self.fc2(out)

        # ReLU para garantir output >= 0
        out = torch.relu(out)

        return out


class SurvivalTrainer:
    """
    Treinador para LSTM Survival (regressão)
    """
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device

        # Loss: MSE (regressão)
        self.criterion = nn.MSELoss()

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
        total_mae = 0

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
            total_mae += torch.abs(pred - y).mean().item()

        return total_loss / len(train_loader), total_mae / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        total_mae = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)

                pred = self.model(x)
                loss = self.criterion(pred, y)

                total_loss += loss.item()
                total_mae += torch.abs(pred - y).mean().item()

        return total_loss / len(val_loader), total_mae / len(val_loader)

    def train(self, train_loader, val_loader, epochs=50, patience=10):
        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(epochs):
            train_loss, train_mae = self.train_epoch(train_loader)
            val_loss, val_mae = self.validate(val_loader)

            self.scheduler.step(val_loss)

            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.2f} - "
                  f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.2f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # Save model
                model_path = Path(__file__).parent / "models" / "crash_survival_lstm.pth"
                torch.save(self.model.state_dict(), model_path)
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"\nEarly stopping apos {epoch+1} epocas")
                break

        print(f"\n[OK] Treinamento concluido! Best val loss: {best_val_loss:.4f}")


def backtest_strategy(model, test_loader, device, threshold=20):
    """
    Backtesta estratégia: entrar LONG se pred >= threshold
    """
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            all_preds.extend(pred.cpu().numpy().flatten())
            all_labels.extend(y.cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Estratégia: entrar se pred >= threshold
    trades = all_preds >= threshold
    n_trades = trades.sum()

    # Win = label também >= threshold (estava realmente seguro)
    wins = (all_labels[trades] >= threshold).sum() if n_trades > 0 else 0
    win_rate = wins / n_trades if n_trades > 0 else 0

    # MAE e R²
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)

    return {
        'n_trades': n_trades,
        'wins': wins,
        'win_rate': win_rate,
        'mae': mae,
        'r2': r2,
        'preds': all_preds,
        'labels': all_labels
    }


def main():
    print("="*70)
    print("CRASH 500 - LSTM SURVIVAL MODEL")
    print("="*70)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[DEVICE] {device}")

    # 1. Carregar dataset
    print(f"\n[LOAD] Carregando dataset...")
    data_dir = Path(__file__).parent / "data"
    df = pd.read_csv(data_dir / "CRASH500_5min_survival_labeled.csv")

    # Remover NaNs
    df = df.dropna()
    print(f"   Candles: {len(df):,}")

    # 2. Criar datasets
    print(f"\n[DATASET] Criando datasets...")
    dataset = CrashSurvivalDataset(df, lookback=50)

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
    print(f"\n[MODEL] Criando LSTM Survival...")
    model = LSTMSurvivalModel(input_dim=5, hidden_dim1=128, hidden_dim2=64)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parametros: {total_params:,}")

    # 4. Treinar
    print(f"\n[TRAIN] Treinando LSTM Survival...")
    trainer = SurvivalTrainer(model, device=device)
    trainer.train(train_loader, val_loader, epochs=50, patience=10)

    # 5. Backtest
    print(f"\n[BACKTEST] Testando estratégia...")
    model_path = Path(__file__).parent / "models" / "crash_survival_lstm.pth"
    model.load_state_dict(torch.load(model_path))

    results = backtest_strategy(model, test_loader, device, threshold=20)

    print(f"\n{'='*70}")
    print(f"RESULTADOS FINAIS - CRASH 500 SURVIVAL")
    print(f"{'='*70}")
    print(f"\n   Estratégia: Entrar LONG se prever >= 20 candles até alta vol")
    print(f"\n   Trades executados: {results['n_trades']:,}")
    print(f"   Wins: {results['wins']:,}")
    print(f"   WIN RATE: {results['win_rate']:.2%}")
    print(f"\n   MAE (candles): {results['mae']:.2f}")
    print(f"   R² Score: {results['r2']:.4f}")

    # Comparação com baseline
    print(f"\n{'='*70}")
    print(f"COMPARACAO: V100 Scalping vs CRASH 500 Survival")
    print(f"{'='*70}")
    print(f"\n   Modelo           | Abordagem      | Win Rate  | Status")
    print(f"   -----------------+----------------+-----------+--------")
    print(f"   V100 LSTM        | Predict LONG/SHORT | 54.3% | Colapso")
    print(f"   V100 MCA         | Predict LONG/SHORT | 49-51% | Colapso")
    print(f"   V100 LSTM Rich   | Predict LONG/SHORT | 0.0%  | Falhou")
    print(f"   CRASH 500 LSTM   | Predict RISK   | {results['win_rate']:.1%}  | ???")

    if results['win_rate'] >= 0.70:
        status = "[META ATINGIDA] 70%+"
    elif results['win_rate'] >= 0.60:
        status = "[META ATINGIDA] 60%+"
    elif results['win_rate'] > 0.543:
        status = "[MELHOR] > V100 baseline"
    else:
        status = "[FALHOU] <= V100"

    print(f"\n   STATUS: {status}")

    print(f"\n{'='*70}")

if __name__ == "__main__":
    main()
