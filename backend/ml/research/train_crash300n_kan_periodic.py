"""
CRASH300N - Kolmogorov-Arnold Networks (KAN) - VERSÃO PERIÓDICA
Implementação com Splines Senoidais para Detectar Periodicidade em PRNG
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


# --- KAN com Splines Senoidais (para capturar periodicidade) ---
class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Pesos base (linear) + Pesos Spline (não-linear)
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size))

        nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5))
        nn.init.normal_(self.spline_weight, std=0.1)

        self.grid_size = grid_size

    def forward(self, x):
        # x: [batch, in_features]

        # Base (Linear - SiLU)
        base_output = F.linear(F.silu(x), self.base_weight)

        # Spline SENOIDAL (para capturar periodicidade de PRNG)
        # Map to -pi, pi range
        x_norm = torch.tanh(x) * 3.14

        # Stack de senos com frequências diferentes
        spline_basis = torch.stack([
            torch.sin(x_norm * (i+1)) for i in range(self.grid_size)
        ], dim=-1)  # [batch, in_features, grid_size]

        # Combine com pesos
        spline_output = torch.einsum('big,oig->bo', spline_basis, self.spline_weight)

        return base_output + spline_output


class KAN(nn.Module):
    def __init__(self, layers_hidden):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers_hidden) - 1):
            self.layers.append(KANLayer(layers_hidden[i], layers_hidden[i+1]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def prepare_interval_data(df):
    """Transforma OHLC em sequencia de intervalos entre crashes"""
    print("\n[PREP] Calculando intervalos entre crashes...")

    # Detectar crashes
    df['log_ret'] = np.log(df['close'] / df['open'])
    df['is_crash'] = (df['log_ret'] <= -0.005).astype(int)

    # Pegar índices onde ocorreu crash
    crash_indices = df[df['is_crash'] == 1].index.values

    # Calcular diferenca (intervalos)
    intervals = np.diff(crash_indices)

    print(f"  Total Crashes: {len(crash_indices)}")
    print(f"  Total Intervalos: {len(intervals)}")
    print(f"  Media: {intervals.mean():.2f} candles")
    print(f"  Std: {intervals.std():.2f} candles")
    print(f"  Min: {intervals.min()} | Max: {intervals.max()}")

    return intervals


def train_kan():
    print("="*70)
    print("CRASH300N - KAN PERIÓDICO (Splines Senoidais)")
    print("Objetivo: Detectar ciclos/periodicidade no PRNG")
    print("="*70)

    # 1. Load Data
    data_dir = Path(__file__).parent / "data"
    files = list(data_dir.glob("*CRASH300*.csv"))
    if not files:
        print("\nERRO: Nenhum arquivo CRASH300N encontrado")
        return

    print(f"\n[LOAD] {files[0].name}")
    df = pd.read_csv(files[0])

    # 2. Prepare Data (Sequence of Intervals)
    intervals = prepare_interval_data(df)

    # Normalizar (Clip + Division) para estabilidade
    max_val = 1000.0  # Clip em 1000 candles
    intervals_clipped = np.clip(intervals, 0, max_val)
    data_norm = intervals_clipped / max_val

    print(f"\n[NORM] Valores normalizados em [0, 1]")
    print(f"  Max clip: {max_val} candles")

    # Criar sequencias: [I_t-5, I_t-4, I_t-3, I_t-2, I_t-1] -> Predizer I_t
    SEQ_LEN = 5
    X = []
    y = []

    for i in range(len(data_norm) - SEQ_LEN):
        X.append(data_norm[i:i+SEQ_LEN])
        y.append(data_norm[i+SEQ_LEN])

    X = torch.FloatTensor(np.array(X))
    y = torch.FloatTensor(np.array(y)).unsqueeze(1)

    # Split (80/20)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"\n[DATASET] Sequences: {len(X)}")
    print(f"  Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Test:  {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

    # 3. Model (KAN com Splines Senoidais)
    # Input: 5 ultimos intervalos -> Hidden 16 -> Output 1 (Proximo intervalo)
    print(f"\n[MODEL] KAN Periódico")
    print(f"  Architecture: [5, 16, 1]")
    print(f"  - Input: 5 intervalos anteriores")
    print(f"  - Hidden: 16 nodes (KANLayer com splines senoidais)")
    print(f"  - Output: 1 (próximo intervalo)")

    model = KAN([SEQ_LEN, 16, 1])

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Optimizer: AdamW (melhor que LBFGS para generalização)
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-5)
    criterion = nn.MSELoss()

    print(f"\n[TRAINING] Otimizador: AdamW (lr=0.01, weight_decay=1e-5)")
    print(f"  Epochs: 200")
    print(f"  Buscando por periodicidade no PRNG...\n")

    losses_train = []
    losses_val = []

    for epoch in range(200):
        # Train
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(X_test)
            val_loss = criterion(val_out, y_test)

        losses_train.append(loss.item())
        losses_val.append(val_loss.item())

        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d} | Train Loss: {loss.item():.6f} | Val Loss: {val_loss.item():.6f}")

    print(f"\n  Training completo!")

    # 4. Avaliacao Final
    print(f"\n{'='*70}")
    print(f"AVALIAÇÃO FINAL")
    print(f"{'='*70}")

    model.eval()
    with torch.no_grad():
        preds = model(X_test)

    # Desnormalizar
    y_true = y_test.numpy() * max_val
    y_pred = preds.numpy() * max_val

    # Metrics
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # Baseline (sempre prever média)
    baseline_pred = np.full_like(y_true, y_train.numpy().mean() * max_val)
    baseline_mae = np.mean(np.abs(y_true - baseline_pred))
    baseline_rmse = np.sqrt(np.mean((y_true - baseline_pred) ** 2))

    print(f"\n[RESULTADOS]")
    print(f"  KAN Periódico:")
    print(f"    MAE:  {mae:.2f} candles")
    print(f"    RMSE: {rmse:.2f} candles")
    print(f"\n  Baseline (prever média):")
    print(f"    MAE:  {baseline_mae:.2f} candles")
    print(f"    RMSE: {baseline_rmse:.2f} candles")

    # Improvement
    improvement_mae = (baseline_mae - mae) / baseline_mae * 100
    improvement_rmse = (baseline_rmse - rmse) / baseline_rmse * 100

    print(f"\n[IMPROVEMENT]")
    print(f"  MAE:  {improvement_mae:+.2f}%")
    print(f"  RMSE: {improvement_rmse:+.2f}%")

    # Plot
    plt.figure(figsize=(14, 8))

    # Plot 1: Predictions vs Actual
    plt.subplot(2, 2, 1)
    plt.plot(y_true[:200], 'b-', alpha=0.7, label='Real Interval', linewidth=2)
    plt.plot(y_pred[:200], 'r--', alpha=0.7, label='KAN Predicted', linewidth=1.5)
    baseline_mean = float(baseline_pred[0])
    plt.axhline(y=baseline_mean, color='g', linestyle=':', linewidth=1, label=f'Baseline (Mean={baseline_mean:.1f})')
    plt.title(f"KAN Periódico: Predictions vs Actual (First 200)", fontsize=12, fontweight='bold')
    plt.xlabel('Sequence Index')
    plt.ylabel('Interval (candles)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Residuals
    plt.subplot(2, 2, 2)
    residuals = y_true.flatten() - y_pred.flatten()
    plt.hist(residuals, bins=50, alpha=0.7, color='purple', edgecolor='black')
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    plt.title('Residual Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Residuals (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: Training Curves
    plt.subplot(2, 2, 3)
    plt.plot(losses_train, label='Train Loss', alpha=0.7)
    plt.plot(losses_val, label='Val Loss', alpha=0.7)
    plt.title('Training Curves', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # Plot 4: Scatter (Actual vs Predicted)
    plt.subplot(2, 2, 4)
    plt.scatter(y_true, y_pred, alpha=0.3, s=10)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', linewidth=2, label='Perfect Prediction')
    plt.title('Actual vs Predicted', fontsize=12, fontweight='bold')
    plt.xlabel('Actual Interval (candles)')
    plt.ylabel('Predicted Interval (candles)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = Path(__file__).parent / "reports" / "crash300n_kan_periodic_results.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\n[PLOT] Gráfico salvo em: {plot_path}")

    # VEREDICTO FINAL
    print(f"\n{'='*70}")
    print("VEREDICTO FINAL")
    print(f"{'='*70}\n")

    if improvement_mae > 15.0 and improvement_rmse > 15.0:
        print(f"SUCESSO! PERIODICIDADE DETECTADA!")
        print(f"  KAN Periodico descobriu padroes nos intervalos!")
        print(f"  Melhoria: MAE {improvement_mae:.1f}%, RMSE {improvement_rmse:.1f}%")
        print(f"\n  INTERPRETACAO:")
        print(f"  >> O PRNG tem CICLOS detectaveis")
        print(f"  >> Splines senoidais capturaram frequencias")
        print(f"  >> Algoritmo da Deriv e EXPLORAVEL")
        print(f"\n  PROXIMA ACAO:")
        print(f"  >> Implementar estrategia baseada em predicao de intervalos")
        print(f"  >> Entrar LONG quando KAN prever intervalo longo (>60 candles)")

    elif improvement_mae > 5.0 or improvement_rmse > 5.0:
        print(f"EDGE FRACO DETECTADO")
        print(f"  KAN teve melhoria marginal ({improvement_mae:.1f}% MAE, {improvement_rmse:.1f}% RMSE)")
        print(f"  Existe alguma estrutura, mas e fraca")
        print(f"\n  POSSIBILIDADE:")
        print(f"  >> PRNG tem periodo muito longo (impossivel de detectar com dataset atual)")
        print(f"  >> Ou periodicidade e mascarada por ruido adicional")

    else:
        print(f"FALHOU - SEM PERIODICIDADE")
        print(f"  KAN Periodico NAO melhorou vs baseline ({improvement_mae:.1f}% MAE)")
        print(f"  Splines senoidais NAO encontraram ciclos")
        print(f"\n  CONCLUSAO:")
        print(f"  >> PRNG e aperiodico OU periodo > tamanho do dataset")
        print(f"  >> Deriv usa CSPRNG cryptographically secure")
        print(f"  >> Intervalos sao matematicamente imprevisiveis")
        print(f"\n  EVIDENCIA:")
        print(f"  >> LSTM (correlacoes) falhou")
        print(f"  >> XGBoost (particoes) falhou")
        print(f"  >> KAN B-splines (funcoes smooth) falhou")
        print(f"  >> KAN Senoidais (periodicidade) FALHOU")
        print(f"\n  >> 4/4 abordagens convergem: IMPOSSIVEL")

    print(f"\n{'='*70}\n")

    # Save model
    model_path = Path(__file__).parent / "models" / "crash300n_kan_periodic.pth"
    torch.save(model.state_dict(), model_path)
    print(f"[MODEL] Modelo salvo em: {model_path}")


if __name__ == "__main__":
    train_kan()
