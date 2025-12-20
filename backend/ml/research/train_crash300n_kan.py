"""
CRASH300N - KAN (Kolmogorov-Arnold Networks) - O Experimento Final

Hipotese: O algoritmo da Deriv usa PRNG (Pseudo-Random Number Generator)
         Se o PRNG for fraco, os intervalos entre crashes tem relacao funcional:
         t_4 = f(t_1, t_2, t_3)

Estrategia: Usar KAN (2024) para descobrir a formula matematica
           que governa os intervalos entre crashes

Se KAN descobrir funcao: PRNG e fraco (EXPLORAVEL!)
Se KAN falhar (erro = baseline): CSPRNG ou hardware RNG (IMPOSSIVEL)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from kan import KAN
import torch


def extract_crash_intervals(df, crash_threshold=-0.005):
    """
    Extrai sequencia de intervalos entre crashes (em candles)

    Returns:
        intervals: array de intervalos (candles_between_crashes)
    """
    print("\n[EXTRACT] Detectando crashes e calculando intervalos...")

    # Definir crash
    df['log_ret'] = np.log(df['close'] / df['open'])
    df['is_crash'] = (df['log_ret'] <= crash_threshold).astype(int)

    # Indices dos crashes
    crash_indices = df[df['is_crash'] == 1].index.tolist()

    # Calcular intervalos
    intervals = []
    for i in range(1, len(crash_indices)):
        interval = crash_indices[i] - crash_indices[i-1]
        intervals.append(interval)

    intervals = np.array(intervals)

    print(f"  Total crashes detectados: {len(crash_indices):,}")
    print(f"  Total intervalos: {len(intervals):,}")
    print(f"  Intervalo minimo: {intervals.min()} candles")
    print(f"  Intervalo maximo: {intervals.max()} candles")
    print(f"  Intervalo medio: {intervals.mean():.1f} candles")
    print(f"  Std: {intervals.std():.1f} candles")

    return intervals


def create_interval_sequences(intervals, lookback=3):
    """
    Cria sequencias [t_{n-3}, t_{n-2}, t_{n-1}] -> t_n

    Returns:
        X: features (last N intervals)
        y: target (next interval)
    """
    print(f"\n[SEQUENCES] Criando sequencias com lookback={lookback}...")

    X = []
    y = []

    for i in range(lookback, len(intervals)):
        X.append(intervals[i-lookback:i])
        y.append(intervals[i])

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)

    print(f"  Total sequences: {len(X):,}")
    print(f"  Shape X: {X.shape}")
    print(f"  Shape y: {y.shape}")

    return X, y


def train_kan_model(X, y):
    """
    Treina KAN para descobrir funcao matematica t_n = f(t_{n-3}, t_{n-2}, t_{n-1})
    """
    print("\n" + "="*70)
    print("KAN - SYMBOLIC REGRESSION (Descoberta de Formula)")
    print("="*70)

    # Split temporal (70/15/15)
    train_size = int(0.70 * len(X))
    val_size = int(0.15 * len(X))

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]

    print(f"\n[SPLIT]")
    print(f"  Train: {len(X_train):,} sequences")
    print(f"  Val:   {len(X_val):,} sequences")
    print(f"  Test:  {len(X_test):,} sequences")

    # Normalizar para [0, 1] (KAN funciona melhor normalizado)
    x_min = X_train.min()
    x_max = X_train.max()
    y_min = y_train.min()
    y_max = y_train.max()

    X_train_norm = (X_train - x_min) / (x_max - x_min)
    X_val_norm = (X_val - x_min) / (x_max - x_min)
    X_test_norm = (X_test - x_min) / (x_max - x_min)

    y_train_norm = (y_train - y_min) / (y_max - y_min)
    y_val_norm = (y_val - y_min) / (y_max - y_min)
    y_test_norm = (y_test - y_min) / (y_max - y_min)

    # Convert to torch tensors
    X_train_t = torch.FloatTensor(X_train_norm)
    y_train_t = torch.FloatTensor(y_train_norm)
    X_val_t = torch.FloatTensor(X_val_norm)
    y_val_t = torch.FloatTensor(y_val_norm)
    X_test_t = torch.FloatTensor(X_test_norm)

    # Dataset dict for KAN
    dataset = {
        'train_input': X_train_t,
        'train_label': y_train_t,
        'test_input': X_val_t,
        'test_label': y_val_t
    }

    # Create KAN model
    # Architecture: [3, 5, 1] - 3 inputs, 5 hidden nodes, 1 output
    print(f"\n[MODEL] Criando KAN...")
    print(f"  Architecture: [3, 5, 1]")
    print(f"  - Input: 3 (t_{{n-3}}, t_{{n-2}}, t_{{n-1}})")
    print(f"  - Hidden: 5 nodes")
    print(f"  - Output: 1 (t_n)")

    model = KAN(width=[3, 5, 1], grid=5, k=3, seed=42)

    # Train
    print(f"\n[TRAINING] Treinando KAN (100 epochs)...")
    model.fit(dataset, opt="LBFGS", steps=100, lamb=0.01, lamb_entropy=2.0)

    # Predict on test set
    y_pred_norm = model(X_test_t)

    # Denormalize
    y_pred = y_pred_norm.detach().numpy() * (y_max - y_min) + y_min

    # Metrics
    mae = np.mean(np.abs(y_test - y_pred))
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

    # Baseline (always predict mean)
    baseline_pred = np.full_like(y_test, y_train.mean())
    baseline_mae = np.mean(np.abs(y_test - baseline_pred))
    baseline_rmse = np.sqrt(np.mean((y_test - baseline_pred) ** 2))

    print(f"\n[RESULTS] Test Set Performance")
    print(f"  KAN MAE:       {mae:.2f} candles")
    print(f"  KAN RMSE:      {rmse:.2f} candles")
    print(f"\n[BASELINE] Always Predict Mean ({y_train.mean():.1f})")
    print(f"  Baseline MAE:  {baseline_mae:.2f} candles")
    print(f"  Baseline RMSE: {baseline_rmse:.2f} candles")

    # Improvement
    improvement_mae = (baseline_mae - mae) / baseline_mae * 100
    improvement_rmse = (baseline_rmse - rmse) / baseline_rmse * 100

    print(f"\n[IMPROVEMENT]")
    print(f"  MAE:  {improvement_mae:+.2f}%")
    print(f"  RMSE: {improvement_rmse:+.2f}%")

    # Plot predictions vs actual
    plt.figure(figsize=(14, 6))

    # Plot 1: Predictions vs Actual
    plt.subplot(1, 2, 1)
    test_indices = range(len(y_test))
    plt.plot(test_indices[:200], y_test[:200], 'b-', linewidth=2, alpha=0.7, label='Actual')
    plt.plot(test_indices[:200], y_pred[:200], 'r--', linewidth=1.5, alpha=0.7, label='KAN Prediction')
    plt.axhline(y=y_train.mean(), color='g', linestyle=':', linewidth=1, label=f'Baseline (Mean={y_train.mean():.1f})')
    plt.xlabel('Sequence Index', fontsize=12)
    plt.ylabel('Interval (candles)', fontsize=12)
    plt.title('KAN Predictions vs Actual (First 200 sequences)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Residuals
    plt.subplot(1, 2, 2)
    residuals = y_test.flatten() - y_pred.flatten()
    plt.hist(residuals, bins=50, alpha=0.7, color='purple', edgecolor='black')
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    plt.xlabel('Residuals (Actual - Predicted)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Residual Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = Path(__file__).parent / "reports" / "crash300n_kan_predictions.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\n[PLOT] Grafico salvo em: {plot_path}")

    # Veredicto
    print(f"\n{'='*70}")
    print("VEREDICTO FINAL")
    print(f"{'='*70}\n")

    # Criterio de decisao: KAN precisa melhorar pelo menos 10% vs baseline
    if improvement_mae > 10.0 and improvement_rmse > 10.0:
        print(f"SUCESSO! PRNG FRACO DETECTADO!")
        print(f"  KAN descobriu padroes nos intervalos!")
        print(f"  Melhoria: MAE {improvement_mae:.1f}%, RMSE {improvement_rmse:.1f}%")
        print(f"\n  INTERPRETACAO:")
        print(f"  >> O algoritmo da Deriv usa PRNG fraco")
        print(f"  >> Existe relacao funcional: t_n = f(t_{{n-3}}, t_{{n-2}}, t_{{n-1}})")
        print(f"  >> Crash timing e PREVISIVEL (com margem de erro)")
        print(f"\n  PROXIMA ACAO:")
        print(f"  >> Usar KAN para prever quando entrar LONG")
        print(f"  >> Entrar apenas quando KAN prever intervalo longo (>50 candles)")
        print(f"  >> Expectativa: Win rate >50% com TP 2%")

        # Try to extract symbolic formula
        print(f"\n[SYMBOLIC] Tentando extrair formula matematica...")
        try:
            formula = model.symbolic_formula()
            print(f"  Formula descoberta:")
            print(f"  {formula}")
        except:
            print(f"  (Formula muito complexa para extrair simbolicamente)")

    elif improvement_mae > 5.0 or improvement_rmse > 5.0:
        print(f"EDGE FRACO DETECTADO")
        print(f"  KAN teve melhoria marginal ({improvement_mae:.1f}% MAE, {improvement_rmse:.1f}% RMSE)")
        print(f"  Existe algum padrao, mas fraco demais para explorar")
        print(f"\n  CONCLUSAO:")
        print(f"  >> PRNG e relativamente robusto")
        print(f"  >> Edge existe mas e pequeno demais para trading lucrativo")

    else:
        print(f"FALHOU - CSPRNG OU HARDWARE RNG")
        print(f"  KAN NAO melhorou vs baseline ({improvement_mae:.1f}% MAE, {improvement_rmse:.1f}% RMSE)")
        print(f"  Intervalos sao verdadeiramente aleatorios")
        print(f"\n  CONCLUSAO FINAL:")
        print(f"  >> Deriv usa CSPRNG (Cryptographically Secure) OU hardware RNG")
        print(f"  >> Nao existe relacao funcional entre intervalos")
        print(f"  >> Processo e matematicamente IMPREVISIVEL")
        print(f"  >> ML scalping em CRASH300N e IMPOSSIVEL")
        print(f"\n  RECOMENDACAO:")
        print(f"  >> Migrar para Forex/Indices reais")
        print(f"  >> Ativos sinteticos da Deriv sao aleatorios por design")

    print(f"\n{'='*70}\n")

    # Save model
    model_path = Path(__file__).parent / "models" / "crash300n_kan.pth"
    torch.save(model.state_dict(), model_path)
    print(f"[MODEL] Modelo KAN salvo em: {model_path}")

    return model, mae, rmse, baseline_mae, baseline_rmse, improvement_mae


def main():
    print("="*70)
    print("CRASH300N - KAN (O EXPERIMENTO FINAL)")
    print("="*70)
    print("\nHipotese: Deriv usa PRNG fraco, intervalos tem relacao funcional")
    print("Objetivo: Descobrir formula t_n = f(t_{n-3}, t_{n-2}, t_{n-1})")

    # Load data
    data_dir = Path(__file__).parent / "data"
    files = list(data_dir.glob("*CRASH300*.csv"))
    if not files:
        print("\nERRO: Nenhum arquivo CRASH300N encontrado")
        return

    input_file = files[0]
    print(f"\n[LOAD] {input_file.name}")
    df = pd.read_csv(input_file)

    # Sort temporal
    if 'epoch' in df.columns:
        df = df.sort_values('epoch').reset_index(drop=True)
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"  Total candles: {len(df):,}")

    # Step 1: Extract crash intervals
    intervals = extract_crash_intervals(df, crash_threshold=-0.005)

    # Step 2: Create sequences
    X, y = create_interval_sequences(intervals, lookback=3)

    # Step 3: Train KAN
    model, mae, rmse, baseline_mae, baseline_rmse, improvement = train_kan_model(X, y)

    # Final summary
    print("\n" + "="*70)
    print("RESUMO DA JORNADA COMPLETA")
    print("="*70)
    print("\nAbordagens Testadas:")
    print("  1. TP-Before-SL (CRASH1000 M5)      >> Win Rate: 40.12%  [REPROVADO]")
    print("  2. Undersampling 50/50 (CRASH1000)  >> Colapsou          [REPROVADO]")
    print("  3. TP Reduzido 0.5% (CRASH1000)     >> Win Rate: 34.37%  [REPROVADO]")
    print("  4. Survival LSTM (CRASH500)         >> 7 crashes          [IMPOSSIVEL]")
    print("  5. Survival LSTM (CRASH300N)        >> Prob constante     [REPROVADO]")
    print("  6. Hazard Analysis (CRASH300N)      >> P-value: 0.8448    [INCERTO]")
    print("  7. XGBoost Non-Linear (CRASH300N)   >> AUC: 0.5012        [REPROVADO]")
    print(f"  8. KAN Symbolic Regression          >> Melhoria: {improvement:.1f}%   [?]")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
