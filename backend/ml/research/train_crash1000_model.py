"""
Treinar modelo LSTM para CRASH1000 M5

CRASH1000 venceu a competição de win rate natural:
- CRASH1000: 40.12%
- CRASH500: 39.69%
- BOOM500: 39.39%

Apesar de < 45% (threshold ideal), CRASH1000 tem:
- 5x mais dados (51,787 vs 9,958 candles)
- Menos volatilidade (movimento médio 0.046% vs 0.074%)
- Padrão mais estável para ML aprender
"""
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Importar classes do modelo original
from crash_tp_before_sl_model import (
    TPBeforeSLDataset,
    LSTMTPBeforeSLModel,
    TPBeforeSLTrainer,
    evaluate_on_test_set
)

def main():
    print("="*70)
    print("CRASH1000 M5 - TP BEFORE SL LSTM CLASSIFIER")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[DEVICE] {device}")

    # 1. Carregar dados CRASH1000
    print(f"\n[DATA] Loading CRASH1000 dataset...")
    data_dir = Path(__file__).parent / "data"
    df = pd.read_csv(data_dir / "CRASH1000_M5_tp_before_sl_labeled.csv")

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
        input_dim=7,
        hidden_dim1=128,
        hidden_dim2=64,
        dropout=0.3
    )

    print(f"\n[MODEL] LSTM Classifier")
    print(f"  Input: 7 features x {lookback} candles")
    print(f"  Architecture: LSTM(7->128->64) + FC(64->32->1)")
    print(f"  Output: Probability [0, 1]")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 5. Train
    trainer = TPBeforeSLTrainer(model, device=device, class_weights=None)
    best_acc = trainer.train(
        train_loader,
        val_loader,
        epochs=100,
        patience=15
    )

    # 6. Load best model
    model_path = Path(__file__).parent / "models" / "crash_tp_before_sl_lstm.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))

    # 7. Evaluate on test set
    print(f"\n{'='*70}")
    print(f"TESTING PHASE")
    print(f"{'='*70}")

    # Teste com múltiplos thresholds
    thresholds = [0.5, 0.6, 0.7]

    for threshold in thresholds:
        print(f"\n\n--- THRESHOLD: {threshold*100:.0f}% ---")
        results = evaluate_on_test_set(model, test_loader, device, threshold=threshold)

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETED")
    print(f"{'='*70}")

    # Salvar modelo renomeado
    final_model_path = Path(__file__).parent / "models" / "crash1000_tp_before_sl_lstm.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"\n[SAVE] Model saved: {final_model_path.name}")

if __name__ == "__main__":
    main()
