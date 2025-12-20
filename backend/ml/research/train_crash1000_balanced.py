"""
CRASH1000 - Treinamento com UNDERSAMPLING (Balanceamento 50/50)

PROBLEMA IDENTIFICADO:
- Dataset desbalanceado: 60% LOSS vs 40% WIN
- Modelo colapsa para classe majoritaria (sempre preve LOSS)
- Accuracy 61% mas TP = 0

SOLUCAO: UNDERSAMPLING
- Remover LOSS aleatorios do TRAIN set ate 50/50
- NUNCA mexer no VAL/TEST (manter distribuicao real)
- Forcar modelo a aprender diferencas reais nas features
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

def undersample_train_set(df_train):
    """
    Aplica undersampling no train set para balancear 50/50

    Args:
        df_train: DataFrame de treino desbalanceado

    Returns:
        df_train_balanced: DataFrame de treino balanceado 50/50
    """
    print(f"\n[UNDERSAMPLING] Balanceando train set...")

    # Separar classes
    df_win = df_train[df_train['tp_before_sl'] == 1]
    df_loss = df_train[df_train['tp_before_sl'] == 0]

    n_win = len(df_win)
    n_loss = len(df_loss)

    print(f"  Original:")
    print(f"    WIN: {n_win:,} ({n_win/(n_win+n_loss)*100:.1f}%)")
    print(f"    LOSS: {n_loss:,} ({n_loss/(n_win+n_loss)*100:.1f}%)")

    # Undersampling da classe majoritaria (LOSS)
    # Escolher aleatoriamente n_win amostras de LOSS
    df_loss_sampled = df_loss.sample(n=n_win, random_state=42)

    # Combinar
    df_balanced = pd.concat([df_win, df_loss_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)

    n_win_new = (df_balanced['tp_before_sl'] == 1).sum()
    n_loss_new = (df_balanced['tp_before_sl'] == 0).sum()

    print(f"\n  Balanceado:")
    print(f"    WIN: {n_win_new:,} ({n_win_new/len(df_balanced)*100:.1f}%)")
    print(f"    LOSS: {n_loss_new:,} ({n_loss_new/len(df_balanced)*100:.1f}%)")
    print(f"    Total: {len(df_balanced):,} (removidos {len(df_train) - len(df_balanced):,} LOSS)")

    return df_balanced


def main():
    print("="*70)
    print("CRASH1000 M5 - LSTM CLASSIFIER COM UNDERSAMPLING")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[DEVICE] {device}")

    # 1. Carregar dados CRASH1000
    print(f"\n[DATA] Loading CRASH1000 dataset...")
    data_dir = Path(__file__).parent / "data"
    df = pd.read_csv(data_dir / "CRASH1000_M5_tp_before_sl_labeled.csv")

    print(f"  Total candles: {len(df):,}")

    # Class distribution (original)
    n_wins = (df['tp_before_sl'] == 1).sum()
    n_losses = (df['tp_before_sl'] == 0).sum()
    print(f"\n  Class distribution (original dataset):")
    print(f"    LOSS (0): {n_losses:,} ({n_losses/len(df)*100:.1f}%)")
    print(f"    WIN (1): {n_wins:,} ({n_wins/len(df)*100:.1f}%)")

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
    df_train_balanced = undersample_train_set(df_train)

    print(f"\n  Split (DEPOIS do undersampling):")
    print(f"    Train: {len(df_train_balanced):,} (50% WIN / 50% LOSS)")
    print(f"    Val:   {len(df_val):,} (distribuicao real mantida)")
    print(f"    Test:  {len(df_test):,} (distribuicao real mantida)")

    # 4. Create datasets
    lookback = 50
    batch_size = 64

    train_dataset = TPBeforeSLDataset(df_train_balanced, lookback=lookback)
    val_dataset = TPBeforeSLDataset(df_val, lookback=lookback)
    test_dataset = TPBeforeSLDataset(df_test, lookback=lookback)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"\n  Lookback: {lookback} candles")
    print(f"  Batch size: {batch_size}")
    print(f"  Train batches: {len(train_loader)} (balanceado 50/50)")
    print(f"  Val batches: {len(val_loader)} (distribuicao real)")
    print(f"  Test batches: {len(test_loader)} (distribuicao real)")

    # 5. Create model
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

    # 6. Train (SEM class weights, pois dataset ja esta balanceado)
    print(f"\n[TRAINING] Treinando com dataset balanceado (50/50)...")
    print(f"  IMPORTANTE: Nao usar class weights (dataset ja balanceado)")

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

    # 8. Evaluate on test set (DESBALANCEADO - distribuicao real!)
    print(f"\n{'='*70}")
    print(f"TESTING PHASE (Test set com distribuicao REAL: 60% LOSS / 40% WIN)")
    print(f"{'='*70}")

    # Teste com multiplos thresholds
    thresholds = [0.5, 0.6, 0.7]

    for threshold in thresholds:
        print(f"\n\n--- THRESHOLD: {threshold*100:.0f}% ---")
        results = evaluate_on_test_set(model, test_loader, device, threshold=threshold)

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETED")
    print(f"{'='*70}")

    # Salvar modelo balanceado
    final_model_path = Path(__file__).parent / "models" / "crash1000_balanced_lstm.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"\n[SAVE] Balanced model saved: {final_model_path.name}")

    # Comparacao com modelo anterior
    print(f"\n{'='*70}")
    print(f"COMPARACAO: ANTES vs DEPOIS DO BALANCEAMENTO")
    print(f"{'='*70}")

    print(f"\nMODELO ANTERIOR (Dataset Desbalanceado):")
    print(f"  Train: 60% LOSS / 40% WIN")
    print(f"  Best Val Acc: 61.01%")
    print(f"  PROBLEMA: TP = 0 (sempre preve LOSS)")
    print(f"  Trades com P(WIN) >= 50%: 0")

    print(f"\nMODELO ATUAL (Dataset Balanceado 50/50):")
    print(f"  Train: 50% LOSS / 50% WIN (undersampling)")
    print(f"  Best Val Acc: {best_acc:.4f}")
    print(f"  Resultado: Verificar acima")

if __name__ == "__main__":
    main()
