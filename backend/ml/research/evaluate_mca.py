"""
Avaliação rápida do modelo MCA treinado
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from scalping_mamba_hybrid import ScalpingMasterMCA, ScalpingDataset, ScalpingLabeler

def main():
    print("="*70)
    print("AVALIAÇÃO: ScalpingMaster-MCA")
    print("="*70)

    # Configuração
    data_dir = Path(__file__).parent / "data"
    model_path = Path(__file__).parent / "models" / "best_scalping_mca.pth"

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[DEVICE] {device}")

    # 1. Carregar dados
    print(f"\n[LOAD] Carregando dataset...")
    df = pd.read_csv(data_dir / "1HZ100V_5min_180days_labeled_pessimista.csv")
    print(f"   Dataset: {len(df)} candles")

    # 2. Gerar labels
    print(f"\n[LABEL] Gerando labels...")
    labeler = ScalpingLabeler(df, tp_pct=0.2, sl_pct=0.1, max_candles=20)
    df_labeled = labeler.generate_labels()

    # 3. Criar test dataset
    print(f"\n[DATASET] Criando test dataset...")
    dataset = ScalpingDataset(df_labeled, long_window=100)

    # Split temporal
    train_size = int(0.70 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    test_dataset = torch.utils.data.Subset(dataset, range(train_size + val_size, len(dataset)))
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    print(f"   Test samples: {len(test_dataset)}")

    # 4. Carregar modelo
    print(f"\n[MODEL] Carregando modelo...")
    model = ScalpingMasterMCA(
        input_channels=4,
        hidden_dim=64,
        mamba_state_dim=16,
        short_window=10,
        long_window=100
    ).to(device)

    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"   Modelo carregado: {model_path}")

    # 5. Avaliar
    print(f"\n[EVAL] Avaliando...")
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

    # Accuracy geral
    accuracy = (all_preds == all_labels).sum() / len(all_labels)
    print(f"\n   Accuracy Geral: {accuracy:.2%}")

    # Filtrar apenas trades (LONG/SHORT)
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
        print(f"RESULTADOS FINAIS - MCA")
        print(f"{'='*70}")
        print(f"\n   WIN RATE (LONG+SHORT): {win_rate:.2%}")
        print(f"   LONG Accuracy: {long_accuracy:.2%}")
        print(f"   SHORT Accuracy: {short_accuracy:.2%}")
        print(f"\n   Total trades no test: {len(trade_labels)}")
        print(f"   LONG: {long_mask.sum()} ({long_mask.sum()/len(trade_labels):.1%})")
        print(f"   SHORT: {short_mask.sum()} ({short_mask.sum()/len(trade_labels):.1%})")

        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(trade_labels, trade_preds, labels=[1, 2])
        print(f"\n   Confusion Matrix (LONG=1, SHORT=2):")
        print(f"   Predicted:  LONG  SHORT")
        print(f"   Real LONG:  {cm[0,0]:4d}  {cm[0,1]:4d}  ({cm[0,0]/(cm[0,0]+cm[0,1]):.1%} recall)")
        print(f"   Real SHORT: {cm[1,0]:4d}  {cm[1,1]:4d}  ({cm[1,1]/(cm[1,0]+cm[1,1]):.1%} recall)")

        # Comparação com LSTM
        print(f"\n{'='*70}")
        print(f"COMPARAÇÃO: MCA vs LSTM")
        print(f"{'='*70}")
        print(f"\n   Métrica          | LSTM    | MCA      | Delta")
        print(f"   -----------------+---------+----------+-------")
        print(f"   Win Rate         | 54.3%   | {win_rate:.1%}   | {(win_rate-0.543)*100:+.1f}pp")
        print(f"   LONG Accuracy    | 100.0%  | {long_accuracy:.1%}  | {(long_accuracy-1.0)*100:+.1f}pp")
        print(f"   SHORT Accuracy   | 0.0%    | {short_accuracy:.1%}  | {(short_accuracy-0.0)*100:+.1f}pp")

        # Status
        if win_rate >= 0.60:
            status = "🎯 META ATINGIDA!"
        elif win_rate >= 0.55:
            status = "✅ Viável (próximo da meta)"
        else:
            status = "⚠️ Abaixo da meta"

        print(f"\n   STATUS: {status}")

    print(f"\n{'='*70}")

if __name__ == "__main__":
    main()
