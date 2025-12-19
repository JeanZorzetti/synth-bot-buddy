"""
Modelo LSTM para Scalping V100 M5

MUDANÇA DE PARADIGMA:
- ML tradicional (XGBoost): 50.5% win rate com 88 features
- Deep Learning (LSTM): Aprende sequencias de candles (sem feature engineering)

ARQUITETURA:
Input: [batch_size, 50 candles, 4 features (OHLC)]
↓
LSTM Layer 1 (128 units, return_sequences=True)
↓
Dropout (0.3)
↓
LSTM Layer 2 (64 units)
↓
Dropout (0.3)
↓
Dense (32 units, ReLU)
↓
Output (3 units, Softmax) → [NO_TRADE, LONG, SHORT]

EXPECTATIVA:
Win rate: 58-65% (baseado em literatura de forex/synthetic indices)

Autor: Claude Sonnet 4.5
Data: 18/12/2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Import labeling
from scalping_labeling import ScalpingLabeler


class LSTMScalpingModel:
    """
    Modelo LSTM para previsao de sinais de scalping
    """

    def __init__(self, lookback=50, learning_rate=0.001):
        """
        Args:
            lookback: Numero de candles para lookback (50 = 250 min de M5)
            learning_rate: Taxa de aprendizado inicial
        """
        self.lookback = lookback
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = StandardScaler()
        self.history = None

    def create_sequences(self, df, label_col='label'):
        """
        Converte DataFrame de candles em sequencias para LSTM

        Args:
            df: DataFrame com OHLC e labels
            label_col: Nome da coluna de labels

        Returns:
            X: Array [samples, lookback, 4] (OHLC normalizado)
            y: Array [samples, 3] (one-hot encoded labels)
        """
        print(f"\n[SEQUENCE] Criando sequencias de {self.lookback} candles...")

        # Extrair OHLC
        ohlc_cols = ['open', 'high', 'low', 'close']
        ohlc_data = df[ohlc_cols].values
        labels = df[label_col].values

        # Normalizar OHLC (por candle, usando close como referencia)
        # Isso preserva a estrutura de cada candle
        normalized_ohlc = np.zeros_like(ohlc_data)
        for i in range(len(ohlc_data)):
            close = ohlc_data[i, 3]  # close price
            if close > 0:
                normalized_ohlc[i] = (ohlc_data[i] - close) / close * 100  # % change

        # Criar sequencias
        X = []
        y = []

        for i in range(self.lookback, len(normalized_ohlc)):
            # Sequencia de lookback candles
            X.append(normalized_ohlc[i-self.lookback:i])
            # Label do candle atual
            y.append(labels[i])

        X = np.array(X)
        y = np.array(y)

        # Converter labels: -1 (SHORT) -> 2, manter 0 (NO_TRADE) e 1 (LONG)
        y = np.where(y == -1, 2, y)

        # One-hot encode labels (NO_TRADE=0, LONG=1, SHORT=2)
        y_categorical = to_categorical(y, num_classes=3)

        print(f"   Sequencias criadas: {len(X)}")
        print(f"   Shape X: {X.shape}")  # [samples, lookback, 4]
        print(f"   Shape y: {y_categorical.shape}")  # [samples, 3]

        # Distribuicao de labels
        unique, counts = np.unique(y, return_counts=True)
        print(f"\n   Distribuicao de labels:")
        label_names = {-1: 'SHORT', 0: 'NO_TRADE', 1: 'LONG', 2: 'SHORT'}
        for label, count in zip(unique, counts):
            label_int = int(label)
            label_name = label_names.get(label_int, f'UNKNOWN({label_int})')
            print(f"      {label_name}: {count} ({count/len(y)*100:.1f}%)")

        return X, y_categorical

    def build_model(self):
        """
        Constroi arquitetura LSTM

        Returns:
            Compiled Keras model
        """
        print(f"\n[BUILD] Construindo arquitetura LSTM...")

        model = Sequential([
            # LSTM Layer 1 (128 units, return sequences for next LSTM)
            LSTM(128, return_sequences=True, input_shape=(self.lookback, 4)),
            BatchNormalization(),
            Dropout(0.3),

            # LSTM Layer 2 (64 units)
            LSTM(64),
            BatchNormalization(),
            Dropout(0.3),

            # Dense layers
            Dense(32, activation='relu'),
            Dropout(0.2),

            # Output layer (3 classes: NO_TRADE, LONG, SHORT)
            Dense(3, activation='softmax')
        ])

        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        print(f"\n   Total parametros: {model.count_params():,}")

        # Print architecture
        model.summary()

        self.model = model
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=256):
        """
        Treina modelo LSTM

        Args:
            X_train, y_train: Dados de treino
            X_val, y_val: Dados de validacao
            epochs: Maximo de epocas
            batch_size: Tamanho do batch

        Returns:
            History object
        """
        print(f"\n[TRAIN] Iniciando treinamento...")
        print(f"   Epocas: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {self.learning_rate}")

        # Callbacks
        callbacks = [
            # Early stopping (para se val_loss nao melhorar por 10 epocas)
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),

            # Reduce learning rate quando val_loss parar de melhorar
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),

            # Save best model
            ModelCheckpoint(
                'models/best_lstm_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]

        # Train
        start_time = time.time()

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        elapsed_time = time.time() - start_time
        print(f"\n   Tempo de treinamento: {elapsed_time/60:.1f} min")

        self.history = history
        return history

    def evaluate(self, X_test, y_test):
        """
        Avalia modelo em test set

        Args:
            X_test, y_test: Dados de teste

        Returns:
            Dictionary com metricas
        """
        print(f"\n[EVAL] Avaliando modelo em Test Set...")

        # Predict
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)

        # Filtrar apenas trades (LONG/SHORT, excluir NO_TRADE)
        tradeable_mask = y_true != 0
        y_true_tradeable = y_true[tradeable_mask]
        y_pred_tradeable = y_pred[tradeable_mask]

        # Metricas gerais
        accuracy = np.mean(y_pred == y_true)

        # Metricas de trading (apenas LONG/SHORT)
        if len(y_true_tradeable) > 0:
            accuracy_tradeable = np.mean(y_pred_tradeable == y_true_tradeable)

            # Separar LONG vs SHORT
            long_mask = y_true_tradeable == 1
            short_mask = y_true_tradeable == 2

            long_accuracy = np.mean(y_pred_tradeable[long_mask] == y_true_tradeable[long_mask]) if long_mask.sum() > 0 else 0
            short_accuracy = np.mean(y_pred_tradeable[short_mask] == y_true_tradeable[short_mask]) if short_mask.sum() > 0 else 0

            # Confusion matrix (apenas LONG/SHORT)
            from sklearn.metrics import confusion_matrix, classification_report
            cm = confusion_matrix(y_true_tradeable, y_pred_tradeable, labels=[1, 2])

            print(f"\n   Accuracy geral: {accuracy*100:.2f}%")
            print(f"   Accuracy tradeable (LONG/SHORT): {accuracy_tradeable*100:.2f}%")
            print(f"   LONG accuracy: {long_accuracy*100:.2f}%")
            print(f"   SHORT accuracy: {short_accuracy*100:.2f}%")

            print(f"\n   Confusion Matrix (LONG/SHORT):")
            print(f"   Predicted:    LONG    SHORT")
            print(f"   Real LONG:    {cm[0,0]:<7} {cm[0,1]:<7} = {cm[0,0]/(cm[0,0]+cm[0,1])*100:.1f}% acerto")
            print(f"   Real SHORT:   {cm[1,0]:<7} {cm[1,1]:<7} = {cm[1,1]/(cm[1,0]+cm[1,1])*100:.1f}% acerto")

            # Classification report
            print(f"\n   Classification Report (LONG/SHORT):")
            report = classification_report(y_true_tradeable, y_pred_tradeable,
                                          target_names=['LONG', 'SHORT'],
                                          digits=3)
            print(report)

        else:
            accuracy_tradeable = 0
            long_accuracy = 0
            short_accuracy = 0

        # Retornar metricas
        metrics = {
            'accuracy': float(accuracy),
            'win_rate': float(accuracy_tradeable),  # Win rate = accuracy em trades
            'long_accuracy': float(long_accuracy),
            'short_accuracy': float(short_accuracy),
            'total_samples': int(len(y_test)),
            'tradeable_samples': int(len(y_true_tradeable))
        }

        return metrics

    def plot_training_history(self, save_path='reports/lstm_training_history.png'):
        """
        Plota curvas de treino/validacao
        """
        if self.history is None:
            print("[WARN] Nenhum historico de treino disponivel")
            return

        print(f"\n[PLOT] Salvando graficos de treino...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss
        ax1.plot(self.history.history['loss'], label='Train Loss')
        ax1.plot(self.history.history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        # Accuracy
        ax2.plot(self.history.history['accuracy'], label='Train Accuracy')
        ax2.plot(self.history.history['val_accuracy'], label='Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   Grafico salvo: {save_path}")
        plt.close()


def main():
    """
    Pipeline completo de treinamento LSTM
    """
    print("="*70)
    print("TREINAMENTO LSTM - SCALPING V100 M5")
    print("="*70)

    data_dir = Path(__file__).parent / "data"
    models_dir = Path(__file__).parent / "models"
    reports_dir = Path(__file__).parent / "reports"

    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    # 1. Carregar dataset base
    base_features_path = data_dir / "1HZ100V_5min_180days.csv"

    if not base_features_path.exists():
        print(f"\n[ERRO] Dataset base nao encontrado: {base_features_path}")
        return

    print(f"\n[LOAD] Carregando dataset: {base_features_path.name}")
    df = pd.read_csv(base_features_path)
    print(f"   Dataset: {len(df)} candles")

    # 2. Gerar labels (TP 0.2% / SL 0.1%)
    print("\n[LABEL] Gerando labels LONG/SHORT/NO_TRADE...")
    print("   Configuracao: TP 0.2% / SL 0.1% / R:R 1:2")

    labeler = ScalpingLabeler(
        df,
        tp_pct=0.2,
        sl_pct=0.1,
        max_candles=20
    )
    df_labeled = labeler.generate_labels()

    # Estatisticas de labels
    label_dist = df_labeled['label'].value_counts()
    print(f"\n   Distribuicao de labels:")
    print(f"      NO_TRADE (0): {label_dist.get(0, 0)} ({label_dist.get(0, 0)/len(df_labeled)*100:.1f}%)")
    print(f"      LONG (1): {label_dist.get(1, 0)} ({label_dist.get(1, 0)/len(df_labeled)*100:.1f}%)")
    print(f"      SHORT (2): {label_dist.get(2, 0)} ({label_dist.get(2, 0)/len(df_labeled)*100:.1f}%)")

    # 3. Criar modelo LSTM
    lstm_model = LSTMScalpingModel(lookback=50, learning_rate=0.001)

    # 4. Criar sequencias
    X, y = lstm_model.create_sequences(df_labeled)

    # 5. Split temporal (70% train, 15% val, 15% test)
    # IMPORTANTE: NAO shuffle (manter ordem temporal)
    train_size = int(0.70 * len(X))
    val_size = int(0.15 * len(X))

    X_train = X[:train_size]
    y_train = y[:train_size]

    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]

    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]

    print(f"\n[SPLIT] Divisao temporal:")
    print(f"   Train: {len(X_train)} sequencias ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Val: {len(X_val)} sequencias ({len(X_val)/len(X)*100:.1f}%)")
    print(f"   Test: {len(X_test)} sequencias ({len(X_test)/len(X)*100:.1f}%)")

    # 6. Build e treinar modelo
    lstm_model.build_model()

    history = lstm_model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=100,
        batch_size=256
    )

    # 7. Avaliar em test set
    metrics = lstm_model.evaluate(X_test, y_test)

    # 8. Comparar com baseline XGBoost
    print("\n" + "="*70)
    print("COMPARACAO LSTM vs XGBoost")
    print("="*70)

    xgboost_baseline = 0.509
    xgboost_advanced = 0.505
    lstm_win_rate = metrics['win_rate']

    improvement_vs_baseline = (lstm_win_rate - xgboost_baseline) * 100
    improvement_vs_advanced = (lstm_win_rate - xgboost_advanced) * 100

    print(f"\nXGBoost Baseline (62 features):")
    print(f"   Win rate: {xgboost_baseline*100:.1f}%")
    print(f"\nXGBoost Advanced (88 features):")
    print(f"   Win rate: {xgboost_advanced*100:.1f}%")
    print(f"\nLSTM (apenas OHLC, 50 candles lookback):")
    print(f"   Win rate: {lstm_win_rate*100:.1f}%")
    print(f"   Melhoria vs XGBoost baseline: {improvement_vs_baseline:+.1f}pp")
    print(f"   Melhoria vs XGBoost advanced: {improvement_vs_advanced:+.1f}pp")

    # 9. Verificar se atingiu meta
    print("\n" + "="*70)
    print("RESULTADO FINAL")
    print("="*70)

    meta_atingida = lstm_win_rate >= 0.60

    if meta_atingida:
        print(f"\n[SUCESSO] META DE 60% ATINGIDA!")
        print(f"   Win rate: {lstm_win_rate*100:.1f}%")
        print(f"   Status: PRONTO PARA BACKTESTING")
        print(f"\nProximo passo: Backtesting completo (3 meses out-of-sample)")
    else:
        gap_to_goal = (0.60 - lstm_win_rate) * 100
        print(f"\n[AVISO] Meta nao atingida")
        print(f"   Win rate: {lstm_win_rate*100:.1f}%")
        print(f"   Faltam: {gap_to_goal:.1f}pp para 60%")

        if lstm_win_rate >= 0.55:
            print(f"\n   Status: PARCIAL (55-60%)")
            print(f"   Proximos passos:")
            print(f"   1. Testar em backtesting mesmo assim")
            print(f"   2. Se backtest OK, considerar forward testing")
            print(f"   3. Ou testar arquitetura Transformer (melhor que LSTM)")
        elif lstm_win_rate >= 0.52:
            print(f"\n   Status: MELHORIA MODERADA (52-55%)")
            print(f"   Proximos passos:")
            print(f"   1. Testar arquitetura Transformer")
            print(f"   2. Adicionar attention mechanism ao LSTM")
            print(f"   3. Testar M15/M30 (timeframes mais estaveis)")
        else:
            print(f"\n   Status: SEM MELHORIA SIGNIFICATIVA (<52%)")
            print(f"   Proximos passos:")
            print(f"   1. LSTM nao funcionou, considerar Transformer")
            print(f"   2. Testar BOOM/CRASH (padroes mais claros)")
            print(f"   3. Reavaliar se scalping e viavel em V100 M5")

    # 10. Plot training curves
    lstm_model.plot_training_history()

    # 11. Salvar resultados
    print("\n[SAVE] Salvando resultados...")

    results = {
        'experiment': 'lstm_scalping',
        'model': 'LSTM',
        'architecture': '128-64-32-3 (2 LSTM layers + Dense)',
        'lookback': 50,
        'learning_rate': 0.001,
        'batch_size': 256,
        'epochs_trained': len(history.history['loss']),
        'tp_pct': 0.2,
        'sl_pct': 0.1,
        'win_rate': float(metrics['win_rate']),
        'accuracy': float(metrics['accuracy']),
        'long_accuracy': float(metrics['long_accuracy']),
        'short_accuracy': float(metrics['short_accuracy']),
        'improvement_vs_xgboost_baseline': float(improvement_vs_baseline),
        'improvement_vs_xgboost_advanced': float(improvement_vs_advanced),
        'goal_achieved': meta_atingida,
        'total_parameters': int(lstm_model.model.count_params()),
        'timestamp': pd.Timestamp.now().isoformat()
    }

    results_path = reports_dir / 'lstm_scalping_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   Resultados salvos: {results_path}")

    print("\n" + "="*70)
    print("TREINAMENTO LSTM CONCLUIDO!")
    print("="*70)


if __name__ == "__main__":
    main()
