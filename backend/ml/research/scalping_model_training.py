"""
Treinamento de Modelo XGBoost para Scalping

Este script treina um modelo XGBoost otimizado com Optuna para prever
setups de scalping LONG/SHORT/NO_TRADE.

Meta: Win rate > 60% para classes LONG e SHORT

Autor: Claude Sonnet 4.5
Data: 18/12/2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import joblib
import json
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class ScalpingModelTrainer:
    """
    Treinador de modelo XGBoost para scalping
    """

    def __init__(self, df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.2):
        """
        Args:
            df: DataFrame com features + label
            test_size: % para test set (out-of-sample final)
            val_size: % para validation set (otimização)
        """
        self.df = df.copy()
        self.test_size = test_size
        self.val_size = val_size

        # Separar features e target
        self.feature_cols = [col for col in df.columns
                            if col not in ['label', 'timestamp', 'epoch', 'label_metadata']]

        print(f"[OK] Trainer inicializado:")
        print(f"   - Total amostras: {len(df)}")
        print(f"   - Features: {len(self.feature_cols)}")
        print(f"   - Test size: {test_size*100:.0f}%")
        print(f"   - Val size: {val_size*100:.0f}%")

    def prepare_data(self) -> Tuple:
        """
        Prepara dados com split temporal

        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print("\n[DATA] Preparando dados...")

        # Remover NaN se houver
        df_clean = self.df.dropna()
        print(f"   - Linhas após dropna: {len(df_clean)}")

        # Extrair features e labels
        X = df_clean[self.feature_cols].values
        y = df_clean['label'].values

        # Converter labels (-1, 0, 1) para (0, 1, 2) para XGBoost
        # 0 (NO_TRADE) -> 0
        # 1 (LONG) -> 1
        # -1 (SHORT) -> 2
        y_mapped = np.where(y == -1, 2, y)

        # Split temporal (não shuffling!)
        n_samples = len(X)
        test_split = int(n_samples * (1 - self.test_size))
        val_split = int(test_split * (1 - self.val_size))

        X_train = X[:val_split]
        X_val = X[val_split:test_split]
        X_test = X[test_split:]

        y_train = y_mapped[:val_split]
        y_val = y_mapped[val_split:test_split]
        y_test = y_mapped[test_split:]

        print(f"\n[SPLIT]")
        print(f"   Train: {len(X_train)} ({len(X_train)/n_samples*100:.1f}%)")
        print(f"   Val:   {len(X_val)} ({len(X_val)/n_samples*100:.1f}%)")
        print(f"   Test:  {len(X_test)} ({len(X_test)/n_samples*100:.1f}%)")

        # Verificar distribuição de classes
        print(f"\n[DISTRIBUICAO Train]")
        unique, counts = np.unique(y_train, return_counts=True)
        for label, count in zip(unique, counts):
            label_name = {0: 'NO_TRADE', 1: 'LONG', 2: 'SHORT'}[label]
            print(f"   {label_name}: {count} ({count/len(y_train)*100:.1f}%)")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val, n_trials=50):
        """
        Otimiza hiperparâmetros com Optuna

        Args:
            n_trials: Número de tentativas

        Returns:
            best_params: Melhores hiperparâmetros encontrados
        """
        print(f"\n[OPTUNA] Otimizando hiperparâmetros ({n_trials} trials)...")

        def objective(trial):
            params = {
                'objective': 'multi:softmax',
                'num_class': 3,
                'eval_metric': 'mlogloss',
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
                'random_state': 42,
                'tree_method': 'hist',  # Mais rápido
                'verbosity': 0
            }

            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, verbose=False)

            y_pred = model.predict(X_val)

            # Otimizar F1-score macro para classes 1 e 2 (LONG e SHORT)
            # Ignorar classe 0 (NO_TRADE)
            mask = y_val != 0  # Apenas LONG e SHORT
            if mask.sum() == 0:
                return 0

            f1 = f1_score(y_val[mask], y_pred[mask], average='macro', labels=[1, 2])

            return f1

        study = optuna.create_study(direction='maximize', study_name='scalping_xgboost')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        print(f"\n[OPTUNA] Otimização concluída!")
        print(f"   - Melhor F1-score: {study.best_value:.4f}")
        print(f"   - Melhores params:")
        for key, value in study.best_params.items():
            print(f"      {key}: {value}")

        return study.best_params

    def train_final_model(self, X_train, y_train, X_val, y_val, params):
        """
        Treina modelo final com melhores hiperparâmetros

        Returns:
            model: Modelo treinado
        """
        print(f"\n[TRAIN] Treinando modelo final...")

        # Adicionar parâmetros fixos
        full_params = {
            'objective': 'multi:softmax',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'random_state': 42,
            'tree_method': 'hist',
            'verbosity': 1,
            **params
        }

        model = xgb.XGBClassifier(**full_params)

        # Treinar com early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=True
        )

        print(f"[OK] Modelo treinado!")

        return model

    def evaluate_model(self, model, X, y, split_name='Test'):
        """
        Avalia modelo e imprime métricas

        Returns:
            metrics: Dict com métricas
        """
        print(f"\n[EVAL] Avaliando em {split_name} set...")

        y_pred = model.predict(X)

        # Accuracy geral
        acc = accuracy_score(y, y_pred)
        print(f"   Accuracy geral: {acc:.4f}")

        # Classification report
        target_names = ['NO_TRADE', 'LONG', 'SHORT']
        print(f"\n{classification_report(y, y_pred, target_names=target_names, digits=4)}")

        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)

        # Métricas específicas para LONG e SHORT (ignorar NO_TRADE)
        mask_tradeable = y != 0
        if mask_tradeable.sum() > 0:
            y_tradeable = y[mask_tradeable]
            y_pred_tradeable = y_pred[mask_tradeable]

            acc_tradeable = accuracy_score(y_tradeable, y_pred_tradeable)
            f1_tradeable = f1_score(y_tradeable, y_pred_tradeable, average='macro', labels=[1, 2])

            print(f"\n[METRICAS TRADEABLE] (apenas LONG/SHORT)")
            print(f"   Accuracy: {acc_tradeable:.4f}")
            print(f"   F1-score: {f1_tradeable:.4f}")

            # Win rate simulado
            # Assumindo que modelo prevê corretamente → win
            # Modelo prevê errado → loss
            # Modelo prevê NO_TRADE quando deveria tradear → miss
            correct = (y_tradeable == y_pred_tradeable).sum()
            win_rate = correct / len(y_tradeable)
            print(f"   Win rate estimado: {win_rate:.4f} ({win_rate*100:.1f}%)")

            if win_rate >= 0.60:
                print(f"   [OK] WIN RATE > 60%! META ATINGIDA!")
            else:
                print(f"   [AVISO] Win rate {win_rate*100:.1f}% < 60% (meta)")

        return {
            'accuracy': acc,
            'accuracy_tradeable': acc_tradeable if mask_tradeable.sum() > 0 else 0,
            'f1_tradeable': f1_tradeable if mask_tradeable.sum() > 0 else 0,
            'win_rate': win_rate if mask_tradeable.sum() > 0 else 0,
            'confusion_matrix': cm
        }

    def plot_feature_importance(self, model, output_path=None):
        """Plota importância de features"""
        print(f"\n[PLOT] Gerando gráfico de feature importance...")

        importance = model.feature_importances_
        feature_names = self.feature_cols

        # Pegar top 20
        indices = np.argsort(importance)[::-1][:20]
        top_features = [feature_names[i] for i in indices]
        top_importance = importance[indices]

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_importance)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Importance')
        plt.title('Top 20 Features mais Importantes')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path)
            print(f"[SAVE] Feature importance salvo em {output_path}")
        else:
            plt.show()

    def save_model(self, model, params, metrics, output_dir):
        """Salva modelo e metadados"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Salvar modelo
        model_path = output_dir / 'scalping_xgboost_model.pkl'
        joblib.dump(model, model_path)
        print(f"\n[SAVE] Modelo salvo em {model_path}")

        # Salvar hiperparâmetros
        params_path = output_dir / 'model_params.json'
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=2)
        print(f"[SAVE] Params salvos em {params_path}")

        # Salvar métricas
        metrics_serializable = {k: float(v) if not isinstance(v, np.ndarray) else v.tolist()
                               for k, v in metrics.items()}
        metrics_path = output_dir / 'model_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        print(f"[SAVE] Métricas salvas em {metrics_path}")


def train_scalping_model(symbol='1HZ100V', timeframe='5min', n_trials=50):
    """
    Pipeline completo de treinamento

    Args:
        symbol: Símbolo a treinar
        timeframe: Timeframe
        n_trials: Número de trials Optuna
    """
    print(f"\n{'='*70}")
    print(f"TREINAMENTO MODELO SCALPING: {symbol} ({timeframe})")
    print(f"{'='*70}")

    # Carregar dados com labels
    data_dir = Path(__file__).parent / "data"
    labeled_path = data_dir / f"{symbol}_{timeframe}_180days_labeled.csv"

    if not labeled_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {labeled_path}")

    df = pd.read_csv(labeled_path)
    print(f"[OK] Dados carregados: {len(df)} amostras")

    # Criar trainer
    trainer = ScalpingModelTrainer(df, test_size=0.2, val_size=0.2)

    # Preparar dados
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data()

    # Otimizar hiperparâmetros
    best_params = trainer.optimize_hyperparameters(X_train, y_train, X_val, y_val, n_trials=n_trials)

    # Treinar modelo final
    model = trainer.train_final_model(X_train, y_train, X_val, y_val, best_params)

    # Avaliar em validation set
    val_metrics = trainer.evaluate_model(model, X_val, y_val, split_name='Validation')

    # Avaliar em test set (out-of-sample final)
    test_metrics = trainer.evaluate_model(model, X_test, y_test, split_name='Test')

    # Feature importance
    importance_path = data_dir.parent / "reports" / f"{symbol}_{timeframe}_feature_importance.png"
    trainer.plot_feature_importance(model, output_path=str(importance_path))

    # Salvar modelo
    model_dir = data_dir.parent / "models"
    trainer.save_model(model, best_params, test_metrics, model_dir)

    print(f"\n{'='*70}")
    print("TREINAMENTO CONCLUÍDO!")
    print(f"{'='*70}")
    print(f"Win rate (Test set): {test_metrics['win_rate']*100:.1f}%")

    if test_metrics['win_rate'] >= 0.60:
        print("[OK] META ATINGIDA! Win rate > 60%!")
        print("Próximo passo: Backtesting completo")
    else:
        print(f"[AVISO] Win rate {test_metrics['win_rate']*100:.1f}% < 60%")
        print("Considere: ajustar features, aumentar trials, ou testar ensemble")

    return model, test_metrics


if __name__ == "__main__":
    """
    Executar treinamento para V100 M5
    """

    # Treinar modelo
    model, metrics = train_scalping_model(
        symbol='1HZ100V',
        timeframe='5min',
        n_trials=50  # Pode aumentar para 100-200 se quiser melhor otimização
    )
