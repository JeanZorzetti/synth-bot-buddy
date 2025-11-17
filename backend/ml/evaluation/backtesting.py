"""
Backtesting Walk-Forward - Validação do Modelo XGBoost
Valida performance em dados temporais não vistos
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s: %(message)s'
)

logger = logging.getLogger(__name__)


class WalkForwardBacktester:
    """
    Realiza backtesting walk-forward para validar modelos ML
    """

    def __init__(self, model_path: str):
        """
        Inicializa backtester com modelo treinado

        Args:
            model_path: Caminho para modelo .pkl
        """
        self.model_path = Path(model_path)
        self.model = self._load_model()
        self.results = []

    def _load_model(self):
        """Carrega modelo treinado"""
        logger.info(f"Carregando modelo de {self.model_path}")
        with open(self.model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info("Modelo carregado com sucesso")
        return model

    def walk_forward_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        train_size: int = 100000,  # 100k candles (~69 dias)
        test_size: int = 20000,    # 20k candles (~14 dias)
        step_size: int = 10000     # 10k candles (~7 dias)
    ) -> Dict:
        """
        Realiza walk-forward validation

        Args:
            X: Features
            y: Target
            train_size: Tamanho da janela de treino
            test_size: Tamanho da janela de teste
            step_size: Passo entre janelas

        Returns:
            Dict com resultados de todas as janelas
        """
        logger.info("="*60)
        logger.info("WALK-FORWARD VALIDATION")
        logger.info("="*60)
        logger.info(f"Total de amostras: {len(X):,}")
        logger.info(f"Train size: {train_size:,} ({train_size/1440:.1f} dias)")
        logger.info(f"Test size: {test_size:,} ({test_size/1440:.1f} dias)")
        logger.info(f"Step size: {step_size:,} ({step_size/1440:.1f} dias)")

        results = []
        n_samples = len(X)

        # Calcular número de janelas
        max_end = n_samples
        n_windows = ((max_end - train_size - test_size) // step_size) + 1

        logger.info(f"\nTotal de janelas: {n_windows}")
        logger.info("="*60)

        for i in range(n_windows):
            # Definir janelas
            train_start = i * step_size
            train_end = train_start + train_size
            test_start = train_end
            test_end = test_start + test_size

            # Verificar se há dados suficientes
            if test_end > n_samples:
                logger.warning(f"Janela {i+1}: dados insuficientes, pulando")
                break

            # Extrair janelas
            X_train_window = X.iloc[train_start:train_end]
            y_train_window = y.iloc[train_start:train_end]
            X_test_window = X.iloc[test_start:test_end]
            y_test_window = y.iloc[test_start:test_end]

            logger.info(f"\nJanela {i+1}/{n_windows}:")
            logger.info(f"  Train: {train_start:,} -> {train_end:,}")
            logger.info(f"  Test:  {test_start:,} -> {test_end:,}")

            # Fazer predições (usando modelo PRÉ-TREINADO, não retreinando)
            y_pred = self.model.predict(X_test_window)
            y_pred_proba = self.model.predict_proba(X_test_window)[:, 1]

            # Calcular métricas
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score,
                f1_score, roc_auc_score, confusion_matrix
            )

            accuracy = accuracy_score(y_test_window, y_pred)
            precision = precision_score(y_test_window, y_pred, zero_division=0)
            recall = recall_score(y_test_window, y_pred, zero_division=0)
            f1 = f1_score(y_test_window, y_pred, zero_division=0)
            auc_roc = roc_auc_score(y_test_window, y_pred_proba)
            cm = confusion_matrix(y_test_window, y_pred)

            # Calcular métricas de trading simulado
            trading_metrics = self._calculate_trading_metrics(
                y_test_window, y_pred, y_pred_proba
            )

            logger.info(f"  Accuracy:  {accuracy*100:.2f}%")
            logger.info(f"  Precision: {precision*100:.2f}%")
            logger.info(f"  Recall:    {recall*100:.2f}%")
            logger.info(f"  F1-Score:  {f1*100:.2f}%")
            logger.info(f"  Profit:    {trading_metrics['total_profit']:.2f}%")

            # Armazenar resultados
            result = {
                'window': i + 1,
                'train_start': int(train_start),
                'train_end': int(train_end),
                'test_start': int(test_start),
                'test_end': int(test_end),
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'auc_roc': float(auc_roc),
                'confusion_matrix': cm.tolist(),
                'trading': trading_metrics
            }

            results.append(result)

        # Calcular estatísticas agregadas
        summary = self._calculate_summary(results)

        # Imprimir resumo
        self._print_summary(summary)

        return {
            'summary': summary,
            'windows': results
        }

    def _calculate_trading_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        risk_reward: float = 2.0,  # 1:2 risk/reward
        threshold_movement: float = 0.003  # 0.3% movement
    ) -> Dict:
        """
        Simula resultados de trading

        Assume:
        - Cada previsão correta de UP = +0.6% (2x o threshold de 0.3%)
        - Cada previsão incorreta de UP = -0.3% (stop loss)
        - Previsões de NO MOVE não geram trades
        """
        # Contar trades
        n_trades = (y_pred == 1).sum()

        if n_trades == 0:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_profit': 0.0,
                'avg_profit_per_trade': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0
            }

        # Calcular trades vencedores e perdedores
        correct_ups = ((y_pred == 1) & (y_true == 1)).sum()
        incorrect_ups = ((y_pred == 1) & (y_true == 0)).sum()

        # Simular P&L
        profit_per_win = threshold_movement * risk_reward * 100  # 0.6%
        loss_per_loss = -threshold_movement * 100  # -0.3%

        total_profit = (correct_ups * profit_per_win) + (incorrect_ups * loss_per_loss)
        avg_profit_per_trade = total_profit / n_trades if n_trades > 0 else 0

        # Simular drawdown
        pnl_series = []
        cumulative_pnl = 0

        for pred, true in zip(y_pred, y_true):
            if pred == 1:  # Trade executado
                if true == 1:
                    cumulative_pnl += profit_per_win
                else:
                    cumulative_pnl += loss_per_loss
                pnl_series.append(cumulative_pnl)

        # Calcular max drawdown
        if pnl_series:
            peak = pnl_series[0]
            max_dd = 0
            for pnl in pnl_series:
                if pnl > peak:
                    peak = pnl
                dd = peak - pnl
                if dd > max_dd:
                    max_dd = dd
        else:
            max_dd = 0

        # Calcular Sharpe Ratio simplificado
        if pnl_series and len(pnl_series) > 1:
            returns = np.diff(pnl_series)
            if returns.std() > 0:
                sharpe = (returns.mean() / returns.std()) * np.sqrt(252)  # Anualizado
            else:
                sharpe = 0
        else:
            sharpe = 0

        return {
            'total_trades': int(n_trades),
            'winning_trades': int(correct_ups),
            'losing_trades': int(incorrect_ups),
            'win_rate': float(correct_ups / n_trades if n_trades > 0 else 0),
            'total_profit': float(total_profit),
            'avg_profit_per_trade': float(avg_profit_per_trade),
            'max_drawdown': float(max_dd),
            'sharpe_ratio': float(sharpe)
        }

    def _calculate_summary(self, results: List[Dict]) -> Dict:
        """Calcula estatísticas agregadas de todas as janelas"""
        if not results:
            return {}

        accuracies = [r['accuracy'] for r in results]
        precisions = [r['precision'] for r in results]
        recalls = [r['recall'] for r in results]
        f1_scores = [r['f1_score'] for r in results]
        profits = [r['trading']['total_profit'] for r in results]
        sharpes = [r['trading']['sharpe_ratio'] for r in results]

        return {
            'n_windows': len(results),
            'accuracy': {
                'mean': float(np.mean(accuracies)),
                'std': float(np.std(accuracies)),
                'min': float(np.min(accuracies)),
                'max': float(np.max(accuracies)),
                'median': float(np.median(accuracies))
            },
            'precision': {
                'mean': float(np.mean(precisions)),
                'std': float(np.std(precisions))
            },
            'recall': {
                'mean': float(np.mean(recalls)),
                'std': float(np.std(recalls))
            },
            'f1_score': {
                'mean': float(np.mean(f1_scores)),
                'std': float(np.std(f1_scores))
            },
            'trading': {
                'total_profit': float(np.sum(profits)),
                'avg_profit_per_window': float(np.mean(profits)),
                'std_profit': float(np.std(profits)),
                'best_window_profit': float(np.max(profits)),
                'worst_window_profit': float(np.min(profits)),
                'avg_sharpe': float(np.mean(sharpes)),
                'consistency': float(1 - (np.std(profits) / np.mean(np.abs(profits)) if np.mean(np.abs(profits)) > 0 else 0))
            }
        }

    def _print_summary(self, summary: Dict):
        """Imprime resumo dos resultados"""
        print("\n" + "="*60)
        print("RESUMO DO WALK-FORWARD VALIDATION")
        print("="*60)

        print(f"\nTotal de janelas testadas: {summary['n_windows']}")

        print("\nACCURACY:")
        print(f"  Média:   {summary['accuracy']['mean']*100:.2f}%")
        print(f"  Desvio:  {summary['accuracy']['std']*100:.2f}%")
        print(f"  Mínimo:  {summary['accuracy']['min']*100:.2f}%")
        print(f"  Máximo:  {summary['accuracy']['max']*100:.2f}%")
        print(f"  Mediana: {summary['accuracy']['median']*100:.2f}%")

        print("\nRECALL:")
        print(f"  Média:   {summary['recall']['mean']*100:.2f}%")
        print(f"  Desvio:  {summary['recall']['std']*100:.2f}%")

        print("\nTRADING PERFORMANCE (Simulado):")
        print(f"  Profit total:        {summary['trading']['total_profit']:.2f}%")
        print(f"  Profit médio/janela: {summary['trading']['avg_profit_per_window']:.2f}%")
        print(f"  Melhor janela:       {summary['trading']['best_window_profit']:.2f}%")
        print(f"  Pior janela:         {summary['trading']['worst_window_profit']:.2f}%")
        print(f"  Sharpe médio:        {summary['trading']['avg_sharpe']:.2f}")
        print(f"  Consistência:        {summary['trading']['consistency']*100:.1f}%")

        # Avaliação
        print("\n" + "="*60)
        print("AVALIAÇÃO")
        print("="*60)

        acc_mean = summary['accuracy']['mean']
        acc_std = summary['accuracy']['std']
        profit_total = summary['trading']['total_profit']

        if acc_mean >= 0.65:
            print(f"[OK] Accuracy media >= 65%: {acc_mean*100:.2f}%")
        else:
            print(f"[FAIL] Accuracy media < 65%: {acc_mean*100:.2f}%")

        if acc_std <= 0.05:
            print(f"[OK] Consistencia alta (std <= 5%): {acc_std*100:.2f}%")
        elif acc_std <= 0.10:
            print(f"[WARN] Consistencia moderada (std <= 10%): {acc_std*100:.2f}%")
        else:
            print(f"[FAIL] Consistencia baixa (std > 10%): {acc_std*100:.2f}%")

        if profit_total > 0:
            print(f"[OK] Lucrativo: +{profit_total:.2f}%")
        else:
            print(f"[FAIL] Prejuizo: {profit_total:.2f}%")

        print("="*60)

    def save_results(self, results: Dict, output_path: str):
        """Salva resultados em JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Resultados salvos em: {output_path}")


def main():
    """Executa backtesting do modelo XGBoost"""

    # Paths
    models_dir = Path(__file__).parent.parent / "models"
    data_dir = Path(__file__).parent.parent / "data"

    # Carregar modelo XGBoost otimizado
    # Encontrar o arquivo mais recente
    model_files = list(models_dir.glob("xgboost_improved_learning_rate_*.pkl"))
    if not model_files:
        logger.error("Modelo XGBoost não encontrado!")
        return

    model_path = sorted(model_files, key=lambda x: x.stat().st_mtime)[-1]
    logger.info(f"Usando modelo: {model_path.name}")

    # Carregar dataset
    dataset_path = data_dir / "ml_dataset_R100_1m_6months.pkl"
    logger.info(f"Carregando dataset de {dataset_path}")

    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    # Dataset é DataFrame completo
    # Separar features (X) e target (y)
    target_col = 'target'
    exclude_cols = ['timestamp', 'close', 'high', 'low', 'open', 'symbol', 'timeframe', target_col]
    feature_cols = [col for col in data.columns if col not in exclude_cols]

    X = data[feature_cols]
    y = data[target_col]

    # Garantir que X tem apenas tipos numéricos
    for col in X.columns:
        if X[col].dtype == 'object':
            logger.warning(f"Removendo coluna não-numérica: {col}")
            X = X.drop(columns=[col])

    # Remover NaN
    valid_idx = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_idx]
    y = y[valid_idx]

    logger.info(f"Dataset carregado: X={X.shape}, y={y.shape}")

    # Criar backtester
    backtester = WalkForwardBacktester(model_path)

    # Garantir que X tem as mesmas features que o modelo espera
    model_features = backtester.model.get_booster().feature_names
    logger.info(f"Modelo espera {len(model_features)} features")

    # Usar apenas features que o modelo conhece
    X = X[model_features]
    logger.info(f"Dataset ajustado: X={X.shape}")

    # Executar walk-forward validation
    results = backtester.walk_forward_validation(
        X, y,
        train_size=100000,  # ~69 dias
        test_size=20000,    # ~14 dias
        step_size=10000     # ~7 dias
    )

    # Salvar resultados
    output_path = models_dir / f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    backtester.save_results(results, output_path)

    print(f"\n[SUCESSO] Backtesting concluído!")
    print(f"Resultados salvos em: {output_path}")


if __name__ == "__main__":
    main()
