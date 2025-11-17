"""
Threshold Optimization - Encontrar threshold ótimo para maximizar profit

Testa múltiplos thresholds (0.25, 0.30, 0.35, 0.40, 0.45, 0.50) e executa
backtesting walk-forward para cada um, selecionando o threshold que maximiza
profit total enquanto mantém drawdown aceitável.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import pickle
from datetime import datetime
from typing import Dict, List, Tuple
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s: %(message)s'
)

logger = logging.getLogger(__name__)


class ThresholdOptimizer:
    """
    Otimiza threshold de classificação para maximizar profit em trading
    """

    def __init__(self, model_path: str):
        """
        Inicializa otimizador com modelo treinado

        Args:
            model_path: Caminho para modelo .pkl
        """
        self.model_path = Path(model_path)
        self.model = self._load_model()
        self.results = {}

    def _load_model(self):
        """Carrega modelo treinado"""
        logger.info(f"Carregando modelo de {self.model_path}")
        with open(self.model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info("Modelo carregado com sucesso")
        return model

    def optimize_threshold(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        thresholds: List[float] = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        train_size: int = 100000,
        test_size: int = 20000,
        step_size: int = 10000
    ) -> Dict:
        """
        Testa múltiplos thresholds e retorna métricas para cada um

        Args:
            X: Features
            y: Target
            thresholds: Lista de thresholds para testar
            train_size: Tamanho janela de treino
            test_size: Tamanho janela de teste
            step_size: Passo entre janelas

        Returns:
            Dict com resultados para cada threshold
        """
        logger.info("="*60)
        logger.info("THRESHOLD OPTIMIZATION")
        logger.info("="*60)
        logger.info(f"Thresholds a testar: {thresholds}")
        logger.info(f"Total de amostras: {len(X):,}")

        results = {}

        for threshold in thresholds:
            logger.info(f"\n{'='*60}")
            logger.info(f"TESTANDO THRESHOLD: {threshold}")
            logger.info(f"{'='*60}")

            # Executar walk-forward validation com este threshold
            threshold_results = self._walk_forward_with_threshold(
                X, y, threshold, train_size, test_size, step_size
            )

            results[threshold] = threshold_results

            # Imprimir resumo
            summary = threshold_results['summary']
            logger.info(f"\nRESULTADO THRESHOLD {threshold}:")
            logger.info(f"  Accuracy:  {summary['accuracy']['mean']*100:.2f}%")
            logger.info(f"  Recall:    {summary['recall']['mean']*100:.2f}%")
            logger.info(f"  Precision: {summary['precision']['mean']*100:.2f}%")
            logger.info(f"  Profit:    {summary['trading']['total_profit']:.2f}%")
            logger.info(f"  Max DD:    {summary['trading']['max_drawdown']:.2f}%")
            logger.info(f"  Sharpe:    {summary['trading']['avg_sharpe']:.2f}")

        # Comparar resultados
        self._compare_thresholds(results)

        return results

    def _walk_forward_with_threshold(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: float,
        train_size: int,
        test_size: int,
        step_size: int
    ) -> Dict:
        """
        Executa walk-forward validation com threshold específico
        """
        results = []
        n_samples = len(X)

        # Calcular número de janelas
        max_end = n_samples
        n_windows = ((max_end - train_size - test_size) // step_size) + 1

        logger.info(f"Executando {n_windows} janelas...")

        for i in range(n_windows):
            # Definir janelas
            train_start = i * step_size
            train_end = train_start + train_size
            test_start = train_end
            test_end = test_start + test_size

            if test_end > n_samples:
                break

            # Extrair janelas
            X_test_window = X.iloc[test_start:test_end]
            y_test_window = y.iloc[test_start:test_end]

            # Fazer predições com threshold customizado
            y_pred_proba = self.model.predict_proba(X_test_window)[:, 1]
            y_pred = (y_pred_proba >= threshold).astype(int)

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

            # Calcular métricas de trading
            trading_metrics = self._calculate_trading_metrics(
                y_test_window, y_pred, y_pred_proba
            )

            # Armazenar resultados
            result = {
                'window': i + 1,
                'threshold': threshold,
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

        return {
            'threshold': threshold,
            'summary': summary,
            'windows': results
        }

    def _calculate_trading_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        risk_reward: float = 2.0,
        threshold_movement: float = 0.003
    ) -> Dict:
        """
        Simula resultados de trading
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

        # Calcular Sharpe Ratio
        if pnl_series and len(pnl_series) > 1:
            returns = np.diff(pnl_series)
            if returns.std() > 0:
                sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
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
        """Calcula estatísticas agregadas"""
        if not results:
            return {}

        accuracies = [r['accuracy'] for r in results]
        precisions = [r['precision'] for r in results]
        recalls = [r['recall'] for r in results]
        f1_scores = [r['f1_score'] for r in results]
        profits = [r['trading']['total_profit'] for r in results]
        sharpes = [r['trading']['sharpe_ratio'] for r in results]
        drawdowns = [r['trading']['max_drawdown'] for r in results]

        return {
            'n_windows': len(results),
            'accuracy': {
                'mean': float(np.mean(accuracies)),
                'std': float(np.std(accuracies)),
                'min': float(np.min(accuracies)),
                'max': float(np.max(accuracies))
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
                'max_drawdown': float(np.max(drawdowns)),
                'avg_drawdown': float(np.mean(drawdowns))
            }
        }

    def _compare_thresholds(self, results: Dict):
        """Compara resultados de todos os thresholds"""
        print("\n" + "="*80)
        print("COMPARACAO DE THRESHOLDS")
        print("="*80)

        # Criar tabela comparativa
        print("\n{:<12} {:<10} {:<10} {:<10} {:<12} {:<12}".format(
            "Threshold", "Accuracy", "Recall", "Precision", "Profit", "Max DD"
        ))
        print("-" * 80)

        best_profit = -float('inf')
        best_threshold = None

        for threshold, data in sorted(results.items()):
            summary = data['summary']
            acc = summary['accuracy']['mean'] * 100
            recall = summary['recall']['mean'] * 100
            precision = summary['precision']['mean'] * 100
            profit = summary['trading']['total_profit']
            max_dd = summary['trading']['max_drawdown']

            print("{:<12.2f} {:<10.2f} {:<10.2f} {:<10.2f} {:<12.2f} {:<12.2f}".format(
                threshold, acc, recall, precision, profit, max_dd
            ))

            # Encontrar melhor threshold (profit > 0 e drawdown aceitável)
            if profit > best_profit and max_dd < 50:  # Max DD < 50%
                best_profit = profit
                best_threshold = threshold

        print("-" * 80)

        # Recomendar melhor threshold
        if best_threshold is not None:
            print(f"\n[RECOMENDACAO] Melhor threshold: {best_threshold}")
            print(f"  Profit: {best_profit:.2f}%")
            print(f"  Accuracy: {results[best_threshold]['summary']['accuracy']['mean']*100:.2f}%")
            print(f"  Recall: {results[best_threshold]['summary']['recall']['mean']*100:.2f}%")
            print(f"  Max DD: {results[best_threshold]['summary']['trading']['max_drawdown']:.2f}%")
        else:
            print("\n[ALERTA] Nenhum threshold atingiu profit positivo com DD < 50%")
            print("Recomendacao: Considerar redefinicao do target ou feature engineering")

        print("="*80)

    def save_results(self, results: Dict, output_path: str):
        """Salva resultados em JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Resultados salvos em: {output_path}")


def main():
    """Executa threshold optimization"""

    # Paths
    models_dir = Path(__file__).parent.parent / "models"
    data_dir = Path(__file__).parent.parent / "data"

    # Carregar modelo XGBoost
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

    # Separar features e target
    target_col = 'target'
    exclude_cols = ['timestamp', 'close', 'high', 'low', 'open', 'symbol', 'timeframe', target_col]
    feature_cols = [col for col in data.columns if col not in exclude_cols]

    X = data[feature_cols]
    y = data[target_col]

    # Garantir apenas tipos numéricos
    for col in X.columns:
        if X[col].dtype == 'object':
            logger.warning(f"Removendo coluna não-numérica: {col}")
            X = X.drop(columns=[col])

    # Remover NaN
    valid_idx = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_idx]
    y = y[valid_idx]

    logger.info(f"Dataset carregado: X={X.shape}, y={y.shape}")

    # Criar optimizer
    optimizer = ThresholdOptimizer(model_path)

    # Ajustar features
    model_features = optimizer.model.get_booster().feature_names
    X = X[model_features]
    logger.info(f"Dataset ajustado: X={X.shape}")

    # Executar otimização
    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    results = optimizer.optimize_threshold(
        X, y,
        thresholds=thresholds,
        train_size=100000,
        test_size=20000,
        step_size=10000
    )

    # Salvar resultados
    output_path = models_dir / f"threshold_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    optimizer.save_results(results, output_path)

    print(f"\n[SUCESSO] Threshold optimization concluido!")
    print(f"Resultados salvos em: {output_path}")


if __name__ == "__main__":
    main()
