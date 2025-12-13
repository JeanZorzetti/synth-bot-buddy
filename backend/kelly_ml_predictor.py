"""
Kelly Criterion ML Predictor

Este módulo usa Machine Learning para prever win_rate e avg_win/avg_loss
dinamicamente, ajustando o Kelly Criterion baseado em condições de mercado.

Features:
- Prevê win_rate usando histórico de trades
- Ajusta Kelly Criterion em tempo real
- Usa Random Forest para capturar padrões não-lineares
- Features: consecutive_wins/losses, volatility, time_of_day, etc.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import os

logger = logging.getLogger(__name__)


class KellyMLPredictor:
    """
    Preditor de Kelly Criterion usando Machine Learning

    Responsabilidades:
    - Extrair features do histórico de trades
    - Treinar modelo para prever win_rate
    - Prever avg_win e avg_loss
    - Calcular Kelly Criterion ajustado dinamicamente
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Inicializa KellyMLPredictor

        Args:
            model_path: Path para salvar/carregar modelos treinados
        """
        self.model_path = model_path or "models/kelly_ml_model.pkl"

        # Modelos
        self.win_rate_model: Optional[RandomForestClassifier] = None
        self.avg_win_model: Optional[RandomForestRegressor] = None
        self.avg_loss_model: Optional[RandomForestRegressor] = None
        self.scaler = StandardScaler()

        # Estado
        self.is_trained = False
        self.min_trades_for_training = 50  # Mínimo de trades para treinar
        self.feature_names: List[str] = []

        logger.info("KellyMLPredictor inicializado")

    def extract_features(self, trade_history: List[Dict]) -> pd.DataFrame:
        """
        Extrai features do histórico de trades

        Features extraídas:
        - consecutive_wins: Número de wins consecutivos
        - consecutive_losses: Número de losses consecutivos
        - recent_win_rate: Win rate dos últimos 10 trades
        - avg_trade_duration: Duração média dos trades (em minutos)
        - volatility: Volatilidade dos últimos 20 trades
        - hour_of_day: Hora do dia (0-23)
        - day_of_week: Dia da semana (0-6)
        - total_trades: Total de trades até o momento
        - avg_position_size: Tamanho médio de posição (últimos 10 trades)
        - sharpe_ratio: Sharpe ratio dos últimos 20 trades

        Args:
            trade_history: Lista de trades históricos

        Returns:
            DataFrame com features extraídas
        """
        if len(trade_history) < 2:
            logger.warning("Trade history muito curto para extrair features")
            return pd.DataFrame()

        features_list = []

        for i in range(1, len(trade_history)):
            trade = trade_history[i]
            recent_trades = trade_history[max(0, i-20):i]
            last_10_trades = trade_history[max(0, i-10):i]

            # Consecutive wins/losses
            consecutive_wins = 0
            consecutive_losses = 0
            for t in reversed(recent_trades):
                if t['is_win']:
                    if consecutive_losses > 0:
                        break
                    consecutive_wins += 1
                else:
                    if consecutive_wins > 0:
                        break
                    consecutive_losses += 1

            # Recent win rate (últimos 10 trades)
            recent_win_rate = sum(1 for t in last_10_trades if t['is_win']) / len(last_10_trades) if last_10_trades else 0.5

            # Volatility (std dos PnLs)
            pnls = [t['pnl'] for t in recent_trades]
            volatility = np.std(pnls) if len(pnls) > 1 else 0.0

            # Time features
            timestamp = trade.get('timestamp', datetime.now())
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            hour_of_day = timestamp.hour
            day_of_week = timestamp.weekday()

            # Position size
            avg_position_size = np.mean([t['position_size'] for t in last_10_trades]) if last_10_trades else 0.0

            # Sharpe ratio (últimos 20 trades)
            if len(pnls) > 1:
                mean_pnl = np.mean(pnls)
                std_pnl = np.std(pnls)
                sharpe_ratio = mean_pnl / std_pnl if std_pnl > 0 else 0.0
            else:
                sharpe_ratio = 0.0

            features = {
                'consecutive_wins': consecutive_wins,
                'consecutive_losses': consecutive_losses,
                'recent_win_rate': recent_win_rate,
                'volatility': volatility,
                'hour_of_day': hour_of_day,
                'day_of_week': day_of_week,
                'total_trades': i,
                'avg_position_size': avg_position_size,
                'sharpe_ratio': sharpe_ratio,
                # Target (será usado no treinamento)
                'is_win': 1 if trade['is_win'] else 0,
                'pnl': trade['pnl']
            }

            features_list.append(features)

        df = pd.DataFrame(features_list)

        logger.info(f"Features extraídas: {len(df)} amostras com {len(df.columns)-2} features")

        return df

    def train(self, trade_history: List[Dict]) -> Dict:
        """
        Treina modelos de ML usando histórico de trades

        Args:
            trade_history: Lista de trades históricos

        Returns:
            Dict com métricas de treinamento
        """
        if len(trade_history) < self.min_trades_for_training:
            raise ValueError(f"Mínimo de {self.min_trades_for_training} trades necessários para treinamento. Atual: {len(trade_history)}")

        # Extrair features
        df = self.extract_features(trade_history)

        if df.empty:
            raise ValueError("Falha ao extrair features do trade_history")

        # Separar features e targets
        self.feature_names = [col for col in df.columns if col not in ['is_win', 'pnl']]
        X = df[self.feature_names].values
        y_win = df['is_win'].values
        y_pnl = df['pnl'].values

        # Normalizar features
        X_scaled = self.scaler.fit_transform(X)

        # Treinar modelo de win_rate (classificação)
        logger.info("Treinando modelo de win_rate...")
        self.win_rate_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        self.win_rate_model.fit(X_scaled, y_win)

        # Treinar modelo de avg_win (regressão - apenas wins)
        logger.info("Treinando modelo de avg_win...")
        X_wins = X_scaled[y_win == 1]
        y_wins = y_pnl[y_win == 1]

        if len(y_wins) > 10:
            self.avg_win_model = RandomForestRegressor(
                n_estimators=50,
                max_depth=8,
                random_state=42
            )
            self.avg_win_model.fit(X_wins, y_wins)
        else:
            logger.warning("Poucos wins para treinar avg_win_model, usando média simples")
            self.avg_win_model = None

        # Treinar modelo de avg_loss (regressão - apenas losses)
        logger.info("Treinando modelo de avg_loss...")
        X_losses = X_scaled[y_win == 0]
        y_losses = np.abs(y_pnl[y_win == 0])  # Valor absoluto das perdas

        if len(y_losses) > 10:
            self.avg_loss_model = RandomForestRegressor(
                n_estimators=50,
                max_depth=8,
                random_state=42
            )
            self.avg_loss_model.fit(X_losses, y_losses)
        else:
            logger.warning("Poucas losses para treinar avg_loss_model, usando média simples")
            self.avg_loss_model = None

        self.is_trained = True

        # Calcular métricas
        win_rate_pred = self.win_rate_model.predict(X_scaled)
        accuracy = np.mean(win_rate_pred == y_win)

        metrics = {
            'accuracy': accuracy,
            'total_samples': len(df),
            'total_wins': int(np.sum(y_win)),
            'total_losses': int(len(y_win) - np.sum(y_win)),
            'actual_win_rate': float(np.mean(y_win)),
            'feature_importance': dict(zip(self.feature_names, self.win_rate_model.feature_importances_))
        }

        logger.info(f"Treinamento completo! Accuracy: {accuracy:.2%}, Win Rate Real: {metrics['actual_win_rate']:.2%}")

        return metrics

    def predict(self, current_state: Dict) -> Dict:
        """
        Prevê win_rate, avg_win, avg_loss e calcula Kelly Criterion ajustado

        Args:
            current_state: Estado atual do mercado/trader
                - consecutive_wins: int
                - consecutive_losses: int
                - recent_win_rate: float (0-1)
                - volatility: float
                - hour_of_day: int (0-23)
                - day_of_week: int (0-6)
                - total_trades: int
                - avg_position_size: float
                - sharpe_ratio: float

        Returns:
            Dict com previsões e Kelly Criterion ajustado
        """
        if not self.is_trained:
            raise ValueError("Modelo não treinado! Execute train() primeiro")

        # Preparar features
        X = np.array([[
            current_state.get('consecutive_wins', 0),
            current_state.get('consecutive_losses', 0),
            current_state.get('recent_win_rate', 0.5),
            current_state.get('volatility', 0.0),
            current_state.get('hour_of_day', 12),
            current_state.get('day_of_week', 0),
            current_state.get('total_trades', 0),
            current_state.get('avg_position_size', 0.0),
            current_state.get('sharpe_ratio', 0.0)
        ]])

        X_scaled = self.scaler.transform(X)

        # Prever win_rate (probabilidade de win)
        win_rate_proba = self.win_rate_model.predict_proba(X_scaled)[0]
        predicted_win_rate = win_rate_proba[1]  # Probabilidade de classe 1 (win)

        # Prever avg_win
        if self.avg_win_model is not None:
            predicted_avg_win = float(self.avg_win_model.predict(X_scaled)[0])
        else:
            predicted_avg_win = current_state.get('fallback_avg_win', 0.0)

        # Prever avg_loss
        if self.avg_loss_model is not None:
            predicted_avg_loss = float(self.avg_loss_model.predict(X_scaled)[0])
        else:
            predicted_avg_loss = current_state.get('fallback_avg_loss', 0.0)

        # Calcular Kelly Criterion ajustado
        if predicted_avg_loss == 0 or predicted_avg_win == 0:
            kelly_criterion = 0.02  # Fallback conservador
        else:
            p = predicted_win_rate
            q = 1 - p
            b = abs(predicted_avg_win / predicted_avg_loss)

            kelly = (p * b - q) / b

            # Quarter Kelly para segurança
            kelly_criterion = kelly * 0.25

            # Limitar entre 1% e 5%
            kelly_criterion = max(0.01, min(kelly_criterion, 0.05))

        result = {
            'predicted_win_rate': predicted_win_rate,
            'predicted_avg_win': predicted_avg_win,
            'predicted_avg_loss': predicted_avg_loss,
            'kelly_criterion': kelly_criterion,
            'kelly_full': kelly if predicted_avg_loss > 0 and predicted_avg_win > 0 else 0.0,
            'confidence': float(max(win_rate_proba))  # Confiança da previsão
        }

        logger.debug(f"Previsão ML: Win Rate={predicted_win_rate:.2%}, Kelly={kelly_criterion:.2%}")

        return result

    def save_model(self, path: Optional[str] = None) -> str:
        """
        Salva modelo treinado em disco

        Args:
            path: Caminho para salvar (usa self.model_path se None)

        Returns:
            Path onde foi salvo
        """
        if not self.is_trained:
            raise ValueError("Modelo não treinado! Nada para salvar")

        save_path = path or self.model_path

        # Criar diretório se não existir
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        model_data = {
            'win_rate_model': self.win_rate_model,
            'avg_win_model': self.avg_win_model,
            'avg_loss_model': self.avg_loss_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }

        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Modelo salvo em: {save_path}")

        return save_path

    def load_model(self, path: Optional[str] = None) -> bool:
        """
        Carrega modelo treinado do disco

        Args:
            path: Caminho para carregar (usa self.model_path se None)

        Returns:
            True se carregou com sucesso
        """
        load_path = path or self.model_path

        if not os.path.exists(load_path):
            logger.warning(f"Modelo não encontrado em: {load_path}")
            return False

        try:
            with open(load_path, 'rb') as f:
                model_data = pickle.load(f)

            self.win_rate_model = model_data['win_rate_model']
            self.avg_win_model = model_data['avg_win_model']
            self.avg_loss_model = model_data['avg_loss_model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.is_trained = model_data['is_trained']

            logger.info(f"Modelo carregado de: {load_path}")

            return True
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}", exc_info=True)
            return False


# Singleton global
_kelly_ml_predictor: Optional[KellyMLPredictor] = None


def get_kelly_ml_predictor() -> KellyMLPredictor:
    """Retorna instância singleton do KellyMLPredictor"""
    global _kelly_ml_predictor
    if _kelly_ml_predictor is None:
        _kelly_ml_predictor = KellyMLPredictor()
    return _kelly_ml_predictor


def initialize_kelly_ml_predictor(model_path: Optional[str] = None) -> KellyMLPredictor:
    """Inicializa e retorna KellyMLPredictor"""
    global _kelly_ml_predictor
    _kelly_ml_predictor = KellyMLPredictor(model_path)
    return _kelly_ml_predictor
