"""
ML Predictor - CRASH 500 Survival Analysis

Integra modelo LSTM Survival (91.81% win rate) com sistema de trading para
prever risco de alta volatilidade no CRASH 500.

ESTRATÉGIA:
- Se modelo prever >= 20 candles até alta vol → ENTRAR LONG
- Se modelo prever < 20 candles → FICAR FORA (zona perigosa)
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMSurvivalModel(nn.Module):
    """
    LSTM para Survival Analysis (mesmo do research)
    """
    def __init__(self, input_dim=5, hidden_dim1=128, hidden_dim2=64):
        super().__init__()

        self.lstm1 = nn.LSTM(input_dim, hidden_dim1, batch_first=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.dropout1 = nn.Dropout(0.3)

        self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.dropout2 = nn.Dropout(0.3)

        self.fc1 = nn.Linear(hidden_dim2, 32)
        self.dropout3 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = out[:, -1, :]
        out = self.bn1(out)
        out = self.dropout1(out)

        out = out.unsqueeze(1)

        out, _ = self.lstm2(out)
        out = out[:, -1, :]
        out = self.bn2(out)
        out = self.dropout2(out)

        out = torch.relu(self.fc1(out))
        out = self.dropout3(out)
        out = self.fc2(out)

        out = torch.relu(out)

        return out


class CRASH500Predictor:
    """
    Predictor de risco para CRASH 500 usando Survival Analysis
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        safe_threshold: int = 20,  # >= 20 candles = zona segura
        lookback: int = 50  # Janela de 50 candles
    ):
        """
        Inicializa predictor CRASH 500

        Args:
            model_path: Caminho para modelo .pth
            safe_threshold: Candles mínimos para considerar seguro
            lookback: Janela de observação (50 candles)
        """
        self.safe_threshold = safe_threshold
        self.lookback = lookback
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Carregar modelo
        if model_path is None:
            model_path = Path(__file__).parent / "ml" / "research" / "models" / "crash_survival_lstm.pth"

        self.model_path = Path(model_path)  # Salvar path para compatibilidade com ForwardTestingEngine

        logger.info(f"Carregando modelo CRASH 500 de {self.model_path}")
        self.model = LSTMSurvivalModel(input_dim=5).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()

        logger.info("Modelo CRASH 500 Survival carregado com sucesso!")
        logger.info(f"  Safe threshold: {safe_threshold} candles")
        logger.info(f"  Lookback window: {lookback} candles")

    def prepare_features(self, candles_df: pd.DataFrame) -> Optional[torch.Tensor]:
        """
        Prepara features para predição

        Args:
            candles_df: DataFrame com OHLC dos últimos candles

        Returns:
            Tensor normalizado [1, lookback, 5] ou None se insuficiente
        """
        if len(candles_df) < self.lookback:
            logger.warning(f"Candles insuficientes: {len(candles_df)} < {self.lookback}")
            return None

        # Pegar últimos lookback candles
        recent_candles = candles_df.iloc[-self.lookback:].copy()

        # Calcular volatilidade realizada
        recent_candles['return'] = recent_candles['close'].pct_change()
        recent_candles['realized_vol'] = recent_candles['return'].rolling(window=20).std()

        # Remover NaNs
        recent_candles = recent_candles.dropna()

        if len(recent_candles) < self.lookback:
            logger.warning(f"Candles insuficientes após remover NaNs: {len(recent_candles)}")
            return None

        # Features: OHLC + realized_vol
        features = recent_candles[['open', 'high', 'low', 'close', 'realized_vol']].values.astype(np.float32)

        # Normalização Min-Max por janela (mesmo do treinamento)
        window_min = features.min(axis=0)
        window_max = features.max(axis=0)
        window_range = window_max - window_min + 1e-8
        features_norm = (features - window_min) / window_range

        # Converter para tensor [1, lookback, 5]
        tensor = torch.FloatTensor(features_norm).unsqueeze(0)

        return tensor

    def predict(self, candles_df: pd.DataFrame, return_confidence: bool = True) -> Dict:
        """
        Gera predição de risco

        Args:
            candles_df: DataFrame com OHLC histórico
            return_confidence: Compatibilidade com MLPredictor (sempre retorna confidence)

        Returns:
            Dict com:
                - signal: 'LONG' (entrar) ou 'WAIT' (ficar fora)
                - candles_to_risk: Candles previstos até alta vol
                - is_safe: True se >= safe_threshold
                - confidence: Confiança relativa (0-1)
                - timestamp: Timestamp da predição
        """
        try:
            # Preparar features
            features = self.prepare_features(candles_df)

            if features is None:
                return {
                    'signal': 'WAIT',
                    'candles_to_risk': 0,
                    'is_safe': False,
                    'confidence': 0.0,
                    'reason': 'Candles insuficientes',
                    'timestamp': datetime.now().isoformat()
                }

            # Predição
            with torch.no_grad():
                features = features.to(self.device)
                candles_pred = self.model(features).cpu().item()

            # Decisão
            is_safe = candles_pred >= self.safe_threshold
            signal = 'LONG' if is_safe else 'WAIT'

            # Confiança: quanto mais longe do threshold, maior
            distance_from_threshold = abs(candles_pred - self.safe_threshold)
            confidence = min(distance_from_threshold / 20.0, 1.0)  # Normalizar por 20

            result = {
                'signal': signal,
                'candles_to_risk': round(candles_pred, 1),
                'is_safe': is_safe,
                'confidence': round(confidence, 2),
                'timestamp': datetime.now().isoformat()
            }

            logger.info(f"Predição CRASH 500: {result}")

            return result

        except Exception as e:
            logger.error(f"Erro na predição: {e}")
            return {
                'signal': 'WAIT',
                'candles_to_risk': 0,
                'is_safe': False,
                'confidence': 0.0,
                'reason': f'Erro: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }

    def get_strategy_description(self) -> Dict:
        """
        Retorna descrição da estratégia
        """
        return {
            'name': 'CRASH 500 Survival Analysis',
            'model': 'LSTM (121k params)',
            'win_rate': '91.81%',
            'strategy': f'Entrar LONG se prever >= {self.safe_threshold} candles até alta volatilidade',
            'asset': 'CRASH 500',
            'timeframe': 'M5',
            'lookback': f'{self.lookback} candles',
            'features': ['OHLC', 'Realized Volatility'],
            'risk_management': 'Evita zona de perigo (< 20 candles até spike)'
        }


# Para compatibilidade com código existente
MLPredictor = CRASH500Predictor
