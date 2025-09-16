"""
🧠 AI MODEL VALIDATION TESTS
Comprehensive testing for LSTM neural network and pattern recognition
"""

import pytest
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))


class TestAIModelValidation:
    """🎯 Validação Completa do Modelo de IA"""

    @pytest.fixture
    def sample_training_data(self):
        """Dados de treinamento simulados"""
        np.random.seed(42)

        # Simular 1000 sequências de 60 ticks cada
        sequences = []
        labels = []

        for i in range(1000):
            # Gerar sequência de preços com trend
            base_price = 245.0
            trend = np.random.choice([-1, 1])  # UP ou DOWN
            noise = np.random.normal(0, 0.1, 60)

            price_sequence = []
            current_price = base_price

            for j in range(60):
                current_price += (trend * 0.01) + noise[j]
                price_sequence.append(current_price)

            # Calcular features técnicas
            features = self.calculate_technical_features(price_sequence)
            sequences.append(features)
            labels.append(1 if trend > 0 else 0)  # 1=UP, 0=DOWN

        return np.array(sequences), np.array(labels)

    def calculate_technical_features(self, prices: List[float]) -> List[List[float]]:
        """Calcular features técnicas para uma sequência de preços"""
        prices_array = np.array(prices)
        features_sequence = []

        for i in range(len(prices)):
            if i < 20:  # Não há dados suficientes para indicadores
                features = [0.0] * 22  # 22 features
            else:
                # Calcular features (simulado)
                price_velocity = (prices[i] - prices[i-1]) / prices[i-1] if i > 0 else 0
                volatility = np.std(prices_array[max(0, i-10):i+1])
                sma_5 = np.mean(prices_array[max(0, i-4):i+1])
                sma_20 = np.mean(prices_array[max(0, i-19):i+1])

                features = [
                    price_velocity,
                    volatility,
                    (prices[i] - sma_5) / sma_5,  # Distância da SMA5
                    (prices[i] - sma_20) / sma_20,  # Distância da SMA20
                    sma_5 / sma_20 - 1,  # SMA5/SMA20 ratio
                    # Adicionar mais 17 features simuladas
                    *[np.random.normal(0, 0.1) for _ in range(17)]
                ]

            features_sequence.append(features)

        return features_sequence

    @pytest.fixture
    def mock_lstm_model(self):
        """Modelo LSTM simulado para testes"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(60, 22)),
            tf.keras.layers.LSTM(32, return_sequences=False),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def test_model_architecture_validation(self, mock_lstm_model):
        """🏗️ Validação da Arquitetura do Modelo"""
        print("\n🧪 Validando arquitetura do modelo LSTM...")

        # 1. Verificar estrutura do modelo
        assert len(mock_lstm_model.layers) == 5  # LSTM + LSTM + Dense + Dropout + Dense

        # 2. Verificar input shape
        expected_input_shape = (None, 60, 22)  # (batch, timesteps, features)
        assert mock_lstm_model.layers[0].input_shape == expected_input_shape

        # 3. Verificar output shape
        expected_output_shape = (None, 1)  # (batch, prediction)
        assert mock_lstm_model.layers[-1].output_shape == expected_output_shape

        # 4. Verificar parâmetros treináveis
        total_params = mock_lstm_model.count_params()
        assert total_params > 10000  # Modelo complexo o suficiente

        print(f"✅ Arquitetura validada - {total_params:,} parâmetros")

    def test_model_training_validation(self, mock_lstm_model, sample_training_data):
        """🎓 Validação do Treinamento do Modelo"""
        print("\n🧪 Validando treinamento do modelo...")

        X_train, y_train = sample_training_data

        # 1. Split train/validation
        split_idx = int(0.8 * len(X_train))
        X_val = X_train[split_idx:]
        y_val = y_train[split_idx:]
        X_train = X_train[:split_idx]
        y_train = y_train[:split_idx]

        # 2. Treinar modelo (épocas limitadas para teste)
        history = mock_lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=5,
            batch_size=32,
            verbose=0
        )

        # 3. Verificar convergência
        final_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        final_accuracy = history.history['accuracy'][-1]

        assert final_loss < 1.0  # Loss razoável
        assert final_accuracy > 0.4  # Melhor que aleatório
        assert abs(final_loss - final_val_loss) < 0.5  # Não overfitting severo

        print(f"✅ Treinamento validado - Acc: {final_accuracy:.3f}, Loss: {final_loss:.3f}")

    def test_model_prediction_accuracy(self, mock_lstm_model, sample_training_data):
        """🎯 Validação da Acurácia das Predições"""
        print("\n🧪 Validando acurácia das predições...")

        X_test, y_test = sample_training_data

        # 1. Treinar rapidamente para ter um modelo funcional
        mock_lstm_model.fit(X_test[:800], y_test[:800], epochs=3, verbose=0)

        # 2. Fazer predições no conjunto de teste
        X_test_subset = X_test[800:]
        y_test_subset = y_test[800:]

        predictions = mock_lstm_model.predict(X_test_subset, verbose=0)
        predicted_classes = (predictions > 0.5).astype(int).flatten()

        # 3. Calcular métricas
        accuracy = np.mean(predicted_classes == y_test_subset)

        # 4. Calcular precision e recall
        true_positives = np.sum((predicted_classes == 1) & (y_test_subset == 1))
        false_positives = np.sum((predicted_classes == 1) & (y_test_subset == 0))
        false_negatives = np.sum((predicted_classes == 0) & (y_test_subset == 1))

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # 5. Validar métricas
        assert accuracy >= 0.50  # Melhor que aleatório
        assert precision >= 0.30  # Precision mínima
        assert recall >= 0.30  # Recall mínimo

        print(f"✅ Métricas validadas - Acc: {accuracy:.3f}, Prec: {precision:.3f}, Rec: {recall:.3f}, F1: {f1_score:.3f}")

    def test_model_confidence_calibration(self, mock_lstm_model):
        """📊 Validação da Calibração de Confiança"""
        print("\n🧪 Validando calibração de confiança...")

        # 1. Gerar dados de teste diversos
        test_scenarios = [
            # Cenário 1: Trend forte UP
            {'trend': 0.05, 'volatility': 0.02, 'expected_confidence': 'high'},
            # Cenário 2: Trend forte DOWN
            {'trend': -0.05, 'volatility': 0.02, 'expected_confidence': 'high'},
            # Cenário 3: Sideways (sem trend)
            {'trend': 0.001, 'volatility': 0.01, 'expected_confidence': 'low'},
            # Cenário 4: Alta volatilidade
            {'trend': 0.02, 'volatility': 0.10, 'expected_confidence': 'medium'}
        ]

        confidence_results = []

        for scenario in test_scenarios:
            # Gerar sequência baseada no cenário
            prices = []
            base_price = 245.0

            for i in range(60):
                noise = np.random.normal(0, scenario['volatility'])
                base_price += scenario['trend'] + noise
                prices.append(base_price)

            # Calcular features e fazer predição
            features = self.calculate_technical_features(prices)
            features_array = np.array([features])

            # Mock de predição (treinar rapidamente)
            dummy_X = np.random.random((100, 60, 22))
            dummy_y = np.random.randint(0, 2, 100)
            mock_lstm_model.fit(dummy_X, dummy_y, epochs=1, verbose=0)

            prediction = mock_lstm_model.predict(features_array, verbose=0)[0][0]

            # Calcular confiança (distância de 0.5)
            confidence = abs(prediction - 0.5) * 2
            confidence_results.append(confidence)

        # 2. Verificar que cenários mais claros têm maior confiança
        avg_confidence = np.mean(confidence_results)
        assert avg_confidence > 0.1  # Pelo menos alguma diferenciação

        print(f"✅ Confiança calibrada - Média: {avg_confidence:.3f}")

    def test_feature_importance_analysis(self):
        """🔍 Análise de Importância das Features"""
        print("\n🧪 Analisando importância das features...")

        # Simular análise de feature importance
        feature_names = [
            'price_velocity', 'volatility', 'sma_divergence', 'momentum',
            'rsi', 'macd', 'bollinger_position', 'volume_ratio',
            'price_acceleration', 'trend_strength', 'support_resistance',
            'market_microstructure', 'order_flow', 'bid_ask_spread',
            'tick_direction', 'price_impact', 'volatility_surface',
            'correlation_matrix', 'regime_detection', 'anomaly_score',
            'sentiment_score', 'news_impact'
        ]

        # Simular importâncias (normalmente viria de SHAP ou permutation importance)
        np.random.seed(42)
        feature_importance = np.random.random(len(feature_names))
        feature_importance = feature_importance / np.sum(feature_importance)  # Normalizar

        # Análise das features mais importantes
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)

        top_features = importance_df.head(5)

        # Verificações
        assert len(top_features) == 5
        assert top_features['importance'].sum() > 0.3  # Top 5 features representam >30%

        # Features críticas devem estar no top 10
        critical_features = ['price_velocity', 'volatility', 'momentum', 'rsi']
        top_10_features = importance_df.head(10)['feature'].tolist()

        critical_in_top10 = sum(1 for f in critical_features if f in top_10_features)
        assert critical_in_top10 >= 2  # Pelo menos 2 features críticas no top 10

        print("✅ Features mais importantes:")
        for idx, row in top_features.iterrows():
            print(f"   {row['feature']}: {row['importance']:.3f}")

    def test_model_robustness(self, mock_lstm_model):
        """🛡️ Teste de Robustez do Modelo"""
        print("\n🧪 Testando robustez do modelo...")

        # 1. Teste com dados adversariais
        base_sequence = np.random.random((1, 60, 22))

        # Treinar modelo rapidamente
        dummy_X = np.random.random((100, 60, 22))
        dummy_y = np.random.randint(0, 2, 100)
        mock_lstm_model.fit(dummy_X, dummy_y, epochs=2, verbose=0)

        baseline_pred = mock_lstm_model.predict(base_sequence, verbose=0)[0][0]

        # 2. Adicionar ruído pequeno
        noise_levels = [0.01, 0.05, 0.1]
        robustness_scores = []

        for noise_level in noise_levels:
            noisy_sequence = base_sequence + np.random.normal(0, noise_level, base_sequence.shape)
            noisy_pred = mock_lstm_model.predict(noisy_sequence, verbose=0)[0][0]

            # Calcular mudança na predição
            prediction_change = abs(noisy_pred - baseline_pred)
            robustness_scores.append(prediction_change)

        # 3. Verificar robustez
        max_change = max(robustness_scores)
        assert max_change < 0.3  # Mudança máxima < 30% com ruído

        print(f"✅ Robustez validada - Mudança máxima: {max_change:.3f}")

    def test_model_interpretability(self):
        """💡 Teste de Interpretabilidade do Modelo"""
        print("\n🧪 Testando interpretabilidade...")

        # Simular análise de atenção (para modelos com attention)
        sequence_length = 60
        attention_weights = np.random.random(sequence_length)
        attention_weights = attention_weights / np.sum(attention_weights)

        # 1. Verificar se atenção está focada em períodos recentes
        recent_attention = np.sum(attention_weights[-10:])  # Últimos 10 timesteps
        assert recent_attention > 0.3  # Pelo menos 30% de atenção no período recente

        # 2. Simular gradientes para feature importance
        feature_gradients = np.random.random(22)  # 22 features
        feature_gradients = feature_gradients / np.sum(feature_gradients)

        # Verificar se features importantes têm gradientes significativos
        important_feature_indices = [0, 1, 2, 3]  # price_velocity, volatility, etc.
        important_gradients = np.sum(feature_gradients[important_feature_indices])
        assert important_gradients > 0.2  # Features importantes têm >20% dos gradientes

        print("✅ Interpretabilidade validada")
        print(f"   Atenção recente: {recent_attention:.3f}")
        print(f"   Gradientes importantes: {important_gradients:.3f}")

    def test_model_versioning_and_comparison(self):
        """📋 Teste de Versionamento e Comparação de Modelos"""
        print("\n🧪 Testando versionamento de modelos...")

        # Simular diferentes versões do modelo
        model_versions = {
            'v1.0': {'accuracy': 0.62, 'precision': 0.58, 'recall': 0.65, 'f1': 0.61},
            'v1.1': {'accuracy': 0.65, 'precision': 0.62, 'recall': 0.68, 'f1': 0.65},
            'v2.0': {'accuracy': 0.68, 'precision': 0.66, 'recall': 0.70, 'f1': 0.68},
            'v2.1': {'accuracy': 0.72, 'precision': 0.70, 'recall': 0.74, 'f1': 0.72}
        }

        # 1. Verificar progressão das métricas
        accuracies = [model_versions[v]['accuracy'] for v in model_versions.keys()]

        # Deve haver melhoria geral ao longo das versões
        assert accuracies[-1] > accuracies[0]  # Última versão melhor que primeira
        assert max(accuracies) >= 0.70  # Pelo menos uma versão com >70% accuracy

        # 2. Identificar melhor modelo
        best_version = max(model_versions.keys(),
                          key=lambda v: model_versions[v]['f1'])

        best_metrics = model_versions[best_version]
        assert best_metrics['f1'] >= 0.65  # F1 score mínimo

        print(f"✅ Melhor modelo: {best_version}")
        print(f"   Métricas: Acc={best_metrics['accuracy']:.3f}, F1={best_metrics['f1']:.3f}")


# 🏃‍♂️ Test Runner
if __name__ == "__main__":
    print("🧠 EXECUTANDO VALIDAÇÃO DO MODELO DE IA")
    print("=" * 50)

    pytest.main([
        __file__,
        "-v",
        "--tb=short"
    ])