"""
🧪 INTEGRATION TESTS - AI Trading Bot
End-to-End System Integration Testing
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from autonomous_trading_engine import AutonomousTradingEngine
from deriv_api_client import DerivAPIClient
from risk_management import RiskManager
from execution_system import AutonomousTradingBot
from deriv_trading_api import DerivTradingAPI


class TestSystemIntegration:
    """🔧 Testes de Integração do Sistema Completo"""

    @pytest.fixture
    async def trading_bot(self):
        """Setup do bot de trading para testes"""
        bot = AutonomousTradingBot()
        await bot.initialize()
        yield bot
        await bot.shutdown()

    @pytest.fixture
    def sample_tick_data(self):
        """Dados de tick simulados para testes"""
        return [
            {
                'symbol': 'R_100',
                'tick': 245.67,
                'epoch': int(time.time()),
                'quote': 245.67,
                'pip_size': 0.01
            },
            {
                'symbol': 'R_100',
                'tick': 245.89,
                'epoch': int(time.time()) + 1,
                'quote': 245.89,
                'pip_size': 0.01
            },
            {
                'symbol': 'R_100',
                'tick': 246.12,
                'epoch': int(time.time()) + 2,
                'quote': 246.12,
                'pip_size': 0.01
            }
        ]

    @pytest.mark.asyncio
    async def test_full_system_integration(self, trading_bot, sample_tick_data):
        """🎯 Teste de Integração Completa do Sistema"""
        print("\n🧪 Testando integração completa do sistema...")

        # 1. Verificar inicialização do sistema
        assert trading_bot.is_initialized
        assert trading_bot.risk_manager is not None
        assert trading_bot.trading_engine is not None

        # 2. Processar dados de tick
        for tick in sample_tick_data:
            await trading_bot.process_tick(tick)

        # 3. Verificar processamento de features
        features = await trading_bot.get_latest_features()
        assert len(features) > 0
        assert 'price_velocity' in features
        assert 'volatility' in features

        print("✅ Sistema integrado funcionando corretamente")

    @pytest.mark.asyncio
    async def test_ai_model_validation(self, trading_bot):
        """🧠 Validação do Modelo de IA"""
        print("\n🧪 Validando modelo de IA...")

        # 1. Verificar carregamento do modelo
        model_status = await trading_bot.get_model_status()
        assert model_status['loaded'] == True
        assert model_status['version'] is not None

        # 2. Testar predição com dados simulados
        test_features = {
            'price_velocity': [0.1, 0.2, -0.1],
            'volatility': [0.05, 0.04, 0.06],
            'momentum': [0.3, 0.4, 0.2],
            'rsi': [65.0, 70.0, 60.0],
            'macd': [0.02, 0.01, 0.03]
        }

        prediction = await trading_bot.make_prediction(test_features)

        # 3. Verificar formato da predição
        assert 'signal_strength' in prediction
        assert 'confidence' in prediction
        assert 'direction' in prediction
        assert prediction['confidence'] >= 0 and prediction['confidence'] <= 1
        assert prediction['direction'] in ['UP', 'DOWN', 'HOLD']

        print(f"✅ Modelo IA validado - Confiança: {prediction['confidence']:.2f}")

    @pytest.mark.asyncio
    async def test_trading_execution_system(self, trading_bot):
        """⚡ Teste do Sistema de Execução de Trading"""
        print("\n🧪 Testando sistema de execução...")

        # 1. Configurar modo de teste (sem execução real)
        await trading_bot.set_test_mode(True)

        # 2. Simular sinal de trading forte
        strong_signal = {
            'signal_strength': 0.85,
            'confidence': 0.92,
            'direction': 'UP',
            'symbol': 'R_100',
            'risk_score': 0.25
        }

        # 3. Processar decisão de trading
        decision = await trading_bot.process_trading_signal(strong_signal)

        # 4. Verificar decisão
        assert decision is not None
        assert decision['action'] in ['BUY', 'SELL', 'HOLD']
        assert decision['position_size'] > 0
        assert decision['confidence'] >= 0.75  # Threshold mínimo

        print(f"✅ Execução validada - Ação: {decision['action']}")

    @pytest.mark.asyncio
    async def test_risk_management_system(self, trading_bot):
        """🛡️ Teste do Sistema de Risk Management"""
        print("\n🧪 Testando risk management...")

        # 1. Configurar cenário de alto risco
        high_risk_scenario = {
            'current_drawdown': 8.5,  # Alto drawdown
            'consecutive_losses': 4,   # Perdas consecutivas
            'position_concentration': 0.15,  # 15% em uma posição
            'market_volatility': 0.08  # Alta volatilidade
        }

        # 2. Avaliar risco
        risk_assessment = await trading_bot.assess_current_risk(high_risk_scenario)

        # 3. Verificar medidas de proteção
        assert risk_assessment['risk_level'] >= 0.7  # Alto risco detectado
        assert risk_assessment['emergency_stop'] == True
        assert risk_assessment['max_position_size'] < 0.02  # Posição reduzida

        print(f"✅ Risk management ativo - Nível: {risk_assessment['risk_level']:.2f}")

    @pytest.mark.asyncio
    async def test_performance_benchmarking(self, trading_bot):
        """📊 Benchmarking de Performance"""
        print("\n🧪 Executando benchmarks de performance...")

        # 1. Teste de latência de processamento
        start_time = time.time()

        for i in range(100):  # 100 ticks
            tick = {
                'symbol': 'R_100',
                'tick': 245.0 + (i * 0.01),
                'epoch': int(time.time()) + i,
                'quote': 245.0 + (i * 0.01),
                'pip_size': 0.01
            }
            await trading_bot.process_tick(tick)

        processing_time = time.time() - start_time
        avg_latency = (processing_time / 100) * 1000  # ms

        # 2. Verificar performance targets
        assert avg_latency < 50  # < 50ms por tick (target: <200ms)

        # 3. Teste de throughput
        start_time = time.time()

        # Simular processamento intenso
        tasks = []
        for i in range(50):
            tick = {
                'symbol': f'R_{100 + (i % 5)}',
                'tick': 245.0 + (i * 0.01),
                'epoch': int(time.time()) + i,
                'quote': 245.0 + (i * 0.01),
                'pip_size': 0.01
            }
            tasks.append(trading_bot.process_tick(tick))

        await asyncio.gather(*tasks)

        total_time = time.time() - start_time
        throughput = 50 / total_time  # ticks por segundo

        assert throughput > 20  # > 20 ticks/segundo (target: >100 ticks/s)

        print(f"✅ Performance OK - Latência: {avg_latency:.1f}ms, Throughput: {throughput:.1f} ticks/s")

    @pytest.mark.asyncio
    async def test_data_pipeline_efficiency(self, trading_bot):
        """🔄 Teste de Eficiência do Pipeline de Dados"""
        print("\n🧪 Testando pipeline de dados...")

        # 1. Simular fluxo contínuo de dados
        data_points = []
        start_time = time.time()

        for i in range(1000):  # 1000 data points
            data_point = {
                'timestamp': datetime.now().isoformat(),
                'price': 245.0 + (i * 0.001),
                'volume': 100 + (i % 50),
                'features': [0.1, 0.2, 0.3, 0.4, 0.5]
            }
            data_points.append(data_point)

        # 2. Processar pipeline
        processed_data = await trading_bot.process_data_pipeline(data_points)

        processing_time = time.time() - start_time

        # 3. Verificar eficiência
        assert len(processed_data) == len(data_points)
        assert processing_time < 5.0  # < 5 segundos para 1000 pontos

        # 4. Verificar qualidade dos dados processados
        for data in processed_data[:10]:  # Verificar primeiros 10
            assert 'normalized_price' in data
            assert 'technical_indicators' in data
            assert 'ml_features' in data

        print(f"✅ Pipeline eficiente - {len(data_points)} pontos em {processing_time:.2f}s")

    @pytest.mark.asyncio
    async def test_memory_usage_optimization(self, trading_bot):
        """💾 Teste de Otimização de Memória"""
        print("\n🧪 Testando otimização de memória...")

        import psutil
        import gc

        # 1. Baseline de memória
        gc.collect()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB

        # 2. Simular carga intensa de dados
        large_dataset = []
        for i in range(10000):  # 10k records
            record = {
                'timestamp': datetime.now().isoformat(),
                'features': [float(j) for j in range(50)],  # 50 features
                'metadata': {'symbol': f'R_{i % 10}', 'session': i // 100}
            }
            large_dataset.append(record)

        # 3. Processar com sistema de limpeza de memória
        await trading_bot.process_large_dataset(large_dataset)

        # 4. Verificar uso de memória após processamento
        gc.collect()
        final_memory = process.memory_info().rss / (1024 * 1024)  # MB
        memory_increase = final_memory - initial_memory

        # 5. Target: < 100MB de aumento para 10k records
        assert memory_increase < 100, f"Uso de memória muito alto: {memory_increase:.1f}MB"

        print(f"✅ Memória otimizada - Aumento: {memory_increase:.1f}MB")

    def test_error_handling_and_recovery(self, trading_bot):
        """🔧 Teste de Tratamento de Erros e Recuperação"""
        print("\n🧪 Testando recuperação de erros...")

        # 1. Simular falha de API
        with pytest.raises(Exception):
            asyncio.run(trading_bot.simulate_api_failure())

        # 2. Verificar recuperação automática
        recovery_status = asyncio.run(trading_bot.check_recovery_status())
        assert recovery_status['auto_recovery_enabled'] == True
        assert recovery_status['max_retry_attempts'] >= 3

        # 3. Simular falha de modelo
        model_backup_status = asyncio.run(trading_bot.test_model_failover())
        assert model_backup_status['backup_model_loaded'] == True

        print("✅ Sistema de recuperação funcionando")


# 🏃‍♂️ Test Runner Personalizado
if __name__ == "__main__":
    print("🚀 EXECUTANDO TESTES DE INTEGRAÇÃO - AI TRADING BOT")
    print("=" * 60)

    # Executar testes com pytest
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto"
    ])