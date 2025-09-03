#!/usr/bin/env python3
"""
Script de teste para o Sistema de Gestão de Capital
Demonstra o funcionamento do reinvestimento progressivo e martingale
"""

import asyncio
from capital_manager import CapitalManager, TradeResult

def print_separator(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_trade_info(trade_num, amount, result, profit_loss, next_amount, is_martingale=False):
    result_icon = "✅ WIN" if result == TradeResult.WIN else "❌ LOSS"
    martingale_info = " (Martingale)" if is_martingale else ""
    
    print(f"Trade {trade_num}: ${amount:.2f} → {result_icon} → P/L: ${profit_loss:.2f}{martingale_info}")
    print(f"          Next amount: ${next_amount:.2f}")

def test_basic_functionality():
    """Teste básico do funcionamento"""
    print_separator("TESTE 1: Funcionalidade Básica")
    
    # Inicializar com capital de $10
    cm = CapitalManager(initial_capital=10.0, reinvestment_rate=0.20, martingale_multiplier=1.25)
    
    print(f"💰 Capital inicial: ${cm.initial_capital}")
    print(f"📈 Taxa de reinvestimento: {cm.reinvestment_rate*100}%")
    print(f"🎰 Multiplicador martingale: {cm.martingale_multiplier}x")
    
    # Cenário: WIN → WIN → LOSS → WIN
    scenarios = [
        (TradeResult.WIN, 18.0),   # Win: payout $18 (profit $8)
        (TradeResult.WIN, 18.0),   # Win: payout $18 (profit varies)
        (TradeResult.LOSS, 0.0),   # Loss: no payout
        (TradeResult.WIN, 18.0),   # Win: recovery
    ]
    
    print("\n📊 Simulando sequência: WIN → WIN → LOSS → WIN")
    print("-" * 50)
    
    for i, (result, payout) in enumerate(scenarios, 1):
        amount = cm.get_next_trade_amount()
        trade = cm.record_trade(f"T{i}", amount, result, payout)
        next_amount = cm.get_next_trade_amount()
        
        print_trade_info(i, amount, result, trade.profit_loss, next_amount, trade.is_martingale)
    
    stats = cm.get_stats()
    print(f"\n📈 Resultado final:")
    print(f"   Lucro acumulado: ${stats['capital_info']['accumulated_profit']:.2f}")
    print(f"   Taxa de vitória: {stats['session_stats']['win_rate']:.1f}%")
    print(f"   Total investido: ${stats['session_stats']['total_invested']:.2f}")

def test_loss_sequence():
    """Teste de sequência de perdas com martingale"""
    print_separator("TESTE 2: Sequência de Perdas (Martingale)")
    
    cm = CapitalManager(initial_capital=10.0)
    
    # Cenário: LOSS → LOSS → LOSS → WIN
    scenarios = [
        (TradeResult.LOSS, 0.0),   # Loss 1
        (TradeResult.LOSS, 0.0),   # Loss 2 
        (TradeResult.LOSS, 0.0),   # Loss 3
        (TradeResult.WIN, 30.0),   # Win: recovery with good payout
    ]
    
    print("🎰 Simulando sequência de martingale: L → L → L → W")
    print("-" * 50)
    
    for i, (result, payout) in enumerate(scenarios, 1):
        amount = cm.get_next_trade_amount()
        trade = cm.record_trade(f"L{i}", amount, result, payout)
        next_amount = cm.get_next_trade_amount()
        
        print_trade_info(i, amount, result, trade.profit_loss, next_amount, trade.is_martingale)
    
    stats = cm.get_stats()
    risk = cm.get_risk_assessment()
    
    print(f"\n⚠️  Análise de risco:")
    print(f"   Sequência máxima de perdas: {stats['session_stats']['max_sequence_length']}")
    print(f"   Drawdown máximo: ${stats['session_stats']['max_drawdown']:.2f}")
    print(f"   Nível de risco: {risk['risk_level']}")
    for rec in risk['recommendations'][:2]:  # Show first 2 recommendations
        print(f"   {rec}")

def test_reinvestment_progression():
    """Teste do sistema de reinvestimento progressivo"""
    print_separator("TESTE 3: Reinvestimento Progressivo")
    
    cm = CapitalManager(initial_capital=10.0)
    
    print("💹 Simulando sequência de vitórias com reinvestimento:")
    print("-" * 50)
    
    # Sequência de vitórias para demonstrar reinvestimento
    for i in range(5):
        amount = cm.get_next_trade_amount()
        payout = amount * 1.8  # 80% de lucro
        trade = cm.record_trade(f"W{i+1}", amount, TradeResult.WIN, payout)
        next_amount = cm.get_next_trade_amount()
        
        print_trade_info(i+1, amount, TradeResult.WIN, trade.profit_loss, next_amount)
    
    stats = cm.get_stats()
    print(f"\n🚀 Crescimento do capital:")
    print(f"   Capital inicial: ${cm.initial_capital}")
    print(f"   Próxima entrada: ${cm.get_next_trade_amount():.2f}")
    print(f"   Crescimento: {((cm.get_next_trade_amount() / cm.initial_capital) - 1) * 100:.1f}%")
    print(f"   Lucro acumulado: ${stats['capital_info']['accumulated_profit']:.2f}")

def test_simulation_feature():
    """Teste da funcionalidade de simulação"""
    print_separator("TESTE 4: Simulação de Estratégia")
    
    cm = CapitalManager(initial_capital=10.0)
    
    # Simular diferentes cenários
    scenarios = [
        ([TradeResult.WIN] * 5, "5 vitórias seguidas"),
        ([TradeResult.LOSS] * 3 + [TradeResult.WIN], "3 perdas + 1 vitória"),
        ([TradeResult.WIN, TradeResult.LOSS, TradeResult.WIN, TradeResult.LOSS, TradeResult.WIN], "Alternando W-L-W-L-W"),
    ]
    
    for results, description in scenarios:
        simulation = cm.simulate_sequence(results)
        final_pnl = simulation["total_profit_loss"]
        
        print(f"\n🎯 Cenário: {description}")
        print(f"   Resultado: ${final_pnl:.2f} ({'Lucro' if final_pnl > 0 else 'Prejuízo'})")
        print(f"   Trades: {len(results)} | Win Rate: {(results.count(TradeResult.WIN)/len(results))*100:.1f}%")

def main():
    """Executar todos os testes"""
    print_separator("🤖 TESTE DO SISTEMA DE GESTÃO DE CAPITAL - BOTDERIV")
    
    print("""
📋 Sistema implementado:
   • Capital inicial: $10 (configurável)
   • Reinvestimento: 20% do lucro anterior
   • Martingale: 1.25x na perda
   • Reset: volta ao inicial após recuperar sequência de perdas
   
🔍 Executando testes...
    """)
    
    test_basic_functionality()
    test_loss_sequence() 
    test_reinvestment_progression()
    test_simulation_feature()
    
    print_separator("✅ TODOS OS TESTES CONCLUÍDOS")
    print("""
🎯 Sistema pronto para integração!

Próximos passos:
1. Testar API endpoints: /capital/stats, /capital/simulate
2. Integrar com frontend para visualização 
3. Configurar alertas de risco
4. Implementar estratégias de trading automático

Use: python test_capital_management.py
    """)

if __name__ == "__main__":
    main()