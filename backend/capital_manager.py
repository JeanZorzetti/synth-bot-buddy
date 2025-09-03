"""
Capital Management System - BotDeriv
Sistema inteligente de gestão de capital com:
- Reinvestimento progressivo de 20% do lucro
- Martingale de 1.25x em perdas
- Reset ao capital inicial após recuperação de perdas
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class TradeResult(Enum):
    WIN = "win"
    LOSS = "loss"
    PENDING = "pending"

@dataclass
class TradeRecord:
    """Registro individual de um trade"""
    id: str
    timestamp: datetime
    amount: float
    result: TradeResult
    payout: float = 0.0
    profit_loss: float = 0.0
    sequence_number: int = 0
    is_martingale: bool = False
    
    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat(),
            'result': self.result.value
        }

class CapitalManager:
    """
    Gerenciador de Capital com Reinvestimento Progressivo e Martingale
    
    Regras:
    1. Capital inicial (x) = base de cálculo
    2. Win: reinveste capital + 20% do último lucro
    3. Loss: aplica martingale 1.25x na última entrada perdida
    4. Após sequência de loss + win: retorna ao capital inicial
    """
    
    def __init__(self, initial_capital: float = 10.0, reinvestment_rate: float = 0.20, martingale_multiplier: float = 1.25):
        # Configurações
        self.initial_capital = initial_capital
        self.reinvestment_rate = reinvestment_rate
        self.martingale_multiplier = martingale_multiplier
        
        # Estado atual
        self.current_capital = initial_capital
        self.total_balance = 0.0
        self.accumulated_profit = 0.0
        self.last_profit = 0.0
        
        # Controle de sequências
        self.current_sequence = 0
        self.is_in_loss_sequence = False
        self.loss_sequence_count = 0
        self.win_streak = 0
        
        # Histórico
        self.trade_history: List[TradeRecord] = []
        self.session_stats = {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0.0,
            'total_invested': 0.0,
            'total_returned': 0.0,
            'net_profit': 0.0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0,
            'max_sequence_length': 0,
            'current_sequence_length': 0
        }
        
        logger.info(f"Capital Manager initialized - Initial Capital: ${initial_capital}")
    
    def calculate_next_amount(self) -> float:
        """
        Calcula o próximo valor a ser investido baseado nas regras:
        - Se ganhou: capital inicial + 20% do último lucro
        - Se perdeu: martingale 1.25x da última entrada
        - Se recuperou de sequência de perdas: volta ao capital inicial
        """
        
        # Se não há histórico, usar capital inicial
        if not self.trade_history:
            return self.initial_capital
        
        last_trade = self.trade_history[-1]
        
        # Se a última operação foi WIN
        if last_trade.result == TradeResult.WIN:
            if self.is_in_loss_sequence:
                # Acabou de sair de uma sequência de perdas - volta ao inicial
                logger.info(f"Recovered from loss sequence. Returning to initial capital: ${self.initial_capital}")
                self.is_in_loss_sequence = False
                self.loss_sequence_count = 0
                return self.initial_capital
            else:
                # Estava ganhando - reinveste com 20% do lucro
                reinvestment = self.initial_capital + (self.last_profit * self.reinvestment_rate)
                logger.info(f"Win streak continues. Reinvesting: ${reinvestment:.2f} (20% of ${self.last_profit:.2f} profit)")
                return round(reinvestment, 2)
        
        # Se a última operação foi LOSS
        elif last_trade.result == TradeResult.LOSS:
            self.is_in_loss_sequence = True
            self.loss_sequence_count += 1
            
            # Aplica martingale na última entrada
            martingale_amount = last_trade.amount * self.martingale_multiplier
            logger.info(f"Loss detected. Applying martingale: ${martingale_amount:.2f} (${last_trade.amount} x {self.martingale_multiplier})")
            return round(martingale_amount, 2)
        
        # Fallback - não deveria chegar aqui
        return self.initial_capital
    
    def record_trade(self, trade_id: str, amount: float, result: TradeResult, payout: float = 0.0) -> TradeRecord:
        """
        Registra um trade e atualiza as estatísticas
        """
        self.current_sequence += 1
        profit_loss = 0.0
        
        if result == TradeResult.WIN:
            profit_loss = payout - amount
            self.last_profit = profit_loss
            self.win_streak += 1
            self.session_stats['wins'] += 1
        elif result == TradeResult.LOSS:
            profit_loss = -amount
            self.win_streak = 0
            self.session_stats['losses'] += 1
        
        # Criar registro do trade
        trade_record = TradeRecord(
            id=trade_id,
            timestamp=datetime.now(),
            amount=amount,
            result=result,
            payout=payout,
            profit_loss=profit_loss,
            sequence_number=self.current_sequence,
            is_martingale=self.is_in_loss_sequence
        )
        
        self.trade_history.append(trade_record)
        
        # Atualizar estatísticas
        self._update_stats(trade_record)
        
        logger.info(f"Trade recorded: {trade_record.id} | Amount: ${amount} | Result: {result.value} | P/L: ${profit_loss:.2f}")
        
        return trade_record
    
    def _update_stats(self, trade: TradeRecord):
        """Atualiza estatísticas da sessão"""
        self.session_stats['total_trades'] += 1
        self.session_stats['total_invested'] += trade.amount
        self.session_stats['total_returned'] += trade.payout
        self.session_stats['net_profit'] += trade.profit_loss
        
        self.accumulated_profit += trade.profit_loss
        
        # Win rate
        if self.session_stats['total_trades'] > 0:
            self.session_stats['win_rate'] = (self.session_stats['wins'] / self.session_stats['total_trades']) * 100
        
        # Drawdown
        if self.accumulated_profit < 0:
            self.session_stats['current_drawdown'] = abs(self.accumulated_profit)
            if self.session_stats['current_drawdown'] > self.session_stats['max_drawdown']:
                self.session_stats['max_drawdown'] = self.session_stats['current_drawdown']
        else:
            self.session_stats['current_drawdown'] = 0
        
        # Sequence tracking
        if trade.result == TradeResult.LOSS:
            self.session_stats['current_sequence_length'] += 1
            if self.session_stats['current_sequence_length'] > self.session_stats['max_sequence_length']:
                self.session_stats['max_sequence_length'] = self.session_stats['current_sequence_length']
        else:
            self.session_stats['current_sequence_length'] = 0
    
    def get_next_trade_amount(self) -> float:
        """Retorna o valor para o próximo trade"""
        return self.calculate_next_amount()
    
    def get_stats(self) -> Dict:
        """Retorna estatísticas completas da sessão"""
        return {
            'capital_info': {
                'initial_capital': self.initial_capital,
                'current_capital': self.get_next_trade_amount(),
                'accumulated_profit': self.accumulated_profit,
                'last_profit': self.last_profit,
                'is_in_loss_sequence': self.is_in_loss_sequence,
                'loss_sequence_count': self.loss_sequence_count,
                'win_streak': self.win_streak
            },
            'session_stats': self.session_stats,
            'recent_trades': [trade.to_dict() for trade in self.trade_history[-5:]]  # Last 5 trades
        }
    
    def get_risk_assessment(self) -> Dict:
        """Avalia o risco atual da estratégia"""
        next_amount = self.get_next_trade_amount()
        risk_percentage = (next_amount / self.initial_capital) * 100
        
        risk_level = "LOW"
        if risk_percentage > 200:
            risk_level = "HIGH"
        elif risk_percentage > 150:
            risk_level = "MEDIUM"
        
        return {
            'next_amount': next_amount,
            'risk_percentage': round(risk_percentage, 1),
            'risk_level': risk_level,
            'recommendations': self._get_risk_recommendations(risk_level, risk_percentage)
        }
    
    def _get_risk_recommendations(self, risk_level: str, risk_percentage: float) -> List[str]:
        """Gera recomendações baseadas no nível de risco"""
        recommendations = []
        
        if risk_level == "HIGH":
            recommendations.extend([
                "⚠️ Alto risco detectado! Considere parar e avaliar.",
                f"💰 Próxima entrada: {risk_percentage:.1f}% do capital inicial",
                "🛑 Considere definir um stop loss para a sessão"
            ])
        elif risk_level == "MEDIUM":
            recommendations.extend([
                "⚡ Risco moderado. Monitore cuidadosamente.",
                f"📊 Próxima entrada: {risk_percentage:.1f}% do capital inicial"
            ])
        else:
            recommendations.extend([
                "✅ Risco baixo. Continue operando normalmente.",
                "📈 Sistema de reinvestimento ativo"
            ])
        
        return recommendations
    
    def reset_session(self):
        """Reseta a sessão mantendo configurações"""
        logger.info("Resetting capital management session")
        
        self.current_capital = self.initial_capital
        self.accumulated_profit = 0.0
        self.last_profit = 0.0
        self.current_sequence = 0
        self.is_in_loss_sequence = False
        self.loss_sequence_count = 0
        self.win_streak = 0
        
        self.trade_history.clear()
        self.session_stats = {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0.0,
            'total_invested': 0.0,
            'total_returned': 0.0,
            'net_profit': 0.0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0,
            'max_sequence_length': 0,
            'current_sequence_length': 0
        }
    
    def export_history(self) -> List[Dict]:
        """Exporta histórico de trades"""
        return [trade.to_dict() for trade in self.trade_history]
    
    def simulate_sequence(self, results: List[TradeResult], payout_multiplier: float = 1.8) -> Dict:
        """
        Simula uma sequência de resultados para testar a estratégia
        """
        original_state = {
            'trade_history': self.trade_history.copy(),
            'session_stats': self.session_stats.copy(),
            'accumulated_profit': self.accumulated_profit,
            'is_in_loss_sequence': self.is_in_loss_sequence,
            'loss_sequence_count': self.loss_sequence_count,
            'win_streak': self.win_streak,
            'current_sequence': self.current_sequence,
            'last_profit': self.last_profit
        }
        
        simulation_results = []
        
        for i, result in enumerate(results):
            amount = self.get_next_trade_amount()
            payout = amount * payout_multiplier if result == TradeResult.WIN else 0.0
            
            trade = self.record_trade(f"SIM_{i+1}", amount, result, payout)
            simulation_results.append({
                'trade_number': i + 1,
                'amount': amount,
                'result': result.value,
                'profit_loss': trade.profit_loss,
                'accumulated_profit': self.accumulated_profit
            })
        
        final_stats = self.get_stats()
        
        # Restaurar estado original
        self.trade_history = original_state['trade_history']
        self.session_stats = original_state['session_stats']
        self.accumulated_profit = original_state['accumulated_profit']
        self.is_in_loss_sequence = original_state['is_in_loss_sequence']
        self.loss_sequence_count = original_state['loss_sequence_count']
        self.win_streak = original_state['win_streak']
        self.current_sequence = original_state['current_sequence']
        self.last_profit = original_state['last_profit']
        
        return {
            'simulation_results': simulation_results,
            'final_stats': final_stats,
            'total_profit_loss': sum(r['profit_loss'] for r in simulation_results)
        }