"""
ANALYTICS ROUTES - Análise estatística de histórico de trades

Endpoints para análise avançada de performance, risco e sobrevivência
"""
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import logging
import pandas as pd
import numpy as np

from database.abutre_repository_async import get_async_repository

router = APIRouter(prefix="/api/abutre/analytics", tags=["Analytics"])
logger = logging.getLogger(__name__)


class SurvivalMetrics(BaseModel):
    """Métricas de sobrevivência do bot"""
    max_level_reached: int
    max_level_frequency: int  # Quantas vezes chegou no nível máximo
    death_sequences: List[Dict[str, Any]]  # Sequências que chegaram perto da morte
    recovery_factor: float  # Lucro / Risco máximo
    critical_hours: List[int]  # Horários com mais risco


class PerformanceMetrics(BaseModel):
    """Métricas de performance gerais"""
    total_trades: int
    win_rate: float
    profit_factor: float
    total_profit: float
    max_drawdown: float
    avg_win: float
    avg_loss: float
    max_win_streak: int
    max_loss_streak: int
    sharpe_ratio: Optional[float] = None


class HourlyAnalysis(BaseModel):
    """Análise por horário"""
    hour: int
    trades: int
    win_rate: float
    avg_profit: float
    risk_score: float  # 0-10, baseado em max_level


@router.get("/survival", response_model=SurvivalMetrics)
async def get_survival_metrics(
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None)
):
    """
    Análise de sobrevivência: detecta quando o bot chegou perto da "morte"

    Morte = Atingir nível máximo permitido (ex: nível 9 com stake $89.60)
    """
    try:
        repo = await get_async_repository()

        # Buscar trades
        if date_from and date_to:
            dt_from = datetime.fromisoformat(date_from + "T00:00:00")
            dt_to = datetime.fromisoformat(date_to + "T23:59:59")
            trades = await repo.get_trades_by_period(dt_from, dt_to, limit=10000)
        else:
            trades = await repo.get_recent_trades(limit=10000)

        if not trades:
            raise HTTPException(status_code=404, detail="No trades found")

        # Converter para DataFrame
        df = pd.DataFrame(trades)

        # 1. Nível máximo atingido
        max_level = df['max_level_reached'].max()
        max_level_freq = len(df[df['max_level_reached'] == max_level])

        # 2. Sequências de morte (nível >= 7)
        death_threshold = 7
        death_trades = df[df['max_level_reached'] >= death_threshold].to_dict('records')
        death_sequences = [
            {
                "trade_id": t['trade_id'],
                "level": t['max_level_reached'],
                "stake": t['initial_stake'] * (2 ** (t['max_level_reached'] - 1)),
                "time": t['entry_time'],
                "result": t['result']
            }
            for t in death_trades
        ]

        # 3. Fator de recuperação (Total Profit / Max Risco)
        total_profit = df['profit'].sum()
        max_stake = df['initial_stake'].max() * (2 ** (max_level - 1))
        recovery_factor = total_profit / max_stake if max_stake > 0 else 0

        # 4. Horários críticos (mais trades com nível alto)
        df['hour'] = pd.to_datetime(df['entry_time']).dt.hour
        critical_df = df[df['max_level_reached'] >= 5].groupby('hour').size()
        critical_hours = critical_df.nlargest(5).index.tolist() if len(critical_df) > 0 else []

        return SurvivalMetrics(
            max_level_reached=int(max_level),
            max_level_frequency=max_level_freq,
            death_sequences=death_sequences,
            recovery_factor=round(recovery_factor, 2),
            critical_hours=critical_hours
        )

    except Exception as e:
        logger.error(f"Error calculating survival metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate survival metrics: {str(e)}"
        )


@router.get("/performance", response_model=PerformanceMetrics)
async def get_performance_metrics(
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None)
):
    """
    Métricas de performance: win rate, profit factor, drawdown, streaks
    """
    try:
        repo = await get_async_repository()

        # Buscar trades
        if date_from and date_to:
            dt_from = datetime.fromisoformat(date_from + "T00:00:00")
            dt_to = datetime.fromisoformat(date_to + "T23:59:59")
            trades = await repo.get_trades_by_period(dt_from, dt_to, limit=10000)
        else:
            trades = await repo.get_recent_trades(limit=10000)

        if not trades:
            raise HTTPException(status_code=404, detail="No trades found")

        df = pd.DataFrame(trades)

        # 1. Total de trades
        total_trades = len(df)

        # 2. Win Rate
        wins = df[df['result'] == 'WIN']
        losses = df[df['result'] == 'LOSS']
        win_rate = len(wins) / total_trades if total_trades > 0 else 0

        # 3. Profit Factor
        total_wins = wins['profit'].sum() if len(wins) > 0 else 0
        total_losses = abs(losses['profit'].sum()) if len(losses) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        # 4. Total Profit
        total_profit = df['profit'].sum()

        # 5. Max Drawdown
        df = df.sort_values('entry_time')
        df['cumulative_profit'] = df['profit'].cumsum()
        df['running_max'] = df['cumulative_profit'].cummax()
        df['drawdown'] = df['cumulative_profit'] - df['running_max']
        max_drawdown = df['drawdown'].min()

        # 6. Avg Win/Loss
        avg_win = wins['profit'].mean() if len(wins) > 0 else 0
        avg_loss = losses['profit'].mean() if len(losses) > 0 else 0

        # 7. Max Streaks
        df['is_win'] = df['result'] == 'WIN'
        max_win_streak = 0
        max_loss_streak = 0
        current_win_streak = 0
        current_loss_streak = 0

        for is_win in df['is_win']:
            if is_win:
                current_win_streak += 1
                current_loss_streak = 0
                max_win_streak = max(max_win_streak, current_win_streak)
            else:
                current_loss_streak += 1
                current_win_streak = 0
                max_loss_streak = max(max_loss_streak, current_loss_streak)

        # 8. Sharpe Ratio (opcional, precisa de mais dados)
        returns = df['profit']
        sharpe_ratio = None
        if len(returns) > 1:
            sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0

        return PerformanceMetrics(
            total_trades=total_trades,
            win_rate=round(win_rate * 100, 2),
            profit_factor=round(profit_factor, 2),
            total_profit=round(total_profit, 2),
            max_drawdown=round(max_drawdown, 2),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            max_win_streak=max_win_streak,
            max_loss_streak=max_loss_streak,
            sharpe_ratio=round(sharpe_ratio, 2) if sharpe_ratio is not None else None
        )

    except Exception as e:
        logger.error(f"Error calculating performance metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate performance metrics: {str(e)}"
        )


@router.get("/hourly", response_model=List[HourlyAnalysis])
async def get_hourly_analysis(
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None)
):
    """
    Análise por horário: identifica horários mais arriscados e lucrativos
    """
    try:
        repo = await get_async_repository()

        # Buscar trades
        if date_from and date_to:
            dt_from = datetime.fromisoformat(date_from + "T00:00:00")
            dt_to = datetime.fromisoformat(date_to + "T23:59:59")
            trades = await repo.get_trades_by_period(dt_from, dt_to, limit=10000)
        else:
            trades = await repo.get_recent_trades(limit=10000)

        if not trades:
            raise HTTPException(status_code=404, detail="No trades found")

        df = pd.DataFrame(trades)
        df['hour'] = pd.to_datetime(df['entry_time']).dt.hour

        # Agrupar por horário
        hourly_stats = []

        for hour in range(24):
            hour_df = df[df['hour'] == hour]

            if len(hour_df) == 0:
                continue

            trades_count = len(hour_df)
            wins = len(hour_df[hour_df['result'] == 'WIN'])
            win_rate = wins / trades_count if trades_count > 0 else 0
            avg_profit = hour_df['profit'].mean()

            # Risk score baseado em max_level (0-10)
            avg_level = hour_df['max_level_reached'].mean()
            risk_score = min(10, avg_level)  # Nível 10+ = score 10

            hourly_stats.append(HourlyAnalysis(
                hour=hour,
                trades=trades_count,
                win_rate=round(win_rate * 100, 2),
                avg_profit=round(avg_profit, 2),
                risk_score=round(risk_score, 1)
            ))

        return hourly_stats

    except Exception as e:
        logger.error(f"Error calculating hourly analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate hourly analysis: {str(e)}"
        )


@router.get("/equity-curve")
async def get_equity_curve(
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None)
):
    """
    Retorna a curva de equity (saldo acumulado ao longo do tempo)
    """
    try:
        repo = await get_async_repository()

        # Buscar trades
        if date_from and date_to:
            dt_from = datetime.fromisoformat(date_from + "T00:00:00")
            dt_to = datetime.fromisoformat(date_to + "T23:59:59")
            trades = await repo.get_trades_by_period(dt_from, dt_to, limit=10000)
        else:
            trades = await repo.get_recent_trades(limit=10000)

        if not trades:
            raise HTTPException(status_code=404, detail="No trades found")

        df = pd.DataFrame(trades)
        df = df.sort_values('entry_time')
        df['cumulative_profit'] = df['profit'].cumsum()

        # Preparar dados para gráfico
        equity_data = [
            {
                "timestamp": row['entry_time'],
                "balance": row['balance_after'],
                "cumulative_profit": row['cumulative_profit'],
                "trade_id": row['trade_id']
            }
            for _, row in df.iterrows()
        ]

        return {
            "status": "success",
            "data": equity_data,
            "summary": {
                "initial_balance": df.iloc[0]['balance_after'] - df.iloc[0]['profit'] if len(df) > 0 else 0,
                "final_balance": df.iloc[-1]['balance_after'] if len(df) > 0 else 0,
                "total_profit": df['profit'].sum(),
                "peak_balance": df['balance_after'].max(),
                "lowest_balance": df['balance_after'].min()
            }
        }

    except Exception as e:
        logger.error(f"Error generating equity curve: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate equity curve: {str(e)}"
        )
