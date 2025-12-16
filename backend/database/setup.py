#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para criar database SQLite trades_history.db com schema inicial

Execução: python database/setup.py
"""

import sys
import io
import sqlite3

# Force UTF-8 encoding for Windows compatibility
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os
from pathlib import Path
from datetime import datetime, timedelta
import json

# Path para o database (DEVE SER trades_history.db para coincidir com trades_history_manager.py)
DB_PATH = Path(__file__).parent.parent / "trades_history.db"

# Verificar se já existe
if DB_PATH.exists():
    print(f"⚠️  Database já existe em: {DB_PATH}")
    print("   Pulando criação para não sobrescrever dados existentes.")
    print(f"   Para recriar, delete manualmente: rm {DB_PATH}")
    exit(0)

# Criar database
print(f"✅ Criando database em: {DB_PATH}")
conn = sqlite3.connect(str(DB_PATH))
cursor = conn.cursor()

# Schema da tabela trades_history (EXATO como em trades_history_manager.py)
CREATE_TRADES_TABLE = """
CREATE TABLE IF NOT EXISTS trades_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    symbol TEXT NOT NULL,
    trade_type TEXT NOT NULL CHECK(trade_type IN ('BUY', 'SELL', 'CALL', 'PUT')),
    entry_price REAL NOT NULL,
    exit_price REAL,
    stake REAL NOT NULL,
    profit_loss REAL,
    result TEXT CHECK(result IN ('win', 'loss', 'pending')),
    confidence REAL CHECK(confidence >= 0 AND confidence <= 100),
    strategy TEXT CHECK(strategy IN ('ml', 'technical', 'hybrid', 'order_flow')),
    indicators_used TEXT,
    ml_prediction REAL,
    order_flow_signal TEXT,
    stop_loss REAL,
    take_profit REAL,
    exit_reason TEXT,
    notes TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

# Índices para performance
CREATE_INDEXES = [
    'CREATE INDEX IF NOT EXISTS idx_timestamp ON trades_history(timestamp DESC);',
    'CREATE INDEX IF NOT EXISTS idx_symbol ON trades_history(symbol);',
    'CREATE INDEX IF NOT EXISTS idx_result ON trades_history(result);',
    'CREATE INDEX IF NOT EXISTS idx_strategy ON trades_history(strategy);',
    'CREATE INDEX IF NOT EXISTS idx_created_at ON trades_history(created_at DESC);'
]

# Executar criação de tabela
cursor.execute(CREATE_TRADES_TABLE)
print("✅ Tabela trades_history criada")

# Criar índices
for i, index_sql in enumerate(CREATE_INDEXES, 1):
    cursor.execute(index_sql)
    print(f"✅ Índice {i}/{len(CREATE_INDEXES)} criado")

# Trades de exemplo (schema correto: trade_type, stake, indicators_used, ml_prediction REAL)
now = datetime.now()
sample_trades = [
    {
        'timestamp': (now - timedelta(days=1)).isoformat(),
        'symbol': 'R_100',
        'trade_type': 'CALL',
        'entry_price': 100.50,
        'exit_price': 101.25,
        'stake': 10.0,
        'profit_loss': 7.5,
        'result': 'win',
        'confidence': 75.0,
        'strategy': 'ml',
        'indicators_used': json.dumps({'rsi': 65, 'macd': 'bullish'}),
        'ml_prediction': 0.75,
        'order_flow_signal': 'bullish',
        'stop_loss': 99.50,
        'take_profit': 102.50,
        'exit_reason': 'take_profit',
        'notes': 'Trade de exemplo - Paper Trading'
    },
    {
        'timestamp': (now - timedelta(hours=12)).isoformat(),
        'symbol': 'R_100',
        'trade_type': 'PUT',
        'entry_price': 99.75,
        'exit_price': 99.50,
        'stake': 10.0,
        'profit_loss': 2.5,
        'result': 'win',
        'confidence': 68.0,
        'strategy': 'ml',
        'indicators_used': json.dumps({'rsi': 35, 'macd': 'bearish'}),
        'ml_prediction': 0.68,
        'order_flow_signal': 'bearish',
        'stop_loss': 100.75,
        'take_profit': 97.75,
        'exit_reason': 'take_profit',
        'notes': 'Trade de exemplo - Paper Trading'
    },
    {
        'timestamp': (now - timedelta(hours=2)).isoformat(),
        'symbol': 'R_100',
        'trade_type': 'CALL',
        'entry_price': 101.00,
        'exit_price': 100.50,
        'stake': 10.0,
        'profit_loss': -5.0,
        'result': 'loss',
        'confidence': 62.0,
        'strategy': 'ml',
        'indicators_used': json.dumps({'rsi': 70, 'macd': 'bullish'}),
        'ml_prediction': 0.62,
        'order_flow_signal': 'bullish',
        'stop_loss': 100.00,
        'take_profit': 103.00,
        'exit_reason': 'stop_loss',
        'notes': 'Trade de exemplo - Stop Loss acionado'
    }
]

# Inserir trades de exemplo
INSERT_SQL = """
    INSERT INTO trades_history (
        timestamp, symbol, trade_type, entry_price, exit_price, stake,
        profit_loss, result, confidence, strategy, indicators_used,
        ml_prediction, order_flow_signal, stop_loss, take_profit, exit_reason, notes
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

for i, trade in enumerate(sample_trades, 1):
    cursor.execute(INSERT_SQL, (
        trade['timestamp'], trade['symbol'], trade['trade_type'], trade['entry_price'],
        trade['exit_price'], trade['stake'], trade['profit_loss'], trade['result'],
        trade['confidence'], trade['strategy'], trade['indicators_used'],
        trade['ml_prediction'], trade['order_flow_signal'], trade['stop_loss'],
        trade['take_profit'], trade['exit_reason'], trade['notes']
    ))
    print(f"✅ Trade de exemplo {i}/{len(sample_trades)} inserido")

# Commit e fechar
conn.commit()

# Verificar
cursor.execute('SELECT COUNT(*) FROM trades_history')
count = cursor.fetchone()[0]

conn.close()

print(f"\n✅ Setup completo! Database tem {count} trades de exemplo")
print("\nPróximos passos:")
print("1. Verificar arquivo existe: ls -lh backend/trades_history.db")
print("2. Testar endpoint: curl http://localhost:8000/api/trades/stats")
print("3. Verificar frontend: https://botderiv.roilabs.com.br/trade-history\n")
print("🔒 Database trades_history.db criado com sucesso")
