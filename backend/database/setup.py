#!/usr/bin/env python3
"""
Script para criar database SQLite trades.db com schema inicial

Execução: python database/setup.py
"""

import sqlite3
import os
from pathlib import Path
from datetime import datetime, timedelta
import json

# Path para o database
DB_PATH = Path(__file__).parent.parent / "trades.db"

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

# Schema da tabela trades_history
CREATE_TRADES_TABLE = """
CREATE TABLE IF NOT EXISTS trades_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL CHECK(direction IN ('UP', 'DOWN', 'CALL', 'PUT')),
    entry_price REAL NOT NULL,
    exit_price REAL,
    quantity REAL NOT NULL DEFAULT 1.0,
    position_size REAL NOT NULL,
    stop_loss REAL,
    take_profit REAL,
    profit_loss REAL,
    profit_loss_pct REAL,
    result TEXT CHECK(result IN ('win', 'loss', 'pending', 'breakeven')),
    strategy TEXT,
    confidence REAL,
    ml_prediction TEXT,
    indicators TEXT,
    notes TEXT,
    closed_at TEXT,
    duration_seconds INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
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

# Trades de exemplo
now = datetime.now()
sample_trades = [
    {
        'timestamp': (now - timedelta(days=1)).isoformat(),
        'symbol': 'R_100',
        'direction': 'UP',
        'entry_price': 100.50,
        'exit_price': 101.25,
        'quantity': 1.0,
        'position_size': 1000,
        'stop_loss': 99.50,
        'take_profit': 102.50,
        'profit_loss': 7.5,
        'profit_loss_pct': 0.75,
        'result': 'win',
        'strategy': 'ML_Predictor',
        'confidence': 0.75,
        'ml_prediction': 'UP',
        'indicators': json.dumps({'rsi': 65, 'macd': 'bullish'}),
        'notes': 'Trade de exemplo - Paper Trading',
        'closed_at': (now - timedelta(hours=23)).isoformat(),
        'duration_seconds': 3600
    },
    {
        'timestamp': (now - timedelta(hours=12)).isoformat(),
        'symbol': 'R_100',
        'direction': 'DOWN',
        'entry_price': 99.75,
        'exit_price': 99.50,
        'quantity': 1.0,
        'position_size': 1000,
        'stop_loss': 100.75,
        'take_profit': 97.75,
        'profit_loss': 2.5,
        'profit_loss_pct': 0.25,
        'result': 'win',
        'strategy': 'ML_Predictor',
        'confidence': 0.68,
        'ml_prediction': 'DOWN',
        'indicators': json.dumps({'rsi': 35, 'macd': 'bearish'}),
        'notes': 'Trade de exemplo - Paper Trading',
        'closed_at': (now - timedelta(hours=11)).isoformat(),
        'duration_seconds': 3600
    },
    {
        'timestamp': (now - timedelta(hours=2)).isoformat(),
        'symbol': 'R_100',
        'direction': 'UP',
        'entry_price': 101.00,
        'exit_price': 100.50,
        'quantity': 1.0,
        'position_size': 1000,
        'stop_loss': 100.00,
        'take_profit': 103.00,
        'profit_loss': -5.0,
        'profit_loss_pct': -0.50,
        'result': 'loss',
        'strategy': 'ML_Predictor',
        'confidence': 0.62,
        'ml_prediction': 'UP',
        'indicators': json.dumps({'rsi': 70, 'macd': 'bullish'}),
        'notes': 'Trade de exemplo - Stop Loss acionado',
        'closed_at': (now - timedelta(hours=1)).isoformat(),
        'duration_seconds': 3600
    }
]

# Inserir trades de exemplo
INSERT_SQL = """
    INSERT INTO trades_history (
        timestamp, symbol, direction, entry_price, exit_price, quantity,
        position_size, stop_loss, take_profit, profit_loss, profit_loss_pct,
        result, strategy, confidence, ml_prediction, indicators, notes,
        closed_at, duration_seconds
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

for i, trade in enumerate(sample_trades, 1):
    cursor.execute(INSERT_SQL, (
        trade['timestamp'], trade['symbol'], trade['direction'], trade['entry_price'],
        trade['exit_price'], trade['quantity'], trade['position_size'], trade['stop_loss'],
        trade['take_profit'], trade['profit_loss'], trade['profit_loss_pct'], trade['result'],
        trade['strategy'], trade['confidence'], trade['ml_prediction'], trade['indicators'],
        trade['notes'], trade['closed_at'], trade['duration_seconds']
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
print("1. Verificar arquivo existe: ls -lh backend/trades.db")
print("2. Testar endpoint: curl http://localhost:8000/api/trades/stats")
print("3. Verificar frontend: https://botderiv.roilabs.com.br/trade-history\n")
print("🔒 Database criado com sucesso")
