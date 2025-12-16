#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick script to verify database contents"""
import sqlite3
import sys
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

conn = sqlite3.connect('trades_history.db')
cursor = conn.cursor()

cursor.execute('SELECT id, symbol, trade_type, result, profit_loss, confidence FROM trades_history')
print('\n✅ Trades in database:')
for row in cursor.fetchall():
    print(f'  ID {row[0]}: {row[1]} {row[2]} - {row[3]} (P&L: ${row[4]}, Confidence: {row[5]}%)')

cursor.execute('SELECT COUNT(*) FROM trades_history')
count = cursor.fetchone()[0]
print(f'\n✅ Total: {count} trades')

conn.close()
