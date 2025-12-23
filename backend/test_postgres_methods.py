#!/usr/bin/env python3
"""
Quick test to verify PostgreSQL repository methods work with kwargs
"""
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

from database.abutre_repository_postgres import AbutreRepositoryPostgres

def test_methods():
    """Test all fixed methods"""
    print("=" * 60)
    print("TESTE DOS MÉTODOS POSTGRESQL")
    print("=" * 60)

    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        print("❌ DATABASE_URL não configurada!")
        return False

    print(f"Database: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else DATABASE_URL}")

    repo = AbutreRepositoryPostgres(database_url=DATABASE_URL)

    # Test 1: insert_candle with kwargs
    print("\n1. Testando insert_candle()...")
    try:
        candle_id = repo.insert_candle(
            timestamp=datetime.now(),
            open=100.0,
            high=105.0,
            low=99.0,
            close=102.0,
            color='green',
            symbol='TEST',
            source='test'
        )
        print(f"   ✅ insert_candle() funcionou! ID: {candle_id}")
    except Exception as e:
        print(f"   ❌ insert_candle() falhou: {e}")
        return False

    # Test 2: insert_trigger with kwargs
    print("\n2. Testando insert_trigger()...")
    try:
        trigger_id = repo.insert_trigger(
            timestamp=datetime.now(),
            streak_count=3,
            direction='CALL',
            source='test'
        )
        print(f"   ✅ insert_trigger() funcionou! ID: {trigger_id}")
    except Exception as e:
        print(f"   ❌ insert_trigger() falhou: {e}")
        return False

    # Test 3: insert_trade_opened with kwargs
    print("\n3. Testando insert_trade_opened()...")
    try:
        trade_id = repo.insert_trade_opened(
            trade_id='TEST_TRADE_001',
            timestamp=datetime.now(),
            direction='CALL',
            stake=10.0,
            level=1,
            contract_id='CONTRACT_001',
            source='test'
        )
        print(f"   ✅ insert_trade_opened() funcionou! ID: {trade_id}")
    except Exception as e:
        print(f"   ❌ insert_trade_opened() falhou: {e}")
        return False

    # Test 4: update_trade_closed with kwargs
    print("\n4. Testando update_trade_closed()...")
    try:
        success = repo.update_trade_closed(
            trade_id='TEST_TRADE_001',
            exit_time=datetime.now(),
            result='win',
            profit=5.0,
            balance=10005.0,
            max_level=1
        )
        print(f"   ✅ update_trade_closed() funcionou! Success: {success}")
    except Exception as e:
        print(f"   ❌ update_trade_closed() falhou: {e}")
        return False

    # Test 5: insert_balance_snapshot with kwargs
    print("\n5. Testando insert_balance_snapshot()...")
    try:
        snapshot_id = repo.insert_balance_snapshot(
            timestamp=datetime.now(),
            balance=10005.0,
            peak_balance=10005.0,
            drawdown_pct=0.0,
            total_trades=1,
            wins=1,
            losses=0,
            roi_pct=0.05
        )
        print(f"   ✅ insert_balance_snapshot() funcionou! ID: {snapshot_id}")
    except Exception as e:
        print(f"   ❌ insert_balance_snapshot() falhou: {e}")
        return False

    print("\n" + "=" * 60)
    print("✅ TODOS OS MÉTODOS FUNCIONARAM CORRETAMENTE!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_methods()
    sys.exit(0 if success else 1)
