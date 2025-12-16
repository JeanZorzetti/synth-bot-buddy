/**
 * Script para criar database SQLite trades.db com schema inicial
 *
 * Execução: node database/setup.js
 */

const sqlite3 = require('sqlite3').verbose();
const path = require('path');
const fs = require('fs');

// Path para o database
const DB_PATH = path.join(__dirname, '..', 'trades.db');

// Remover database antigo se existir (apenas para setup limpo)
if (fs.existsSync(DB_PATH)) {
    console.log('⚠️  Database já existe em:', DB_PATH);
    console.log('   Pulando criação para não sobrescrever dados existentes.');
    console.log('   Para recriar, delete manualmente: rm backend/trades.db');
    process.exit(0);
}

// Criar database
const db = new sqlite3.Database(DB_PATH, (err) => {
    if (err) {
        console.error('❌ Erro ao criar database:', err.message);
        process.exit(1);
    }
    console.log('✅ Database criado em:', DB_PATH);
});

// Schema da tabela trades_history
const CREATE_TRADES_TABLE = `
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
`;

// Índices para performance
const CREATE_INDEXES = [
    'CREATE INDEX IF NOT EXISTS idx_timestamp ON trades_history(timestamp DESC);',
    'CREATE INDEX IF NOT EXISTS idx_symbol ON trades_history(symbol);',
    'CREATE INDEX IF NOT EXISTS idx_result ON trades_history(result);',
    'CREATE INDEX IF NOT EXISTS idx_strategy ON trades_history(strategy);',
    'CREATE INDEX IF NOT EXISTS idx_created_at ON trades_history(created_at DESC);'
];

// Executar criação de tabela
db.run(CREATE_TRADES_TABLE, (err) => {
    if (err) {
        console.error('❌ Erro ao criar tabela trades_history:', err.message);
        db.close();
        process.exit(1);
    }
    console.log('✅ Tabela trades_history criada');

    // Criar índices
    let indexCount = 0;
    CREATE_INDEXES.forEach((indexSQL, i) => {
        db.run(indexSQL, (err) => {
            if (err) {
                console.error(`❌ Erro ao criar índice ${i + 1}:`, err.message);
            } else {
                console.log(`✅ Índice ${i + 1}/${CREATE_INDEXES.length} criado`);
            }

            indexCount++;
            if (indexCount === CREATE_INDEXES.length) {
                // Inserir trade de exemplo
                insertSampleTrades();
            }
        });
    });
});

// Inserir alguns trades de exemplo para testar
function insertSampleTrades() {
    const sampleTrades = [
        {
            timestamp: new Date(Date.now() - 86400000).toISOString(), // 1 dia atrás
            symbol: 'R_100',
            direction: 'UP',
            entry_price: 100.50,
            exit_price: 101.25,
            quantity: 1.0,
            position_size: 1000,
            stop_loss: 99.50,
            take_profit: 102.50,
            profit_loss: 7.5,
            profit_loss_pct: 0.75,
            result: 'win',
            strategy: 'ML_Predictor',
            confidence: 0.75,
            ml_prediction: 'UP',
            indicators: JSON.stringify({rsi: 65, macd: 'bullish'}),
            notes: 'Trade de exemplo - Paper Trading',
            closed_at: new Date(Date.now() - 82800000).toISOString(),
            duration_seconds: 3600
        },
        {
            timestamp: new Date(Date.now() - 43200000).toISOString(), // 12 horas atrás
            symbol: 'R_100',
            direction: 'DOWN',
            entry_price: 99.75,
            exit_price: 99.50,
            quantity: 1.0,
            position_size: 1000,
            stop_loss: 100.75,
            take_profit: 97.75,
            profit_loss: 2.5,
            profit_loss_pct: 0.25,
            result: 'win',
            strategy: 'ML_Predictor',
            confidence: 0.68,
            ml_prediction: 'DOWN',
            indicators: JSON.stringify({rsi: 35, macd: 'bearish'}),
            notes: 'Trade de exemplo - Paper Trading',
            closed_at: new Date(Date.now() - 39600000).toISOString(),
            duration_seconds: 3600
        },
        {
            timestamp: new Date(Date.now() - 7200000).toISOString(), // 2 horas atrás
            symbol: 'R_100',
            direction: 'UP',
            entry_price: 101.00,
            exit_price: 100.50,
            quantity: 1.0,
            position_size: 1000,
            stop_loss: 100.00,
            take_profit: 103.00,
            profit_loss: -5.0,
            profit_loss_pct: -0.50,
            result: 'loss',
            strategy: 'ML_Predictor',
            confidence: 0.62,
            ml_prediction: 'UP',
            indicators: JSON.stringify({rsi: 70, macd: 'bullish'}),
            notes: 'Trade de exemplo - Stop Loss acionado',
            closed_at: new Date(Date.now() - 3600000).toISOString(),
            duration_seconds: 3600
        }
    ];

    let insertCount = 0;
    const insertSQL = `
        INSERT INTO trades_history (
            timestamp, symbol, direction, entry_price, exit_price, quantity,
            position_size, stop_loss, take_profit, profit_loss, profit_loss_pct,
            result, strategy, confidence, ml_prediction, indicators, notes,
            closed_at, duration_seconds
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `;

    sampleTrades.forEach((trade) => {
        db.run(insertSQL, [
            trade.timestamp, trade.symbol, trade.direction, trade.entry_price,
            trade.exit_price, trade.quantity, trade.position_size, trade.stop_loss,
            trade.take_profit, trade.profit_loss, trade.profit_loss_pct, trade.result,
            trade.strategy, trade.confidence, trade.ml_prediction, trade.indicators,
            trade.notes, trade.closed_at, trade.duration_seconds
        ], (err) => {
            if (err) {
                console.error('❌ Erro ao inserir trade de exemplo:', err.message);
            } else {
                insertCount++;
                console.log(`✅ Trade de exemplo ${insertCount}/${sampleTrades.length} inserido`);
            }

            if (insertCount === sampleTrades.length) {
                finishSetup();
            }
        });
    });
}

function finishSetup() {
    // Verificar quantos trades foram inseridos
    db.get('SELECT COUNT(*) as count FROM trades_history', [], (err, row) => {
        if (err) {
            console.error('❌ Erro ao verificar trades:', err.message);
        } else {
            console.log(`\n✅ Setup completo! Database tem ${row.count} trades de exemplo`);
            console.log('\nPróximos passos:');
            console.log('1. Reiniciar backend: cd backend && uvicorn main:app --reload');
            console.log('2. Testar endpoint: curl http://localhost:8000/api/trades/stats');
            console.log('3. Verificar frontend: https://botderiv.roilabs.com.br/trade-history\n');
        }

        // Fechar conexão
        db.close((err) => {
            if (err) {
                console.error('❌ Erro ao fechar database:', err.message);
            } else {
                console.log('🔒 Database fechado com sucesso');
            }
        });
    });
}
