#!/usr/bin/env node

/**
 * Script de inicialização do banco de dados SQLite para controle de roadmap
 *
 * Este script:
 * 1. Cria o arquivo de banco de dados se não existir
 * 2. Executa o script SQL de inicialização
 * 3. Valida a criação das tabelas
 *
 * Uso: node setup.js
 */

const sqlite3 = require('sqlite3').verbose();
const fs = require('fs');
const path = require('path');

const DB_PATH = path.join(__dirname, 'roadmap.db');
const SQL_SCRIPT = path.join(__dirname, 'init_roadmap.sql');

console.log('🗄️  Inicializando banco de dados SQLite para controle de roadmap...\n');

// Verificar se o script SQL existe
if (!fs.existsSync(SQL_SCRIPT)) {
    console.error('❌ Erro: Arquivo init_roadmap.sql não encontrado!');
    console.error(`   Esperado em: ${SQL_SCRIPT}`);
    process.exit(1);
}

// Ler o script SQL
const sqlScript = fs.readFileSync(SQL_SCRIPT, 'utf8');

// Criar/abrir banco de dados
const db = new sqlite3.Database(DB_PATH, (err) => {
    if (err) {
        console.error('❌ Erro ao criar banco de dados:', err.message);
        process.exit(1);
    }
    console.log(`✅ Banco de dados criado/aberto: ${DB_PATH}`);
});

// Executar script SQL
db.exec(sqlScript, (err) => {
    if (err) {
        console.error('❌ Erro ao executar script SQL:', err.message);
        db.close();
        process.exit(1);
    }

    console.log('✅ Script SQL executado com sucesso!\n');

    // Validar criação das tabelas
    db.all(`
        SELECT name, type
        FROM sqlite_master
        WHERE type IN ('table', 'view', 'trigger', 'index')
        ORDER BY type, name
    `, [], (err, rows) => {
        if (err) {
            console.error('❌ Erro ao validar tabelas:', err.message);
            db.close();
            process.exit(1);
        }

        console.log('📋 Objetos criados no banco de dados:\n');

        const groupedByType = rows.reduce((acc, row) => {
            if (!acc[row.type]) acc[row.type] = [];
            acc[row.type].push(row.name);
            return acc;
        }, {});

        Object.keys(groupedByType).forEach(type => {
            console.log(`  ${type.toUpperCase()}S:`);
            groupedByType[type].forEach(name => {
                console.log(`    - ${name}`);
            });
            console.log();
        });

        // Mostrar estatísticas iniciais
        db.get(`
            SELECT
                COUNT(*) as total_tasks,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status = 'in_progress' THEN 1 ELSE 0 END) as in_progress,
                SUM(CASE WHEN status = 'todo' THEN 1 ELSE 0 END) as todo
            FROM roadmap_tasks
        `, [], (err, stats) => {
            if (err) {
                console.error('❌ Erro ao obter estatísticas:', err.message);
            } else {
                console.log('📊 Estatísticas iniciais do roadmap:\n');
                console.log(`  Total de tasks: ${stats.total_tasks}`);
                console.log(`  Completadas:    ${stats.completed}`);
                console.log(`  Em progresso:   ${stats.in_progress}`);
                console.log(`  A fazer:        ${stats.todo}`);
                console.log();
            }

            // Mostrar milestones
            db.all(`SELECT * FROM milestones ORDER BY target_date`, [], (err, milestones) => {
                if (err) {
                    console.error('❌ Erro ao obter milestones:', err.message);
                } else {
                    console.log('🎯 Milestones criados:\n');
                    milestones.forEach(m => {
                        console.log(`  ${m.name} (${m.status})`);
                        console.log(`    Target: ${m.target_date}`);
                        console.log(`    ${m.description}`);
                        console.log();
                    });
                }

                // Fechar banco de dados
                db.close((err) => {
                    if (err) {
                        console.error('❌ Erro ao fechar banco de dados:', err.message);
                        process.exit(1);
                    }
                    console.log('✨ Banco de dados inicializado com sucesso!');
                    console.log('\n💡 Para usar com Claude Code, reinicie o Claude Code para carregar o MCP server.');
                    console.log('   Execute /mcp para verificar se o servidor sqlite está rodando.');
                });
            });
        });
    });
});
