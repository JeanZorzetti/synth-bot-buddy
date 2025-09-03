# Arquitetura do Projeto: Synth Bot Buddy

## 1. Visão Geral

Este documento descreve a arquitetura "Cérebro como Operador Total", onde uma aplicação backend escrita em Python centraliza toda a lógica de análise de mercado, tomada de decisão e execução de ordens de negociação.

O sistema é composto por um backend inteligente ("Cérebro") e um frontend ("Painel de Controle"). A plataforma Deriv Bot (XML) não é utilizada para operações em tempo real, servindo apenas como uma ferramenta de prototipagem visual.

## 2. Componentes da Arquitetura

### 2.1. Backend "Cérebro" (Python / FastAPI)

O coração do sistema. É uma aplicação Python responsável por toda a lógica de negócio.

-   **API REST (FastAPI):** Fornece endpoints para o frontend se comunicar com o Cérebro (ex: `/start`, `/stop`, `/status`, `/parameters`).
-   **Módulo de Conexão WebSocket:** Gerencia a conexão persistente e segura (`wss://`) com a API da Deriv. É responsável por:
    -   Autenticar a sessão usando o Token API.
    -   Subscrever aos streams de dados de ticks para os ativos desejados.
    -   Enviar ordens de compra (`buy`) e venda (`sell`).
    -   Manter a conexão ativa (heartbeat/ping).
-   **Módulo de Análise e Decisão:** Onde a estratégia de negociação reside. Processa cada tick recebido em tempo real, aplica os filtros e a lógica definida (ex: análise de fluxo, médias móveis, etc.) e decide se uma ordem deve ser executada.
-   **Módulo de Execução:** Formata a requisição de compra/venda no formato JSON esperado pela API da Deriv e a envia através do Módulo WebSocket.
-   **Módulo de Backtesting e Otimização ("Fase 1"):**
    -   Coleta e armazena dados históricos de ticks.
    -   Simula a estratégia contra esses dados com milhares de combinações de parâmetros para encontrar o "DNA Vencedor" (conjunto de parâmetros mais lucrativo).
-   **Módulo de Logging e Aprendizado ("Fase 2"):**
    -   Registra cada operação (data, hora, motivo, resultado) em um banco de dados ou arquivo de log estruturado.
    -   Periodicamente, reanalisa os logs junto com os dados históricos para refinar e otimizar os parâmetros da estratégia.
-   **Dockerfile:** Um arquivo de configuração para empacotar toda a aplicação backend em um contêiner Docker, facilitando o deploy e a escalabilidade na plataforma Easypanel.

### 2.2. Frontend "Painel de Controle" (React)

A interface web com a qual o usuário interage.

-   **Controle do Bot:** Botões para iniciar e parar a execução do "Cérebro".
-   **Configuração de Parâmetros:** Formulários para ajustar os parâmetros da estratégia (ex: stake, fator de Martingale, limiares de sinal) que serão consumidos pelo backend.
-   **Visualização de Dados:** Exibe em tempo real o status do bot, o saldo da conta, e um log das operações realizadas.
-   **Relatórios de Performance:** Mostra gráficos e estatísticas sobre o desempenho histórico do bot.

### 2.3. Ferramentas Auxiliares (Fora de Produção)

-   **Deriv Bot (Plataforma Visual com XML):** Utilizado estritamente para fins de prototipagem e validação visual de novas ideias de estratégias. A lógica criada aqui serve como um rascunho para a implementação real em código Python. Não tem nenhuma função no sistema em produção.

## 3. Fluxo de Operação (Produção)

1.  O usuário acessa o **Frontend**.
2.  O usuário clica em "Iniciar Bot" no painel.
3.  O **Frontend** envia uma requisição para o endpoint `/start` da **API REST** do backend.
4.  O **Backend ("Cérebro")** recebe a requisição e ativa o **Módulo WebSocket**.
5.  O **Módulo WebSocket** estabelece uma conexão com a API da Deriv.
6.  O Cérebro começa a receber ticks em tempo real.
7.  O **Módulo de Análise e Decisão** processa cada tick.
8.  Ao identificar uma oportunidade, o **Módulo de Decisão** aciona o **Módulo de Execução**.
9.  O **Módulo de Execução** envia a ordem de compra/venda via **WebSocket**.
10. O resultado da operação é recebido e registrado pelo **Módulo de Logging**.
11. O **Frontend** atualiza a interface com o novo status e log da operação.

## 4. Segurança

-   **Gerenciamento de Token:** O Token da API Deriv é a chave da conta. Ele deve ser configurado como uma variável de ambiente no ambiente de produção (Easypanel) e lido pelo backend. **NUNCA** deve ser exposto no código do frontend ou versionado no Git.
-   **Escopos de Permissão:** O token deve ser gerado com o mínimo de permissões necessárias. Para esta arquitetura, os escopos essenciais são `Read` (para ler dados da conta e histórico) e `Trade` (para executar ordens). Os escopos `Payments` e `Admin` devem ser desabilitados para mitigar riscos.
