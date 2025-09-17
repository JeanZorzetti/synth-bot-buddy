"""
🎧 REAL TECHNICAL SUPPORT SYSTEM
Sistema completo de suporte técnico com múltiplos canais e automação
"""

import asyncio
import json
import logging
import smtplib
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import asyncpg
import aiofiles
from pathlib import Path
import requests
import websockets
from jinja2 import Template

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TicketStatus(Enum):
    """🎫 Status dos tickets"""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    WAITING_CUSTOMER = "waiting_customer"
    WAITING_AGENT = "waiting_agent"
    RESOLVED = "resolved"
    CLOSED = "closed"
    CANCELLED = "cancelled"


class TicketPriority(Enum):
    """⚡ Prioridade dos tickets"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


class TicketCategory(Enum):
    """📂 Categorias dos tickets"""
    TECHNICAL_ISSUE = "technical_issue"
    TRADING_PROBLEM = "trading_problem"
    ACCOUNT_ISSUE = "account_issue"
    BILLING_QUESTION = "billing_question"
    FEATURE_REQUEST = "feature_request"
    BUG_REPORT = "bug_report"
    API_SUPPORT = "api_support"
    GENERAL_INQUIRY = "general_inquiry"


class SupportChannel(Enum):
    """📞 Canais de suporte"""
    EMAIL = "email"
    CHAT = "chat"
    PHONE = "phone"
    WHATSAPP = "whatsapp"
    DISCORD = "discord"
    TICKET_SYSTEM = "ticket_system"


@dataclass
class SupportTicket:
    """🎫 Ticket de suporte"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    title: str = ""
    description: str = ""
    category: TicketCategory = TicketCategory.GENERAL_INQUIRY
    priority: TicketPriority = TicketPriority.MEDIUM
    status: TicketStatus = TicketStatus.OPEN
    channel: SupportChannel = SupportChannel.EMAIL
    assigned_agent_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    first_response_at: Optional[datetime] = None
    customer_email: str = ""
    customer_name: str = ""
    customer_phone: str = ""
    metadata: Dict = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    attachments: List[str] = field(default_factory=list)
    satisfaction_rating: Optional[int] = None
    satisfaction_feedback: str = ""


@dataclass
class TicketMessage:
    """💬 Mensagem do ticket"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ticket_id: str = ""
    sender_id: str = ""
    sender_type: str = "customer"  # customer, agent, system
    message: str = ""
    message_type: str = "text"  # text, image, file, system
    timestamp: datetime = field(default_factory=datetime.now)
    is_internal: bool = False
    metadata: Dict = field(default_factory=dict)


@dataclass
class SupportAgent:
    """👤 Agente de suporte"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    email: str = ""
    role: str = "agent"  # agent, senior_agent, supervisor, admin
    is_active: bool = True
    is_online: bool = False
    specialties: List[TicketCategory] = field(default_factory=list)
    max_concurrent_tickets: int = 10
    current_ticket_count: int = 0
    languages: List[str] = field(default_factory=lambda: ["en", "pt"])
    created_at: datetime = field(default_factory=datetime.now)
    last_active_at: Optional[datetime] = None


class KnowledgeBase:
    """📚 Base de conhecimento"""

    def __init__(self):
        self.articles = {
            "login_issues": {
                "title": "Como resolver problemas de login",
                "content": """
                1. Verifique se seu email e senha estão corretos
                2. Limpe o cache do navegador
                3. Tente usar modo privado/incógnito
                4. Verifique se sua conta não está bloqueada
                5. Use a opção "Esqueci minha senha" se necessário
                """,
                "tags": ["login", "password", "account"],
                "category": "account_issue"
            },
            "api_connection": {
                "title": "Problemas de conexão com API",
                "content": """
                1. Verifique se sua API key está ativa
                2. Confirme se você tem as permissões necessárias
                3. Verifique os rate limits
                4. Teste a conectividade de rede
                5. Consulte a documentação da API
                """,
                "tags": ["api", "connection", "troubleshooting"],
                "category": "technical_issue"
            },
            "trading_errors": {
                "title": "Erros comuns de trading",
                "content": """
                1. Saldo insuficiente
                2. Mercado fechado
                3. Ativo não disponível
                4. Limite de posição excedido
                5. Configuração de risco inadequada
                """,
                "tags": ["trading", "errors", "troubleshooting"],
                "category": "trading_problem"
            }
        }

    def search_articles(self, query: str, category: TicketCategory = None) -> List[Dict]:
        """Buscar artigos na base de conhecimento"""
        results = []
        query_lower = query.lower()

        for article_id, article in self.articles.items():
            # Filtrar por categoria se especificada
            if category and article["category"] != category.value:
                continue

            # Buscar por palavra-chave
            if (query_lower in article["title"].lower() or
                query_lower in article["content"].lower() or
                any(query_lower in tag.lower() for tag in article["tags"])):

                results.append({
                    "id": article_id,
                    "title": article["title"],
                    "content": article["content"],
                    "relevance_score": self._calculate_relevance(query_lower, article)
                })

        # Ordenar por relevância
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:5]  # Top 5 resultados

    def _calculate_relevance(self, query: str, article: Dict) -> float:
        """Calcular relevância do artigo"""
        score = 0.0

        # Título tem peso maior
        if query in article["title"].lower():
            score += 2.0

        # Conteúdo tem peso médio
        if query in article["content"].lower():
            score += 1.0

        # Tags têm peso alto
        for tag in article["tags"]:
            if query in tag.lower():
                score += 1.5

        return score


class NotificationManager:
    """📧 Gerenciador de notificações"""

    def __init__(self, smtp_config: Dict = None, slack_webhook: str = "", discord_webhook: str = ""):
        self.smtp_config = smtp_config or {}
        self.slack_webhook = slack_webhook
        self.discord_webhook = discord_webhook

    async def send_email_notification(self, to_email: str, subject: str, content: str, is_html: bool = False) -> bool:
        """Enviar notificação por email"""
        try:
            if not self.smtp_config:
                logger.warning("SMTP not configured, skipping email notification")
                return False

            msg = MimeMultipart()
            msg['From'] = self.smtp_config['from_email']
            msg['To'] = to_email
            msg['Subject'] = subject

            msg.attach(MimeText(content, 'html' if is_html else 'plain'))

            server = smtplib.SMTP(self.smtp_config['smtp_server'], self.smtp_config['smtp_port'])
            server.starttls()
            server.login(self.smtp_config['username'], self.smtp_config['password'])
            server.send_message(msg)
            server.quit()

            logger.info(f"✅ Email sent to: {to_email}")
            return True

        except Exception as e:
            logger.error(f"❌ Error sending email: {e}")
            return False

    async def send_slack_notification(self, message: str, channel: str = "#support") -> bool:
        """Enviar notificação para Slack"""
        try:
            if not self.slack_webhook:
                logger.warning("Slack webhook not configured")
                return False

            payload = {
                "channel": channel,
                "text": message,
                "username": "TradingBot Support"
            }

            response = requests.post(self.slack_webhook, json=payload)
            return response.status_code == 200

        except Exception as e:
            logger.error(f"❌ Error sending Slack notification: {e}")
            return False

    async def send_discord_notification(self, message: str) -> bool:
        """Enviar notificação para Discord"""
        try:
            if not self.discord_webhook:
                logger.warning("Discord webhook not configured")
                return False

            payload = {
                "content": message,
                "username": "TradingBot Support"
            }

            response = requests.post(self.discord_webhook, json=payload)
            return response.status_code == 204

        except Exception as e:
            logger.error(f"❌ Error sending Discord notification: {e}")
            return False


class AutomatedResponder:
    """🤖 Respondedor automático"""

    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base
        self.auto_responses = {
            "greeting": "Olá! Obrigado por entrar em contato. Como posso ajudá-lo hoje?",
            "thanks": "De nada! Fico feliz em poder ajudar. Há mais alguma coisa em que posso auxiliá-lo?",
            "escalation": "Entendo que precisa de assistência adicional. Vou transferir seu caso para um especialista que entrará em contato em breve."
        }

    def detect_intent(self, message: str) -> str:
        """Detectar intenção da mensagem"""
        message_lower = message.lower()

        # Saudações
        if any(word in message_lower for word in ["olá", "oi", "hello", "hi"]):
            return "greeting"

        # Agradecimentos
        if any(word in message_lower for word in ["obrigado", "obrigada", "thanks", "valeu"]):
            return "thanks"

        # Problemas de login
        if any(word in message_lower for word in ["login", "senha", "password", "entrar"]):
            return "login_help"

        # Problemas de API
        if any(word in message_lower for word in ["api", "conexão", "connection", "endpoint"]):
            return "api_help"

        # Problemas de trading
        if any(word in message_lower for word in ["trade", "trading", "negociação", "posição"]):
            return "trading_help"

        return "general"

    def generate_auto_response(self, message: str, ticket_category: TicketCategory) -> Optional[str]:
        """Gerar resposta automática"""
        intent = self.detect_intent(message)

        if intent in self.auto_responses:
            return self.auto_responses[intent]

        # Buscar na base de conhecimento
        articles = self.knowledge_base.search_articles(message, ticket_category)

        if articles:
            best_article = articles[0]
            return f"""
Encontrei algumas informações que podem ajudar:

**{best_article['title']}**

{best_article['content']}

Se isso não resolver seu problema, nossa equipe de suporte entrará em contato em breve.
            """

        return None


class SupportMetrics:
    """📊 Métricas de suporte"""

    def __init__(self):
        self.metrics = {}

    def calculate_sla_metrics(self, tickets: List[SupportTicket]) -> Dict:
        """Calcular métricas de SLA"""
        if not tickets:
            return {}

        total_tickets = len(tickets)
        resolved_tickets = [t for t in tickets if t.status == TicketStatus.RESOLVED]

        # Tempo médio de primeira resposta
        first_response_times = []
        for ticket in tickets:
            if ticket.first_response_at:
                response_time = (ticket.first_response_at - ticket.created_at).total_seconds() / 3600
                first_response_times.append(response_time)

        # Tempo médio de resolução
        resolution_times = []
        for ticket in resolved_tickets:
            if ticket.resolved_at:
                resolution_time = (ticket.resolved_at - ticket.created_at).total_seconds() / 3600
                resolution_times.append(resolution_time)

        # Satisfação do cliente
        satisfaction_ratings = [t.satisfaction_rating for t in resolved_tickets if t.satisfaction_rating]

        return {
            'total_tickets': total_tickets,
            'resolved_tickets': len(resolved_tickets),
            'resolution_rate': (len(resolved_tickets) / total_tickets) * 100 if total_tickets > 0 else 0,
            'avg_first_response_hours': sum(first_response_times) / len(first_response_times) if first_response_times else 0,
            'avg_resolution_hours': sum(resolution_times) / len(resolution_times) if resolution_times else 0,
            'avg_satisfaction_rating': sum(satisfaction_ratings) / len(satisfaction_ratings) if satisfaction_ratings else 0,
            'tickets_by_priority': {
                priority.value: len([t for t in tickets if t.priority == priority])
                for priority in TicketPriority
            },
            'tickets_by_category': {
                category.value: len([t for t in tickets if t.category == category])
                for category in TicketCategory
            }
        }


class TechnicalSupportSystem:
    """🎧 Sistema principal de suporte técnico"""

    def __init__(self, database_url: str, smtp_config: Dict = None):
        self.database_url = database_url
        self.pool = None
        self.knowledge_base = KnowledgeBase()
        self.notification_manager = NotificationManager(smtp_config)
        self.auto_responder = AutomatedResponder(self.knowledge_base)
        self.metrics = SupportMetrics()
        self.agents = {}  # {agent_id: SupportAgent}

    async def initialize(self):
        """Inicializar sistema de suporte"""
        try:
            self.pool = await asyncpg.create_pool(self.database_url)
            await self._create_tables()
            await self._load_agents()
            logger.info("✅ Technical Support System initialized")
        except Exception as e:
            logger.error(f"❌ Support system initialization failed: {e}")
            await self._initialize_file_storage()

    async def _create_tables(self):
        """Criar tabelas necessárias"""
        async with self.pool.acquire() as conn:
            # Tabela de tickets
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS support_tickets (
                    id UUID PRIMARY KEY,
                    user_id UUID NOT NULL,
                    title VARCHAR NOT NULL,
                    description TEXT NOT NULL,
                    category VARCHAR NOT NULL,
                    priority VARCHAR NOT NULL,
                    status VARCHAR DEFAULT 'open',
                    channel VARCHAR NOT NULL,
                    assigned_agent_id UUID,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    resolved_at TIMESTAMP,
                    closed_at TIMESTAMP,
                    first_response_at TIMESTAMP,
                    customer_email VARCHAR NOT NULL,
                    customer_name VARCHAR,
                    customer_phone VARCHAR,
                    metadata JSONB DEFAULT '{}',
                    tags JSONB DEFAULT '[]',
                    attachments JSONB DEFAULT '[]',
                    satisfaction_rating INTEGER,
                    satisfaction_feedback TEXT
                )
            """)

            # Tabela de mensagens
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ticket_messages (
                    id UUID PRIMARY KEY,
                    ticket_id UUID REFERENCES support_tickets(id) ON DELETE CASCADE,
                    sender_id UUID,
                    sender_type VARCHAR NOT NULL,
                    message TEXT NOT NULL,
                    message_type VARCHAR DEFAULT 'text',
                    timestamp TIMESTAMP DEFAULT NOW(),
                    is_internal BOOLEAN DEFAULT FALSE,
                    metadata JSONB DEFAULT '{}'
                )
            """)

            # Tabela de agentes
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS support_agents (
                    id UUID PRIMARY KEY,
                    name VARCHAR NOT NULL,
                    email VARCHAR UNIQUE NOT NULL,
                    role VARCHAR DEFAULT 'agent',
                    is_active BOOLEAN DEFAULT TRUE,
                    is_online BOOLEAN DEFAULT FALSE,
                    specialties JSONB DEFAULT '[]',
                    max_concurrent_tickets INTEGER DEFAULT 10,
                    current_ticket_count INTEGER DEFAULT 0,
                    languages JSONB DEFAULT '["en"]',
                    created_at TIMESTAMP DEFAULT NOW(),
                    last_active_at TIMESTAMP
                )
            """)

            # Índices
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_tickets_user_id ON support_tickets(user_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_tickets_status ON support_tickets(status)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_tickets_priority ON support_tickets(priority)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_ticket_id ON ticket_messages(ticket_id)")

    async def _initialize_file_storage(self):
        """Inicializar armazenamento em arquivo"""
        self.storage_path = Path("data/support")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        logger.info("📁 Using file-based storage for support system")

    async def _load_agents(self):
        """Carregar agentes padrão"""
        default_agents = [
            SupportAgent(
                name="Ana Silva",
                email="ana@tradingbot.com",
                role="senior_agent",
                specialties=[TicketCategory.TECHNICAL_ISSUE, TicketCategory.API_SUPPORT],
                languages=["pt", "en"]
            ),
            SupportAgent(
                name="Carlos Santos",
                email="carlos@tradingbot.com",
                role="agent",
                specialties=[TicketCategory.TRADING_PROBLEM, TicketCategory.ACCOUNT_ISSUE],
                languages=["pt"]
            ),
            SupportAgent(
                name="Maria Costa",
                email="maria@tradingbot.com",
                role="supervisor",
                specialties=[TicketCategory.BILLING_QUESTION, TicketCategory.GENERAL_INQUIRY],
                languages=["pt", "en", "es"]
            )
        ]

        for agent in default_agents:
            self.agents[agent.id] = agent
            await self._save_agent(agent)

    async def create_ticket(self, user_id: str, title: str, description: str, category: TicketCategory,
                           customer_email: str, customer_name: str = "", priority: TicketPriority = TicketPriority.MEDIUM,
                           channel: SupportChannel = SupportChannel.EMAIL) -> Optional[SupportTicket]:
        """Criar novo ticket de suporte"""
        try:
            ticket = SupportTicket(
                user_id=user_id,
                title=title,
                description=description,
                category=category,
                priority=priority,
                channel=channel,
                customer_email=customer_email,
                customer_name=customer_name
            )

            # Tentar resposta automática
            auto_response = self.auto_responder.generate_auto_response(description, category)

            # Atribuir agente automaticamente
            assigned_agent = await self._auto_assign_agent(category, priority)
            if assigned_agent:
                ticket.assigned_agent_id = assigned_agent.id

            # Salvar ticket
            success = await self._save_ticket(ticket)

            if success:
                # Enviar resposta automática se disponível
                if auto_response:
                    await self.add_message_to_ticket(
                        ticket.id,
                        "system",
                        auto_response,
                        sender_type="system"
                    )
                    ticket.first_response_at = datetime.now()
                    await self._save_ticket(ticket)

                # Notificar agente atribuído
                if assigned_agent:
                    await self.notification_manager.send_email_notification(
                        assigned_agent.email,
                        f"Novo ticket atribuído: {title}",
                        f"Um novo ticket foi atribuído a você:\n\nTítulo: {title}\nCategoria: {category.value}\nPrioridade: {priority.value}\n\nDescrição:\n{description}"
                    )

                logger.info(f"✅ Support ticket created: {ticket.id}")
                return ticket

        except Exception as e:
            logger.error(f"❌ Error creating support ticket: {e}")

        return None

    async def add_message_to_ticket(self, ticket_id: str, sender_id: str, message: str,
                                   sender_type: str = "customer", is_internal: bool = False) -> bool:
        """Adicionar mensagem ao ticket"""
        try:
            message_obj = TicketMessage(
                ticket_id=ticket_id,
                sender_id=sender_id,
                sender_type=sender_type,
                message=message,
                is_internal=is_internal
            )

            success = await self._save_message(message_obj)

            if success:
                # Atualizar timestamp do ticket
                await self._update_ticket_timestamp(ticket_id)

                # Se for primeira resposta de agente, marcar timestamp
                if sender_type == "agent":
                    ticket = await self.get_ticket(ticket_id)
                    if ticket and not ticket.first_response_at:
                        ticket.first_response_at = datetime.now()
                        await self._save_ticket(ticket)

                logger.info(f"✅ Message added to ticket: {ticket_id}")
                return True

        except Exception as e:
            logger.error(f"❌ Error adding message to ticket: {e}")

        return False

    async def get_ticket(self, ticket_id: str) -> Optional[SupportTicket]:
        """Obter ticket por ID"""
        try:
            if self.pool:
                async with self.pool.acquire() as conn:
                    row = await conn.fetchrow(
                        "SELECT * FROM support_tickets WHERE id = $1",
                        ticket_id
                    )
                    if row:
                        return self._row_to_ticket(row)

        except Exception as e:
            logger.error(f"❌ Error getting ticket: {e}")

        return None

    async def get_user_tickets(self, user_id: str, status: TicketStatus = None) -> List[SupportTicket]:
        """Obter tickets do usuário"""
        try:
            tickets = []

            if self.pool:
                async with self.pool.acquire() as conn:
                    if status:
                        rows = await conn.fetch(
                            "SELECT * FROM support_tickets WHERE user_id = $1 AND status = $2 ORDER BY created_at DESC",
                            user_id, status.value
                        )
                    else:
                        rows = await conn.fetch(
                            "SELECT * FROM support_tickets WHERE user_id = $1 ORDER BY created_at DESC",
                            user_id
                        )

                    for row in rows:
                        tickets.append(self._row_to_ticket(row))

            return tickets

        except Exception as e:
            logger.error(f"❌ Error getting user tickets: {e}")
            return []

    async def resolve_ticket(self, ticket_id: str, agent_id: str, resolution_notes: str = "") -> bool:
        """Resolver ticket"""
        try:
            if self.pool:
                async with self.pool.acquire() as conn:
                    await conn.execute("""
                        UPDATE support_tickets
                        SET status = 'resolved', resolved_at = NOW(), updated_at = NOW()
                        WHERE id = $1
                    """, ticket_id)

                    # Adicionar nota de resolução
                    if resolution_notes:
                        await self.add_message_to_ticket(
                            ticket_id,
                            agent_id,
                            f"Ticket resolvido: {resolution_notes}",
                            sender_type="agent"
                        )

                    logger.info(f"✅ Ticket resolved: {ticket_id}")
                    return True

        except Exception as e:
            logger.error(f"❌ Error resolving ticket: {e}")

        return False

    async def get_support_metrics(self, days: int = 30) -> Dict:
        """Obter métricas de suporte"""
        try:
            if self.pool:
                async with self.pool.acquire() as conn:
                    # Buscar tickets do período
                    rows = await conn.fetch("""
                        SELECT * FROM support_tickets
                        WHERE created_at >= NOW() - INTERVAL '%s days'
                    """, days)

                    tickets = [self._row_to_ticket(row) for row in rows]
                    return self.metrics.calculate_sla_metrics(tickets)

        except Exception as e:
            logger.error(f"❌ Error getting support metrics: {e}")

        return {}

    async def _auto_assign_agent(self, category: TicketCategory, priority: TicketPriority) -> Optional[SupportAgent]:
        """Atribuir agente automaticamente"""
        # Filtrar agentes disponíveis
        available_agents = [
            agent for agent in self.agents.values()
            if (agent.is_active and
                agent.current_ticket_count < agent.max_concurrent_tickets and
                (not agent.specialties or category in agent.specialties))
        ]

        if not available_agents:
            return None

        # Priorizar por especialidade e carga de trabalho
        best_agent = min(available_agents, key=lambda a: a.current_ticket_count)
        best_agent.current_ticket_count += 1

        return best_agent

    async def _save_ticket(self, ticket: SupportTicket) -> bool:
        """Salvar ticket no banco"""
        try:
            if self.pool:
                async with self.pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO support_tickets (
                            id, user_id, title, description, category, priority, status,
                            channel, assigned_agent_id, created_at, updated_at, resolved_at,
                            closed_at, first_response_at, customer_email, customer_name,
                            customer_phone, metadata, tags, attachments, satisfaction_rating,
                            satisfaction_feedback
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22)
                        ON CONFLICT (id) DO UPDATE SET
                            status = EXCLUDED.status,
                            assigned_agent_id = EXCLUDED.assigned_agent_id,
                            updated_at = EXCLUDED.updated_at,
                            resolved_at = EXCLUDED.resolved_at,
                            closed_at = EXCLUDED.closed_at,
                            first_response_at = EXCLUDED.first_response_at,
                            satisfaction_rating = EXCLUDED.satisfaction_rating,
                            satisfaction_feedback = EXCLUDED.satisfaction_feedback
                    """,
                        ticket.id, ticket.user_id, ticket.title, ticket.description,
                        ticket.category.value, ticket.priority.value, ticket.status.value,
                        ticket.channel.value, ticket.assigned_agent_id, ticket.created_at,
                        ticket.updated_at, ticket.resolved_at, ticket.closed_at,
                        ticket.first_response_at, ticket.customer_email, ticket.customer_name,
                        ticket.customer_phone, json.dumps(ticket.metadata),
                        json.dumps(ticket.tags), json.dumps(ticket.attachments),
                        ticket.satisfaction_rating, ticket.satisfaction_feedback
                    )
                return True

        except Exception as e:
            logger.error(f"❌ Error saving ticket: {e}")

        return False

    async def _save_message(self, message: TicketMessage) -> bool:
        """Salvar mensagem no banco"""
        try:
            if self.pool:
                async with self.pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO ticket_messages (
                            id, ticket_id, sender_id, sender_type, message, message_type,
                            timestamp, is_internal, metadata
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,
                        message.id, message.ticket_id, message.sender_id, message.sender_type,
                        message.message, message.message_type, message.timestamp,
                        message.is_internal, json.dumps(message.metadata)
                    )
                return True

        except Exception as e:
            logger.error(f"❌ Error saving message: {e}")

        return False

    async def _save_agent(self, agent: SupportAgent) -> bool:
        """Salvar agente no banco"""
        try:
            if self.pool:
                async with self.pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO support_agents (
                            id, name, email, role, is_active, is_online, specialties,
                            max_concurrent_tickets, current_ticket_count, languages,
                            created_at, last_active_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                        ON CONFLICT (email) DO UPDATE SET
                            name = EXCLUDED.name,
                            role = EXCLUDED.role,
                            is_active = EXCLUDED.is_active,
                            is_online = EXCLUDED.is_online,
                            specialties = EXCLUDED.specialties,
                            max_concurrent_tickets = EXCLUDED.max_concurrent_tickets,
                            current_ticket_count = EXCLUDED.current_ticket_count
                    """,
                        agent.id, agent.name, agent.email, agent.role, agent.is_active,
                        agent.is_online, json.dumps([s.value for s in agent.specialties]),
                        agent.max_concurrent_tickets, agent.current_ticket_count,
                        json.dumps(agent.languages), agent.created_at, agent.last_active_at
                    )
                return True

        except Exception as e:
            logger.error(f"❌ Error saving agent: {e}")

        return False

    async def _update_ticket_timestamp(self, ticket_id: str) -> bool:
        """Atualizar timestamp do ticket"""
        try:
            if self.pool:
                async with self.pool.acquire() as conn:
                    await conn.execute(
                        "UPDATE support_tickets SET updated_at = NOW() WHERE id = $1",
                        ticket_id
                    )
                return True

        except Exception as e:
            logger.error(f"❌ Error updating ticket timestamp: {e}")

        return False

    def _row_to_ticket(self, row) -> SupportTicket:
        """Converter row para SupportTicket"""
        return SupportTicket(
            id=str(row['id']),
            user_id=str(row['user_id']),
            title=row['title'],
            description=row['description'],
            category=TicketCategory(row['category']),
            priority=TicketPriority(row['priority']),
            status=TicketStatus(row['status']),
            channel=SupportChannel(row['channel']),
            assigned_agent_id=str(row['assigned_agent_id']) if row['assigned_agent_id'] else None,
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            resolved_at=row['resolved_at'],
            closed_at=row['closed_at'],
            first_response_at=row['first_response_at'],
            customer_email=row['customer_email'],
            customer_name=row['customer_name'] or '',
            customer_phone=row['customer_phone'] or '',
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
            tags=json.loads(row['tags']) if row['tags'] else [],
            attachments=json.loads(row['attachments']) if row['attachments'] else [],
            satisfaction_rating=row['satisfaction_rating'],
            satisfaction_feedback=row['satisfaction_feedback'] or ''
        )


# 🧪 Função de teste
async def test_technical_support_system():
    """Testar sistema de suporte técnico"""
    database_url = "postgresql://trading_user:password@localhost:5432/trading_db"

    smtp_config = {
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'username': 'your_email@gmail.com',
        'password': 'your_app_password',
        'from_email': 'your_email@gmail.com'
    }

    # Inicializar sistema
    support_system = TechnicalSupportSystem(database_url, smtp_config)
    await support_system.initialize()

    print("\n" + "="*80)
    print("🎧 TECHNICAL SUPPORT SYSTEM TEST")
    print("="*80)

    test_user_id = str(uuid.uuid4())

    # 1. Criar ticket de suporte
    print(f"\n🎫 CREATING SUPPORT TICKET for user: {test_user_id}")
    ticket = await support_system.create_ticket(
        user_id=test_user_id,
        title="Problema com login na plataforma",
        description="Não consigo fazer login na minha conta. Aparece erro de senha incorreta, mas tenho certeza que está correta.",
        category=TicketCategory.ACCOUNT_ISSUE,
        customer_email="cliente@example.com",
        customer_name="João Silva",
        priority=TicketPriority.HIGH,
        channel=SupportChannel.EMAIL
    )

    if ticket:
        print(f"✅ Support ticket created: {ticket.id}")
        print(f"   Title: {ticket.title}")
        print(f"   Category: {ticket.category.value}")
        print(f"   Priority: {ticket.priority.value}")
        print(f"   Status: {ticket.status.value}")
        print(f"   Assigned Agent: {ticket.assigned_agent_id}")

    # 2. Adicionar mensagem ao ticket
    if ticket:
        print(f"\n💬 ADDING MESSAGE TO TICKET...")
        message_added = await support_system.add_message_to_ticket(
            ticket.id,
            test_user_id,
            "Já tentei redefinir a senha várias vezes, mas o problema persiste.",
            sender_type="customer"
        )

        if message_added:
            print(f"✅ Message added to ticket")

    # 3. Buscar na base de conhecimento
    print(f"\n📚 SEARCHING KNOWLEDGE BASE...")
    kb_results = support_system.knowledge_base.search_articles(
        "problema login senha",
        TicketCategory.ACCOUNT_ISSUE
    )

    for i, article in enumerate(kb_results, 1):
        print(f"   {i}. {article['title']} (relevância: {article['relevance_score']:.1f})")

    # 4. Resposta automática
    print(f"\n🤖 TESTING AUTO RESPONSE...")
    auto_response = support_system.auto_responder.generate_auto_response(
        "não consigo fazer login",
        TicketCategory.ACCOUNT_ISSUE
    )

    if auto_response:
        print(f"✅ Auto response generated:")
        print(f"   {auto_response[:100]}...")

    # 5. Listar tickets do usuário
    print(f"\n📋 LISTING USER TICKETS...")
    user_tickets = await support_system.get_user_tickets(test_user_id)

    for ticket in user_tickets:
        print(f"   🎫 {ticket.title}")
        print(f"      Status: {ticket.status.value}")
        print(f"      Created: {ticket.created_at}")

    # 6. Métricas de suporte
    print(f"\n📊 SUPPORT METRICS...")
    metrics = await support_system.get_support_metrics(days=30)

    print(f"   Total Tickets: {metrics.get('total_tickets', 0)}")
    print(f"   Resolution Rate: {metrics.get('resolution_rate', 0):.1f}%")
    print(f"   Avg First Response: {metrics.get('avg_first_response_hours', 0):.1f}h")
    print(f"   Avg Resolution: {metrics.get('avg_resolution_hours', 0):.1f}h")

    print("\n" + "="*80)
    print("✅ TECHNICAL SUPPORT SYSTEM TEST COMPLETED")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(test_technical_support_system())