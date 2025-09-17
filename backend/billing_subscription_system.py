"""
💳 REAL BILLING & SUBSCRIPTION SYSTEM
Sistema completo de cobrança e assinaturas com integração de pagamento real
"""

import asyncio
import json
import logging
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
import stripe
import requests
from decimal import Decimal
import asyncpg
from pathlib import Path
import aiofiles

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PaymentProvider(Enum):
    """💳 Provedores de pagamento"""
    STRIPE = "stripe"
    PAYPAL = "paypal"
    PIX = "pix"
    CRYPTO = "crypto"
    BANK_TRANSFER = "bank_transfer"


class PaymentStatus(Enum):
    """📊 Status de pagamento"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"


class SubscriptionTier(Enum):
    """🏆 Níveis de assinatura"""
    TRIAL = "trial"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


@dataclass
class SubscriptionPlan:
    """📋 Plano de assinatura"""
    id: str
    name: str
    tier: SubscriptionTier
    price_monthly: Decimal
    price_yearly: Decimal
    features: List[str]
    max_api_calls: int
    max_positions: int
    max_assets: int
    support_level: str
    analytics_retention_days: int
    custom_strategies: bool
    priority_execution: bool
    risk_limits: Dict[str, float]
    description: str
    is_active: bool = True


@dataclass
class Payment:
    """💰 Registro de pagamento"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    subscription_id: str = ""
    amount: Decimal = Decimal('0.00')
    currency: str = "USD"
    provider: PaymentProvider = PaymentProvider.STRIPE
    provider_payment_id: str = ""
    status: PaymentStatus = PaymentStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    processed_at: Optional[datetime] = None
    metadata: Dict = field(default_factory=dict)
    description: str = ""
    invoice_url: str = ""
    receipt_url: str = ""
    failure_reason: str = ""


@dataclass
class Subscription:
    """📱 Assinatura do usuário"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    plan_id: str = ""
    status: str = "active"
    current_period_start: datetime = field(default_factory=datetime.now)
    current_period_end: datetime = field(default_factory=lambda: datetime.now() + timedelta(days=30))
    trial_start: Optional[datetime] = None
    trial_end: Optional[datetime] = None
    cancel_at_period_end: bool = False
    cancelled_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    payment_method: Dict = field(default_factory=dict)
    billing_cycle: str = "monthly"  # monthly, yearly
    discount_percent: float = 0.0
    usage_stats: Dict = field(default_factory=dict)
    next_billing_amount: Decimal = Decimal('0.00')


class SubscriptionPlansManager:
    """📋 Gerenciador de planos de assinatura"""

    def __init__(self):
        self.plans = self._initialize_plans()

    def _initialize_plans(self) -> Dict[str, SubscriptionPlan]:
        """Inicializar planos padrão"""
        return {
            "trial": SubscriptionPlan(
                id="trial",
                name="Trial Gratuito",
                tier=SubscriptionTier.TRIAL,
                price_monthly=Decimal('0.00'),
                price_yearly=Decimal('0.00'),
                features=[
                    "Acesso básico ao bot",
                    "1 ativo simultâneo",
                    "Dados em tempo real limitados",
                    "Suporte via email"
                ],
                max_api_calls=1000,
                max_positions=1,
                max_assets=1,
                support_level="email",
                analytics_retention_days=7,
                custom_strategies=False,
                priority_execution=False,
                risk_limits={"max_position_size": 0.01, "max_daily_risk": 0.02},
                description="Teste grátis por 14 dias com funcionalidades básicas"
            ),

            "basic": SubscriptionPlan(
                id="basic",
                name="Plano Básico",
                tier=SubscriptionTier.BASIC,
                price_monthly=Decimal('49.99'),
                price_yearly=Decimal('499.99'),
                features=[
                    "Acesso completo ao bot",
                    "5 ativos simultâneos",
                    "Dados em tempo real completos",
                    "Estratégias pré-configuradas",
                    "Suporte via email e chat"
                ],
                max_api_calls=10000,
                max_positions=5,
                max_assets=5,
                support_level="email_chat",
                analytics_retention_days=30,
                custom_strategies=False,
                priority_execution=False,
                risk_limits={"max_position_size": 0.05, "max_daily_risk": 0.05},
                description="Ideal para traders iniciantes e intermediários"
            ),

            "premium": SubscriptionPlan(
                id="premium",
                name="Plano Premium",
                tier=SubscriptionTier.PREMIUM,
                price_monthly=Decimal('149.99'),
                price_yearly=Decimal('1499.99'),
                features=[
                    "Todos os recursos do Básico",
                    "Ativos ilimitados",
                    "Estratégias customizadas",
                    "Análise avançada de risco",
                    "Execução prioritária",
                    "Suporte 24/7"
                ],
                max_api_calls=100000,
                max_positions=20,
                max_assets=-1,  # Ilimitado
                support_level="priority_24_7",
                analytics_retention_days=90,
                custom_strategies=True,
                priority_execution=True,
                risk_limits={"max_position_size": 0.10, "max_daily_risk": 0.10},
                description="Para traders profissionais e algoritmos avançados"
            ),

            "enterprise": SubscriptionPlan(
                id="enterprise",
                name="Plano Enterprise",
                tier=SubscriptionTier.ENTERPRISE,
                price_monthly=Decimal('499.99'),
                price_yearly=Decimal('4999.99'),
                features=[
                    "Todos os recursos Premium",
                    "API dedicada",
                    "Servidor dedicado",
                    "Customização completa",
                    "Consultoria especializada",
                    "SLA garantido",
                    "Integração personalizada"
                ],
                max_api_calls=-1,  # Ilimitado
                max_positions=-1,  # Ilimitado
                max_assets=-1,     # Ilimitado
                support_level="dedicated_manager",
                analytics_retention_days=365,
                custom_strategies=True,
                priority_execution=True,
                risk_limits={"max_position_size": 1.0, "max_daily_risk": 0.20},
                description="Solução completa para instituições e fundos"
            )
        }

    def get_plan(self, plan_id: str) -> Optional[SubscriptionPlan]:
        """Obter plano por ID"""
        return self.plans.get(plan_id)

    def get_all_plans(self) -> List[SubscriptionPlan]:
        """Obter todos os planos ativos"""
        return [plan for plan in self.plans.values() if plan.is_active]

    def calculate_price(self, plan_id: str, billing_cycle: str = "monthly", discount_percent: float = 0.0) -> Decimal:
        """Calcular preço com desconto"""
        plan = self.get_plan(plan_id)
        if not plan:
            return Decimal('0.00')

        base_price = plan.price_yearly if billing_cycle == "yearly" else plan.price_monthly
        discount_amount = base_price * Decimal(str(discount_percent / 100))

        return base_price - discount_amount


class PaymentProcessor:
    """💳 Processador de pagamentos"""

    def __init__(self, stripe_secret_key: str = "", webhook_secret: str = ""):
        self.stripe_secret_key = stripe_secret_key
        self.webhook_secret = webhook_secret

        if stripe_secret_key:
            stripe.api_key = stripe_secret_key

    async def create_stripe_payment_intent(self, amount: Decimal, currency: str = "USD", metadata: Dict = None) -> Optional[Dict]:
        """Criar payment intent no Stripe"""
        try:
            intent = stripe.PaymentIntent.create(
                amount=int(amount * 100),  # Stripe usa centavos
                currency=currency.lower(),
                metadata=metadata or {},
                automatic_payment_methods={'enabled': True}
            )

            return {
                'client_secret': intent.client_secret,
                'payment_intent_id': intent.id,
                'status': intent.status,
                'amount': amount,
                'currency': currency
            }

        except stripe.error.StripeError as e:
            logger.error(f"❌ Stripe error: {e}")
            return None

    async def create_stripe_subscription(self, customer_id: str, price_id: str, trial_period_days: int = 0) -> Optional[Dict]:
        """Criar assinatura no Stripe"""
        try:
            subscription_data = {
                'customer': customer_id,
                'items': [{'price': price_id}],
                'expand': ['latest_invoice.payment_intent']
            }

            if trial_period_days > 0:
                subscription_data['trial_period_days'] = trial_period_days

            subscription = stripe.Subscription.create(**subscription_data)

            return {
                'subscription_id': subscription.id,
                'status': subscription.status,
                'client_secret': subscription.latest_invoice.payment_intent.client_secret if subscription.latest_invoice.payment_intent else None,
                'current_period_start': datetime.fromtimestamp(subscription.current_period_start),
                'current_period_end': datetime.fromtimestamp(subscription.current_period_end)
            }

        except stripe.error.StripeError as e:
            logger.error(f"❌ Stripe subscription error: {e}")
            return None

    async def process_pix_payment(self, amount: Decimal, user_email: str) -> Optional[Dict]:
        """Processar pagamento PIX (simulado)"""
        try:
            # Simulação de integração PIX
            pix_code = f"PIX{secrets.token_hex(8).upper()}"
            qr_code_data = f"00020101021126580014br.gov.bcb.pix0136{user_email}520400005303986540{amount}5802BR5909TradingBot6009SaoPaulo{pix_code}"

            return {
                'pix_code': pix_code,
                'qr_code_data': qr_code_data,
                'expires_at': (datetime.now() + timedelta(minutes=30)).isoformat(),
                'amount': amount,
                'status': 'pending'
            }

        except Exception as e:
            logger.error(f"❌ PIX payment error: {e}")
            return None

    async def verify_webhook_signature(self, payload: str, signature: str) -> bool:
        """Verificar assinatura do webhook"""
        try:
            stripe.Webhook.construct_event(payload, signature, self.webhook_secret)
            return True
        except (ValueError, stripe.error.SignatureVerificationError):
            return False


class BillingManager:
    """📊 Gerenciador de cobrança"""

    def __init__(self, database_url: str, stripe_secret_key: str = ""):
        self.database_url = database_url
        self.pool = None
        self.plans_manager = SubscriptionPlansManager()
        self.payment_processor = PaymentProcessor(stripe_secret_key)

    async def initialize(self):
        """Inicializar sistema de billing"""
        try:
            self.pool = await asyncpg.create_pool(self.database_url)
            await self._create_tables()
            logger.info("✅ Billing system initialized")
        except Exception as e:
            logger.error(f"❌ Billing initialization failed: {e}")
            await self._initialize_file_storage()

    async def _create_tables(self):
        """Criar tabelas de billing"""
        async with self.pool.acquire() as conn:
            # Tabela de assinaturas
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS subscriptions (
                    id UUID PRIMARY KEY,
                    user_id UUID NOT NULL,
                    plan_id VARCHAR NOT NULL,
                    status VARCHAR DEFAULT 'active',
                    current_period_start TIMESTAMP NOT NULL,
                    current_period_end TIMESTAMP NOT NULL,
                    trial_start TIMESTAMP,
                    trial_end TIMESTAMP,
                    cancel_at_period_end BOOLEAN DEFAULT FALSE,
                    cancelled_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    payment_method JSONB DEFAULT '{}',
                    billing_cycle VARCHAR DEFAULT 'monthly',
                    discount_percent DECIMAL DEFAULT 0.0,
                    usage_stats JSONB DEFAULT '{}',
                    next_billing_amount DECIMAL DEFAULT 0.00
                )
            """)

            # Tabela de pagamentos
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS payments (
                    id UUID PRIMARY KEY,
                    user_id UUID NOT NULL,
                    subscription_id UUID,
                    amount DECIMAL NOT NULL,
                    currency VARCHAR DEFAULT 'USD',
                    provider VARCHAR NOT NULL,
                    provider_payment_id VARCHAR,
                    status VARCHAR DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    processed_at TIMESTAMP,
                    metadata JSONB DEFAULT '{}',
                    description TEXT,
                    invoice_url VARCHAR,
                    receipt_url VARCHAR,
                    failure_reason TEXT
                )
            """)

            # Índices
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_subscriptions_user_id ON subscriptions(user_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_payments_user_id ON payments(user_id)")

    async def _initialize_file_storage(self):
        """Inicializar armazenamento em arquivo"""
        self.storage_path = Path("data/billing")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        logger.info("📁 Using file-based billing storage")

    async def create_subscription(self, user_id: str, plan_id: str, billing_cycle: str = "monthly", payment_method: Dict = None) -> Optional[Subscription]:
        """Criar nova assinatura"""
        try:
            plan = self.plans_manager.get_plan(plan_id)
            if not plan:
                logger.error(f"❌ Plan not found: {plan_id}")
                return None

            # Calcular datas
            start_date = datetime.now()

            if billing_cycle == "yearly":
                end_date = start_date + timedelta(days=365)
                amount = plan.price_yearly
            else:
                end_date = start_date + timedelta(days=30)
                amount = plan.price_monthly

            # Período de trial para novos usuários
            trial_end = None
            if plan_id == "trial":
                trial_end = start_date + timedelta(days=14)
                end_date = trial_end

            subscription = Subscription(
                user_id=user_id,
                plan_id=plan_id,
                current_period_start=start_date,
                current_period_end=end_date,
                trial_end=trial_end,
                billing_cycle=billing_cycle,
                payment_method=payment_method or {},
                next_billing_amount=amount
            )

            success = await self._save_subscription(subscription)
            if success:
                logger.info(f"✅ Subscription created: {subscription.id} for user {user_id}")
                return subscription

        except Exception as e:
            logger.error(f"❌ Error creating subscription: {e}")

        return None

    async def process_payment(self, user_id: str, subscription_id: str, amount: Decimal, provider: PaymentProvider, currency: str = "USD") -> Optional[Payment]:
        """Processar pagamento"""
        try:
            payment = Payment(
                user_id=user_id,
                subscription_id=subscription_id,
                amount=amount,
                currency=currency,
                provider=provider,
                description=f"Subscription payment - {amount} {currency}"
            )

            # Processar pagamento baseado no provedor
            if provider == PaymentProvider.STRIPE:
                result = await self.payment_processor.create_stripe_payment_intent(
                    amount=amount,
                    currency=currency,
                    metadata={
                        'user_id': user_id,
                        'subscription_id': subscription_id,
                        'payment_id': payment.id
                    }
                )

                if result:
                    payment.provider_payment_id = result['payment_intent_id']
                    payment.metadata = result
                    payment.status = PaymentStatus.PROCESSING

            elif provider == PaymentProvider.PIX:
                result = await self.payment_processor.process_pix_payment(
                    amount=amount,
                    user_email=f"user_{user_id}@example.com"
                )

                if result:
                    payment.provider_payment_id = result['pix_code']
                    payment.metadata = result

            # Salvar pagamento
            success = await self._save_payment(payment)
            if success:
                logger.info(f"✅ Payment processed: {payment.id}")
                return payment

        except Exception as e:
            logger.error(f"❌ Error processing payment: {e}")

        return None

    async def get_user_subscription(self, user_id: str) -> Optional[Subscription]:
        """Obter assinatura ativa do usuário"""
        try:
            if self.pool:
                async with self.pool.acquire() as conn:
                    row = await conn.fetchrow(
                        "SELECT * FROM subscriptions WHERE user_id = $1 AND status = 'active' ORDER BY created_at DESC LIMIT 1",
                        user_id
                    )
                    if row:
                        return self._row_to_subscription(row)
            else:
                # Busca em arquivos
                subscription_file = self.storage_path / f"subscription_{user_id}.json"
                if subscription_file.exists():
                    async with aiofiles.open(subscription_file, 'r') as f:
                        data = json.loads(await f.read())
                        return self._dict_to_subscription(data)

        except Exception as e:
            logger.error(f"❌ Error getting user subscription: {e}")

        return None

    async def cancel_subscription(self, subscription_id: str, immediate: bool = False) -> bool:
        """Cancelar assinatura"""
        try:
            if self.pool:
                async with self.pool.acquire() as conn:
                    if immediate:
                        await conn.execute(
                            "UPDATE subscriptions SET status = 'cancelled', cancelled_at = NOW(), updated_at = NOW() WHERE id = $1",
                            subscription_id
                        )
                    else:
                        await conn.execute(
                            "UPDATE subscriptions SET cancel_at_period_end = TRUE, updated_at = NOW() WHERE id = $1",
                            subscription_id
                        )

                    logger.info(f"✅ Subscription cancelled: {subscription_id}")
                    return True

        except Exception as e:
            logger.error(f"❌ Error cancelling subscription: {e}")

        return False

    async def get_billing_stats(self) -> Dict:
        """Estatísticas de billing"""
        try:
            stats = {
                'total_subscriptions': 0,
                'active_subscriptions': 0,
                'trial_subscriptions': 0,
                'premium_subscriptions': 0,
                'monthly_revenue': Decimal('0.00'),
                'yearly_revenue': Decimal('0.00'),
                'total_payments': 0,
                'successful_payments': 0,
                'failed_payments': 0
            }

            if self.pool:
                async with self.pool.acquire() as conn:
                    # Estatísticas de assinaturas
                    sub_stats = await conn.fetchrow("""
                        SELECT
                            COUNT(*) as total,
                            COUNT(*) FILTER (WHERE status = 'active') as active,
                            COUNT(*) FILTER (WHERE plan_id = 'trial') as trial,
                            COUNT(*) FILTER (WHERE plan_id = 'premium') as premium
                        FROM subscriptions
                    """)

                    stats.update({
                        'total_subscriptions': sub_stats['total'],
                        'active_subscriptions': sub_stats['active'],
                        'trial_subscriptions': sub_stats['trial'],
                        'premium_subscriptions': sub_stats['premium']
                    })

                    # Estatísticas de pagamentos
                    payment_stats = await conn.fetchrow("""
                        SELECT
                            COUNT(*) as total,
                            COUNT(*) FILTER (WHERE status = 'completed') as successful,
                            COUNT(*) FILTER (WHERE status = 'failed') as failed,
                            COALESCE(SUM(amount) FILTER (WHERE status = 'completed'), 0) as revenue
                        FROM payments
                        WHERE created_at >= date_trunc('month', CURRENT_DATE)
                    """)

                    stats.update({
                        'total_payments': payment_stats['total'],
                        'successful_payments': payment_stats['successful'],
                        'failed_payments': payment_stats['failed'],
                        'monthly_revenue': Decimal(str(payment_stats['revenue']))
                    })

            return stats

        except Exception as e:
            logger.error(f"❌ Error getting billing stats: {e}")
            return {'error': str(e)}

    async def _save_subscription(self, subscription: Subscription) -> bool:
        """Salvar assinatura"""
        try:
            if self.pool:
                async with self.pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO subscriptions (
                            id, user_id, plan_id, status, current_period_start, current_period_end,
                            trial_start, trial_end, cancel_at_period_end, cancelled_at,
                            created_at, updated_at, payment_method, billing_cycle,
                            discount_percent, usage_stats, next_billing_amount
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
                        ON CONFLICT (id) DO UPDATE SET
                            status = EXCLUDED.status,
                            current_period_start = EXCLUDED.current_period_start,
                            current_period_end = EXCLUDED.current_period_end,
                            trial_end = EXCLUDED.trial_end,
                            cancel_at_period_end = EXCLUDED.cancel_at_period_end,
                            cancelled_at = EXCLUDED.cancelled_at,
                            updated_at = EXCLUDED.updated_at,
                            payment_method = EXCLUDED.payment_method,
                            billing_cycle = EXCLUDED.billing_cycle,
                            discount_percent = EXCLUDED.discount_percent,
                            usage_stats = EXCLUDED.usage_stats,
                            next_billing_amount = EXCLUDED.next_billing_amount
                    """,
                        subscription.id, subscription.user_id, subscription.plan_id,
                        subscription.status, subscription.current_period_start, subscription.current_period_end,
                        subscription.trial_start, subscription.trial_end, subscription.cancel_at_period_end,
                        subscription.cancelled_at, subscription.created_at, subscription.updated_at,
                        json.dumps(subscription.payment_method), subscription.billing_cycle,
                        subscription.discount_percent, json.dumps(subscription.usage_stats),
                        subscription.next_billing_amount
                    )
                return True

        except Exception as e:
            logger.error(f"❌ Error saving subscription: {e}")

        return False

    async def _save_payment(self, payment: Payment) -> bool:
        """Salvar pagamento"""
        try:
            if self.pool:
                async with self.pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO payments (
                            id, user_id, subscription_id, amount, currency, provider,
                            provider_payment_id, status, created_at, updated_at,
                            processed_at, metadata, description, invoice_url,
                            receipt_url, failure_reason
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                        ON CONFLICT (id) DO UPDATE SET
                            status = EXCLUDED.status,
                            updated_at = EXCLUDED.updated_at,
                            processed_at = EXCLUDED.processed_at,
                            metadata = EXCLUDED.metadata,
                            invoice_url = EXCLUDED.invoice_url,
                            receipt_url = EXCLUDED.receipt_url,
                            failure_reason = EXCLUDED.failure_reason
                    """,
                        payment.id, payment.user_id, payment.subscription_id,
                        payment.amount, payment.currency, payment.provider.value,
                        payment.provider_payment_id, payment.status.value,
                        payment.created_at, payment.updated_at, payment.processed_at,
                        json.dumps(payment.metadata), payment.description,
                        payment.invoice_url, payment.receipt_url, payment.failure_reason
                    )
                return True

        except Exception as e:
            logger.error(f"❌ Error saving payment: {e}")

        return False

    def _row_to_subscription(self, row) -> Subscription:
        """Converter row para Subscription"""
        return Subscription(
            id=str(row['id']),
            user_id=str(row['user_id']),
            plan_id=row['plan_id'],
            status=row['status'],
            current_period_start=row['current_period_start'],
            current_period_end=row['current_period_end'],
            trial_start=row['trial_start'],
            trial_end=row['trial_end'],
            cancel_at_period_end=row['cancel_at_period_end'],
            cancelled_at=row['cancelled_at'],
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            payment_method=json.loads(row['payment_method']) if row['payment_method'] else {},
            billing_cycle=row['billing_cycle'],
            discount_percent=float(row['discount_percent']),
            usage_stats=json.loads(row['usage_stats']) if row['usage_stats'] else {},
            next_billing_amount=Decimal(str(row['next_billing_amount']))
        )

    def _dict_to_subscription(self, data: Dict) -> Subscription:
        """Converter dict para Subscription"""
        return Subscription(
            id=data['id'],
            user_id=data['user_id'],
            plan_id=data['plan_id'],
            status=data['status'],
            current_period_start=datetime.fromisoformat(data['current_period_start']),
            current_period_end=datetime.fromisoformat(data['current_period_end']),
            trial_start=datetime.fromisoformat(data['trial_start']) if data.get('trial_start') else None,
            trial_end=datetime.fromisoformat(data['trial_end']) if data.get('trial_end') else None,
            cancel_at_period_end=data.get('cancel_at_period_end', False),
            cancelled_at=datetime.fromisoformat(data['cancelled_at']) if data.get('cancelled_at') else None,
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            payment_method=data.get('payment_method', {}),
            billing_cycle=data.get('billing_cycle', 'monthly'),
            discount_percent=data.get('discount_percent', 0.0),
            usage_stats=data.get('usage_stats', {}),
            next_billing_amount=Decimal(str(data.get('next_billing_amount', '0.00')))
        )


# 🧪 Função de teste
async def test_billing_subscription_system():
    """Testar sistema de billing e assinaturas"""
    # Configurações
    database_url = "postgresql://trading_user:password@localhost:5432/trading_db"
    stripe_secret_key = "sk_test_..."  # Chave de teste do Stripe

    # Inicializar sistema
    billing_manager = BillingManager(database_url, stripe_secret_key)
    await billing_manager.initialize()

    print("\n" + "="*80)
    print("💳 BILLING & SUBSCRIPTION SYSTEM TEST")
    print("="*80)

    # 1. Listar planos disponíveis
    print("\n📋 AVAILABLE SUBSCRIPTION PLANS:")
    plans = billing_manager.plans_manager.get_all_plans()
    for plan in plans:
        print(f"   📱 {plan.name} ({plan.id})")
        print(f"      💰 Monthly: ${plan.price_monthly} | Yearly: ${plan.price_yearly}")
        print(f"      🎯 Features: {len(plan.features)} features")
        print(f"      📊 Max API Calls: {plan.max_api_calls}")
        print()

    # 2. Criar assinatura de teste
    test_user_id = str(uuid.uuid4())
    print(f"\n📱 CREATING TEST SUBSCRIPTION for user: {test_user_id}")

    subscription = await billing_manager.create_subscription(
        user_id=test_user_id,
        plan_id="premium",
        billing_cycle="monthly"
    )

    if subscription:
        print(f"✅ Subscription created: {subscription.id}")
        print(f"   Plan: {subscription.plan_id}")
        print(f"   Status: {subscription.status}")
        print(f"   Period: {subscription.current_period_start} to {subscription.current_period_end}")
        print(f"   Next billing: ${subscription.next_billing_amount}")

    # 3. Processar pagamento
    if subscription:
        print(f"\n💳 PROCESSING PAYMENT...")
        payment = await billing_manager.process_payment(
            user_id=test_user_id,
            subscription_id=subscription.id,
            amount=Decimal('149.99'),
            provider=PaymentProvider.STRIPE
        )

        if payment:
            print(f"✅ Payment processed: {payment.id}")
            print(f"   Amount: ${payment.amount} {payment.currency}")
            print(f"   Provider: {payment.provider.value}")
            print(f"   Status: {payment.status.value}")

    # 4. Estatísticas de billing
    print(f"\n📊 BILLING STATISTICS:")
    stats = await billing_manager.get_billing_stats()
    print(f"   Total Subscriptions: {stats.get('total_subscriptions', 0)}")
    print(f"   Active Subscriptions: {stats.get('active_subscriptions', 0)}")
    print(f"   Premium Subscriptions: {stats.get('premium_subscriptions', 0)}")
    print(f"   Monthly Revenue: ${stats.get('monthly_revenue', 0)}")
    print(f"   Successful Payments: {stats.get('successful_payments', 0)}")

    # 5. Testar PIX
    print(f"\n🇧🇷 TESTING PIX PAYMENT...")
    pix_payment = await billing_manager.process_payment(
        user_id=test_user_id,
        subscription_id=subscription.id if subscription else "test_sub",
        amount=Decimal('149.99'),
        provider=PaymentProvider.PIX,
        currency="BRL"
    )

    if pix_payment:
        print(f"✅ PIX payment created: {pix_payment.id}")
        print(f"   PIX Code: {pix_payment.metadata.get('pix_code', 'N/A')}")
        print(f"   Expires at: {pix_payment.metadata.get('expires_at', 'N/A')}")

    print("\n" + "="*80)
    print("✅ BILLING & SUBSCRIPTION SYSTEM TEST COMPLETED")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(test_billing_subscription_system())