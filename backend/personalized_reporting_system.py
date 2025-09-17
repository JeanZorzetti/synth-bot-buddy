"""
📋 PERSONALIZED REPORTING SYSTEM
Sistema completo de relatórios personalizados com automação e distribuição
"""

import asyncio
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import uuid
import asyncpg
import aiofiles
from pathlib import Path
import schedule
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from email.mime.application import MimeApplication
from jinja2 import Template, Environment, FileSystemLoader
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from weasyprint import HTML, CSS
import base64
import io
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportFormat(Enum):
    """📄 Formatos de relatório"""
    HTML = "html"
    PDF = "pdf"
    CSV = "csv"
    EXCEL = "excel"
    JSON = "json"
    POWERPOINT = "powerpoint"


class ReportFrequency(Enum):
    """⏰ Frequência de relatórios"""
    MANUAL = "manual"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


class ReportStatus(Enum):
    """📊 Status do relatório"""
    DRAFT = "draft"
    SCHEDULED = "scheduled"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"
    SENT = "sent"


@dataclass
class ReportTemplate:
    """📋 Template de relatório"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    category: str = ""
    sql_query: str = ""
    chart_configs: List[Dict] = field(default_factory=list)
    template_html: str = ""
    parameters: Dict = field(default_factory=dict)
    required_permissions: List[str] = field(default_factory=list)
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    is_public: bool = False
    tags: List[str] = field(default_factory=list)


@dataclass
class PersonalizedReport:
    """📊 Relatório personalizado"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    template_id: str = ""
    name: str = ""
    description: str = ""
    parameters: Dict = field(default_factory=dict)
    frequency: ReportFrequency = ReportFrequency.MANUAL
    format: ReportFormat = ReportFormat.HTML
    recipients: List[str] = field(default_factory=list)
    schedule_time: str = "08:00"  # HH:MM
    schedule_day: Optional[int] = None  # For weekly (1-7) or monthly (1-31)
    last_generated_at: Optional[datetime] = None
    next_run_at: Optional[datetime] = None
    status: ReportStatus = ReportStatus.DRAFT
    file_path: str = ""
    error_message: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)


@dataclass
class ReportData:
    """📈 Dados do relatório"""
    query_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    charts: List[Dict] = field(default_factory=list)
    summary_stats: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.now)


class ReportTemplateManager:
    """📋 Gerenciador de templates"""

    def __init__(self):
        self.templates = self._initialize_default_templates()

    def _initialize_default_templates(self) -> Dict[str, ReportTemplate]:
        """Inicializar templates padrão"""
        templates = {}

        # Template: Trading Performance Summary
        templates["trading_performance"] = ReportTemplate(
            id="trading_performance",
            name="Trading Performance Summary",
            description="Resumo completo de performance de trading",
            category="trading",
            sql_query="""
                SELECT
                    DATE(timestamp) as date,
                    symbol,
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(profit_loss) as total_pnl,
                    AVG(profit_loss) as avg_pnl,
                    MAX(profit_loss) as max_profit,
                    MIN(profit_loss) as max_loss,
                    AVG(duration_minutes) as avg_duration,
                    strategy_name
                FROM trading_metrics
                WHERE timestamp >= %(start_date)s
                  AND timestamp <= %(end_date)s
                  AND (%(user_id)s IS NULL OR user_id = %(user_id)s)
                GROUP BY DATE(timestamp), symbol, strategy_name
                ORDER BY date DESC, total_pnl DESC
            """,
            chart_configs=[
                {
                    "type": "line",
                    "x": "date",
                    "y": "total_pnl",
                    "title": "P&L Daily",
                    "cumulative": True
                },
                {
                    "type": "bar",
                    "x": "symbol",
                    "y": "total_trades",
                    "title": "Trades por Símbolo"
                },
                {
                    "type": "pie",
                    "values": "total_pnl",
                    "names": "strategy_name",
                    "title": "P&L por Estratégia"
                }
            ],
            parameters={
                "start_date": {"type": "date", "default": "30_days_ago"},
                "end_date": {"type": "date", "default": "today"},
                "user_id": {"type": "string", "default": None, "optional": True}
            },
            tags=["trading", "performance", "pnl"]
        )

        # Template: User Activity Report
        templates["user_activity"] = ReportTemplate(
            id="user_activity",
            name="User Activity Report",
            description="Relatório detalhado de atividade dos usuários",
            category="analytics",
            sql_query="""
                SELECT
                    DATE(timestamp) as date,
                    COUNT(DISTINCT user_id) as active_users,
                    COUNT(*) as total_events,
                    COUNT(DISTINCT session_id) as unique_sessions,
                    event_type,
                    COUNT(*) as event_count
                FROM user_analytics
                WHERE timestamp >= %(start_date)s
                  AND timestamp <= %(end_date)s
                GROUP BY DATE(timestamp), event_type
                ORDER BY date DESC, event_count DESC
            """,
            chart_configs=[
                {
                    "type": "line",
                    "x": "date",
                    "y": "active_users",
                    "title": "Usuários Ativos Diários"
                },
                {
                    "type": "bar",
                    "x": "event_type",
                    "y": "event_count",
                    "title": "Eventos por Tipo"
                }
            ],
            parameters={
                "start_date": {"type": "date", "default": "7_days_ago"},
                "end_date": {"type": "date", "default": "today"}
            },
            tags=["users", "activity", "analytics"]
        )

        # Template: Financial Summary
        templates["financial_summary"] = ReportTemplate(
            id="financial_summary",
            name="Financial Summary",
            description="Resumo financeiro completo",
            category="financial",
            sql_query="""
                SELECT
                    DATE(timestamp) as date,
                    SUM(profit_loss) as daily_pnl,
                    COUNT(*) as trades_count,
                    AVG(profit_loss) as avg_trade_pnl,
                    SUM(CASE WHEN profit_loss > 0 THEN profit_loss ELSE 0 END) as gross_profit,
                    SUM(CASE WHEN profit_loss < 0 THEN ABS(profit_loss) ELSE 0 END) as gross_loss,
                    COUNT(CASE WHEN profit_loss > 0 THEN 1 END) as winning_trades,
                    COUNT(CASE WHEN profit_loss < 0 THEN 1 END) as losing_trades
                FROM trading_metrics
                WHERE timestamp >= %(start_date)s
                  AND timestamp <= %(end_date)s
                  AND (%(user_id)s IS NULL OR user_id = %(user_id)s)
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
            """,
            chart_configs=[
                {
                    "type": "waterfall",
                    "x": "date",
                    "y": "daily_pnl",
                    "title": "P&L Waterfall Chart"
                },
                {
                    "type": "line",
                    "x": "date",
                    "y": ["gross_profit", "gross_loss"],
                    "title": "Profit vs Loss"
                }
            ],
            parameters={
                "start_date": {"type": "date", "default": "30_days_ago"},
                "end_date": {"type": "date", "default": "today"},
                "user_id": {"type": "string", "default": None, "optional": True}
            },
            tags=["financial", "pnl", "summary"]
        )

        # Template: Risk Analysis
        templates["risk_analysis"] = ReportTemplate(
            id="risk_analysis",
            name="Risk Analysis Report",
            description="Análise detalhada de risco e drawdown",
            category="risk",
            sql_query="""
                WITH daily_pnl AS (
                    SELECT
                        DATE(timestamp) as date,
                        user_id,
                        SUM(profit_loss) as daily_pnl,
                        COUNT(*) as trades_count,
                        MAX(profit_loss) as max_win,
                        MIN(profit_loss) as max_loss
                    FROM trading_metrics
                    WHERE timestamp >= %(start_date)s
                      AND timestamp <= %(end_date)s
                      AND (%(user_id)s IS NULL OR user_id = %(user_id)s)
                    GROUP BY DATE(timestamp), user_id
                ),
                cumulative_pnl AS (
                    SELECT *,
                        SUM(daily_pnl) OVER (PARTITION BY user_id ORDER BY date) as cumulative_pnl
                    FROM daily_pnl
                ),
                drawdown AS (
                    SELECT *,
                        cumulative_pnl - MAX(cumulative_pnl) OVER (
                            PARTITION BY user_id
                            ORDER BY date
                            ROWS UNBOUNDED PRECEDING
                        ) as drawdown
                    FROM cumulative_pnl
                )
                SELECT
                    date,
                    user_id,
                    daily_pnl,
                    cumulative_pnl,
                    drawdown,
                    trades_count,
                    max_win,
                    max_loss,
                    ABS(drawdown) / NULLIF(cumulative_pnl, 0) * 100 as drawdown_percent
                FROM drawdown
                ORDER BY date DESC
            """,
            chart_configs=[
                {
                    "type": "line",
                    "x": "date",
                    "y": "drawdown",
                    "title": "Drawdown Analysis"
                },
                {
                    "type": "histogram",
                    "x": "daily_pnl",
                    "title": "Daily P&L Distribution"
                }
            ],
            parameters={
                "start_date": {"type": "date", "default": "90_days_ago"},
                "end_date": {"type": "date", "default": "today"},
                "user_id": {"type": "string", "default": None, "optional": True}
            },
            tags=["risk", "drawdown", "analysis"]
        )

        return templates

    def get_template(self, template_id: str) -> Optional[ReportTemplate]:
        """Obter template por ID"""
        return self.templates.get(template_id)

    def list_templates(self, category: str = None, tags: List[str] = None) -> List[ReportTemplate]:
        """Listar templates com filtros"""
        templates = list(self.templates.values())

        if category:
            templates = [t for t in templates if t.category == category]

        if tags:
            templates = [t for t in templates if any(tag in t.tags for tag in tags)]

        return templates


class ChartGenerator:
    """📊 Gerador de gráficos"""

    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3

    def generate_chart(self, data: pd.DataFrame, chart_config: Dict) -> Dict:
        """Gerar gráfico baseado na configuração"""
        try:
            chart_type = chart_config.get("type", "line")
            title = chart_config.get("title", "Chart")

            if chart_type == "line":
                return self._create_line_chart(data, chart_config, title)
            elif chart_type == "bar":
                return self._create_bar_chart(data, chart_config, title)
            elif chart_type == "pie":
                return self._create_pie_chart(data, chart_config, title)
            elif chart_type == "histogram":
                return self._create_histogram(data, chart_config, title)
            elif chart_type == "waterfall":
                return self._create_waterfall_chart(data, chart_config, title)
            else:
                return self._create_line_chart(data, chart_config, title)

        except Exception as e:
            logger.error(f"❌ Error generating chart: {e}")
            return {"error": str(e)}

    def _create_line_chart(self, data: pd.DataFrame, config: Dict, title: str) -> Dict:
        """Criar gráfico de linha"""
        fig = go.Figure()

        x_col = config.get("x")
        y_cols = config.get("y", []) if isinstance(config.get("y"), list) else [config.get("y")]

        for i, y_col in enumerate(y_cols):
            if y_col in data.columns:
                y_data = data[y_col]

                # Aplicar cumulativo se solicitado
                if config.get("cumulative", False):
                    y_data = y_data.cumsum()

                fig.add_trace(go.Scatter(
                    x=data[x_col] if x_col in data.columns else data.index,
                    y=y_data,
                    mode='lines+markers',
                    name=y_col,
                    line=dict(color=self.color_palette[i % len(self.color_palette)])
                ))

        fig.update_layout(
            title=title,
            xaxis_title=x_col,
            yaxis_title="Value",
            hovermode='x unified'
        )

        return fig.to_dict()

    def _create_bar_chart(self, data: pd.DataFrame, config: Dict, title: str) -> Dict:
        """Criar gráfico de barras"""
        x_col = config.get("x")
        y_col = config.get("y")

        # Agrupar dados se necessário
        if len(data) > 20:  # Limitar número de barras
            data = data.groupby(x_col)[y_col].sum().reset_index()

        fig = go.Figure(data=[
            go.Bar(
                x=data[x_col],
                y=data[y_col],
                marker_color=self.color_palette[0]
            )
        ])

        fig.update_layout(
            title=title,
            xaxis_title=x_col,
            yaxis_title=y_col
        )

        return fig.to_dict()

    def _create_pie_chart(self, data: pd.DataFrame, config: Dict, title: str) -> Dict:
        """Criar gráfico de pizza"""
        values_col = config.get("values")
        names_col = config.get("names")

        # Agrupar e somar valores
        grouped_data = data.groupby(names_col)[values_col].sum().reset_index()

        fig = go.Figure(data=[
            go.Pie(
                labels=grouped_data[names_col],
                values=grouped_data[values_col],
                marker_colors=self.color_palette
            )
        ])

        fig.update_layout(title=title)

        return fig.to_dict()

    def _create_histogram(self, data: pd.DataFrame, config: Dict, title: str) -> Dict:
        """Criar histograma"""
        x_col = config.get("x")

        fig = go.Figure(data=[
            go.Histogram(
                x=data[x_col],
                nbinsx=30,
                marker_color=self.color_palette[0]
            )
        ])

        fig.update_layout(
            title=title,
            xaxis_title=x_col,
            yaxis_title="Frequency"
        )

        return fig.to_dict()

    def _create_waterfall_chart(self, data: pd.DataFrame, config: Dict, title: str) -> Dict:
        """Criar gráfico waterfall"""
        x_col = config.get("x")
        y_col = config.get("y")

        fig = go.Figure(go.Waterfall(
            name="",
            orientation="v",
            measure=["relative"] * len(data),
            x=data[x_col],
            y=data[y_col],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))

        fig.update_layout(title=title)

        return fig.to_dict()


class ReportGenerator:
    """📄 Gerador de relatórios"""

    def __init__(self, warehouse_manager):
        self.warehouse_manager = warehouse_manager
        self.chart_generator = ChartGenerator()
        self.template_env = Environment(loader=FileSystemLoader('templates'))

    async def generate_report_data(self, template: ReportTemplate, parameters: Dict) -> ReportData:
        """Gerar dados do relatório"""
        try:
            # Processar parâmetros
            processed_params = self._process_parameters(template.parameters, parameters)

            # Executar query
            query_results = await self.warehouse_manager.execute_query(
                template.sql_query % processed_params
            )

            if query_results.empty:
                logger.warning("No data returned from query")
                return ReportData()

            # Gerar gráficos
            charts = []
            for chart_config in template.chart_configs:
                chart = self.chart_generator.generate_chart(query_results, chart_config)
                if "error" not in chart:
                    charts.append(chart)

            # Calcular estatísticas resumo
            summary_stats = self._calculate_summary_stats(query_results)

            return ReportData(
                query_results=query_results,
                charts=charts,
                summary_stats=summary_stats,
                metadata={
                    "template_id": template.id,
                    "parameters": processed_params,
                    "rows_count": len(query_results)
                }
            )

        except Exception as e:
            logger.error(f"❌ Error generating report data: {e}")
            return ReportData(metadata={"error": str(e)})

    def _process_parameters(self, template_params: Dict, user_params: Dict) -> Dict:
        """Processar parâmetros do relatório"""
        processed = {}

        for param_name, param_config in template_params.items():
            if param_name in user_params:
                value = user_params[param_name]
            else:
                value = param_config.get("default")

            # Processar datas especiais
            if param_config.get("type") == "date" and isinstance(value, str):
                if value == "today":
                    value = datetime.now().date()
                elif value == "yesterday":
                    value = (datetime.now() - timedelta(days=1)).date()
                elif value.endswith("_days_ago"):
                    days = int(value.split("_")[0])
                    value = (datetime.now() - timedelta(days=days)).date()

            processed[param_name] = value

        return processed

    def _calculate_summary_stats(self, data: pd.DataFrame) -> Dict:
        """Calcular estatísticas resumo"""
        try:
            stats = {}

            # Estatísticas numéricas
            numeric_columns = data.select_dtypes(include=[np.number]).columns

            for col in numeric_columns:
                stats[f"{col}_sum"] = float(data[col].sum())
                stats[f"{col}_mean"] = float(data[col].mean())
                stats[f"{col}_std"] = float(data[col].std())
                stats[f"{col}_min"] = float(data[col].min())
                stats[f"{col}_max"] = float(data[col].max())

            # Contagens gerais
            stats["total_rows"] = len(data)
            stats["total_columns"] = len(data.columns)

            return stats

        except Exception as e:
            logger.error(f"❌ Error calculating summary stats: {e}")
            return {}

    async def generate_html_report(self, report_data: ReportData, template: ReportTemplate) -> str:
        """Gerar relatório em HTML"""
        try:
            # Template HTML básico
            html_template = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>{{ template.name }}</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .header { background-color: #f0f0f0; padding: 20px; margin-bottom: 20px; }
                    .summary { background-color: #e8f4fd; padding: 15px; margin-bottom: 20px; }
                    .chart { margin-bottom: 30px; }
                    table { border-collapse: collapse; width: 100%; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>{{ template.name }}</h1>
                    <p>{{ template.description }}</p>
                    <p><strong>Generated at:</strong> {{ report_data.generated_at }}</p>
                </div>

                <div class="summary">
                    <h2>Summary Statistics</h2>
                    <ul>
                    {% for key, value in report_data.summary_stats.items() %}
                        <li><strong>{{ key }}:</strong> {{ value }}</li>
                    {% endfor %}
                    </ul>
                </div>

                {% for chart in report_data.charts %}
                <div class="chart">
                    <div id="chart_{{ loop.index }}"></div>
                    <script>
                        Plotly.newPlot('chart_{{ loop.index }}', {{ chart | tojson }});
                    </script>
                </div>
                {% endfor %}

                <div class="data-table">
                    <h2>Data Table</h2>
                    {{ report_data.query_results.to_html(classes='table table-striped', escape=False) }}
                </div>
            </body>
            </html>
            """

            template_obj = Template(html_template)
            html_content = template_obj.render(
                template=template,
                report_data=report_data
            )

            return html_content

        except Exception as e:
            logger.error(f"❌ Error generating HTML report: {e}")
            return f"<html><body><h1>Error generating report: {e}</h1></body></html>"

    async def generate_pdf_report(self, html_content: str) -> bytes:
        """Gerar relatório em PDF"""
        try:
            # Usar WeasyPrint para converter HTML para PDF
            html_doc = HTML(string=html_content)
            pdf_bytes = html_doc.write_pdf()

            return pdf_bytes

        except Exception as e:
            logger.error(f"❌ Error generating PDF report: {e}")
            return b""

    async def generate_csv_report(self, report_data: ReportData) -> str:
        """Gerar relatório em CSV"""
        try:
            return report_data.query_results.to_csv(index=False)

        except Exception as e:
            logger.error(f"❌ Error generating CSV report: {e}")
            return ""

    async def generate_excel_report(self, report_data: ReportData, template: ReportTemplate) -> bytes:
        """Gerar relatório em Excel"""
        try:
            output = io.BytesIO()

            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Dados principais
                report_data.query_results.to_excel(writer, sheet_name='Data', index=False)

                # Estatísticas resumo
                summary_df = pd.DataFrame(
                    list(report_data.summary_stats.items()),
                    columns=['Metric', 'Value']
                )
                summary_df.to_excel(writer, sheet_name='Summary', index=False)

                # Metadados
                metadata_df = pd.DataFrame(
                    list(report_data.metadata.items()),
                    columns=['Key', 'Value']
                )
                metadata_df.to_excel(writer, sheet_name='Metadata', index=False)

            return output.getvalue()

        except Exception as e:
            logger.error(f"❌ Error generating Excel report: {e}")
            return b""


class ReportScheduler:
    """⏰ Agendador de relatórios"""

    def __init__(self, report_manager):
        self.report_manager = report_manager
        self.scheduled_reports = {}

    async def schedule_report(self, report: PersonalizedReport):
        """Agendar relatório"""
        try:
            if report.frequency == ReportFrequency.MANUAL:
                return

            # Calcular próxima execução
            next_run = self._calculate_next_run(report)
            report.next_run_at = next_run

            self.scheduled_reports[report.id] = report

            logger.info(f"✅ Report scheduled: {report.name} - Next run: {next_run}")

        except Exception as e:
            logger.error(f"❌ Error scheduling report: {e}")

    def _calculate_next_run(self, report: PersonalizedReport) -> datetime:
        """Calcular próxima execução"""
        now = datetime.now()
        schedule_time = datetime.strptime(report.schedule_time, "%H:%M").time()

        if report.frequency == ReportFrequency.DAILY:
            next_run = datetime.combine(now.date(), schedule_time)
            if next_run <= now:
                next_run += timedelta(days=1)

        elif report.frequency == ReportFrequency.WEEKLY:
            days_ahead = report.schedule_day - now.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            next_run = datetime.combine(
                now.date() + timedelta(days=days_ahead),
                schedule_time
            )

        elif report.frequency == ReportFrequency.MONTHLY:
            if report.schedule_day <= now.day:
                # Próximo mês
                if now.month == 12:
                    next_month = now.replace(year=now.year + 1, month=1)
                else:
                    next_month = now.replace(month=now.month + 1)
            else:
                next_month = now

            next_run = datetime.combine(
                next_month.replace(day=report.schedule_day),
                schedule_time
            )

        else:
            next_run = now + timedelta(hours=1)  # Default

        return next_run

    async def check_and_run_scheduled_reports(self):
        """Verificar e executar relatórios agendados"""
        now = datetime.now()

        for report_id, report in self.scheduled_reports.items():
            if report.next_run_at and now >= report.next_run_at:
                try:
                    await self.report_manager.generate_and_send_report(report.id)

                    # Reagendar
                    await self.schedule_report(report)

                except Exception as e:
                    logger.error(f"❌ Error running scheduled report {report.name}: {e}")


class PersonalizedReportingSystem:
    """📊 Sistema principal de relatórios personalizados"""

    def __init__(self, database_url: str, warehouse_manager, email_config: Dict = None):
        self.database_url = database_url
        self.warehouse_manager = warehouse_manager
        self.email_config = email_config or {}
        self.pool = None

        self.template_manager = ReportTemplateManager()
        self.report_generator = ReportGenerator(warehouse_manager)
        self.scheduler = ReportScheduler(self)

        self.reports_storage_path = Path("data/reports")
        self.reports_storage_path.mkdir(parents=True, exist_ok=True)

    async def initialize(self):
        """Inicializar sistema de relatórios"""
        try:
            self.pool = await asyncpg.create_pool(self.database_url)
            await self._create_tables()

            logger.info("✅ Personalized Reporting System initialized")

        except Exception as e:
            logger.error(f"❌ Reporting system initialization failed: {e}")

    async def _create_tables(self):
        """Criar tabelas necessárias"""
        async with self.pool.acquire() as conn:
            # Tabela de relatórios personalizados
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS personalized_reports (
                    id UUID PRIMARY KEY,
                    user_id UUID NOT NULL,
                    template_id VARCHAR NOT NULL,
                    name VARCHAR NOT NULL,
                    description TEXT,
                    parameters JSONB DEFAULT '{}',
                    frequency VARCHAR DEFAULT 'manual',
                    format VARCHAR DEFAULT 'html',
                    recipients JSONB DEFAULT '[]',
                    schedule_time VARCHAR DEFAULT '08:00',
                    schedule_day INTEGER,
                    last_generated_at TIMESTAMP,
                    next_run_at TIMESTAMP,
                    status VARCHAR DEFAULT 'draft',
                    file_path VARCHAR,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    metadata JSONB DEFAULT '{}'
                )
            """)

            # Índices
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_reports_user_id ON personalized_reports(user_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_reports_next_run ON personalized_reports(next_run_at)")

    async def create_personalized_report(self, user_id: str, template_id: str, name: str,
                                       parameters: Dict = None, frequency: ReportFrequency = ReportFrequency.MANUAL,
                                       format: ReportFormat = ReportFormat.HTML, recipients: List[str] = None) -> Optional[PersonalizedReport]:
        """Criar relatório personalizado"""
        try:
            # Verificar se template existe
            template = self.template_manager.get_template(template_id)
            if not template:
                logger.error(f"Template not found: {template_id}")
                return None

            report = PersonalizedReport(
                user_id=user_id,
                template_id=template_id,
                name=name,
                parameters=parameters or {},
                frequency=frequency,
                format=format,
                recipients=recipients or []
            )

            # Salvar no banco
            success = await self._save_report(report)

            if success:
                # Agendar se necessário
                if frequency != ReportFrequency.MANUAL:
                    await self.scheduler.schedule_report(report)

                logger.info(f"✅ Personalized report created: {name}")
                return report

        except Exception as e:
            logger.error(f"❌ Error creating personalized report: {e}")

        return None

    async def generate_and_send_report(self, report_id: str) -> bool:
        """Gerar e enviar relatório"""
        try:
            # Obter relatório
            report = await self._get_report(report_id)
            if not report:
                logger.error(f"Report not found: {report_id}")
                return False

            # Obter template
            template = self.template_manager.get_template(report.template_id)
            if not template:
                logger.error(f"Template not found: {report.template_id}")
                return False

            # Atualizar status
            report.status = ReportStatus.GENERATING
            await self._save_report(report)

            # Gerar dados do relatório
            report_data = await self.report_generator.generate_report_data(template, report.parameters)

            if "error" in report_data.metadata:
                report.status = ReportStatus.FAILED
                report.error_message = report_data.metadata["error"]
                await self._save_report(report)
                return False

            # Gerar arquivo do relatório
            file_content = None
            file_extension = ""

            if report.format == ReportFormat.HTML:
                file_content = await self.report_generator.generate_html_report(report_data, template)
                file_extension = "html"
            elif report.format == ReportFormat.PDF:
                html_content = await self.report_generator.generate_html_report(report_data, template)
                file_content = await self.report_generator.generate_pdf_report(html_content)
                file_extension = "pdf"
            elif report.format == ReportFormat.CSV:
                file_content = await self.report_generator.generate_csv_report(report_data)
                file_extension = "csv"
            elif report.format == ReportFormat.EXCEL:
                file_content = await self.report_generator.generate_excel_report(report_data, template)
                file_extension = "xlsx"

            # Salvar arquivo
            filename = f"report_{report.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_extension}"
            file_path = self.reports_storage_path / filename

            if isinstance(file_content, str):
                async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                    await f.write(file_content)
            else:
                async with aiofiles.open(file_path, 'wb') as f:
                    await f.write(file_content)

            # Atualizar relatório
            report.file_path = str(file_path)
            report.last_generated_at = datetime.now()
            report.status = ReportStatus.COMPLETED

            # Enviar por email se tiver recipients
            if report.recipients:
                email_sent = await self._send_report_email(report, file_path, template)
                if email_sent:
                    report.status = ReportStatus.SENT

            await self._save_report(report)

            logger.info(f"✅ Report generated and sent: {report.name}")
            return True

        except Exception as e:
            logger.error(f"❌ Error generating report: {e}")

            # Atualizar status de erro
            if 'report' in locals():
                report.status = ReportStatus.FAILED
                report.error_message = str(e)
                await self._save_report(report)

            return False

    async def _send_report_email(self, report: PersonalizedReport, file_path: Path, template: ReportTemplate) -> bool:
        """Enviar relatório por email"""
        try:
            if not self.email_config.get('smtp_server'):
                logger.warning("Email not configured")
                return False

            # Preparar email
            msg = MimeMultipart()
            msg['From'] = self.email_config['from_email']
            msg['To'] = ', '.join(report.recipients)
            msg['Subject'] = f"Relatório: {report.name}"

            # Corpo do email
            body = f"""
            Olá,

            Segue em anexo o relatório "{report.name}" gerado automaticamente.

            Descrição: {template.description}
            Gerado em: {report.last_generated_at.strftime('%d/%m/%Y %H:%M:%S')}

            Atenciosamente,
            Sistema de Relatórios Trading Bot
            """

            msg.attach(MimeText(body, 'plain'))

            # Anexar arquivo
            with open(file_path, 'rb') as f:
                attachment = MimeApplication(f.read())
                attachment.add_header(
                    'Content-Disposition',
                    'attachment',
                    filename=file_path.name
                )
                msg.attach(attachment)

            # Enviar email
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            server.send_message(msg)
            server.quit()

            logger.info(f"✅ Report email sent to: {', '.join(report.recipients)}")
            return True

        except Exception as e:
            logger.error(f"❌ Error sending report email: {e}")
            return False

    async def list_user_reports(self, user_id: str) -> List[PersonalizedReport]:
        """Listar relatórios do usuário"""
        try:
            reports = []

            if self.pool:
                async with self.pool.acquire() as conn:
                    rows = await conn.fetch(
                        "SELECT * FROM personalized_reports WHERE user_id = $1 ORDER BY created_at DESC",
                        user_id
                    )

                    for row in rows:
                        reports.append(self._row_to_report(row))

            return reports

        except Exception as e:
            logger.error(f"❌ Error listing user reports: {e}")
            return []

    async def get_report_file(self, report_id: str, user_id: str) -> Optional[Path]:
        """Obter arquivo do relatório"""
        try:
            report = await self._get_report(report_id)

            if report and report.user_id == user_id and report.file_path:
                file_path = Path(report.file_path)
                if file_path.exists():
                    return file_path

        except Exception as e:
            logger.error(f"❌ Error getting report file: {e}")

        return None

    async def _save_report(self, report: PersonalizedReport) -> bool:
        """Salvar relatório no banco"""
        try:
            if self.pool:
                async with self.pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO personalized_reports (
                            id, user_id, template_id, name, description, parameters,
                            frequency, format, recipients, schedule_time, schedule_day,
                            last_generated_at, next_run_at, status, file_path,
                            error_message, created_at, updated_at, metadata
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
                        ON CONFLICT (id) DO UPDATE SET
                            name = EXCLUDED.name,
                            description = EXCLUDED.description,
                            parameters = EXCLUDED.parameters,
                            frequency = EXCLUDED.frequency,
                            format = EXCLUDED.format,
                            recipients = EXCLUDED.recipients,
                            schedule_time = EXCLUDED.schedule_time,
                            schedule_day = EXCLUDED.schedule_day,
                            last_generated_at = EXCLUDED.last_generated_at,
                            next_run_at = EXCLUDED.next_run_at,
                            status = EXCLUDED.status,
                            file_path = EXCLUDED.file_path,
                            error_message = EXCLUDED.error_message,
                            updated_at = EXCLUDED.updated_at,
                            metadata = EXCLUDED.metadata
                    """,
                        report.id, report.user_id, report.template_id, report.name,
                        report.description, json.dumps(report.parameters),
                        report.frequency.value, report.format.value,
                        json.dumps(report.recipients), report.schedule_time,
                        report.schedule_day, report.last_generated_at, report.next_run_at,
                        report.status.value, report.file_path, report.error_message,
                        report.created_at, report.updated_at, json.dumps(report.metadata)
                    )
                return True

        except Exception as e:
            logger.error(f"❌ Error saving report: {e}")

        return False

    async def _get_report(self, report_id: str) -> Optional[PersonalizedReport]:
        """Obter relatório por ID"""
        try:
            if self.pool:
                async with self.pool.acquire() as conn:
                    row = await conn.fetchrow(
                        "SELECT * FROM personalized_reports WHERE id = $1",
                        report_id
                    )
                    if row:
                        return self._row_to_report(row)

        except Exception as e:
            logger.error(f"❌ Error getting report: {e}")

        return None

    def _row_to_report(self, row) -> PersonalizedReport:
        """Converter row para PersonalizedReport"""
        return PersonalizedReport(
            id=str(row['id']),
            user_id=str(row['user_id']),
            template_id=row['template_id'],
            name=row['name'],
            description=row['description'] or '',
            parameters=json.loads(row['parameters']) if row['parameters'] else {},
            frequency=ReportFrequency(row['frequency']),
            format=ReportFormat(row['format']),
            recipients=json.loads(row['recipients']) if row['recipients'] else [],
            schedule_time=row['schedule_time'],
            schedule_day=row['schedule_day'],
            last_generated_at=row['last_generated_at'],
            next_run_at=row['next_run_at'],
            status=ReportStatus(row['status']),
            file_path=row['file_path'] or '',
            error_message=row['error_message'] or '',
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            metadata=json.loads(row['metadata']) if row['metadata'] else {}
        )

    async def run_scheduler(self):
        """Executar agendador de relatórios"""
        await self.scheduler.check_and_run_scheduled_reports()


# 🧪 Função de teste
async def test_personalized_reporting_system():
    """Testar sistema de relatórios personalizados"""
    from real_analytics_system import BigQueryManager, DataWarehouseConfig, DataWarehouseProvider

    # Configuração mock
    config = DataWarehouseConfig(
        provider=DataWarehouseProvider.BIGQUERY,
        project_id="trading-bot-test",
        dataset_id="test_data"
    )

    warehouse_manager = BigQueryManager(config.project_id, config.dataset_id)
    database_url = "postgresql://trading_user:password@localhost:5432/trading_db"

    email_config = {
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'username': 'your_email@gmail.com',
        'password': 'your_password',
        'from_email': 'your_email@gmail.com'
    }

    # Inicializar sistema
    reporting_system = PersonalizedReportingSystem(database_url, warehouse_manager, email_config)
    await reporting_system.initialize()

    print("\n" + "="*80)
    print("📋 PERSONALIZED REPORTING SYSTEM TEST")
    print("="*80)

    test_user_id = str(uuid.uuid4())

    # 1. Listar templates disponíveis
    print("\n📋 AVAILABLE REPORT TEMPLATES:")
    templates = reporting_system.template_manager.list_templates()

    for template in templates:
        print(f"   📊 {template.name} ({template.id})")
        print(f"      Category: {template.category}")
        print(f"      Description: {template.description}")
        print(f"      Tags: {', '.join(template.tags)}")
        print()

    # 2. Criar relatório personalizado
    print(f"\n📊 CREATING PERSONALIZED REPORT for user: {test_user_id}")
    report = await reporting_system.create_personalized_report(
        user_id=test_user_id,
        template_id="trading_performance",
        name="Meu Relatório de Trading",
        parameters={
            "start_date": "30_days_ago",
            "end_date": "today",
            "user_id": test_user_id
        },
        frequency=ReportFrequency.WEEKLY,
        format=ReportFormat.HTML,
        recipients=["user@example.com"]
    )

    if report:
        print(f"✅ Personalized report created: {report.name}")
        print(f"   ID: {report.id}")
        print(f"   Frequency: {report.frequency.value}")
        print(f"   Format: {report.format.value}")
        print(f"   Status: {report.status.value}")

    # 3. Gerar relatório manualmente
    if report:
        print(f"\n📄 GENERATING REPORT...")
        success = await reporting_system.generate_and_send_report(report.id)

        if success:
            print(f"✅ Report generated successfully")
        else:
            print(f"❌ Report generation failed")

    # 4. Listar relatórios do usuário
    print(f"\n📋 LISTING USER REPORTS...")
    user_reports = await reporting_system.list_user_reports(test_user_id)

    for user_report in user_reports:
        print(f"   📊 {user_report.name}")
        print(f"      Status: {user_report.status.value}")
        print(f"      Last Generated: {user_report.last_generated_at}")
        print(f"      File Path: {user_report.file_path}")

    # 5. Testar agendador
    print(f"\n⏰ TESTING SCHEDULER...")
    await reporting_system.run_scheduler()
    print(f"✅ Scheduler check completed")

    print("\n" + "="*80)
    print("✅ PERSONALIZED REPORTING SYSTEM TEST COMPLETED")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(test_personalized_reporting_system())