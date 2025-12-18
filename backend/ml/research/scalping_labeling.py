"""
Labeling para Scalping ML Trading

Este script gera labels LONG/SHORT/NO_TRADE para treinar modelo supervisionado.

Configuração otimizada (baseada em análise M5):
- TP: 0.2% (20 pips)
- SL: 0.1% (10 pips)
- R:R: 1:2
- Max candles: 20 (100 minutos timeout)

Autor: Claude Sonnet 4.5
Data: 18/12/2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class ScalpingLabeler:
    """
    Gerador de labels para scalping
    """

    def __init__(self, df: pd.DataFrame, tp_pct: float = 0.2, sl_pct: float = 0.1, max_candles: int = 20):
        """
        Args:
            df: DataFrame com OHLC + features
            tp_pct: Take profit em % (default: 0.2%)
            sl_pct: Stop loss em % (default: 0.1%)
            max_candles: Máximo de candles para esperar (default: 20 = 100 min em M5)
        """
        self.df = df.copy()
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.max_candles = max_candles

        print(f"[OK] Labeler inicializado:")
        print(f"   - TP: {tp_pct}% / SL: {sl_pct}% (R:R {tp_pct/sl_pct:.1f})")
        print(f"   - Max timeout: {max_candles} candles ({max_candles * 5} min)")
        print(f"   - Total candles: {len(df)}")

    def generate_labels(self) -> pd.DataFrame:
        """
        Gera labels para cada candle

        Labels:
        - 1: LONG setup válido (TP atingido antes de SL)
        - -1: SHORT setup válido (TP atingido antes de SL)
        - 0: NO_TRADE (SL atingido primeiro, timeout, ou fim de dados)

        Returns:
            DataFrame com coluna 'label' adicionada
        """
        print("\n[LABEL] Gerando labels...")

        labels = []
        label_metadata = []  # Para análise posterior

        for i in range(len(self.df) - self.max_candles):
            if i % 5000 == 0:
                print(f"   Processando candle {i}/{len(self.df) - self.max_candles}...")

            entry_price = self.df.iloc[i]['close']

            # Calcular targets para LONG
            tp_long = entry_price * (1 + self.tp_pct / 100)
            sl_long = entry_price * (1 - self.sl_pct / 100)

            # Calcular targets para SHORT
            tp_short = entry_price * (1 - self.tp_pct / 100)
            sl_short = entry_price * (1 + self.sl_pct / 100)

            # Verificar próximos candles para LONG
            long_result = self._check_trade_outcome(
                i, entry_price, tp_long, sl_long, direction='LONG'
            )

            # Verificar próximos candles para SHORT
            short_result = self._check_trade_outcome(
                i, entry_price, tp_short, sl_short, direction='SHORT'
            )

            # Decidir label baseado nos resultados
            label, metadata = self._decide_label(long_result, short_result)
            labels.append(label)
            label_metadata.append(metadata)

        # Preencher últimos candles com 0 (não há dados futuros suficientes)
        labels.extend([0] * self.max_candles)
        label_metadata.extend([{'reason': 'no_future_data'}] * self.max_candles)

        self.df['label'] = labels
        self.df['label_metadata'] = label_metadata

        print(f"[OK] Labels gerados!")
        self._print_label_distribution()

        return self.df

    def _check_trade_outcome(self, start_idx: int, entry_price: float,
                            tp_price: float, sl_price: float,
                            direction: str) -> Dict:
        """
        Verifica o resultado de um trade hipotético

        Returns:
            Dict com resultado: {
                'hit_tp': bool,
                'hit_sl': bool,
                'candles_to_exit': int,
                'exit_price': float,
                'pnl_pct': float
            }
        """
        for j in range(start_idx + 1, min(start_idx + self.max_candles + 1, len(self.df))):
            high = self.df.iloc[j]['high']
            low = self.df.iloc[j]['low']
            candles_elapsed = j - start_idx

            if direction == 'LONG':
                # Verificar se TP foi atingido
                if high >= tp_price:
                    return {
                        'hit_tp': True,
                        'hit_sl': False,
                        'candles_to_exit': candles_elapsed,
                        'exit_price': tp_price,
                        'pnl_pct': self.tp_pct
                    }
                # Verificar se SL foi atingido
                if low <= sl_price:
                    return {
                        'hit_tp': False,
                        'hit_sl': True,
                        'candles_to_exit': candles_elapsed,
                        'exit_price': sl_price,
                        'pnl_pct': -self.sl_pct
                    }
            else:  # SHORT
                # Verificar se TP foi atingido
                if low <= tp_price:
                    return {
                        'hit_tp': True,
                        'hit_sl': False,
                        'candles_to_exit': candles_elapsed,
                        'exit_price': tp_price,
                        'pnl_pct': self.tp_pct
                    }
                # Verificar se SL foi atingido
                if high >= sl_price:
                    return {
                        'hit_tp': False,
                        'hit_sl': True,
                        'candles_to_exit': candles_elapsed,
                        'exit_price': sl_price,
                        'pnl_pct': -self.sl_pct
                    }

        # Timeout (nem TP nem SL atingido)
        return {
            'hit_tp': False,
            'hit_sl': False,
            'candles_to_exit': self.max_candles,
            'exit_price': entry_price,
            'pnl_pct': 0
        }

    def _decide_label(self, long_result: Dict, short_result: Dict) -> Tuple[int, Dict]:
        """
        Decide o label baseado nos resultados LONG e SHORT

        Lógica:
        - Se LONG hit TP e SHORT não hit TP → label = 1 (LONG)
        - Se SHORT hit TP e LONG não hit TP → label = -1 (SHORT)
        - Se ambos hit TP → escolher o mais rápido
        - Se nenhum hit TP → label = 0 (NO_TRADE)

        Returns:
            (label, metadata)
        """
        long_tp = long_result['hit_tp']
        short_tp = short_result['hit_tp']

        # Caso 1: Apenas LONG hit TP
        if long_tp and not short_tp:
            return 1, {
                'direction': 'LONG',
                'reason': 'long_tp_only',
                'candles': long_result['candles_to_exit']
            }

        # Caso 2: Apenas SHORT hit TP
        if short_tp and not long_tp:
            return -1, {
                'direction': 'SHORT',
                'reason': 'short_tp_only',
                'candles': short_result['candles_to_exit']
            }

        # Caso 3: Ambos hit TP → escolher o mais rápido
        if long_tp and short_tp:
            if long_result['candles_to_exit'] <= short_result['candles_to_exit']:
                return 1, {
                    'direction': 'LONG',
                    'reason': 'long_faster',
                    'candles': long_result['candles_to_exit']
                }
            else:
                return -1, {
                    'direction': 'SHORT',
                    'reason': 'short_faster',
                    'candles': short_result['candles_to_exit']
                }

        # Caso 4: Nenhum hit TP → NO_TRADE
        if long_result['hit_sl']:
            reason = 'long_sl_hit'
        elif short_result['hit_sl']:
            reason = 'short_sl_hit'
        else:
            reason = 'timeout'

        return 0, {'direction': 'NO_TRADE', 'reason': reason, 'candles': 0}

    def _print_label_distribution(self):
        """Imprime distribuição de labels"""
        label_counts = self.df['label'].value_counts().sort_index()

        total = len(self.df)
        long_count = label_counts.get(1, 0)
        short_count = label_counts.get(-1, 0)
        no_trade_count = label_counts.get(0, 0)

        print("\n[DISTRIBUICAO DE LABELS]")
        print(f"   LONG (1):     {long_count:6d} ({long_count/total*100:5.1f}%)")
        print(f"   SHORT (-1):   {short_count:6d} ({short_count/total*100:5.1f}%)")
        print(f"   NO_TRADE (0): {no_trade_count:6d} ({no_trade_count/total*100:5.1f}%)")
        print(f"   Total:        {total:6d}")

        # Verificar se distribuição está OK
        tradeable_pct = (long_count + short_count) / total * 100
        print(f"\n[ANALISE]")
        print(f"   Setup viáveis: {tradeable_pct:.1f}%")

        if tradeable_pct < 30:
            print(f"   [AVISO] Apenas {tradeable_pct:.1f}% de setups viaveis!")
            print(f"   Considere aumentar max_candles ou ajustar TP/SL")
        elif tradeable_pct > 70:
            print(f"   [AVISO] {tradeable_pct:.1f}% de setups viaveis (muito alto!)")
            print(f"   Modelo pode overfittar. Considere criterios mais rigorosos.")
        else:
            print(f"   [OK] Distribuicao OK ({tradeable_pct:.1f}% viaveis)")

    def save_labeled_data(self, output_path: str):
        """Salva dados com labels em CSV"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Remover metadata column (não é feature, apenas debug)
        df_to_save = self.df.drop(columns=['label_metadata'])

        df_to_save.to_csv(output_path, index=False)
        print(f"\n[SAVE] Dados com labels salvos em {output_path}")

    def plot_label_analysis(self, output_path: str = None):
        """Gera visualizações da distribuição de labels"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Distribuição de labels (pie chart)
        label_counts = self.df['label'].value_counts()
        axes[0, 0].pie(label_counts.values, labels=['NO_TRADE', 'LONG', 'SHORT'],
                       autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Distribuição de Labels')

        # 2. Labels ao longo do tempo
        axes[0, 1].plot(self.df.index[:1000], self.df['label'][:1000], alpha=0.5)
        axes[0, 1].set_title('Labels ao Longo do Tempo (primeiros 1000)')
        axes[0, 1].set_xlabel('Index')
        axes[0, 1].set_ylabel('Label')

        # 3. Histograma de features por label (exemplo: RSI)
        if 'rsi_14' in self.df.columns:
            for label in [-1, 0, 1]:
                subset = self.df[self.df['label'] == label]['rsi_14']
                axes[1, 0].hist(subset, bins=30, alpha=0.5,
                               label=f'Label {label}')
            axes[1, 0].set_title('Distribuição de RSI por Label')
            axes[1, 0].set_xlabel('RSI')
            axes[1, 0].legend()

        # 4. Correlação de labels com features principais
        if 'rsi_14' in self.df.columns and 'bb_position' in self.df.columns:
            feature_cols = ['rsi_14', 'bb_position', 'macd', 'stoch_k']
            existing_cols = [col for col in feature_cols if col in self.df.columns]
            correlation = self.df[existing_cols + ['label']].corr()['label'].drop('label')
            axes[1, 1].barh(correlation.index, correlation.values)
            axes[1, 1].set_title('Correlação Features vs Label')
            axes[1, 1].set_xlabel('Correlação')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path)
            print(f"[SAVE] Visualização salva em {output_path}")
        else:
            plt.show()


def process_labeling(symbol: str = '1HZ100V', timeframe: str = '5min',
                    tp_pct: float = 0.2, sl_pct: float = 0.1):
    """
    Processa labeling para um símbolo

    Args:
        symbol: '1HZ75V' ou '1HZ100V'
        timeframe: '5min'
        tp_pct: Take profit %
        sl_pct: Stop loss %
    """
    print(f"\n{'='*70}")
    print(f"PROCESSANDO LABELING: {symbol} ({timeframe})")
    print(f"{'='*70}")

    # Carregar dados com features
    data_dir = Path(__file__).parent / "data"
    features_path = data_dir / f"{symbol}_{timeframe}_180days_features.csv"

    if not features_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {features_path}")

    df = pd.read_csv(features_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    print(f"[OK] Dados carregados: {len(df)} candles, {len(df.columns)} features")

    # Criar labels
    labeler = ScalpingLabeler(df, tp_pct=tp_pct, sl_pct=sl_pct, max_candles=20)
    df_labeled = labeler.generate_labels()

    # Salvar
    output_path = data_dir / f"{symbol}_{timeframe}_180days_labeled.csv"
    labeler.save_labeled_data(output_path)

    # Gerar visualização
    viz_path = data_dir.parent / "reports" / f"{symbol}_{timeframe}_label_analysis.png"
    viz_path.parent.mkdir(exist_ok=True)
    labeler.plot_label_analysis(output_path=str(viz_path))

    return df_labeled


if __name__ == "__main__":
    """
    Executar labeling para V100 M5
    """

    # Processar V100 M5 (configuração ótima encontrada)
    df_labeled = process_labeling(
        symbol='1HZ100V',
        timeframe='5min',
        tp_pct=0.2,
        sl_pct=0.1
    )

    print("\n" + "="*70)
    print("LABELING CONCLUÍDO COM SUCESSO!")
    print("="*70)
    print(f"Total de amostras: {len(df_labeled)}")
    print(f"Features + Label: {len(df_labeled.columns)} colunas")
    print("\nPróximo passo: Treinar modelo XGBoost com estes labels")
