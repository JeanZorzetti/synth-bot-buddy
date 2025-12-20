"""
CRASH300N - PRNG Cracking Attempt (Mersenne Twister)
Tese: O gerador de numeros nao e criptograficamente seguro (CSPRNG).
Objetivo: Recuperar o estado interno do RNG para prever o proximo numero exato.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import random
import time

# Tenta importar randcrack (voce precisaria instalar: pip install randcrack)
# Se nao tiver, simula a logica para demonstrar a inviabilidade ou sucesso
try:
    from randcrack import RandCrack
    HAS_RANDCRACK = True
except ImportError:
    HAS_RANDCRACK = False

def attempt_rng_crack():
    print("="*70)
    print("CRASH300N - PRNG CRACKING (CRIPTOGRAFIA VS ESTATISTICA)")
    print("Objetivo: Clonar o estado do Mersenne Twister (se usado)")
    print("="*70)

    if not HAS_RANDCRACK:
        print("WARNING: Biblioteca 'randcrack' nao encontrada.")
        print("   Para teste real, instale: pip install randcrack")
        print("   Continuando com analise de entropia de bits...")

    # 1. Carregar Dados
    data_dir = Path(__file__).parent / "data"
    files = list(data_dir.glob("*CRASH300*.csv"))
    if not files: return
    df = pd.read_csv(files[0])

    # Sort temporal
    if 'epoch' in df.columns:
        df = df.sort_values('epoch').reset_index(drop=True)
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"\n[LOAD] Total candles: {len(df):,}")

    # 2. Engenharia Reversa do Float
    # O preco sobe ticks pequenos. Vamos tentar isolar o componente aleatorio.
    # Hipotese: O tamanho do tick (Tick Size) e derivado de um RNG.
    # Log Returns removem o nivel de preco absoluto.
    df['log_ret'] = np.log(df['close'] / df['open'])

    # Filtrar apenas os ticks normais (sem crash), pois o crash e um evento de reset
    # e pode vir de uma logica separada. Queremos ver se o "ruido" da subida tem padrao.
    normal_ticks = df[df['log_ret'] > 0]['log_ret'].values

    print(f"\n[DADOS] Analisando {len(normal_ticks):,} ticks de subida")
    print("Tese: A magnitude da subida vem de um RNG previsivel?")

    # 3. Normalizacao para Inteiros (32-bit)
    # PRNGs geram inteiros de 32 bits. Precisamos mapear o float de volta para int.
    # Isso e dificil sem saber a formula de escala, mas vamos tentar quantizar.
    # Vamos assumir que o float = int / 2^32

    # Multiplicar por 10^9 para manter precisao e converter para int
    scaled_ticks = (normal_ticks * 1e9).astype(np.int64)

    # 4. Teste de Aleatoriedade de Bits (NIST SP 800-22 simplificado)
    # Se for um PRNG fraco, os bits menos significativos podem ter padrao.
    print("\n[ANALISE DE BITS]")

    # Converter para string binaria gigante
    bits = ""
    # Amostra de 1000 ticks para ser rapido
    for val in scaled_ticks[:1000]:
        bits += format(abs(val), '032b')

    zeros = bits.count('0')
    ones = bits.count('1')
    total = len(bits)

    ratio = ones / total
    print(f"  Total Bits: {total:,}")
    print(f"  Zeros: {zeros:,} ({zeros/total*100:.2f}%)")
    print(f"  Ones:  {ones:,} ({ones/total*100:.2f}%)")
    print(f"  Ratio 1s/0s: {ratio:.5f} (Ideal: 0.50000)")

    # Teste de Chi-Quadrado nos bits
    expected = total / 2
    chi_square = ((zeros - expected)**2 + (ones - expected)**2) / expected
    print(f"  Chi-Square Score: {chi_square:.4f} (Critical: 3.84)")

    if chi_square > 3.84:
        print("  ALERTA: Distribuicao de bits NAO e aleatoria (p < 0.05)")
        print("     Possivel vies no gerador ou na quantizacao!")
    else:
        print("  OK: Distribuicao de bits parece aleatoria.")

    # 5. Tentativa de Predicao (Simulada sem randcrack perfeito)
    # A ideia: Se observarmos X valores, o proximo repete?
    print("\n[BUSCA DE REPETICAO DE SEQUENCIA]")
    # Procurar subsequencias repetidas (Padrao de ciclo curto)

    def find_repeating_sequence(arr, min_len=3):
        n = len(arr)
        # Otimizacao: olhar apenas ultimos 10000
        arr_view = arr[-10000:] if len(arr) > 10000 else arr
        n_view = len(arr_view)

        for length in range(min_len, 20): # Sequencias de tamanho 3 a 20
            # Pegar ultima sequencia
            pattern = arr_view[-length:]
            # Procurar ocorrencias anteriores
            # Isso e lento O(N*M), mas ok para teste
            found = 0
            for i in range(n_view - length - 1):
                if np.array_equal(arr_view[i:i+length], pattern):
                    found += 1

            if found > 0:
                return length, found, pattern
        return 0, 0, None

    # Quantizar mais grosseiramente para achar padroes macro
    # Ex: Tick pequeno (0), Tick medio (1), Tick grande (2)
    bins = np.percentile(normal_ticks, [33, 66])
    digitized = np.digitize(normal_ticks, bins)

    print(f"  Analisando sequencias em {len(digitized):,} ticks...")
    print(f"  Bins: {bins}")

    seq_len, count, pattern = find_repeating_sequence(digitized)

    if count > 0:
        print(f"\n  PADRAO ENCONTRADO! Sequencia de tamanho {seq_len} repetiu {count} vezes.")
        print(f"  Padrao: {pattern[:10]}...")  # Mostrar apenas primeiros 10
        print("  Isso sugere um RNG de ciclo curto ou logica deterministica simples.")
    else:
        print("\n  NENHUMA sequencia repetida encontrada.")
        print("  O gerador tem ciclo longo (provavelmente > 2^19937) ou e CSPRNG.")

    # 6. Teste de Autocorrelacao nos Bits
    print("\n[AUTOCORRELACAO DE BITS]")

    # Converter bits para array
    bit_array = np.array([int(b) for b in bits[:10000]])  # Primeiros 10k bits

    # Calcular autocorrelacao em lags 1, 10, 100
    lags = [1, 10, 100, 1000]
    autocorrs = []

    for lag in lags:
        if lag < len(bit_array):
            x = bit_array[:-lag]
            y = bit_array[lag:]
            corr = np.corrcoef(x, y)[0, 1]
            autocorrs.append(corr)
            print(f"  Lag {lag:4d}: {corr:+.6f}")

    max_autocorr = max(abs(c) for c in autocorrs) if autocorrs else 0

    if max_autocorr > 0.05:
        print(f"\n  ALERTA: Autocorrelacao alta ({max_autocorr:.4f})")
        print("  Bits tem MEMORIA (nao e aleatorio verdadeiro)")
    else:
        print(f"\n  OK: Autocorrelacao baixa ({max_autocorr:.4f})")

    # 7. VERDICTO FINAL
    print(f"\n{'='*70}")
    print("VEREDICTO DE SEGURANCA")
    print(f"{'='*70}")

    is_weak = (chi_square > 3.84) or (count > 0) or (max_autocorr > 0.05)

    if is_weak:
        print("\nVULNERABILIDADE POTENCIAL DETECTADA")
        print("   Os numeros mostram vies estatistico ou repeticao.")
        print("   Isso nao e um CSPRNG perfeito.")
        print("\n   EVIDENCIAS:")
        if chi_square > 3.84:
            print(f"   - Chi-square: {chi_square:.4f} > 3.84 (vies nos bits)")
        if count > 0:
            print(f"   - Sequencias repetidas: {count} ocorrencias")
        if max_autocorr > 0.05:
            print(f"   - Autocorrelacao: {max_autocorr:.4f} > 0.05 (memoria)")
        print("\n   ACAO: Tentar RandCrack com dados brutos de WebSocket (Tick Data).")
    else:
        print("\nSISTEMA SEGURO (CSPRNG Provavel)")
        print("   - Bits distribuidos uniformemente (Chi-Square OK)")
        print("   - Nenhuma repeticao de sequencia encontrada")
        print("   - Autocorrelacao baixa (sem memoria)")
        print("\n   Conclusao: Deriv usa criptografia forte (ex: ChaCha20/AES-CTR).")
        print("   Impossivel prever matematicamente sem a chave (Seed).")

    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    attempt_rng_crack()
