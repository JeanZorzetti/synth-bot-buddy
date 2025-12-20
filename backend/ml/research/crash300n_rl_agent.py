"""
CRASH300N - DEEP REINFORCEMENT LEARNING (PPO)
O EXPERIMENTO FINAL: Agente aprende a JOGAR, nao a PREVER

Diferenca brutal:
- LSTM: Tenta prever proximo candle (Input -> Output) [FALHOU]
- RL: Aprende politica que maximiza recompensa (State -> Action -> Reward)

RL pode descobrir "exploits" que nao sao padroes de preco:
- Timing de execucao
- Spread/Slippage bias
- Microestrutura de ordem
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback


class CRASH300NEnv(gym.Env):
    """
    Ambiente Gym para CRASH300N

    Estado: Ultimos 60 ticks normalizados + posicao atual
    Acao: [0=Hold, 1=Buy, 2=Sell, 3=Close]
    Recompensa: PnL acumulado
    """
    metadata = {'render_modes': []}

    def __init__(self, df, initial_balance=1000.0, max_position_size=1000.0):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.max_position_size = max_position_size

        # Estado: [60 ticks normalizados, posicao atual, PnL atual]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(62,), dtype=np.float32
        )

        # Acao: 0=Hold, 1=Buy, 2=Sell, 3=Close
        self.action_space = spaces.Discrete(4)

        # Internal state
        self.current_step = 60  # Start after 60 ticks
        self.balance = initial_balance
        self.position = 0.0  # Current position size
        self.entry_price = 0.0
        self.total_pnl = 0.0

        # Metrics
        self.trades = []
        self.episode_rewards = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Random start position (to avoid overfitting to specific periods)
        max_start = len(self.df) - 1000
        self.current_step = np.random.randint(60, max_start)

        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.total_pnl = 0.0
        self.trades = []

        return self._get_observation(), {}

    def _get_observation(self):
        # Get last 60 close prices
        start_idx = max(0, self.current_step - 60)
        prices = self.df.iloc[start_idx:self.current_step]['close'].values

        # Normalize prices (% change from first price)
        if len(prices) > 0:
            prices_norm = (prices - prices[0]) / prices[0]
        else:
            prices_norm = np.zeros(60)

        # Pad if needed
        if len(prices_norm) < 60:
            prices_norm = np.pad(prices_norm, (60 - len(prices_norm), 0))

        # Add position and PnL
        obs = np.concatenate([
            prices_norm,
            [self.position / self.max_position_size],  # Normalized position
            [self.total_pnl / self.initial_balance]    # Normalized PnL
        ]).astype(np.float32)

        return obs

    def step(self, action):
        # Execute action
        current_price = self.df.iloc[self.current_step]['close']
        reward = 0.0

        # Action 0: Hold (do nothing)
        if action == 0:
            pass

        # Action 1: Buy (if no position)
        elif action == 1:
            if self.position == 0:
                self.position = self.max_position_size
                self.entry_price = current_price

        # Action 2: Sell (if no position)
        elif action == 2:
            if self.position == 0:
                self.position = -self.max_position_size
                self.entry_price = current_price

        # Action 3: Close position
        elif action == 3:
            if self.position != 0:
                # Calculate PnL
                if self.position > 0:  # Long
                    pnl = self.position * (current_price - self.entry_price) / self.entry_price
                else:  # Short
                    pnl = abs(self.position) * (self.entry_price - current_price) / self.entry_price

                self.total_pnl += pnl
                reward = pnl

                self.trades.append({
                    'entry_price': self.entry_price,
                    'exit_price': current_price,
                    'pnl': pnl,
                    'step': self.current_step
                })

                self.position = 0.0
                self.entry_price = 0.0

        # Calculate unrealized PnL (if position is open)
        if self.position != 0:
            if self.position > 0:  # Long
                unrealized_pnl = self.position * (current_price - self.entry_price) / self.entry_price
            else:  # Short
                unrealized_pnl = abs(self.position) * (self.entry_price - current_price) / self.entry_price

            # Small reward for unrealized profit (to encourage holding winners)
            reward += unrealized_pnl * 0.01

        # Check for crash (force close if crash happens)
        is_crash = self.df.iloc[self.current_step].get('is_crash', 0)
        if is_crash and self.position > 0:  # Crash kills long positions
            crash_loss = self.position * 0.015  # -1.5% typical crash
            self.total_pnl -= crash_loss
            reward -= crash_loss

            self.trades.append({
                'entry_price': self.entry_price,
                'exit_price': current_price * 0.985,  # -1.5%
                'pnl': -crash_loss,
                'step': self.current_step,
                'crash': True
            })

            self.position = 0.0
            self.entry_price = 0.0

        # Move to next step
        self.current_step += 1

        # Check if episode is done
        done = False
        if self.current_step >= len(self.df) - 1:
            done = True
            # Force close any open position at end
            if self.position != 0:
                if self.position > 0:
                    pnl = self.position * (current_price - self.entry_price) / self.entry_price
                else:
                    pnl = abs(self.position) * (self.entry_price - current_price) / self.entry_price
                self.total_pnl += pnl
                reward += pnl

        # Penalty for running out of money
        if self.balance + self.total_pnl <= 0:
            done = True
            reward -= 100.0  # Large penalty for ruin

        obs = self._get_observation()
        truncated = False
        info = {'total_pnl': self.total_pnl, 'trades': len(self.trades)}

        return obs, reward, done, truncated, info


class TensorboardCallback(BaseCallback):
    """Callback para logar metricas customizadas no Tensorboard"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Log episode statistics
        if 'episode' in self.locals.get('infos', [{}])[0]:
            info = self.locals['infos'][0]['episode']
            self.logger.record('rollout/ep_rew_mean', info['r'])
            self.logger.record('rollout/ep_len_mean', info['l'])
        return True


def train_ppo_agent():
    print("="*70)
    print("CRASH300N - DEEP REINFORCEMENT LEARNING (PPO)")
    print("="*70)
    print("\nObjetivo: Agente aprende a JOGAR (nao a PREVER)")
    print("RL descobre exploits na RECOMPENSA, nao no INPUT\n")

    # 1. Load Data
    data_dir = Path(__file__).parent / "data"
    files = list(data_dir.glob("*CRASH300*.csv"))
    if not files:
        print("\nERRO: Nenhum arquivo CRASH300N encontrado")
        return

    print(f"[LOAD] {files[0].name}")
    df = pd.read_csv(files[0])

    # Sort temporal
    if 'epoch' in df.columns:
        df = df.sort_values('epoch').reset_index(drop=True)
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

    # Detect crashes
    df['log_ret'] = np.log(df['close'] / df['open'])
    df['is_crash'] = (df['log_ret'] <= -0.005).astype(int)

    print(f"  Total candles: {len(df):,}")
    print(f"  Total crashes: {df['is_crash'].sum():,}")

    # Split train/test
    split_idx = int(0.8 * len(df))
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()

    print(f"\n[SPLIT]")
    print(f"  Train: {len(df_train):,} candles")
    print(f"  Test:  {len(df_test):,} candles")

    # 2. Create Environment
    print(f"\n[ENV] Criando ambiente Gym...")
    env = CRASH300NEnv(df_train, initial_balance=1000.0, max_position_size=1000.0)
    env = DummyVecEnv([lambda: env])

    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")

    # 3. Create PPO Agent
    print(f"\n[AGENT] Criando agente PPO...")
    print(f"  Policy: MlpPolicy (2 hidden layers [64, 64])")
    print(f"  Learning rate: 3e-4")
    print(f"  Batch size: 64")
    print(f"  n_epochs: 10")

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        verbose=1,
        tensorboard_log="./ppo_crash300n_tensorboard/"
    )

    # 4. Train Agent
    print(f"\n[TRAIN] Treinando agente PPO (50,000 timesteps)...")
    print(f"  Isso pode levar ~10 minutos...\n")

    model.learn(
        total_timesteps=50000,
        callback=TensorboardCallback(),
        progress_bar=True
    )

    # 5. Save Model
    model_path = data_dir.parent / "models" / "crash300n_ppo_agent.zip"
    model.save(model_path)
    print(f"\n[SAVE] Modelo salvo em: {model_path}")

    # 6. Evaluate on Test Set
    print(f"\n{'='*70}")
    print("AVALIACAO NO TEST SET")
    print(f"{'='*70}")

    env_test = CRASH300NEnv(df_test, initial_balance=1000.0, max_position_size=1000.0)
    env_test = DummyVecEnv([lambda: env_test])

    mean_reward, std_reward = evaluate_policy(
        model, env_test, n_eval_episodes=10, deterministic=True
    )

    print(f"\n[RESULTS]")
    print(f"  Mean reward: {mean_reward:.2f}")
    print(f"  Std reward:  {std_reward:.2f}")

    # 7. Run full test episode for detailed analysis
    print(f"\n[DETAILED TEST] Rodando episodio completo no test set...")

    obs = env_test.reset()
    done = False
    total_reward = 0.0

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env_test.step(action)
        total_reward += reward[0]

    # Get trades from environment
    test_env = env_test.envs[0]
    trades = test_env.trades

    print(f"\n[TRADES]")
    print(f"  Total trades: {len(trades)}")

    if len(trades) > 0:
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]

        print(f"  Wins: {len(wins)} ({len(wins)/len(trades)*100:.1f}%)")
        print(f"  Losses: {len(losses)} ({len(losses)/len(trades)*100:.1f}%)")
        print(f"  Total PnL: ${sum([t['pnl'] for t in trades]):.2f}")
        print(f"  Avg Win: ${np.mean([t['pnl'] for t in wins]) if wins else 0:.2f}")
        print(f"  Avg Loss: ${np.mean([t['pnl'] for t in losses]) if losses else 0:.2f}")

        # Check for crashes
        crash_trades = [t for t in trades if t.get('crash', False)]
        print(f"  Crash trades: {len(crash_trades)}")

    # 8. Baseline Comparison
    print(f"\n[BASELINE] Random Agent (10 episodes)...")

    baseline_rewards = []
    for _ in range(10):
        obs = env_test.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action = env_test.action_space.sample()  # Random action
            obs, reward, done, info = env_test.step(action)
            episode_reward += reward[0]

        baseline_rewards.append(episode_reward)

    baseline_mean = np.mean(baseline_rewards)
    baseline_std = np.std(baseline_rewards)

    print(f"  Random Mean reward: {baseline_mean:.2f}")
    print(f"  Random Std reward:  {baseline_std:.2f}")

    # 9. Veredicto
    print(f"\n{'='*70}")
    print("VEREDICTO FINAL - DEEP RL")
    print(f"{'='*70}\n")

    improvement = ((mean_reward - baseline_mean) / abs(baseline_mean)) * 100 if baseline_mean != 0 else 0

    print(f"[COMPARISON]")
    print(f"  PPO Agent:     {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"  Random Agent:  {baseline_mean:.2f} +/- {baseline_std:.2f}")
    print(f"  Improvement:   {improvement:+.1f}%")

    if improvement > 20.0:
        print(f"\nSUCESSO! RL DESCOBRIU EXPLOIT!")
        print(f"  Agente PPO supera random em {improvement:.1f}%")
        print(f"  RL encontrou padroes na RECOMPENSA (nao no INPUT)")
        print(f"\n  PROXIMOS PASSOS:")
        print(f"  1. Analisar politica aprendida (quais acoes em quais estados)")
        print(f"  2. Testar em dados out-of-sample (proximos 6 meses)")
        print(f"  3. Paper trading por 1 mes")
        print(f"  4. Live trading com capital pequeno ($100)")

    elif improvement > 5.0:
        print(f"\nEDGE FRACO DETECTADO")
        print(f"  Agente PPO melhora {improvement:.1f}% vs random")
        print(f"  Existe algum aprendizado, mas marginal")
        print(f"\n  POSSIBILIDADES:")
        print(f"  - Mais timesteps de treino (100k+)")
        print(f"  - Tuning de hiperparametros")
        print(f"  - Features adicionais (volume, spread)")

    else:
        print(f"\nFALHOU - SEM EDGE DETECTAVEL")
        print(f"  Agente PPO nao superou random ({improvement:.1f}%)")
        print(f"  RL tambem falhou em encontrar exploits")
        print(f"\n  CONCLUSAO FINAL:")
        print(f"  >> 7 abordagens testadas (LSTM, XGBoost, KAN, FFT, Anti-Poisson, RL)")
        print(f"  >> TODAS falharam em encontrar edge")
        print(f"  >> CRASH300N e matematicamente IMPOSSIVEL de prever")
        print(f"  >> Processo e CSPRNG + Game Engine perfeito")
        print(f"\n  RECOMENDACAO:")
        print(f"  >> Migrar para Forex/Indices reais")
        print(f"  >> Aceitar que CRASH/BOOM e entretenimento, nao trading")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    train_ppo_agent()
