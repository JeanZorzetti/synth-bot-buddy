"""
Experimentos para Melhorar Win Rate do Scalping

Baseline: 50.9% win rate (abaixo da meta de 60%)

Experimentos:
A) TP/SL menos agressivo (0.3% / 0.15%)
B) Ensemble de modelos (XGBoost + LightGBM + CatBoost)
C) Optuna com 100 trials (mais exploração)

Autor: Claude Sonnet 4.5
Data: 18/12/2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List
import time

# Import dos scripts existentes
from scalping_labeling import ScalpingLabeler
from scalping_model_training import ScalpingModelTrainer
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import VotingClassifier
import joblib


class ScalpingExperiments:
    """
    Gerenciador de experimentos para scalping
    """

    def __init__(self, symbol='1HZ100V', timeframe='5min'):
        self.symbol = symbol
        self.timeframe = timeframe
        self.data_dir = Path(__file__).parent / "data"
        self.reports_dir = Path(__file__).parent / "reports"
        self.models_dir = Path(__file__).parent / "models"

        # Criar diretórios
        self.reports_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)

        # Carregar features base (TP 0.2% / SL 0.1%)
        features_path = self.data_dir / f"{symbol}_{timeframe}_180days_features.csv"
        self.df_features = pd.read_csv(features_path)
        self.df_features['timestamp'] = pd.to_datetime(self.df_features['timestamp'])

        print(f"[OK] Experimentos inicializados para {symbol} {timeframe}")
        print(f"   Dataset: {len(self.df_features)} candles com features")

    def experiment_a_relaxed_tp_sl(self):
        """
        Experimento A: TP 0.3% / SL 0.15% (menos agressivo)

        Hipótese: TP/SL mais largo reduz ruído e aumenta win rate base
        """
        print("\n" + "="*70)
        print("EXPERIMENTO A: TP 0.3% / SL 0.15% (R:R 1:2)")
        print("="*70)

        # 1. Gerar labels com TP/SL relaxado
        labeler = ScalpingLabeler(
            self.df_features,
            tp_pct=0.3,
            sl_pct=0.15,
            max_candles=20
        )
        df_labeled = labeler.generate_labels()

        # Salvar dataset temporário
        temp_path = self.data_dir / f"{self.symbol}_{self.timeframe}_labeled_exp_a.csv"
        labeler.save_labeled_data(temp_path)

        # 2. Treinar modelo XGBoost com Optuna (50 trials)
        trainer = ScalpingModelTrainer(df_labeled, test_size=0.2, val_size=0.2)
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data()

        best_params = trainer.optimize_hyperparameters(
            X_train, y_train, X_val, y_val, n_trials=50
        )

        model = trainer.train_final_model(X_train, y_train, X_val, y_val, best_params)

        # 3. Avaliar
        metrics = trainer.evaluate_model(model, X_test, y_test, split_name='Test')

        # 4. Salvar resultados
        result = {
            'experiment': 'A_relaxed_tp_sl',
            'tp_pct': 0.3,
            'sl_pct': 0.15,
            'rr_ratio': 2.0,
            'n_trials': 50,
            'win_rate': metrics['win_rate'],
            'f1_score': metrics['f1_tradeable'],
            'accuracy': metrics['accuracy_tradeable'],
            'best_params': best_params,
            'timestamp': pd.Timestamp.now().isoformat()
        }

        # Salvar modelo e resultados
        model_path = self.models_dir / 'experiment_a_model.pkl'
        joblib.dump(model, model_path)

        result_path = self.reports_dir / 'experiment_a_results.json'
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\n[EXPERIMENTO A] Concluído!")
        print(f"   Win rate: {metrics['win_rate']*100:.1f}%")
        print(f"   Meta atingida: {'SIM' if metrics['win_rate'] >= 0.60 else 'NAO'}")

        return result

    def experiment_b_ensemble(self):
        """
        Experimento B: Ensemble XGBoost + LightGBM + CatBoost

        Hipótese: Combinar 3 modelos aumenta robustez e win rate
        """
        print("\n" + "="*70)
        print("EXPERIMENTO B: ENSEMBLE (XGBoost + LightGBM + CatBoost)")
        print("="*70)

        # Usar dataset original (TP 0.2% / SL 0.1%)
        labeled_path = self.data_dir / f"{self.symbol}_{self.timeframe}_180days_labeled.csv"
        df_labeled = pd.read_csv(labeled_path)

        trainer = ScalpingModelTrainer(df_labeled, test_size=0.2, val_size=0.2)
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data()

        print("\n[1/3] Treinando XGBoost...")
        # XGBoost (usar melhores params do baseline)
        xgb_params = {
            'max_depth': 9,
            'learning_rate': 0.215,
            'n_estimators': 263,
            'min_child_weight': 9,
            'subsample': 0.73,
            'colsample_bytree': 0.97,
            'gamma': 0.22,
            'reg_alpha': 0.39,
            'reg_lambda': 1.42,
            'objective': 'multi:softmax',
            'num_class': 3,
            'random_state': 42,
            'tree_method': 'hist'
        }
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(X_train, y_train, verbose=False)

        print("[2/3] Treinando LightGBM...")
        # LightGBM (params similares)
        lgb_params = {
            'max_depth': 9,
            'learning_rate': 0.2,
            'n_estimators': 250,
            'num_leaves': 31,
            'subsample': 0.7,
            'colsample_bytree': 0.95,
            'reg_alpha': 0.4,
            'reg_lambda': 1.4,
            'objective': 'multiclass',
            'num_class': 3,
            'random_state': 42,
            'verbose': -1
        }
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        lgb_model.fit(X_train, y_train)

        print("[3/3] Treinando CatBoost...")
        # CatBoost
        try:
            from catboost import CatBoostClassifier
            cat_params = {
                'depth': 9,
                'learning_rate': 0.2,
                'iterations': 250,
                'l2_leaf_reg': 1.4,
                'random_state': 42,
                'verbose': 0,
                'loss_function': 'MultiClass',
                'classes_count': 3
            }
            cat_model = CatBoostClassifier(**cat_params)
            cat_model.fit(X_train, y_train, verbose=False)

            # Criar ensemble com voting
            ensemble = VotingClassifier(
                estimators=[
                    ('xgb', xgb_model),
                    ('lgb', lgb_model),
                    ('cat', cat_model)
                ],
                voting='soft'
            )
            # Fit já foi feito nos modelos individuais, apenas criar wrapper
            ensemble.estimators_ = [xgb_model, lgb_model, cat_model]

        except ImportError:
            print("[AVISO] CatBoost não instalado, usando apenas XGBoost + LightGBM")
            ensemble = VotingClassifier(
                estimators=[
                    ('xgb', xgb_model),
                    ('lgb', lgb_model)
                ],
                voting='soft'
            )
            ensemble.estimators_ = [xgb_model, lgb_model]

        # Avaliar ensemble
        print("\n[EVAL] Avaliando ensemble...")
        y_pred = ensemble.predict(X_test)

        # Calcular métricas (código similar ao trainer.evaluate_model)
        from sklearn.metrics import accuracy_score, f1_score

        mask = y_test != 0
        acc_tradeable = accuracy_score(y_test[mask], y_pred[mask])
        f1_tradeable = f1_score(y_test[mask], y_pred[mask], average='macro', labels=[1, 2])
        win_rate = acc_tradeable

        print(f"   Win rate: {win_rate*100:.1f}%")
        print(f"   F1-score: {f1_tradeable:.4f}")

        # Salvar resultados
        result = {
            'experiment': 'B_ensemble',
            'models': ['XGBoost', 'LightGBM', 'CatBoost'] if 'cat_model' in locals() else ['XGBoost', 'LightGBM'],
            'win_rate': float(win_rate),
            'f1_score': float(f1_tradeable),
            'accuracy': float(acc_tradeable),
            'timestamp': pd.Timestamp.now().isoformat()
        }

        # Salvar ensemble
        model_path = self.models_dir / 'experiment_b_ensemble.pkl'
        joblib.dump(ensemble, model_path)

        result_path = self.reports_dir / 'experiment_b_results.json'
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\n[EXPERIMENTO B] Concluído!")
        print(f"   Meta atingida: {'SIM' if win_rate >= 0.60 else 'NAO'}")

        return result

    def experiment_c_more_trials(self):
        """
        Experimento C: Optuna com 100 trials (mais exploração)

        Hipótese: 50 trials foram insuficientes, 100 trials acharão melhores hiperparâmetros
        """
        print("\n" + "="*70)
        print("EXPERIMENTO C: OPTUNA COM 100 TRIALS")
        print("="*70)

        # Usar dataset original
        labeled_path = self.data_dir / f"{self.symbol}_{self.timeframe}_180days_labeled.csv"
        df_labeled = pd.read_csv(labeled_path)

        trainer = ScalpingModelTrainer(df_labeled, test_size=0.2, val_size=0.2)
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data()

        # Optuna com 100 trials
        best_params = trainer.optimize_hyperparameters(
            X_train, y_train, X_val, y_val, n_trials=100
        )

        model = trainer.train_final_model(X_train, y_train, X_val, y_val, best_params)

        # Avaliar
        metrics = trainer.evaluate_model(model, X_test, y_test, split_name='Test')

        # Salvar resultados
        result = {
            'experiment': 'C_more_trials',
            'n_trials': 100,
            'win_rate': metrics['win_rate'],
            'f1_score': metrics['f1_tradeable'],
            'accuracy': metrics['accuracy_tradeable'],
            'best_params': best_params,
            'timestamp': pd.Timestamp.now().isoformat()
        }

        # Salvar modelo
        model_path = self.models_dir / 'experiment_c_model.pkl'
        joblib.dump(model, model_path)

        result_path = self.reports_dir / 'experiment_c_results.json'
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\n[EXPERIMENTO C] Concluído!")
        print(f"   Win rate: {metrics['win_rate']*100:.1f}%")
        print(f"   Meta atingida: {'SIM' if metrics['win_rate'] >= 0.60 else 'NAO'}")

        return result

    def run_all_experiments(self):
        """
        Executa todos os 3 experimentos e compara resultados
        """
        print("\n" + "="*70)
        print("EXECUTANDO TODOS OS EXPERIMENTOS")
        print("="*70)

        results = {}

        # Experimento A
        print("\n[1/3] Iniciando Experimento A...")
        try:
            results['A'] = self.experiment_a_relaxed_tp_sl()
        except Exception as e:
            print(f"[ERRO] Experimento A falhou: {e}")
            results['A'] = {'win_rate': 0, 'error': str(e)}

        # Experimento B
        print("\n[2/3] Iniciando Experimento B...")
        try:
            results['B'] = self.experiment_b_ensemble()
        except Exception as e:
            print(f"[ERRO] Experimento B falhou: {e}")
            results['B'] = {'win_rate': 0, 'error': str(e)}

        # Experimento C
        print("\n[3/3] Iniciando Experimento C...")
        try:
            results['C'] = self.experiment_c_more_trials()
        except Exception as e:
            print(f"[ERRO] Experimento C falhou: {e}")
            results['C'] = {'win_rate': 0, 'error': str(e)}

        # Comparar resultados
        self._compare_results(results)

        return results

    def _compare_results(self, results: Dict):
        """
        Compara resultados dos experimentos
        """
        print("\n" + "="*70)
        print("COMPARAÇÃO DE RESULTADOS")
        print("="*70)

        # Baseline
        print("\nBASELINE (TP 0.2% / SL 0.1%, 50 trials):")
        print(f"   Win rate: 50.9%")

        # Experimentos
        for exp_name, result in results.items():
            if 'error' in result:
                print(f"\n{exp_name}: FALHOU - {result['error']}")
            else:
                win_rate = result.get('win_rate', 0) * 100
                improvement = win_rate - 50.9
                status = "ATINGIU META!" if win_rate >= 60 else f"Faltam {60 - win_rate:.1f}%"

                print(f"\nEXPERIMENTO {exp_name}:")
                print(f"   Win rate: {win_rate:.1f}%")
                print(f"   Melhoria: {improvement:+.1f}pp")
                print(f"   Status: {status}")

        # Escolher melhor
        best_exp = max(results.items(), key=lambda x: x[1].get('win_rate', 0))
        best_name, best_result = best_exp

        print(f"\n{'='*70}")
        print(f"MELHOR EXPERIMENTO: {best_name}")
        print(f"   Win rate: {best_result.get('win_rate', 0)*100:.1f}%")

        if best_result.get('win_rate', 0) >= 0.60:
            print("\n[SUCESSO] META DE 60% ATINGIDA!")
            print("Próximo passo: Backtesting completo")
        else:
            print(f"\n[AVISO] Meta não atingida (faltam {60 - best_result.get('win_rate', 0)*100:.1f}%)")
            print("Considere: Feature engineering avançada (order flow, tape reading)")

        # Salvar comparação
        comparison_path = self.reports_dir / 'experiments_comparison.json'
        comparison = {
            'baseline': {'win_rate': 0.509, 'tp_pct': 0.2, 'sl_pct': 0.1, 'n_trials': 50},
            'experiments': results,
            'best_experiment': best_name,
            'goal_achieved': best_result.get('win_rate', 0) >= 0.60,
            'timestamp': pd.Timestamp.now().isoformat()
        }

        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)

        print(f"\n[SAVE] Comparação salva em {comparison_path}")


if __name__ == "__main__":
    """
    Executar todos os experimentos
    """

    experiments = ScalpingExperiments(symbol='1HZ100V', timeframe='5min')

    # Executar todos
    results = experiments.run_all_experiments()

    print("\n" + "="*70)
    print("EXPERIMENTOS CONCLUÍDOS!")
    print("="*70)
