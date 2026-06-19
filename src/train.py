"""Treino e avaliação dos modelos.

- Constrói/atualiza features a partir dos dados crus.
- Treina os três modelos (Elo, Poisson, ML).
- Faz um backtest temporal (treina no passado, testa no período recente)
  reportando acurácia, log loss e Brier score por modelo e do ensemble.
- Persiste os modelos em models/.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, log_loss

from data_collection import load_results
from feature_engineering import build_features, load_team_state, save
from models import (CLASSES, CatBoostModel, EloProbModel, LightGBMModel,
                    MLModel, PoissonModel, XGBoostModel, combine,
                    temporal_weights)
from squad_data import add_squad_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


def _proba_table(model, df: pd.DataFrame) -> np.ndarray:
    """Probabilidades [N,3] para um DataFrame de teste, linha a linha."""
    out = np.zeros((len(df), 3))
    for i, row in enumerate(df.to_dict("records")):
        out[i] = model.predict_proba(row)
    return out


def optimize_weights(probs: dict[str, np.ndarray], y_idx: np.ndarray) -> dict[str, float]:
    """Acha os pesos do ensemble (simplex) que minimizam o log loss."""
    names = list(probs.keys())
    stacked = np.stack([probs[n] for n in names], axis=0)  # [M, N, 3]

    def neg_ll(w: np.ndarray) -> float:
        w = np.clip(w, 0, None)
        s = w.sum()
        w = w / s if s > 0 else np.ones_like(w) / len(w)
        ens = (stacked * w[:, None, None]).sum(axis=0)
        ens = ens / ens.sum(axis=1, keepdims=True)
        return log_loss(y_idx, ens, labels=[0, 1, 2])

    x0 = np.ones(len(names)) / len(names)
    cons = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
    bounds = [(0.0, 1.0)] * len(names)
    res = minimize(neg_ll, x0, method="SLSQP", bounds=bounds, constraints=cons)
    w = np.clip(res.x, 0, None)
    w = w / w.sum()
    return {n: float(wi) for n, wi in zip(names, w)}


def evaluate(df: pd.DataFrame, test_fraction: float = 0.15, val_fraction: float = 0.15) -> dict[str, float]:
    """Backtest temporal com decaimento e pesos de ensemble otimizados.

    Divisão temporal: [ treino | validação | teste ].
    - treino: ajusta os modelos (com peso por recência).
    - validação: otimiza os pesos do ensemble (minimiza log loss).
    - teste: mede o desempenho final, fora da amostra.
    """
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    split_test = int(n * (1 - test_fraction))
    split_val = int(split_test * (1 - val_fraction))
    train_df = df.iloc[:split_val]
    val_df = df.iloc[split_val:split_test]
    test_df = df.iloc[split_test:]
    y_true = test_df["result"].to_numpy()
    y_idx = np.array([CLASSES.index(c) for c in y_true])
    yv_idx = np.array([CLASSES.index(c) for c in val_df["result"]])

    print(f"\n=== Backtest temporal ===")
    print(f"Treino : {len(train_df):,} jogos ({train_df['date'].min().date()} -> {train_df['date'].max().date()})")
    print(f"Valida.: {len(val_df):,} jogos ({val_df['date'].min().date()} -> {val_df['date'].max().date()})")
    print(f"Teste  : {len(test_df):,} jogos ({test_df['date'].min().date()} -> {test_df['date'].max().date()})")

    w_train = temporal_weights(train_df["date"])
    models = {
        "Elo": EloProbModel().fit(train_df, sample_weight=w_train),
        "Poisson": PoissonModel().fit(train_df, sample_weight=w_train),
        "ML": MLModel().fit(train_df, sample_weight=w_train),
        "CatBoost": CatBoostModel().fit(train_df, sample_weight=w_train),
        "LightGBM": LightGBMModel().fit(train_df, sample_weight=w_train),
        "XGBoost": XGBoostModel().fit(train_df, sample_weight=w_train),
    }

    # Otimiza os pesos do ensemble na validação.
    val_probs = {name: _proba_table(m, val_df) for name, m in models.items()}
    weights = optimize_weights(val_probs, yv_idx)
    print("\nPesos do ensemble (otimizados na validação):")
    for name, wt in weights.items():
        print(f"  {name:<8} {wt:.3f}")

    all_probs = {}
    print(f"\n{'Modelo':<12}{'Acurácia':>10}{'LogLoss':>10}{'Brier':>10}")
    print("-" * 42)
    for name, model in models.items():
        proba = _proba_table(model, test_df)
        all_probs[name] = proba
        pred = proba.argmax(axis=1)
        acc = accuracy_score(y_idx, pred)
        ll = log_loss(y_idx, proba, labels=[0, 1, 2])
        onehot = np.eye(3)[y_idx]
        brier = float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))
        print(f"{name:<12}{acc:>10.3f}{ll:>10.3f}{brier:>10.3f}")

    # Ensemble simples (média) para comparação.
    ens_mean = np.mean(list(all_probs.values()), axis=0)
    # Ensemble ponderado (pesos otimizados).
    w = np.array([weights[n] for n in all_probs])
    ens_w = (np.stack(list(all_probs.values()), axis=0) * w[:, None, None]).sum(axis=0)
    ens_w = ens_w / ens_w.sum(axis=1, keepdims=True)

    onehot = np.eye(3)[y_idx]
    for label, ens in (("Ens.(média)", ens_mean), ("Ens.(pesos)", ens_w)):
        acc = accuracy_score(y_idx, ens.argmax(axis=1))
        ll = log_loss(y_idx, ens, labels=[0, 1, 2])
        brier = float(np.mean(np.sum((ens - onehot) ** 2, axis=1)))
        print(f"{label:<12}{acc:>10.3f}{ll:>10.3f}{brier:>10.3f}")

    # Baseline: sempre prever o resultado mais comum (mando) para referência.
    base_idx = np.bincount(np.array([CLASSES.index(c) for c in train_df['result']]), minlength=3).argmax()
    base_acc = accuracy_score(y_idx, np.full_like(y_idx, base_idx))
    print(f"\nBaseline (classe majoritária '{CLASSES[base_idx]}'): acurácia {base_acc:.3f}")
    return weights


def train_and_save(df: pd.DataFrame, weights: dict[str, float]) -> None:
    """Treina nos dados completos (com peso por recência) e persiste tudo."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("\n=== Treino final (dados completos) ===")
    w = temporal_weights(df["date"])
    elo = EloProbModel().fit(df, sample_weight=w)
    poisson_m = PoissonModel().fit(df, sample_weight=w)
    ml = MLModel().fit(df, sample_weight=w)
    cat = CatBoostModel().fit(df, sample_weight=w)
    lgb = LightGBMModel().fit(df, sample_weight=w)
    xgb = XGBoostModel().fit(df, sample_weight=w)
    joblib.dump(elo, MODELS_DIR / "elo_model.joblib")
    joblib.dump(poisson_m, MODELS_DIR / "poisson_model.joblib")
    joblib.dump(ml, MODELS_DIR / "ml_model.joblib")
    joblib.dump(cat, MODELS_DIR / "catboost_model.joblib")
    joblib.dump(lgb, MODELS_DIR / "lightgbm_model.joblib")
    joblib.dump(xgb, MODELS_DIR / "xgboost_model.joblib")
    with open(MODELS_DIR / "ensemble_weights.json", "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2)
    print(f"[ok] modelos e pesos do ensemble salvos em {MODELS_DIR}")


def main() -> None:
    print("=== 1/3 Coletando dados ===")
    matches = load_results()
    print(f"{len(matches):,} partidas carregadas.")

    print("\n=== 2/3 Engenharia de features ===")
    training_df, team_state = build_features(matches)
    # Força de elenco (Transfermarkt): diferenças de valor/idade/ranking FIFA.
    training_df = add_squad_features(training_df)
    save(training_df, team_state)

    print("\n=== 3/3 Avaliação + treino ===")
    weights = evaluate(training_df)
    train_and_save(training_df, weights)
    print("\nPronto! Use: python src/predict.py \"Brazil\" \"Argentina\"")


if __name__ == "__main__":
    main()
