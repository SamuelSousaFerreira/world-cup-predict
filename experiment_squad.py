"""Experimento A/B: as features de elenco (Transfermarkt) melhoram o MLModel?

Mesmo split temporal, mesmo decaimento. Compara log loss / acurácia / brier
no bloco de teste, com e sem as 3 features de elenco.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
from sklearn.metrics import accuracy_score, log_loss

import models as M
from feature_engineering import load_training
from models import MLModel, temporal_weights
from squad_data import SQUAD_FEATURES, add_squad_features

CLASSES = ["H", "D", "A"]
Y2I = {"H": 0, "D": 1, "A": 2}


def split(df, test_fraction=0.15):
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    cut = int(n * (1 - test_fraction))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def brier(proba, y_idx):
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y_idx)), y_idx] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def run_ml(train_df, test_df, features):
    m = MLModel()
    # Sobrescreve a lista de features que o MLModel usa, temporariamente.
    orig = M.ML_FEATURES
    M.ML_FEATURES = features
    try:
        w = temporal_weights(train_df["date"])
        m.fit(train_df, sample_weight=w)
        X = test_df[features].to_numpy(dtype=float)
        proba_raw = m.clf.predict_proba(X)
        # reordena para [H, D, A]
        cls = list(m.clf.classes_)
        order = [cls.index(c) for c in CLASSES]
        proba = proba_raw[:, order]
    finally:
        M.ML_FEATURES = orig
    return proba


def main():
    df = load_training()
    df = add_squad_features(df)
    print(f"Partidas de treino: {len(df):,}")
    cov = df[SQUAD_FEATURES[0]].notna().mean()
    print(f"Cobertura de elenco (linhas com dados): {cov*100:.1f}%\n")

    train_df, test_df = split(df, test_fraction=0.15)
    y_idx = test_df["result"].map(Y2I).to_numpy()
    print(f"Treino: {len(train_df):,} | Teste: {len(test_df):,} "
          f"({test_df['date'].min().date()} -> {test_df['date'].max().date()})\n")

    base_feats = list(M.ML_FEATURES)
    aug_feats = base_feats + SQUAD_FEATURES

    print("Treinando ML SEM elenco...")
    p_base = run_ml(train_df, test_df, base_feats)
    print("Treinando ML COM elenco...")
    p_aug = run_ml(train_df, test_df, aug_feats)

    def report(name, p):
        acc = accuracy_score(y_idx, p.argmax(axis=1))
        ll = log_loss(y_idx, p, labels=[0, 1, 2])
        bs = brier(p, y_idx)
        print(f"  {name:<18} acc={acc:.4f}  logloss={ll:.4f}  brier={bs:.4f}")
        return ll

    print("\n=== Resultado no bloco de teste ===")
    ll_base = report("ML base", p_base)
    ll_aug = report("ML + elenco", p_aug)
    delta = ll_base - ll_aug
    print(f"\nDelta log loss (positivo = elenco MELHORA): {delta:+.4f}")
    print("VEREDITO:", "MANTER elenco" if delta > 0.001 else "NAO compensa (remover)")


if __name__ == "__main__":
    main()
