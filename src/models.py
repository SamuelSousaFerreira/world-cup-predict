"""Modelos de previsão de confrontos.

Implementa três abordagens independentes que produzem probabilidades
P(vitória mandante / empate / vitória visitante). Cada uma tem prós e
contras distintos — quando divergem, um humano decide.

1) EloProbModel  (estatístico, baseado em rating)
   + Robusto, poucos parâmetros, interpretável, ótimo para força relativa.
   - Ignora forma recente e estilo; não dá placar provável.

2) PoissonModel  (força de ataque x defesa -> gols esperados)
   + Modela explicitamente ataque/defesa; gera placar provável e
     probabilidade de cada resultado exato.
   - Assume independência entre gols; sensível a goleadas atípicas.

3) MLModel  (HistGradientBoosting sobre todas as features)
   + Captura interações não lineares entre Elo, forma, ataque, defesa e
     estilo; costuma calibrar melhor.
   - "Caixa-preta"; precisa de dados suficientes; pode superajustar.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from squad_data import SQUAD_FEATURES

CLASSES = ["H", "D", "A"]  # ordem canônica das probabilidades

# Meia-vida do decaimento temporal: um jogo de HALF_LIFE_DAYS atrás pesa
# metade de um jogo de hoje. ~3 anos equilibra recência e volume de dados.
HALF_LIFE_DAYS = 365 * 3


def temporal_weights(dates: pd.Series, half_life_days: int = HALF_LIFE_DAYS) -> np.ndarray:
    """Pesos exponenciais por recência (jogos antigos pesam menos)."""
    ref = pd.Timestamp(dates.max())
    age_days = (ref - pd.to_datetime(dates)).dt.days.to_numpy(dtype=float)
    return 0.5 ** (age_days / half_life_days)

ML_FEATURES = [
    "elo_diff", "elo_home", "elo_away",
    "home_form", "away_form",
    "home_attack", "away_attack",
    "home_defense", "away_defense",
    "home_style", "away_style",
    "home_aggression", "away_aggression",
    "neutral", "importance",
]

# Features de contexto/forma avançadas (engenharia v2): ataque/defesa ajustados
# pela força do oponente, forma com EWMA, força de calendário, sequência e
# descanso/amplitude da janela. NaN é tratado nativamente pelos modelos de árvore.
ADV_FEATURES = [
    "home_ewma_form", "away_ewma_form",
    "home_adj_attack", "away_adj_attack",
    "home_adj_defense", "away_adj_defense",
    "home_sos", "away_sos",
    "home_streak", "away_streak",
    "home_rest_days", "away_rest_days",
    "home_window_days", "away_window_days",
]

# Features completas do MLModel: base + força de elenco (Transfermarkt) + avançadas.
# O HistGradientBoosting trata NaN nativamente, então seleções sem cobertura
# de elenco simplesmente entram como "informação ausente".
ML_ALL_FEATURES = ML_FEATURES + SQUAD_FEATURES + ADV_FEATURES


def _order_proba(model_classes, proba_row_classes_dict) -> np.ndarray:
    """Garante a ordem [H, D, A] em um vetor de probabilidades."""
    return np.array([proba_row_classes_dict.get(c, 0.0) for c in CLASSES])


# ----------------------------- 1) Elo --------------------------------------- #
class EloProbModel:
    """Regressão logística multinomial sobre o Elo (diferença + mando)."""

    def __init__(self) -> None:
        # multinomial é o comportamento padrão para problemas multiclasse
        # no scikit-learn >= 1.7 (o parâmetro multi_class foi removido).
        self.clf = LogisticRegression(max_iter=1000, C=1.0)
        self.scaler = StandardScaler()

    def _X(self, df: pd.DataFrame) -> np.ndarray:
        return df[["elo_diff", "neutral"]].to_numpy(dtype=float)

    def fit(self, df: pd.DataFrame, sample_weight: np.ndarray | None = None) -> "EloProbModel":
        X = self.scaler.fit_transform(self._X(df))
        self.clf.fit(X, df["result"].to_numpy(), sample_weight=sample_weight)
        return self

    def predict_proba(self, feats: dict) -> np.ndarray:
        X = self.scaler.transform([[feats["elo_diff"], feats["neutral"]]])
        proba = self.clf.predict_proba(X)[0]
        d = dict(zip(self.clf.classes_, proba))
        return _order_proba(self.clf.classes_, d)


# --------------------------- 2) Poisson ------------------------------------- #
class PoissonModel:
    """Gols esperados a partir de força de ataque/defesa (Poisson independente).

    lambda_mandante = base_home * atk_home * def_away
    lambda_visitante = base_away * atk_visit * def_home

    As forças são derivadas da média de gols dos últimos jogos (já presentes
    nas features) normalizadas pela média da liga, e ajustadas pelo Elo para
    refletir a qualidade relativa do adversário.
    """

    def __init__(self, max_goals: int = 10, damping: float = 0.5, rho: float = -0.08) -> None:
        self.max_goals = max_goals
        # Amortece o quanto as forças (janela de 10 jogos) se afastam da média.
        # damping=1 -> usa a razão crua; damping=0.5 -> raiz quadrada (suaviza
        # extremos como uma defesa de 0.2 gols, evitando superconfiança).
        self.damping = damping
        # rho: parâmetro de Dixon-Coles que corrige a dependência entre os gols
        # em placares baixos (0-0, 1-0, 0-1, 1-1), onde o Poisson puro erra mais.
        self.rho = rho
        self.base_home = 1.5
        self.base_away = 1.1
        self.league_attack = 1.3
        self.league_defense = 1.3

    def fit(self, df: pd.DataFrame, sample_weight: np.ndarray | None = None) -> "PoissonModel":
        w = np.ones(len(df)) if sample_weight is None else np.asarray(sample_weight)
        wsum = w.sum()
        self.base_home = float((df["home_score"].to_numpy() * w).sum() / wsum)
        self.base_away = float((df["away_score"].to_numpy() * w).sum() / wsum)
        # médias de referência das janelas de forma
        self.league_attack = float(
            pd.concat([df["home_attack"], df["away_attack"]]).replace(0, np.nan).mean()
        )
        self.league_defense = float(
            pd.concat([df["home_defense"], df["away_defense"]]).replace(0, np.nan).mean()
        )
        return self

    def expected_goals(self, feats: dict) -> tuple[float, float]:
        la, ld = self.league_attack, self.league_defense
        d = self.damping
        atk_h = ((feats["home_attack"] or la) / la) ** d
        atk_a = ((feats["away_attack"] or la) / la) ** d
        def_h = ((feats["home_defense"] or ld) / ld) ** d
        def_a = ((feats["away_defense"] or ld) / ld) ** d

        # Ajuste pela diferença de Elo (qualidade): ~ +/-25% por 200 pts.
        elo_adj = 10 ** (feats["elo_diff"] / 1000.0)

        neutral = feats["neutral"]
        base_h = self.base_home if not neutral else (self.base_home + self.base_away) / 2
        base_a = self.base_away if not neutral else (self.base_home + self.base_away) / 2

        lam_h = base_h * atk_h * def_a * elo_adj
        lam_a = base_a * atk_a * def_h / elo_adj
        # limites de sanidade
        lam_h = float(np.clip(lam_h, 0.15, 6.0))
        lam_a = float(np.clip(lam_a, 0.15, 6.0))
        return lam_h, lam_a

    def score_matrix(self, feats: dict) -> np.ndarray:
        lam_h, lam_a = self.expected_goals(feats)
        gh = poisson.pmf(np.arange(self.max_goals + 1), lam_h)
        ga = poisson.pmf(np.arange(self.max_goals + 1), lam_a)
        m = np.outer(gh, ga)  # M[i,j] = P(mandante i x j visitante)
        # Correção Dixon-Coles nos placares baixos.
        rho = self.rho
        m[0, 0] *= 1.0 - lam_h * lam_a * rho
        m[0, 1] *= 1.0 + lam_h * rho
        m[1, 0] *= 1.0 + lam_a * rho
        m[1, 1] *= 1.0 - rho
        m = np.clip(m, 0.0, None)
        return m / m.sum()

    def predict_proba(self, feats: dict) -> np.ndarray:
        m = self.score_matrix(feats)
        p_home = np.tril(m, -1).sum()   # i > j
        p_draw = np.trace(m)            # i == j
        p_away = np.triu(m, 1).sum()    # i < j
        total = p_home + p_draw + p_away
        return np.array([p_home, p_draw, p_away]) / total

    def most_likely_score(self, feats: dict) -> tuple[int, int, float]:
        m = self.score_matrix(feats)
        i, j = np.unravel_index(np.argmax(m), m.shape)
        return int(i), int(j), float(m[i, j])


# ----------------------------- 3) ML ---------------------------------------- #
class MLModel:
    """Gradient Boosting calibrado sobre todas as features."""

    def __init__(self) -> None:
        base = HistGradientBoostingClassifier(
            max_iter=400,
            learning_rate=0.05,
            max_depth=None,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        )
        self.clf = CalibratedClassifierCV(base, method="isotonic", cv=3)

    def fit(self, df: pd.DataFrame, sample_weight: np.ndarray | None = None) -> "MLModel":
        X = df[ML_ALL_FEATURES].to_numpy(dtype=float)
        y = df["result"].to_numpy()
        self.clf.fit(X, y, sample_weight=sample_weight)
        self.classes_ = self.clf.classes_
        return self

    def predict_proba(self, feats: dict) -> np.ndarray:
        X = np.array([[feats.get(f, np.nan) for f in ML_ALL_FEATURES]], dtype=float)
        proba = self.clf.predict_proba(X)[0]
        d = dict(zip(self.clf.classes_, proba))
        return _order_proba(self.clf.classes_, d)


# ---------------------- 4-6) Boostings calibrados --------------------------- #
# CatBoost, LightGBM e XGBoost sobre o mesmo conjunto de features do MLModel.
# No A/B (bloco de teste) cada um rendeu ganho marginal mas consistente ao
# ensemble (+0.0016 em log loss com os três juntos). Usam rótulos inteiros
# (H=0, D=1, A=2) porque XGBoost não aceita rótulos string.
_Y2I = {"H": 0, "D": 1, "A": 2}


class _CalibratedBoost:
    """Base: classificador de árvore calibrado (isotônica, cv=3) sobre ML_ALL_FEATURES."""

    def _make_base(self):  # pragma: no cover - implementado nas subclasses
        raise NotImplementedError

    def fit(self, df: pd.DataFrame, sample_weight: np.ndarray | None = None):
        X = df[ML_ALL_FEATURES].to_numpy(dtype=float)
        y = df["result"].map(_Y2I).to_numpy()
        self.clf = CalibratedClassifierCV(self._make_base(), method="isotonic", cv=3)
        self.clf.fit(X, y, sample_weight=sample_weight)
        self.classes_ = self.clf.classes_
        return self

    def predict_proba(self, feats: dict) -> np.ndarray:
        X = np.array([[feats.get(f, np.nan) for f in ML_ALL_FEATURES]], dtype=float)
        proba = self.clf.predict_proba(X)[0]
        # classes_ são inteiros 0/1/2 -> reordena para [H, D, A].
        d = {CLASSES[int(c)]: p for c, p in zip(self.clf.classes_, proba)}
        return np.array([d.get(c, 0.0) for c in CLASSES])


class CatBoostModel(_CalibratedBoost):
    def _make_base(self):
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=400, learning_rate=0.05, depth=6,
                                  l2_leaf_reg=3.0, random_seed=42, verbose=0)


class LightGBMModel(_CalibratedBoost):
    def _make_base(self):
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                              reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8,
                              random_state=42, verbose=-1)


class XGBoostModel(_CalibratedBoost):
    def _make_base(self):
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6,
                             reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8,
                             eval_metric="mlogloss", random_state=42)


# ------------------------- Ensemble / Divergência --------------------------- #
@dataclass
class Prediction:
    probs: dict[str, np.ndarray]          # por modelo: {nome: [H, D, A]}
    ensemble: np.ndarray                  # média (ponderada) [H, D, A]
    most_likely_score: tuple[int, int, float]
    diverges: bool                        # modelos discordam do vencedor?
    disagreement: float                   # dispersão máxima entre modelos


def combine(probs: dict[str, np.ndarray], score: tuple[int, int, float],
            weights: dict[str, float] | None = None) -> Prediction:
    names = list(probs.keys())
    stacked = np.vstack([probs[n] for n in names])
    if weights:
        w = np.array([weights.get(n, 0.0) for n in names], dtype=float)
        w = w / w.sum() if w.sum() > 0 else np.ones(len(names)) / len(names)
    else:
        w = np.ones(len(names)) / len(names)
    ensemble = (stacked * w[:, None]).sum(axis=0)
    ensemble = ensemble / ensemble.sum()
    picks = {name: CLASSES[int(np.argmax(p))] for name, p in probs.items()}
    diverges = len(set(picks.values())) > 1
    disagreement = float(stacked.max(axis=0).max() - stacked.min(axis=0).min())
    return Prediction(
        probs=probs,
        ensemble=ensemble,
        most_likely_score=score,
        diverges=diverges,
        disagreement=disagreement,
    )
