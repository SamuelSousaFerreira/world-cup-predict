"""Probabilidades de um confronto eliminatório (mata-mata).

Encadeia três estágios sobre o resultado do tempo normal, sem alterar os
modelos existentes (recebe o ensemble e os gols esperados que o
``compute_prediction`` já devolve):

  1) 90 min        -> usa o 1X2 do ensemble (vitória / empate / derrota).
  2) prorrogação   -> novo Poisson/Dixon-Coles com lambda reescalado por tempo
                      (30 min = 1/3 dos 90), só se a regulamentar empatar.
  3) pênaltis      -> viés pequeno por Elo, calibrado em shootouts.csv
                      (logística sem intercepto: Elo igual -> 50%).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import poisson

# Coeficiente do viés de Elo nos pênaltis, calibrado offline no shootouts.csv.
# Fallback usado caso o arquivo de calibração não exista.
_CALIB_PATH = Path(__file__).resolve().parent.parent / "models" / "penalty_calib.json"
_DEFAULT_PEN_COEF = 0.0015688822815496835


def _load_pen_coef() -> float:
    try:
        with open(_CALIB_PATH, encoding="utf-8") as f:
            return float(json.load(f)["elo_coef"])
    except Exception:
        return _DEFAULT_PEN_COEF


def _dixon_coles_matrix(lam_h: float, lam_a: float,
                        max_goals: int = 10, rho: float = -0.08) -> np.ndarray:
    """Matriz de placares Poisson com correção Dixon-Coles (igual à do PoissonModel)."""
    lam_h = float(np.clip(lam_h, 0.05, 6.0))
    lam_a = float(np.clip(lam_a, 0.05, 6.0))
    gh = poisson.pmf(np.arange(max_goals + 1), lam_h)
    ga = poisson.pmf(np.arange(max_goals + 1), lam_a)
    m = np.outer(gh, ga)
    m[0, 0] *= 1.0 - lam_h * lam_a * rho
    m[0, 1] *= 1.0 + lam_h * rho
    m[1, 0] *= 1.0 + lam_a * rho
    m[1, 1] *= 1.0 - rho
    m = np.clip(m, 0.0, None)
    return m / m.sum()


def _split(m: np.ndarray) -> tuple[float, float, float]:
    """(P mandante vence, P empate, P visitante vence) de uma matriz de placares."""
    return float(np.tril(m, -1).sum()), float(np.trace(m)), float(np.triu(m, 1).sum())


def shootout_home_prob(elo_diff: float, coef: float | None = None) -> float:
    """P(mandante vence a disputa de pênaltis) ~ logística do Elo (neutro = 50%)."""
    c = _load_pen_coef() if coef is None else coef
    return float(1.0 / (1.0 + np.exp(-c * elo_diff)))


def knockout_probabilities(ensemble, lam_h: float, lam_a: float, elo_diff: float,
                           *, et_scale: float = 1.0 / 3.0, rho: float = -0.08) -> dict:
    """Decompõe a probabilidade de classificação num confronto eliminatório.

    Args:
        ensemble: [P(mandante 90), P(empate 90), P(visitante 90)] do ensemble.
        lam_h, lam_a: gols esperados (lambda Poisson) na regulamentar.
        elo_diff: Elo mandante - Elo visitante (mata-mata é neutro, sem mando).
        et_scale: fração de tempo da prorrogação (30/90 = 1/3).
    """
    wh90, dreg, wa90 = float(ensemble[0]), float(ensemble[1]), float(ensemble[2])

    # Prorrogação: mesmo Poisson, com lambda proporcional ao tempo (30 min).
    met = _dixon_coles_matrix(lam_h * et_scale, lam_a * et_scale, rho=rho)
    wh_et, det, wa_et = _split(met)

    p_home_pen = shootout_home_prob(elo_diff)

    # Três caminhos para cada time se classificar.
    home_90, home_et, home_pen = wh90, dreg * wh_et, dreg * det * p_home_pen
    away_90, away_et, away_pen = wa90, dreg * wa_et, dreg * det * (1.0 - p_home_pen)

    return {
        "p_extra_time": dreg,                 # vai à prorrogação
        "p_penalties": dreg * det,            # vai aos pênaltis
        "decided_90": 1.0 - dreg,             # decidido no tempo normal
        "decided_et": dreg * (1.0 - det),     # decidido na prorrogação
        "decided_pen": dreg * det,            # decidido nos pênaltis
        "home_advance": home_90 + home_et + home_pen,
        "away_advance": away_90 + away_et + away_pen,
        "home_pen_win": p_home_pen,
        "away_pen_win": 1.0 - p_home_pen,
        "home_breakdown": (home_90, home_et, home_pen),
        "away_breakdown": (away_90, away_et, away_pen),
    }
