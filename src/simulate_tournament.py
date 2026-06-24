"""Simulação de Monte Carlo de um torneio (mata-mata).

Usa os modelos treinados para simular milhares de vezes um chaveamento de
eliminatória simples e estimar a probabilidade de cada seleção ser campeã.

Como cada partida é incerta, repetimos a simulação muitas vezes (Monte Carlo)
e contamos com que frequência cada time levanta a taça.

Uso:
    # bracket explícito (potência de 2: 4, 8, 16, 32 times), em ordem do chaveamento
    python src/simulate_tournament.py "Brazil" "Croatia" "Netherlands" "Argentina" ...

    # sem argumentos: usa as 16 seleções de maior Elo como demonstração
    python src/simulate_tournament.py --n 20000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from tabulate import tabulate

from feature_engineering import HOME_ADVANTAGE, REST_CAP_DAYS, load_team_state
from squad_data import load_squad_table, squad_diffs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


def load_models() -> dict:
    needed = ["elo_model.joblib", "poisson_model.joblib", "ml_model.joblib"]
    if any(not (MODELS_DIR / n).exists() for n in needed):
        raise SystemExit("Modelos não encontrados. Rode primeiro: python src/train.py")
    models = {
        "Elo": joblib.load(MODELS_DIR / "elo_model.joblib"),
        "Poisson": joblib.load(MODELS_DIR / "poisson_model.joblib"),
        "ML": joblib.load(MODELS_DIR / "ml_model.joblib"),
    }
    for name, fname in (("CatBoost", "catboost_model.joblib"),
                        ("LightGBM", "lightgbm_model.joblib"),
                        ("XGBoost", "xgboost_model.joblib")):
        p = MODELS_DIR / fname
        if p.exists():
            models[name] = joblib.load(p)
    return models


def load_weights() -> dict | None:
    p = MODELS_DIR / "ensemble_weights.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


class MatchPredictor:
    """Cacheia as probabilidades [P(A vence), P(empate), P(B vence)] por confronto."""

    def __init__(self, state: dict, importance: float = 1.0) -> None:
        self.state = state
        self.models = load_models()
        self.weights = load_weights()
        self.importance = importance
        self.squad_table = load_squad_table()
        self._cache: dict[tuple[str, str], np.ndarray] = {}

    def _features(self, a: str, b: str) -> dict:
        # Torneio em campo neutro: sem vantagem de mando.
        h, g = self.state[a], self.state[b]
        feats = {
            "elo_home": h["elo"], "elo_away": g["elo"],
            "elo_diff": h["elo"] - g["elo"],
            "home_form": h["form"], "away_form": g["form"],
            "home_attack": h["attack"], "away_attack": g["attack"],
            "home_defense": h["defense"], "away_defense": g["defense"],
            "home_style": h["style"], "away_style": g["style"],
            "home_aggression": h["aggression"], "away_aggression": g["aggression"],
            "neutral": 1, "importance": self.importance,
            # ---- features avançadas (v2), disponíveis no team_state ----
            "home_ewma_form": h.get("ewma_form", h["form"]),
            "away_ewma_form": g.get("ewma_form", g["form"]),
            "home_adj_attack": h.get("adj_attack", h["attack"]),
            "away_adj_attack": g.get("adj_attack", g["attack"]),
            "home_adj_defense": h.get("adj_defense", h["defense"]),
            "away_adj_defense": g.get("adj_defense", g["defense"]),
            "home_sos": h.get("sos", 0.0), "away_sos": g.get("sos", 0.0),
            "home_streak": h.get("streak", 0), "away_streak": g.get("streak", 0),
            # descanso/janela indisponíveis em simulação hipotética -> NaN.
            "home_rest_days": np.nan, "away_rest_days": np.nan,
            "home_window_days": np.nan, "away_window_days": np.nan,
        }
        feats.update(squad_diffs(a, b, self.squad_table))
        return feats

    def probs(self, a: str, b: str) -> np.ndarray:
        key = (a, b)
        if key in self._cache:
            return self._cache[key]
        feats = self._features(a, b)
        per_model = {n: m.predict_proba(feats) for n, m in self.models.items()}
        names = list(per_model)
        stacked = np.vstack([per_model[n] for n in names])
        if self.weights:
            w = np.array([self.weights.get(n, 0.0) for n in names])
            w = w / w.sum() if w.sum() > 0 else np.ones(len(names)) / len(names)
        else:
            w = np.ones(len(names)) / len(names)
        ens = (stacked * w[:, None]).sum(axis=0)
        ens = ens / ens.sum()
        self._cache[key] = ens
        return ens

    def knockout_winprob(self, a: str, b: str) -> float:
        """P(a passa de fase) — redistribui o empate por pênaltis."""
        p = self.probs(a, b)
        p_home, p_draw, p_away = p
        # Empate decidido nos pênaltis: ~ proporcional à chance de vitória,
        # mas mais perto de uma moeda (pênaltis nivelam).
        base = p_home / (p_home + p_away) if (p_home + p_away) > 0 else 0.5
        pen = 0.5 + (base - 0.5) * 0.5
        return float(p_home + p_draw * pen)


def simulate_knockout(teams: list[str], predictor: MatchPredictor,
                      rng: np.random.Generator) -> str:
    """Uma simulação de eliminatória simples. Retorna o campeão."""
    alive = list(teams)
    while len(alive) > 1:
        nxt = []
        for i in range(0, len(alive), 2):
            a, b = alive[i], alive[i + 1]
            p = predictor.knockout_winprob(a, b)
            nxt.append(a if rng.random() < p else b)
        alive = nxt
    return alive[0]


def run(teams: list[str], n_sims: int, seed: int = 42) -> list[tuple[str, float]]:
    state = load_team_state()
    # valida nomes
    missing = [t for t in teams if t not in state]
    if missing:
        raise SystemExit(f"Seleções não encontradas (use nomes em inglês): {missing}")
    predictor = MatchPredictor(state)
    rng = np.random.default_rng(seed)
    titles = {t: 0 for t in teams}
    for _ in range(n_sims):
        champ = simulate_knockout(teams, predictor, rng)
        titles[champ] += 1
    out = sorted(((t, c / n_sims) for t, c in titles.items()), key=lambda x: -x[1])
    return out


def run_with_reach(teams: list[str], n_sims: int, seed: int = 42
                   ) -> tuple[list[tuple[str, float]], dict[str, list[float]], int]:
    """Como run(), mas também devolve a probabilidade de cada seleção alcançar
    cada fase (para o diagrama de Sankey).

    Retorna ``(titulo, reach, n_stages)`` onde ``reach[time]`` é uma lista de
    tamanho ``n_stages`` com P(alcançar a fase s): a fase 0 é a rodada inicial
    (sempre 1.0) e a última é o título. A soma de ``reach[*][s]`` é o número de
    seleções vivas naquela fase (n, n/2, ..., 1).
    """
    state = load_team_state()
    missing = [t for t in teams if t not in state]
    if missing:
        raise SystemExit(f"Seleções não encontradas (use nomes em inglês): {missing}")
    predictor = MatchPredictor(state)
    rng = np.random.default_rng(seed)
    n_stages = len(teams).bit_length()              # log2(n) + 1
    counts = {t: np.zeros(n_stages) for t in teams}
    for _ in range(n_sims):
        alive = list(teams)
        stage = 0
        for t in alive:
            counts[t][stage] += 1
        while len(alive) > 1:
            nxt = []
            for i in range(0, len(alive), 2):
                a, b = alive[i], alive[i + 1]
                p = predictor.knockout_winprob(a, b)
                nxt.append(a if rng.random() < p else b)
            alive = nxt
            stage += 1
            for t in alive:
                counts[t][stage] += 1
    reach = {t: (counts[t] / n_sims).tolist() for t in teams}
    title = sorted(((t, reach[t][-1]) for t in teams), key=lambda x: -x[1])
    return title, reach, n_stages


def default_bracket(size: int = 16) -> list[str]:
    """Top-N seleções por Elo, semeadas (1 vs N, 2 vs N-1, ...)."""
    state = load_team_state()
    ranked = sorted(state.items(), key=lambda kv: -kv[1]["elo"])
    top = [t for t, _ in ranked[:size]]
    # Semeadura: melhores não se enfrentam cedo.
    seeded = [None] * size
    lo, hi = 0, size - 1
    for i, team in enumerate(top):
        if i % 2 == 0:
            seeded[lo] = team
            lo += 1
        else:
            seeded[hi] = team
            hi -= 1
    return seeded


def main() -> None:
    ap = argparse.ArgumentParser(description="Simulação Monte Carlo de mata-mata.")
    ap.add_argument("teams", nargs="*", help="Seleções em ordem de chaveamento (potência de 2)")
    ap.add_argument("--n", type=int, default=20000, help="Número de simulações")
    ap.add_argument("--size", type=int, default=16, help="Tamanho do bracket padrão (se sem times)")
    args = ap.parse_args()

    teams = args.teams
    if not teams:
        teams = default_bracket(args.size)
        print(f"[info] Nenhum time informado — usando as {len(teams)} seleções de maior Elo.")

    if len(teams) & (len(teams) - 1) != 0:
        raise SystemExit(f"O número de times deve ser potência de 2 (recebido: {len(teams)}).")

    print(f"\nSimulando {args.n:,} torneios com {len(teams)} seleções...\n")
    results = run(teams, args.n)

    rows = [[i + 1, t, f"{p * 100:5.1f}%"] for i, (t, p) in enumerate(results)]
    print(tabulate(rows, headers=["#", "Seleção", "P(título)"], tablefmt="github"))
    print(f"\nFavorito ao título: {results[0][0]} ({results[0][1] * 100:.1f}%)")


if __name__ == "__main__":
    main()
