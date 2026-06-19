"""Engenharia de features para previsão de partidas.

Processa as partidas em ordem cronológica mantendo, para cada seleção,
um estado incremental:

- Rating Elo (com vantagem de mando e multiplicador por saldo de gols)
- Janela dos últimos N jogos (gols pró/contra/resultado)

Para cada partida registra as features PRÉ-jogo (sem vazamento de dados):

    elo_home, elo_away, elo_diff
    home_form, away_form            -> pontos nos últimos N jogos (3/1/0) / N*3
    home_attack, away_attack        -> média de gols marcados (últimos N)
    home_defense, away_defense      -> média de gols sofridos (últimos N)
    home_style, away_style          -> "tempo": média de gols totais por jogo (ritmo)
    home_aggression, away_aggression-> saldo médio (ataque - defesa): estilo ofensivo/defensivo
    neutral                         -> mando neutro (1/0)
    importance                      -> peso do torneio (Copa do Mundo > eliminatória > amistoso)

Alvo:
    result -> 'H' (mandante vence), 'D' (empate), 'A' (visitante vence)
    home_score, away_score

Ao final salva também o ESTADO ATUAL de cada seleção (data/processed/team_state.json),
usado para prever confrontos futuros.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from data_collection import load_results

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# ----------------------------- Parâmetros Elo ------------------------------ #
ELO_START = 1500.0          # rating inicial de uma seleção nova
ELO_K = 40.0                # velocidade de atualização
HOME_ADVANTAGE = 65.0       # vantagem de mando em pontos de Elo
FORM_WINDOW = 10            # "últimos 10 jogos" pedidos pelo usuário

# Peso por importância do torneio (afeta o K do Elo e vira feature).
TOURNAMENT_WEIGHT = {
    "FIFA World Cup": 1.00,
    "FIFA World Cup qualification": 0.85,
    "UEFA Euro": 0.90,
    "UEFA Euro qualification": 0.75,
    "Copa América": 0.90,
    "African Cup of Nations": 0.85,
    "AFC Asian Cup": 0.80,
    "UEFA Nations League": 0.80,
    "Confederations Cup": 0.85,
    "Friendly": 0.45,
}
DEFAULT_WEIGHT = 0.65


def tournament_weight(name: str) -> float:
    return TOURNAMENT_WEIGHT.get(name, DEFAULT_WEIGHT)


def expected_score(elo_a: float, elo_b: float) -> float:
    """Expectativa de pontuação (probabilidade de vitória + meio empate) de A."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def goal_diff_multiplier(goal_diff: int) -> float:
    """Multiplicador do K conforme o saldo de gols (modelo World Football Elo)."""
    g = abs(goal_diff)
    if g <= 1:
        return 1.0
    if g == 2:
        return 1.5
    return (11 + g) / 8.0


# ------------------------ Parâmetros das novas features --------------------- #
EWMA_HALFLIFE_GAMES = 3.0   # meia-vida (em jogos) p/ recência dentro da janela
SOS_SCALE = 100.0           # normalização da força de calendário (Elo -> ~[-3,3])
STRENGTH_LO, STRENGTH_HI = 0.5, 1.7   # limites do fator de força do oponente
REST_CAP_DAYS = 180         # teto p/ dias de descanso (evita outliers de retorno)


def _ewma(values, halflife: float = EWMA_HALFLIFE_GAMES) -> float:
    """Média exponencial: o jogo mais recente (fim da lista) pesa mais."""
    n = len(values)
    if n == 0:
        return float("nan")
    arr = np.asarray(values, dtype=float)
    ages = np.arange(n - 1, -1, -1)          # mais antigo = maior idade
    w = 0.5 ** (ages / halflife)
    return float(np.sum(arr * w) / np.sum(w))


def _strength_factor(opp_elo) -> np.ndarray:
    """Fator de força do oponente (1.0 = mediano). Marcar contra time forte
    vale mais; sofrer contra time forte 'pesa' menos."""
    arr = np.asarray(opp_elo, dtype=float)
    return np.clip(1.0 + (arr - ELO_START) / 600.0, STRENGTH_LO, STRENGTH_HI)


def _streak(points) -> int:
    """Sequência atual com sinal: +n vitórias seguidas, -n derrotas, 0 se empate."""
    if not points:
        return 0
    last = points[-1]
    if last == 1:
        return 0
    sign = 1 if last == 3 else -1
    run = 0
    for p in reversed(points):
        if (sign == 1 and p == 3) or (sign == -1 and p == 0):
            run += 1
        else:
            break
    return sign * run



@dataclass
class TeamState:
    """Estado incremental de uma seleção."""
    elo: float = ELO_START
    goals_for: deque = field(default_factory=lambda: deque(maxlen=FORM_WINDOW))
    goals_against: deque = field(default_factory=lambda: deque(maxlen=FORM_WINDOW))
    points: deque = field(default_factory=lambda: deque(maxlen=FORM_WINDOW))
    opp_elo: deque = field(default_factory=lambda: deque(maxlen=FORM_WINDOW))
    exp_score: deque = field(default_factory=lambda: deque(maxlen=FORM_WINDOW))
    dates: deque = field(default_factory=lambda: deque(maxlen=FORM_WINDOW))
    last_date: object = None
    games: int = 0

    def features(self) -> dict[str, float]:
        n = len(self.points)
        if n == 0:
            return {
                "elo": self.elo,
                "form": 0.5,
                "attack": 1.2,
                "defense": 1.2,
                "style": 2.4,
                "aggression": 0.0,
                # novas
                "ewma_form": 0.5,
                "adj_attack": 1.2,
                "adj_defense": 1.2,
                "sos": 0.0,
                "streak": 0,
                "games": 0,
            }
        gf = np.mean(self.goals_for)
        ga = np.mean(self.goals_against)
        factor = _strength_factor(self.opp_elo)
        gf_arr = np.asarray(self.goals_for, dtype=float)
        ga_arr = np.asarray(self.goals_against, dtype=float)
        # EWMA da forma (pontos) com mais peso aos jogos recentes.
        pts_frac = [p / 3.0 for p in self.points]
        return {
            "elo": self.elo,
            "form": sum(self.points) / (n * 3.0),     # 0..1 (média simples)
            "attack": float(gf),                       # gols marcados/jogo
            "defense": float(ga),                      # gols sofridos/jogo
            "style": float(gf + ga),                   # ritmo/tempo de jogo
            "aggression": float(gf - ga),              # ofensivo(+)/defensivo(-)
            # ---- novas features ----
            "ewma_form": _ewma(pts_frac),              # forma recente (EWMA)
            "adj_attack": _ewma(gf_arr * factor),      # ataque ajustado p/ oponente
            "adj_defense": _ewma(ga_arr / factor),     # defesa ajustada p/ oponente
            "sos": float((np.mean(self.opp_elo) - ELO_START) / SOS_SCALE),  # força do calendário
            "streak": _streak(self.points),            # sequência com sinal
            "games": n,
        }

    def update(self, gf: int, ga: int, opp_elo: float = ELO_START,
               exp_score: float = 0.5, date=None) -> None:
        self.goals_for.append(gf)
        self.goals_against.append(ga)
        self.opp_elo.append(opp_elo)
        self.exp_score.append(exp_score)
        if date is not None:
            self.dates.append(date)
            self.last_date = date
        if gf > ga:
            self.points.append(3)
        elif gf == ga:
            self.points.append(1)
        else:
            self.points.append(0)
        self.games += 1



def build_features(df: pd.DataFrame, min_date: str | None = "1990-01-01") -> tuple[pd.DataFrame, dict]:
    """Constrói a tabela de treino e o estado final de cada seleção.

    Args:
        df: partidas (de load_results), já ordenadas por data.
        min_date: descarta linhas de treino anteriores a esta data (mantém o
            histórico só para "aquecer" o Elo). None = usa tudo.

    Returns:
        (training_df, team_state) onde team_state é {team: features_atuais}.
    """
    states: dict[str, TeamState] = {}
    rows: list[dict] = []
    cutoff = pd.Timestamp(min_date) if min_date else None

    for r in df.itertuples(index=False):
        home, away = r.home_team, r.away_team
        sh = states.setdefault(home, TeamState())
        sa = states.setdefault(away, TeamState())

        fh, fa = sh.features(), sa.features()
        neutral = 1 if bool(getattr(r, "neutral", False)) else 0
        weight = tournament_weight(r.tournament)

        # Vantagem de mando só conta se não for campo neutro.
        ha = 0.0 if neutral else HOME_ADVANTAGE

        if cutoff is None or r.date >= cutoff:
            if r.home_score > r.away_score:
                result = "H"
            elif r.home_score == r.away_score:
                result = "D"
            else:
                result = "A"
            # Descanso e amplitude da janela (contexto/fadiga), com a data atual.
            h_rest = min((r.date - sh.last_date).days, REST_CAP_DAYS) if sh.last_date is not None else np.nan
            a_rest = min((r.date - sa.last_date).days, REST_CAP_DAYS) if sa.last_date is not None else np.nan
            h_win = (r.date - sh.dates[0]).days if len(sh.dates) > 0 else np.nan
            a_win = (r.date - sa.dates[0]).days if len(sa.dates) > 0 else np.nan
            rows.append({
                "date": r.date,
                "home_team": home,
                "away_team": away,
                "tournament": r.tournament,
                "neutral": neutral,
                "importance": weight,
                "elo_home": fh["elo"],
                "elo_away": fa["elo"],
                "elo_diff": fh["elo"] + ha - fa["elo"],
                "home_form": fh["form"],
                "away_form": fa["form"],
                "home_attack": fh["attack"],
                "away_attack": fa["attack"],
                "home_defense": fh["defense"],
                "away_defense": fa["defense"],
                "home_style": fh["style"],
                "away_style": fa["style"],
                "home_aggression": fh["aggression"],
                "away_aggression": fa["aggression"],
                # ---- novas features ----
                "home_ewma_form": fh["ewma_form"],
                "away_ewma_form": fa["ewma_form"],
                "home_adj_attack": fh["adj_attack"],
                "away_adj_attack": fa["adj_attack"],
                "home_adj_defense": fh["adj_defense"],
                "away_adj_defense": fa["adj_defense"],
                "home_sos": fh["sos"],
                "away_sos": fa["sos"],
                "home_streak": fh["streak"],
                "away_streak": fa["streak"],
                "home_rest_days": h_rest,
                "away_rest_days": a_rest,
                "home_window_days": h_win,
                "away_window_days": a_win,
                "home_games": fh["games"],
                "away_games": fa["games"],
                "home_score": r.home_score,
                "away_score": r.away_score,
                "result": result,
            })

        # ---- Atualiza Elo (após registrar as features pré-jogo) ----
        exp_home = expected_score(fh["elo"] + ha, fa["elo"])
        score_home = 1.0 if r.home_score > r.away_score else (0.5 if r.home_score == r.away_score else 0.0)
        k = ELO_K * weight * goal_diff_multiplier(r.home_score - r.away_score)
        delta = k * (score_home - exp_home)
        sh.elo += delta
        sa.elo -= delta

        sh.update(r.home_score, r.away_score, opp_elo=fa["elo"], exp_score=exp_home, date=r.date)
        sa.update(r.away_score, r.home_score, opp_elo=fh["elo"], exp_score=1.0 - exp_home, date=r.date)


    training_df = pd.DataFrame(rows)
    # Estado atual exportável.
    team_state = {team: st.features() | {
        "last_goals_for": list(st.goals_for),
        "last_goals_against": list(st.goals_against),
        "last_date": st.last_date.isoformat() if st.last_date is not None else None,
        "window_start": st.dates[0].isoformat() if len(st.dates) > 0 else None,
    } for team, st in states.items()}
    return training_df, team_state


def save(training_df: pd.DataFrame, team_state: dict) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    training_df.to_parquet(PROCESSED_DIR / "training_data.parquet", index=False) \
        if _has_parquet() else training_df.to_csv(PROCESSED_DIR / "training_data.csv", index=False)
    with open(PROCESSED_DIR / "team_state.json", "w", encoding="utf-8") as f:
        json.dump(team_state, f, ensure_ascii=False, indent=2)
    print(f"[ok] {len(training_df):,} partidas de treino e {len(team_state)} seleções salvas em {PROCESSED_DIR}")


def _has_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def load_training() -> pd.DataFrame:
    p_parquet = PROCESSED_DIR / "training_data.parquet"
    p_csv = PROCESSED_DIR / "training_data.csv"
    if p_parquet.exists():
        return pd.read_parquet(p_parquet)
    return pd.read_csv(p_csv, parse_dates=["date"])


def load_team_state() -> dict:
    with open(PROCESSED_DIR / "team_state.json", encoding="utf-8") as f:
        return json.load(f)


def recent_matches(team: str, n: int = 5) -> list[dict]:
    """Retorna os últimos *n* jogos oficiais de *team* (ordem cronológica, do mais antigo ao mais recente).

    Cada dict contém: date (str), opponent (str), gf (int), ga (int),
    result ("W"/"D"/"L" da perspectiva do time), tournament (str), neutral (bool).
    """
    df = load_results()
    mask = (df["home_team"] == team) | (df["away_team"] == team)
    sub = df[mask].sort_values("date", ascending=False).head(n)
    records = []
    for r in sub.itertuples(index=False):
        is_home = r.home_team == team
        gf = int(r.home_score) if is_home else int(r.away_score)
        ga = int(r.away_score) if is_home else int(r.home_score)
        opponent = r.away_team if is_home else r.home_team
        if gf > ga:
            result = "W"
        elif gf == ga:
            result = "D"
        else:
            result = "L"
        records.append({
            "date": pd.Timestamp(r.date).strftime("%d/%m/%Y"),
            "opponent": opponent,
            "gf": gf,
            "ga": ga,
            "result": result,
            "tournament": r.tournament,
            "neutral": bool(r.neutral),
        })
    return records


if __name__ == "__main__":
    matches = load_results()
    train, state = build_features(matches)
    save(train, state)
    print("\nDistribuição de resultados (mandante):")
    print(train["result"].value_counts(normalize=True).round(3).to_string())
