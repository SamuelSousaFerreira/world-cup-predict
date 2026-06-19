"""Força de elenco a partir de dados do Transfermarkt (snapshot atual).

Fonte: dcaribou/transfermarkt-datasets (licença CC0), tabela national_teams
com valor de mercado total do elenco, idade média e ranking FIFA por seleção.

LIMITAÇÃO IMPORTANTE (honestidade científica):
    O Transfermarkt fornece um SNAPSHOT ATUAL (não histórico). Aplicamos o
    mesmo valor de elenco a todas as partidas de uma seleção. Como o treino
    usa decaimento temporal (meia-vida de ~3 anos), apenas os jogos recentes
    pesam de fato — e para esses o snapshot é uma boa aproximação. Para o
    bloco de teste (partidas mais recentes), a avaliação é justa.

    Cobertura: 118 seleções. Faltam algumas reais (ex.: Ivory Coast, Cameroon,
    Mali, Cape Verde). Times sem cobertura recebem features NaN — o
    HistGradientBoosting trata NaN nativamente (vira "informação ausente").
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SQUAD_DIR = PROJECT_ROOT / "data" / "squad"
SQUAD_FILE = SQUAD_DIR / "national_teams.csv.gz"
SQUAD_URL = (
    "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/national_teams.csv.gz"
)

# Features de elenco adicionadas ao MLModel (diferenças mandante - visitante).
SQUAD_FEATURES = ["squad_value_diff", "squad_age_diff", "fifa_rank_diff"]

# Nomes que diferem entre a base de resultados (martj42) e o Transfermarkt.
# (nosso_nome -> nome_no_transfermarkt)
NAME_MAP = {
    "Turkey": "Turkiye",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "United States": "United States",
    "South Korea": "South Korea",
    "North Macedonia": "North Macedonia",
    "Republic of Ireland": "Republic of Ireland",
    "Ireland": "Republic of Ireland",
    "Czechia": "Czech Republic",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Cabo Verde": "Cape Verde Islands",
    "USA": "United States",
}


def download(force: bool = False) -> Path:
    """Baixa a tabela national_teams do Transfermarkt (cacheia em data/squad)."""
    if SQUAD_FILE.exists() and not force:
        return SQUAD_FILE
    import requests

    SQUAD_DIR.mkdir(parents=True, exist_ok=True)
    resp = requests.get(SQUAD_URL, timeout=120)
    resp.raise_for_status()
    SQUAD_FILE.write_bytes(resp.content)
    return SQUAD_FILE


def _impute_market_value(nt: pd.DataFrame) -> pd.DataFrame:
    """Imputa total_market_value ausente (ex.: England/France/Spain) a partir
    do ranking FIFA, via regressão log-linear sobre as seleções conhecidas."""
    nt = nt.copy()
    known = nt[nt["total_market_value"].notna() & nt["fifa_ranking"].notna()]
    if len(known) >= 10:
        x = known["fifa_ranking"].to_numpy(dtype=float)
        y = np.log1p(known["total_market_value"].to_numpy(dtype=float))
        # log(valor) ~ a + b * log(rank)  (rank melhor => valor maior)
        lx = np.log1p(x)
        b, a = np.polyfit(lx, y, 1)
        miss = nt["total_market_value"].isna() & nt["fifa_ranking"].notna()
        pred = np.expm1(a + b * np.log1p(nt.loc[miss, "fifa_ranking"].to_numpy(dtype=float)))
        nt.loc[miss, "total_market_value"] = pred
    return nt


def load_squad_table() -> dict[str, dict]:
    """Retorna lookup {nome_transfermarkt: {value, age, fifa_rank}} já imputado."""
    download()
    nt = pd.read_csv(SQUAD_FILE)
    nt = _impute_market_value(nt)
    out: dict[str, dict] = {}
    for _, r in nt.iterrows():
        out[r["name"]] = {
            "value": float(r["total_market_value"]) if pd.notna(r["total_market_value"]) else np.nan,
            "age": float(r["average_age"]) if pd.notna(r["average_age"]) else np.nan,
            "fifa_rank": float(r["fifa_ranking"]) if pd.notna(r["fifa_ranking"]) else np.nan,
        }
    return out


def squad_lookup(team: str, table: dict[str, dict]) -> dict | None:
    """Resolve a seleção no Transfermarkt (mapa de nomes + identidade)."""
    tm_name = NAME_MAP.get(team, team)
    return table.get(tm_name)


def squad_diffs(home: str, away: str, table: dict[str, dict]) -> dict[str, float]:
    """Features de diferença mandante-visitante. NaN se faltar cobertura."""
    h = squad_lookup(home, table)
    a = squad_lookup(away, table)
    if h is None or a is None:
        return {f: np.nan for f in SQUAD_FEATURES}
    value_diff = np.log1p(h["value"]) - np.log1p(a["value"])
    age_diff = h["age"] - a["age"]
    # rank menor é melhor; positivo => mandante melhor ranqueado
    rank_diff = a["fifa_rank"] - h["fifa_rank"]
    return {
        "squad_value_diff": value_diff,
        "squad_age_diff": age_diff,
        "fifa_rank_diff": rank_diff,
    }


def add_squad_features(df: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta as colunas de força de elenco ao DataFrame de treino."""
    table = load_squad_table()
    diffs = [squad_diffs(h, a, table) for h, a in zip(df["home_team"], df["away_team"])]
    sq = pd.DataFrame(diffs, index=df.index)
    out = df.copy()
    for col in SQUAD_FEATURES:
        out[col] = sq[col]
    return out


def coverage_report(team_names) -> tuple[int, int, list[str]]:
    """Quantos times do conjunto têm cobertura; retorna (cobertos, total, faltantes)."""
    table = load_squad_table()
    missing = [t for t in team_names if squad_lookup(t, table) is None]
    return len(team_names) - len(missing), len(team_names), missing
