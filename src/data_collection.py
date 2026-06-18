"""Coleta de dados de partidas internacionais de futebol.

Fonte: martj42/international_results (licença CC0).
Contém ~49 mil partidas oficiais de seleções masculinas desde 1872,
atualizada de hora em hora. Inclui Copa do Mundo, eliminatórias,
amistosos, etc.

Baixa e armazena em cache:
- results.csv   -> data/raw/results.csv
- shootouts.csv -> data/raw/shootouts.csv  (vencedor de disputas de pênaltis)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import requests

RAW_BASE = "https://raw.githubusercontent.com/martj42/international_results/master"
FILES = {
    "results.csv": f"{RAW_BASE}/results.csv",
    "shootouts.csv": f"{RAW_BASE}/shootouts.csv",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def download(force: bool = False) -> dict[str, Path]:
    """Baixa os CSVs da fonte pública e salva em data/raw/.

    Args:
        force: se True, rebaixa mesmo que o arquivo já exista em cache.

    Returns:
        Dicionário {nome_arquivo: caminho_local}.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, url in FILES.items():
        dest = RAW_DIR / name
        if dest.exists() and not force:
            print(f"[cache] {name} já existe em {dest}")
            paths[name] = dest
            continue
        print(f"[download] baixando {name} de {url} ...")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        print(f"[ok] salvo em {dest} ({len(resp.content) / 1024:.0f} KB)")
        paths[name] = dest
    return paths


def load_results(force_download: bool = False) -> pd.DataFrame:
    """Carrega results.csv como DataFrame com a coluna `date` convertida."""
    paths = download(force=force_download)
    df = pd.read_csv(paths["results.csv"], parse_dates=["date"])
    # Remove partidas sem placar (raras) e ordena cronologicamente.
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df = df.sort_values("date").reset_index(drop=True)
    return df


if __name__ == "__main__":
    force = "--force" in sys.argv
    df = load_results(force_download=force)
    print(f"\nTotal de partidas: {len(df):,}")
    print(f"Período: {df['date'].min().date()} a {df['date'].max().date()}")
    print(f"Seleções distintas: {pd.unique(df[['home_team', 'away_team']].values.ravel()).size}")
    print("\nÚltimas 5 partidas:")
    print(df.tail(5).to_string(index=False))
