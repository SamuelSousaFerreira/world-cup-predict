"""Previsão de um confronto com probabilidades.

Uso:
    python src/predict.py "Brazil" "Argentina"
    python src/predict.py "Brazil" "France" --neutral
    python src/predict.py "Portugal" "Spain" --importance 1.0

Carrega os modelos treinados e o estado atual das seleções, monta as
features do confronto e devolve:
- probabilidades de cada modelo (Elo, Poisson, ML)
- probabilidade do ensemble (média)
- placar mais provável (modelo Poisson)
- alerta de DIVERGÊNCIA quando os modelos discordam do vencedor
  (cabe ao humano decidir).
"""
from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path

import joblib
import numpy as np
from tabulate import tabulate

from feature_engineering import HOME_ADVANTAGE, load_team_state
from models import CLASSES, combine
from squad_data import load_squad_table, squad_diffs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


def load_models() -> dict:
    needed = ["elo_model.joblib", "poisson_model.joblib", "ml_model.joblib"]
    missing = [n for n in needed if not (MODELS_DIR / n).exists()]
    if missing:
        raise SystemExit(
            f"Modelos não encontrados: {missing}\n"
            f"Rode primeiro: python src/train.py"
        )
    return {
        "Elo": joblib.load(MODELS_DIR / "elo_model.joblib"),
        "Poisson": joblib.load(MODELS_DIR / "poisson_model.joblib"),
        "ML": joblib.load(MODELS_DIR / "ml_model.joblib"),
    }


def load_weights() -> dict | None:
    """Carrega os pesos otimizados do ensemble (se existirem)."""
    p = MODELS_DIR / "ensemble_weights.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def resolve_team(name: str, state: dict) -> str:
    """Resolve o nome da seleção (tolerante a maiúsculas/erros leves)."""
    if name in state:
        return name
    lower = {k.lower(): k for k in state}
    if name.lower() in lower:
        return lower[name.lower()]
    match = difflib.get_close_matches(name, list(state), n=1, cutoff=0.6)
    if match:
        print(f"[info] '{name}' interpretado como '{match[0]}'")
        return match[0]
    raise SystemExit(f"Seleção '{name}' não encontrada na base. Verifique o nome em inglês.")


def build_matchup_features(home: str, away: str, state: dict,
                           neutral: bool, importance: float) -> dict:
    h, a = state[home], state[away]
    ha = 0.0 if neutral else HOME_ADVANTAGE
    feats = {
        "elo_home": h["elo"],
        "elo_away": a["elo"],
        "elo_diff": h["elo"] + ha - a["elo"],
        "home_form": h["form"],
        "away_form": a["form"],
        "home_attack": h["attack"],
        "away_attack": a["attack"],
        "home_defense": h["defense"],
        "away_defense": a["defense"],
        "home_style": h["style"],
        "away_style": a["style"],
        "home_aggression": h["aggression"],
        "away_aggression": a["aggression"],
        "neutral": 1 if neutral else 0,
        "importance": importance,
    }
    # Força de elenco (Transfermarkt). NaN quando não há cobertura — o
    # MLModel lida com isso nativamente.
    feats.update(squad_diffs(home, away, load_squad_table()))
    return feats


def fmt_pct(p: np.ndarray) -> list[str]:
    return [f"{x * 100:5.1f}%" for x in p]


def predict(home: str, away: str, neutral: bool = False, importance: float = 0.85) -> None:
    state = load_team_state()
    models = load_models()
    home = resolve_team(home, state)
    away = resolve_team(away, state)
    feats = build_matchup_features(home, away, state, neutral, importance)

    probs = {name: m.predict_proba(feats) for name, m in models.items()}
    score = models["Poisson"].most_likely_score(feats)
    weights = load_weights()
    result = combine(probs, score, weights=weights)

    venue = "campo neutro" if neutral else f"{home} como mandante"
    print(f"\n{'=' * 56}")
    print(f"  {home}  x  {away}   ({venue})")
    print(f"{'=' * 56}")

    header = ["Modelo", f"{home} (V)", "Empate", f"{away} (V)"]
    rows = [[name, *fmt_pct(p)] for name, p in probs.items()]
    rows.append(["—" * 6, "—" * 6, "—" * 6, "—" * 6])
    rows.append(["ENSEMBLE", *fmt_pct(result.ensemble)])
    print("\n" + tabulate(rows, headers=header, tablefmt="github", stralign="right"))

    gh, ga, p = result.most_likely_score
    print(f"\nPlacar mais provável (Poisson): {home} {gh} x {ga} {away}  (p={p * 100:.1f}%)")

    fav_idx = int(np.argmax(result.ensemble))
    favs = [f"{home} vencer", "empate", f"{away} vencer"]
    print(f"Resultado favorito (ensemble): {favs[fav_idx]} "
          f"({result.ensemble[fav_idx] * 100:.1f}%)")

    print(f"\nForças atuais:")
    print(f"  {home:<22} Elo {state[home]['elo']:.0f} | forma {state[home]['form']*100:.0f}% | "
          f"ataque {state[home]['attack']:.2f} | defesa {state[home]['defense']:.2f}")
    print(f"  {away:<22} Elo {state[away]['elo']:.0f} | forma {state[away]['form']*100:.0f}% | "
          f"ataque {state[away]['attack']:.2f} | defesa {state[away]['defense']:.2f}")

    if result.diverges:
        picks = {n: CLASSES[int(np.argmax(p))] for n, p in probs.items()}
        legend = {"H": f"{home}", "D": "empate", "A": f"{away}"}
        print(f"\n⚠️  DIVERGÊNCIA entre modelos — decisão humana recomendada:")
        for n, c in picks.items():
            print(f"     - {n}: aposta em {legend[c]}")
        print(f"   (dispersão entre modelos: {result.disagreement*100:.1f} pontos percentuais)")
    else:
        print("\n✓ Modelos concordam no favorito.")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Previsão de confronto da Copa do Mundo.")
    ap.add_argument("home", help="Seleção mandante (nome em inglês, ex: Brazil)")
    ap.add_argument("away", help="Seleção visitante (nome em inglês, ex: Argentina)")
    ap.add_argument("--neutral", action="store_true", help="Jogo em campo neutro")
    ap.add_argument("--importance", type=float, default=0.85,
                    help="Peso do torneio 0..1 (Copa do Mundo=1.0, padrão=0.85)")
    args = ap.parse_args()
    predict(args.home, args.away, neutral=args.neutral, importance=args.importance)


if __name__ == "__main__":
    main()
