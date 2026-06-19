"""Interface Streamlit do previsor da Copa do Mundo.

Duas abas:
  1) Previsão de jogo único — probabilidades por modelo + ensemble, placares
     mais prováveis, forças atuais e comparação opcional com o mercado.
  2) Simulação de mata-mata (Monte Carlo) — probabilidade de título por seleção.

Roda com:
    .\\venv\\Scripts\\streamlit.exe run app.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

from feature_engineering import load_team_state  # noqa: E402
from predict import compute_prediction, load_models, load_weights  # noqa: E402
from squad_data import load_squad_table  # noqa: E402
from simulate_tournament import (MatchPredictor, default_bracket,  # noqa: E402
                                 run as run_tournament)

st.set_page_config(page_title="Previsor Copa do Mundo", page_icon="⚽", layout="wide")



# --------------------------- Cache de recursos ------------------------------ #
@st.cache_resource(show_spinner=False)
def get_state() -> dict:
    return load_team_state()


@st.cache_resource(show_spinner=False)
def get_models() -> dict:
    return load_models()


@st.cache_resource(show_spinner=False)
def get_weights() -> dict | None:
    return load_weights()


@st.cache_resource(show_spinner=False)
def get_squad_table() -> dict:
    return load_squad_table()


def team_names(state: dict) -> list[str]:
    # Ordena por Elo (mais fortes primeiro) para facilitar a escolha.
    return [t for t, _ in sorted(state.items(), key=lambda kv: -kv[1]["elo"])]


def devig(odds: list[float]) -> np.ndarray | None:
    """Converte odds decimais [casa, empate, fora] em probabilidades sem a margem."""
    arr = np.array(odds, dtype=float)
    if np.any(arr <= 1.0):
        return None
    inv = 1.0 / arr
    return inv / inv.sum()


# ------------------------------- Sidebar ------------------------------------ #
state = get_state()
models = get_models()
weights = get_weights()
squad_table = get_squad_table()
names = team_names(state)

st.sidebar.title("⚽ Previsor Copa do Mundo")
st.sidebar.caption(
    f"{len(names)} seleções • ensemble de {len(models)} modelos "
    "(Elo, Poisson, ML, CatBoost, LightGBM, XGBoost)"
)
if weights:
    wtxt = " · ".join(f"{k} {v:.2f}" for k, v in weights.items() if v > 0.01)
    st.sidebar.caption(f"Pesos: {wtxt}")

tab_jogo, tab_torneio = st.tabs(["🎯 Previsão de jogo", "🏆 Simulação de mata-mata"])


# =========================== ABA 1: JOGO ÚNICO ============================== #
with tab_jogo:
    c1, c2, c3 = st.columns([3, 3, 2])
    with c1:
        home = st.selectbox("Mandante", names, index=names.index("Brazil") if "Brazil" in names else 0)
    with c2:
        away_opts = [n for n in names if n != home]
        away = st.selectbox("Visitante", away_opts,
                            index=away_opts.index("Argentina") if "Argentina" in away_opts else 0)
    with c3:
        neutral = st.checkbox("Campo neutro", value=False)
        importance = st.slider("Importância", 0.0, 1.0, 1.0, 0.05,
                               help="1.0 = Copa do Mundo; 0.85 = amistoso")

    if st.button("Prever jogo", type="primary"):
        res = compute_prediction(home, away, neutral=neutral, importance=importance,
                                 state=state, models=models, weights=weights,
                                 squad_table=squad_table, ref_date=date.today())
        st.session_state["pred"] = res

    res = st.session_state.get("pred")
    if res:
        h, a = res["home"], res["away"]
        ens = res["ensemble"]
        venue = "campo neutro" if res["neutral"] else f"{h} como mandante"
        st.subheader(f"{h} × {a} — {venue}")

        m1, m2, m3 = st.columns(3)
        m1.metric(f"{h} vence", f"{ens[0]*100:.1f}%")
        m2.metric("Empate", f"{ens[1]*100:.1f}%")
        m3.metric(f"{a} vence", f"{ens[2]*100:.1f}%")

        left, right = st.columns([3, 2])
        with left:
            st.markdown("**Probabilidades por modelo**")
            rows = []
            for name, p in res["probs"].items():
                rows.append({"Modelo": name, f"{h} (V)": p[0],
                             "Empate": p[1], f"{a} (V)": p[2]})
            rows.append({"Modelo": "ENSEMBLE", f"{h} (V)": ens[0],
                         "Empate": ens[1], f"{a} (V)": ens[2]})
            df = pd.DataFrame(rows).set_index("Modelo")
            st.dataframe(
                df.style.format("{:.1%}").background_gradient(cmap="Greens", axis=1),
                width="stretch",
            )

            st.markdown("**Probabilidade do ensemble**")
            chart_df = pd.DataFrame(
                {"Resultado": [f"{h}", "Empate", f"{a}"], "Probabilidade": ens}
            ).set_index("Resultado")
            st.bar_chart(chart_df, height=220)

        with right:
            st.markdown("**Placares mais prováveis (Poisson)**")
            sc = pd.DataFrame(
                [{"Placar": f"{h} {i} × {j} {a}", "Prob.": p}
                 for i, j, p in res["top_scores"]]
            ).set_index("Placar")
            st.dataframe(sc.style.format({"Prob.": "{:.1%}"}), width="stretch")

            if res["diverges"]:
                st.warning(
                    f"⚠️ Modelos divergem no favorito "
                    f"(dispersão {res['disagreement']*100:.1f} pp). Decisão humana recomendada."
                )
            else:
                st.success("✓ Modelos concordam no favorito.")

        st.divider()
        st.markdown("**Forças atuais**")
        f1, f2 = st.columns(2)
        for col, team, sdata in ((f1, h, res["state_home"]), (f2, a, res["state_away"])):
            with col:
                st.markdown(f"**{team}**")
                st.caption(
                    f"Elo {sdata['elo']:.0f} • forma {sdata['form']*100:.0f}% • "
                    f"ataque {sdata['attack']:.2f} • defesa {sdata['defense']:.2f}"
                )

        st.divider()
        with st.expander("📊 Comparar com o mercado (odds das casas de aposta)"):
            st.caption("Informe as odds decimais para ver as probabilidades sem a margem (devig).")
            o1, o2, o3 = st.columns(3)
            odd_h = o1.number_input(f"Odd {h}", min_value=1.0, value=1.0, step=0.01)
            odd_d = o2.number_input("Odd empate", min_value=1.0, value=1.0, step=0.01)
            odd_a = o3.number_input(f"Odd {a}", min_value=1.0, value=1.0, step=0.01)
            mkt = devig([odd_h, odd_d, odd_a])
            if mkt is not None:
                comp = pd.DataFrame({
                    "Modelo": ens,
                    "Mercado (devig)": mkt,
                    "Diferença": ens - mkt,
                }, index=[f"{h}", "Empate", f"{a}"])
                st.dataframe(
                    comp.style.format("{:.1%}").background_gradient(
                        cmap="RdYlGn", subset=["Diferença"], axis=None),
                    width="stretch",
                )
                edge = ens - mkt
                k = int(np.argmax(edge))
                if edge[k] > 0.03:
                    st.info(
                        f"💡 O modelo vê **{[h,'empate',a][k]}** subvalorizado pelo mercado "
                        f"(+{edge[k]*100:.1f} pp)."
                    )
            else:
                st.caption("Preencha as três odds (valores > 1.0) para comparar.")


# ========================= ABA 2: MATA-MATA ================================= #
with tab_torneio:
    st.markdown(
        "Simulação **Monte Carlo** de um mata-mata em campo neutro. "
        "Informe as seleções na ordem do chaveamento (potência de 2: 2, 4, 8, 16…)."
    )

    colA, colB = st.columns([1, 1])
    with colA:
        size = st.select_slider("Tamanho do bracket padrão", options=[2, 4, 8, 16, 32], value=8)
    with colB:
        n_sims = st.slider("Nº de simulações", 1000, 50000, 20000, 1000)

    default = [t for t in default_bracket(size) if t]
    bracket = st.multiselect(
        "Seleções (ordem do chaveamento)", names, default=default,
        help="As melhores por Elo são pré-carregadas e semeadas.",
    )

    if st.button("Simular torneio", type="primary"):
        n = len(bracket)
        if n < 2 or (n & (n - 1)) != 0:
            st.error(f"O número de seleções deve ser potência de 2 (2, 4, 8, 16…). Você tem {n}.")
        else:
            with st.spinner(f"Rodando {n_sims:,} simulações..."):
                out = run_tournament(bracket, n_sims)
            st.session_state["tourney"] = out

    out = st.session_state.get("tourney")
    if out:
        st.subheader("Probabilidade de título")
        tdf = pd.DataFrame(out, columns=["Seleção", "P(título)"]).set_index("Seleção")
        c1, c2 = st.columns([2, 3])
        with c1:
            st.dataframe(tdf.style.format({"P(título)": "{:.1%}"}), width="stretch")
        with c2:
            st.bar_chart(tdf, height=max(220, len(tdf) * 28))
        champ, p = out[0]
        st.success(f"🏆 Favorito ao título: **{champ}** ({p*100:.1f}%)")
