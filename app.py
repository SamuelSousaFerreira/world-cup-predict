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

import altair as alt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

from feature_engineering import load_team_state  # noqa: E402
from knockout import knockout_probabilities  # noqa: E402
from predict import compute_prediction, load_models, load_weights  # noqa: E402
from squad_data import load_squad_table  # noqa: E402
from simulate_tournament import (MatchPredictor, default_bracket,  # noqa: E402
                                 run as run_tournament, run_with_reach)
from team_assets import (DRAW_COLOR, flag_img_tag, flag_url,  # noqa: E402
                         lighten, pair_colors, team_color)

st.set_page_config(page_title="Previsor Copa do Mundo", page_icon="⚽", layout="wide")

# Marca de build: alterar este valor força o Streamlit Cloud a recarregar o
# entry-script (evita servir código antigo em cache após commits só de dados).
APP_BUILD = "2026-06-28T10:08:16Z"

# ----------------------------- Estilo (CSS) --------------------------------- #
st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; max-width: 1280px;}
      [data-testid="stMetricValue"] {font-size: 1.7rem;}
      [data-testid="stMetricLabel"] {font-weight: 600;}
      div[data-testid="stExpander"] details {border-radius: 10px;}
      .stTabs [data-baseweb="tab"] {font-size: 1rem; font-weight: 600;}
      h3 {margin-top: 0.4rem;}
    </style>
    """,
    unsafe_allow_html=True,
)




# --------------------------- Cache de recursos ------------------------------ #
# O parâmetro `build` (default = APP_BUILD) entra na chave de cache do Streamlit.
# Como o APP_BUILD é bumpado a cada retreino diário, a chave muda e o cache é
# invalidado automaticamente — garantindo que o app recarregue o team_state.json
# e os modelos novos mesmo que o processo do Streamlit Cloud continue "quente"
# (sem isso, o @st.cache_resource serviria dados antigos indefinidamente).
@st.cache_resource(show_spinner=False)
def get_state(build: str = APP_BUILD) -> dict:
    return load_team_state()


@st.cache_resource(show_spinner=False)
def get_models(build: str = APP_BUILD) -> dict:
    return load_models()


@st.cache_resource(show_spinner=False)
def get_weights(build: str = APP_BUILD) -> dict | None:
    return load_weights()


@st.cache_resource(show_spinner=False)
def get_squad_table(build: str = APP_BUILD) -> dict:
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


# --------------------------- Gráficos (Altair) ------------------------------ #
def scoreline_heatmap(matrix: np.ndarray, home: str, away: str,
                      home_color: str, max_goals: int = 6) -> alt.Chart:
    """Mapa de calor das probabilidades de placar exato (gols mandante × visitante)."""
    m = matrix[: max_goals + 1, : max_goals + 1]
    rows = [
        {"home_goals": i, "away_goals": j, "prob": float(m[i, j])}
        for i in range(m.shape[0])
        for j in range(m.shape[1])
    ]
    df = pd.DataFrame(rows)
    peak = df["prob"].max()
    base = alt.Chart(df).encode(
        x=alt.X("away_goals:O", title=f"Gols {away}"),
        y=alt.Y("home_goals:O", title=f"Gols {home}", sort="descending"),
    )
    heat = base.mark_rect().encode(
        color=alt.Color("prob:Q", title="Prob.",
                        scale=alt.Scale(range=["#ffffff", lighten(home_color, 0.20)]),
                        legend=alt.Legend(format=".0%")),
        tooltip=[alt.Tooltip("home_goals:O", title=f"{home}"),
                 alt.Tooltip("away_goals:O", title=f"{away}"),
                 alt.Tooltip("prob:Q", title="Probabilidade", format=".2%")],
    )
    labels = base.mark_text(fontSize=11).encode(
        text=alt.Text("prob:Q", format=".0%"),
        color=alt.condition(alt.datum.prob > peak * 0.55,
                            alt.value("#0f172a"), alt.value("#475569")),
    )
    return (heat + labels).properties(height=300)


def goal_distribution_chart(home_dist: np.ndarray, away_dist: np.ndarray,
                            home: str, away: str,
                            home_color: str, away_color: str,
                            max_goals: int = 6) -> alt.Chart:
    """Barras agrupadas: P(seleção marcar k gols) para mandante e visitante."""
    rows = []
    for k in range(max_goals + 1):
        rows.append({"goals": k, "team": home, "prob": float(home_dist[k])})
        rows.append({"goals": k, "team": away, "prob": float(away_dist[k])})
    df = pd.DataFrame(rows)
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("goals:O", title="Gols marcados"),
            xOffset="team:N",
            y=alt.Y("prob:Q", title="Probabilidade", axis=alt.Axis(format="%")),
            color=alt.Color("team:N", title="Seleção",
                            scale=alt.Scale(domain=[home, away],
                                            range=[home_color, away_color])),
            tooltip=[alt.Tooltip("team:N", title="Seleção"),
                     alt.Tooltip("goals:O", title="Gols"),
                     alt.Tooltip("prob:Q", title="Probabilidade", format=".1%")],
        )
        .properties(height=260)
    )


def outcome_bar(ens: np.ndarray, home: str, away: str,
                home_color: str, away_color: str) -> alt.Chart:
    """Barras horizontais do resultado (1X2) do ensemble, nas cores das seleções."""
    labels = [f"{home} vence", "Empate", f"{away} vence"]
    df = pd.DataFrame({"Resultado": labels, "Probabilidade": ens})
    bars = alt.Chart(df).mark_bar().encode(
        x=alt.X("Probabilidade:Q", axis=alt.Axis(format="%"), title=None),
        y=alt.Y("Resultado:N", sort=labels, title=None),
        color=alt.Color("Resultado:N", legend=None,
                        scale=alt.Scale(domain=labels,
                                        range=[home_color, DRAW_COLOR, away_color])),
        tooltip=[alt.Tooltip("Resultado:N"),
                 alt.Tooltip("Probabilidade:Q", format=".1%")],
    )
    text = alt.Chart(df).mark_text(align="left", dx=4, fontWeight="bold").encode(
        x="Probabilidade:Q", y=alt.Y("Resultado:N", sort=labels),
        text=alt.Text("Probabilidade:Q", format=".1%"),
    )
    return (bars + text).properties(height=150)


# Paleta dos estágios do mata-mata (verde=cedo, âmbar=prorrog., vermelho=pênaltis)
_KO_STAGES = ["Decidido em 90 min", "Na prorrogação", "Nos pênaltis"]
_KO_PATHS = ["Ganha em 90 min", "Na prorrogação", "Nos pênaltis"]
_KO_COLORS = ["#16a34a", "#d97706", "#dc2626"]


def knockout_stage_bar(ko: dict) -> go.Figure:
    """Barra 100% empilhada: onde o confronto é decidido (90 / prorrogação / pênaltis)."""
    vals = [ko["decided_90"], ko["decided_et"], ko["decided_pen"]]
    total = sum(vals) or 1.0
    fig = go.Figure()
    for label, v, color in zip(_KO_STAGES, vals, _KO_COLORS):
        frac = v / total
        fig.add_bar(
            y=["Confronto"], x=[frac], name=label, orientation="h",
            marker=dict(color=color, line=dict(width=0)),
            text=[f"{frac:.0%}"] if frac >= 0.06 else [""],
            textposition="inside", insidetextanchor="middle",
            textfont=dict(color="white", size=13),
            hovertemplate=f"{label}: %{{x:.1%}}<extra></extra>",
        )
    fig.update_layout(
        barmode="stack", height=140,
        margin=dict(l=8, r=8, t=8, b=44),
        xaxis=dict(range=[0, 1], visible=False, fixedrange=True),
        yaxis=dict(visible=False, fixedrange=True),
        legend=dict(orientation="h", traceorder="normal", xanchor="center",
                    x=0.5, yanchor="top", y=-0.05, title=None,
                    font=dict(size=12)),
        bargap=0.35,
    )
    return fig


def knockout_advance_bar(ko: dict, home: str, away: str) -> go.Figure:
    """Barras empilhadas por seleção: como cada uma se classifica."""
    # Plotly desenha de baixo p/ cima: colocar o mandante no topo.
    teams = [away, home]
    breakdowns = {home: ko["home_breakdown"], away: ko["away_breakdown"]}
    fig = go.Figure()
    for j, (label, color) in enumerate(zip(_KO_PATHS, _KO_COLORS)):
        xs = [float(breakdowns[t][j]) for t in teams]
        fig.add_bar(
            y=teams, x=xs, name=label, orientation="h",
            marker=dict(color=color, line=dict(width=0)),
            text=[f"{x:.0%}" if x >= 0.06 else "" for x in xs],
            textposition="inside", insidetextanchor="middle",
            textfont=dict(color="white", size=12),
            hovertemplate="%{y} — " + label + ": %{x:.1%}<extra></extra>",
        )
    fig.update_layout(
        barmode="stack", height=185,
        margin=dict(l=8, r=8, t=8, b=52),
        xaxis=dict(tickformat=".0%", title=None, showgrid=True,
                   gridcolor="#eef2f7", fixedrange=True, range=[0, 1]),
        yaxis=dict(title=None, fixedrange=True, tickfont=dict(size=13)),
        legend=dict(orientation="h", traceorder="normal", xanchor="center",
                    x=0.5, yanchor="top", y=-0.18, title=None,
                    font=dict(size=12)),
        bargap=0.4,
    )
    return fig


def _stage_name(remaining: int) -> str:
    """Nome da fase a partir de quantas seleções seguem vivas."""
    return {1: "Campeão", 2: "Final", 4: "Semifinal", 8: "Quartas",
            16: "Oitavas", 32: "Fase de 32", 64: "Fase de 64"}.get(
        remaining, f"{remaining} times")


def _hex_to_rgba(hexc: str, alpha: float) -> str:
    hexc = hexc.lstrip("#")
    r, g, b = (int(hexc[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def tournament_sankey(teams: list[str], reach: dict[str, list[float]],
                      n_stages: int, threshold: float = 0.01) -> go.Figure:
    """Fluxo das seleções pelas fases do mata-mata.

    A largura de cada faixa é a probabilidade (Monte Carlo) de a seleção alcançar
    aquela fase; as faixas afunilam a cada rodada. Fluxos abaixo de ``threshold``
    são omitidos para reduzir poluição visual nas fases finais.
    """
    n = len(teams)
    labels, colors, node_index = [], [], {}
    for s in range(n_stages):
        for t in teams:
            if reach[t][s] >= threshold:
                node_index[(t, s)] = len(labels)
                labels.append(t)
                colors.append(_hex_to_rgba(team_color(t), 0.95))

    src, dst, val, link_colors, link_labels = [], [], [], [], []
    for s in range(n_stages - 1):
        remaining_next = n >> (s + 1)
        for t in teams:
            flow = reach[t][s + 1]
            if flow >= threshold and (t, s) in node_index and (t, s + 1) in node_index:
                src.append(node_index[(t, s)])
                dst.append(node_index[(t, s + 1)])
                val.append(flow)
                link_colors.append(_hex_to_rgba(team_color(t), 0.45))
                link_labels.append(f"{t} → {_stage_name(remaining_next)}: {flow:.1%}")

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=labels, color=colors, pad=12, thickness=16,
                  line=dict(width=0), hovertemplate="%{label}<extra></extra>"),
        link=dict(source=src, target=dst, value=val, color=link_colors,
                  customdata=link_labels, hovertemplate="%{customdata}<extra></extra>"),
    ))
    annotations = [
        dict(x=(s / (n_stages - 1) if n_stages > 1 else 0.5), y=1.08,
             xref="paper", yref="paper", text=f"<b>{_stage_name(n >> s)}</b>",
             showarrow=False, font=dict(size=13, color="#334155"), xanchor="center")
        for s in range(n_stages)
    ]
    fig.update_layout(
        annotations=annotations,
        margin=dict(l=60, r=60, t=42, b=10),
        height=max(420, n * 26),
        font=dict(size=12),
    )
    return fig


# ------------------------------ Startup guard ------------------------------- #
@st.cache_resource(show_spinner=False)
def _check_models_exist() -> bool:
    from pathlib import Path
    needed = ["elo_model.joblib", "poisson_model.joblib", "ml_model.joblib"]
    return all((Path(__file__).parent / "models" / n).exists() for n in needed)


if not _check_models_exist():
    st.error(
        "**Modelos não encontrados.** Execute primeiro:\n\n"
        "```\npython src/train.py\n```\n\n"
        "O treino baixa os dados, treina os modelos e salva em `models/`. "
        "Leva alguns minutos na primeira vez."
    )
    st.stop()


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

# Selo de versão/dados: permite confirmar visualmente qual build está no ar e
# até quando os dados foram atualizados (útil para diagnosticar cache/redeploy).
try:
    _last = max(
        (s.get("last_date") for s in state.values() if s.get("last_date")),
        default=None,
    )
    _data_txt = str(_last)[:10] if _last else "—"
except Exception:
    _data_txt = "—"
st.sidebar.caption(f"Build: `{APP_BUILD}` • dados até {_data_txt}")

st.sidebar.divider()
st.sidebar.markdown(
    "[![GitHub](https://img.shields.io/badge/GitHub-Repositório-181717?logo=github&logoColor=white)]"
    "(https://github.com/SamuelSousaFerreira/world-cup-predict)"
)
st.sidebar.caption("by **Samuel Sousa Ferreira**")

st.title("⚽ Previsor da Copa do Mundo")
st.caption(
    "Probabilidades de resultado e de gols a partir de um ensemble de seis "
    "modelos treinados em ~49 mil partidas oficiais de seleções."
)

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
        knockout_on = st.checkbox(
            "⚔️ Mata-mata", value=False,
            help="Mostra prorrogação, pênaltis e a chance de cada seleção se classificar.")
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
        eg_h, eg_a = res["expected_goals"]
        gi, gj, gp = res["most_likely_score"]
        hc, ac = pair_colors(h, a)
        venue = "campo neutro" if res["neutral"] else f"{h} como mandante"
        st.markdown(
            f"<h3>{flag_img_tag(h, 30)}{h} "
            f"<span style='color:#94a3b8;font-weight:400'>×</span> "
            f"{flag_img_tag(a, 30)}{a} "
            f"<span style='font-weight:400;color:#64748b;font-size:1rem'>— {venue}</span></h3>",
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"{h} vence", f"{ens[0]*100:.1f}%",
                  help="Probabilidade de vitória do mandante segundo o ensemble (média ponderada dos 6 modelos).")
        m2.metric("Empate", f"{ens[1]*100:.1f}%",
                  help="Probabilidade de empate no tempo normal segundo o ensemble.")
        m3.metric(f"{a} vence", f"{ens[2]*100:.1f}%",
                  help="Probabilidade de vitória do visitante segundo o ensemble (média ponderada dos 6 modelos).")
        m4.metric("Placar provável", f"{gi}–{gj}", f"{gp*100:.1f}% • xG {eg_h:.2f}–{eg_a:.2f}",
                  delta_color="off",
                  help="Placar de maior probabilidade individual (modelo Poisson). xG = gols esperados de cada time (λ Poisson). A % é a probabilidade daquele placar exato.")

        st.divider()
        st.subheader("Perfil das seleções", help="Resumo das estatísticas recentes de cada seleção: rating Elo, forma, força de ataque/defesa ajustados, valor de elenco e estilo de jogo.")
        f1, f2 = st.columns(2)
        _all_elos = [v["elo"] for v in state.values()]
        _elo_min, _elo_max = min(_all_elos), max(_all_elos)
        # Posição no nosso ranking (por Elo, atualizado diariamente). Substitui o
        # antigo "ranking FIFA" do snapshot do Transfermarkt, que ficava defasado.
        _elo_rank = {t: i for i, (t, _) in enumerate(
            sorted(state.items(), key=lambda kv: -kv[1]["elo"]), start=1)}
        for col, team, sdata, tc in ((f1, h, res["state_home"], hc), (f2, a, res["state_away"], ac)):
            sq = squad_table.get(team, {})
            streak = sdata.get("streak", 0)
            if streak > 0:
                streak_txt = f"🔥 {streak} vitória{'s' if streak > 1 else ''} seguida{'s' if streak > 1 else ''}"
            elif streak < 0:
                streak_txt = f"📉 {abs(streak)} derrota{'s' if abs(streak) > 1 else ''} seguida{'s' if abs(streak) > 1 else ''}"
            else:
                streak_txt = "➖ Sem sequência"
            elo_pct = int((sdata["elo"] - _elo_min) / max(_elo_max - _elo_min, 1) * 100)
            form_pct = int(sdata["form"] * 100)
            style_label = (
                "⚔️ Muito ofensivo" if sdata.get("aggression", 0) > 1.5 else
                "⚔️ Ofensivo"      if sdata.get("aggression", 0) > 0.5 else
                "⚖️ Equilibrado"   if sdata.get("aggression", 0) > -0.5 else
                "🛡️ Defensivo"
            )
            tempo_label = (
                "🎯 Ritmo alto"  if sdata.get("style", 0) > 3.5 else
                "🎯 Ritmo médio" if sdata.get("style", 0) > 2.5 else
                "🎯 Ritmo baixo"
            )
            value_str = f"€{sq['value']/1e6:.0f}M" if sq.get("value") else "—"
            age_str   = f"{sq['age']:.1f} anos"    if sq.get("age")   else "—"
            rank_str  = f"#{_elo_rank[team]}"       if team in _elo_rank else "—"
            with col:
                st.markdown(
                    f"<div style='border:1px solid {tc}55;border-radius:12px;"
                    f"padding:16px 18px;background:linear-gradient(135deg,{tc}12 0%,#ffffff00 100%)'>"
                    f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:14px'>"
                    f"{flag_img_tag(team, 32)}"
                    f"<span style='font-size:1.15rem;font-weight:700'>{team}</span>"
                    f"</div>"
                    f"<div style='margin-bottom:10px' title='Rating dinâmico (escala ~1000–2100). Reflete a força histórica da seleção ponderando vitórias por dificuldade do adversário e importância do torneio. A barra mostra a posição relativa entre todas as seleções.'>"
                    f"<div style='display:flex;justify-content:space-between;font-size:.82rem;color:#475569;margin-bottom:3px'>"
                    f"<span>⭐ Elo</span><span style='font-weight:700;color:#0f172a'>{sdata['elo']:.0f}</span></div>"
                    f"<div style='background:#e2e8f0;border-radius:4px;height:7px'>"
                    f"<div style='width:{elo_pct}%;background:{tc};border-radius:4px;height:7px'></div></div>"
                    f"</div>"
                    f"<div style='margin-bottom:14px' title='Pontos acumulados nos últimos 10 jogos (3=vitória, 1=empate, 0=derrota) normalizados para 0–100%. Calculada com EWMA: jogos mais recentes pesam mais.'>"
                    f"<div style='display:flex;justify-content:space-between;font-size:.82rem;color:#475569;margin-bottom:3px'>"
                    f"<span>📈 Forma (últ. 10 jogos)</span><span style='font-weight:700;color:#0f172a'>{form_pct}%</span></div>"
                    f"<div style='background:#e2e8f0;border-radius:4px;height:7px'>"
                    f"<div style='width:{form_pct}%;background:{tc};border-radius:4px;height:7px'></div></div>"
                    f"</div>"
                    f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:14px'>"
                    f"<div style='background:#f8fafc;border-radius:8px;padding:8px 10px' title='Média de gols marcados ajustada pela força defensiva dos adversários enfrentados. Gols contra defesas fortes valem mais. O valor bruto (sem ajuste) é exibido embaixo.'>"
                    f"<div style='font-size:.72rem;color:#64748b'>⚡ Ataque (aj. oponente)</div>"
                    f"<div style='font-size:1rem;font-weight:700;color:{tc}'>{sdata.get('adj_attack', sdata['attack']):.2f}</div>"
                    f"<div style='font-size:.72rem;color:#94a3b8'>bruto {sdata['attack']:.2f} gols/j</div>"
                    f"</div>"
                    f"<div style='background:#f8fafc;border-radius:8px;padding:8px 10px' title='Média de gols sofridos ajustada pela força ofensiva dos adversários. Quanto menor, melhor. O valor bruto (sem ajuste) é exibido embaixo.'>"
                    f"<div style='font-size:.72rem;color:#64748b'>🛡️ Defesa (aj. oponente)</div>"
                    f"<div style='font-size:1rem;font-weight:700;color:{tc}'>{sdata.get('adj_defense', sdata['defense']):.2f}</div>"
                    f"<div style='font-size:.72rem;color:#94a3b8'>bruto {sdata['defense']:.2f} sofridos/j</div>"
                    f"</div>"
                    f"<div style='background:#f8fafc;border-radius:8px;padding:8px 10px' title='Valor de mercado total do elenco em euros (fonte: Transfermarkt). Proxy da qualidade individual dos jogadores. Abaixo: idade média do elenco e posição no nosso ranking (por Elo, atualizado diariamente).'>"
                    f"<div style='font-size:.72rem;color:#64748b'>💪 Valor de elenco</div>"
                    f"<div style='font-size:.95rem;font-weight:700;color:{tc}'>{value_str}</div>"
                    f"<div style='font-size:.72rem;color:#94a3b8'>Idade méd. {age_str} • Rank {rank_str}</div>"
                    f"</div>"
                    f"<div style='background:#f8fafc;border-radius:8px;padding:8px 10px' title='Strength of Schedule (SoS): Elo médio dos adversários nos jogos recentes, normalizado entre 0 e 1. Alto = calendário difícil; baixo = calendário fácil. Contextualiza se a forma foi conquistada com mérito.'>"
                    f"<div style='font-size:.72rem;color:#64748b'>📅 Força de calendário</div>"
                    f"<div style='font-size:.95rem;font-weight:700;color:{tc}'>{sdata.get('sos', 0):.2f}</div>"
                    f"<div style='font-size:.72rem;color:#94a3b8'>qualidade dos oponentes</div>"
                    f"</div>"
                    f"</div>"
                    f"<div style='display:flex;flex-wrap:wrap;gap:6px;font-size:.78rem'>"
                    f"<span style='background:{tc}22;color:{tc};border-radius:20px;padding:3px 10px;font-weight:600' title='Estilo ofensivo/defensivo baseado no saldo entre gols marcados e sofridos em relação à média geral.'>{style_label}</span>"
                    f"<span style='background:{tc}22;color:{tc};border-radius:20px;padding:3px 10px;font-weight:600' title='Ritmo de jogo baseado no total de gols por partida. Alto = jogos com muitos gols; baixo = jogos fechados.'>{tempo_label}</span>"
                    f"<span style='background:#f1f5f9;color:#475569;border-radius:20px;padding:3px 10px' title='Número de vitórias ou derrotas consecutivas até o último jogo da base.'>{streak_txt}</span>"
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.divider()
        st.subheader("Últimos 5 jogos", help="Os 5 jogos mais recentes de cada seleção na base de dados, do mais novo para o mais antigo. V = vitória, E = empate, D = derrota.")
        r1, r2 = st.columns(2)
        for col, team, matches in ((r1, h, res["recent_home"]), (r2, a, res["recent_away"])):
            with col:
                st.markdown(f"{flag_img_tag(team, 20)}**{team}**", unsafe_allow_html=True)
                _COLOR = {"W": ("#16a34a", "#dcfce7"), "D": ("#d97706", "#fef9c3"), "L": ("#dc2626", "#fee2e2")}
                _LABEL = {"W": "V", "D": "E", "L": "D"}
                for m in matches:
                    rc, bg = _COLOR[m["result"]]
                    lbl = _LABEL[m["result"]]
                    opp_flag = flag_img_tag(m["opponent"], 18)
                    tourn = m["tournament"].replace("FIFA World Cup qualification", "Eliminatória WC") \
                                          .replace("FIFA World Cup", "Copa do Mundo") \
                                          .replace("Friendly", "Amistoso")
                    st.markdown(
                        f"<div style='display:flex;align-items:center;gap:10px;"
                        f"padding:6px 10px;margin-bottom:5px;border-radius:8px;"
                        f"background:{bg};border-left:4px solid {rc}'>"
                        f"<span style='background:{rc};color:#fff;font-weight:700;"
                        f"border-radius:5px;padding:2px 8px;font-size:.85rem'>{lbl}</span>"
                        f"<span style='font-size:.95rem;font-weight:600'>"
                        f"{m['gf']}–{m['ga']}</span>"
                        f"<span style='flex:1'>{opp_flag}{m['opponent']}</span>"
                        f"<span style='font-size:.75rem;color:#64748b;text-align:right'>"
                        f"{m['date']}<br><i>{tourn}</i></span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        if res["diverges"]:
            st.warning(
                f"⚠️ Modelos divergem no favorito "
                f"(dispersão {res['disagreement']*100:.1f} pp). Decisão humana recomendada."
            )

        st.subheader("Resultado (1X2)", help="Notação clássica de apostas: 1 = vitória do mandante, X = empate, 2 = vitória do visitante. Mostra as probabilidades de cada desfecho no tempo normal, por modelo e pelo ensemble.")
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
        with right:
            st.markdown("**Probabilidade do ensemble**")
            st.altair_chart(outcome_bar(ens, h, a, hc, ac), width="stretch")

        if knockout_on:
            elo_diff = res["state_home"]["elo"] - res["state_away"]["elo"]
            ko = knockout_probabilities(ens, eg_h, eg_a, elo_diff)
            st.divider()
            st.subheader(
                "⚔️ Cenário de eliminatória",
                help="Se este jogo fosse mata-mata (sem mando): empate na "
                     "regulamentar leva à prorrogação (Poisson com gols esperados "
                     "proporcionais a 30 min) e, persistindo, aos pênaltis (leve "
                     "viés por Elo calibrado em 677 disputas reais; Elo igual = 50%).")
            a1, a2, a3, a4 = st.columns(4)
            a1.metric(f"{h} classifica", f"{ko['home_advance']*100:.1f}%",
                      f"{(ko['home_advance']-ens[0])*100:+.1f} pp vs vencer em 90",
                      delta_color="off")
            a2.metric(f"{a} classifica", f"{ko['away_advance']*100:.1f}%",
                      f"{(ko['away_advance']-ens[2])*100:+.1f} pp vs vencer em 90",
                      delta_color="off")
            a3.metric("Vai à prorrogação", f"{ko['p_extra_time']*100:.1f}%")
            a4.metric("Vai aos pênaltis", f"{ko['p_penalties']*100:.1f}%")
            kc1, kc2 = st.columns(2)
            with kc1:
                st.markdown("**Onde o confronto é decidido**")
                st.plotly_chart(knockout_stage_bar(ko), width="stretch",
                                config={"displayModeBar": False})
            with kc2:
                st.markdown("**Como cada seleção se classifica**")
                st.plotly_chart(knockout_advance_bar(ko, h, a), width="stretch",
                                config={"displayModeBar": False})
            st.caption(
                f"Pênaltis: quase moeda com leve viés por Elo — "
                f"{h} {ko['home_pen_win']*100:.0f}% × {ko['away_pen_win']*100:.0f}% {a}. "
                f"P(prorrogação) usa o empate do ensemble; classificação = vencer em 90 "
                f"+ empate×(vencer na prorrogação) + empate×empate-na-prorrogação×(vencer nos pênaltis).")

        st.divider()
        st.subheader("Probabilidades de gols", help="Derivadas do modelo Poisson com correção Dixon-Coles. Mostra a distribuição de gols de cada time, o mapa de calor dos placares exatos e os mercados clássicos (over/under e ambas marcam).")
        g_left, g_right = st.columns([3, 2])
        with g_left:
            st.markdown("**Mapa de calor dos placares** (gols mandante × visitante)")
            st.altair_chart(
                scoreline_heatmap(res["score_matrix"], h, a, hc),
                width="stretch",
            )
            st.markdown("**Distribuição de gols por seleção**")
            st.altair_chart(
                goal_distribution_chart(res["home_goal_dist"], res["away_goal_dist"],
                                        h, a, hc, ac),
                width="stretch",
            )
        with g_right:

            st.markdown("**Placares mais prováveis**")
            sc = pd.DataFrame(
                [{"Placar": f"{i}–{j}", "Prob.": p} for i, j, p in res["top_scores"]]
            ).set_index("Placar")
            st.dataframe(
                sc.style.format({"Prob.": "{:.1%}"}).background_gradient(
                    cmap="Blues", subset=["Prob."]),
                width="stretch",
            )

            st.markdown("**Mercados de gols**")
            ou = res["over_under"]
            mk = pd.DataFrame(
                [{"Mercado": f"Mais de {thr} gols", "Prob.": p} for thr, p in ou.items()]
                + [{"Mercado": "Ambas marcam", "Prob.": res["btts"]}]
            ).set_index("Mercado")
            st.dataframe(
                mk.style.format({"Prob.": "{:.1%}"}).background_gradient(
                    cmap="Blues", subset=["Prob."]),
                width="stretch",
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
                out, reach, n_stages = run_with_reach(bracket, n_sims)
            st.session_state["tourney"] = out
            st.session_state["tourney_reach"] = (bracket, reach, n_stages)

    out = st.session_state.get("tourney")
    if out:
        st.subheader("Probabilidade de título")
        teams = [t for t, _ in out]
        probs = [p for _, p in out]
        c1, c2 = st.columns([2, 3])
        with c1:
            tdf = pd.DataFrame({
                "": [flag_url(t, 40) for t in teams],
                "Seleção": teams,
                "P(título)": probs,
            })
            st.dataframe(
                tdf,
                width="stretch",
                hide_index=True,
                column_config={
                    "": st.column_config.ImageColumn("🏳️", width="small"),
                    "P(título)": st.column_config.ProgressColumn(
                        "P(título)", format="percent",
                        min_value=0.0, max_value=float(max(probs))),
                },
            )
        with c2:
            cdf = pd.DataFrame({"Seleção": teams, "P(título)": probs})
            chart = (
                alt.Chart(cdf)
                .mark_bar()
                .encode(
                    x=alt.X("P(título):Q", axis=alt.Axis(format="%"), title=None),
                    y=alt.Y("Seleção:N", sort=teams, title=None),
                    color=alt.Color("Seleção:N", legend=None,
                                    scale=alt.Scale(domain=teams,
                                                    range=[team_color(t) for t in teams])),
                    tooltip=[alt.Tooltip("Seleção:N"),
                             alt.Tooltip("P(título):Q", format=".1%")],
                )
                .properties(height=max(220, len(teams) * 30))
            )
            st.altair_chart(chart, width="stretch")
        champ, p = out[0]
        st.markdown(
            f"<div style='font-size:1.05rem'>🏆 Favorito ao título: "
            f"{flag_img_tag(champ, 24)}<b>{champ}</b> ({p*100:.1f}%)</div>",
            unsafe_allow_html=True,
        )

        reach_data = st.session_state.get("tourney_reach")
        if reach_data:
            sk_teams, reach, n_stages = reach_data
            st.subheader("Caminho até o título (Sankey)")
            st.caption(
                "A largura de cada faixa é a probabilidade de a seleção alcançar "
                "aquela fase; as faixas afunilam a cada rodada."
            )
            thr = st.slider(
                "Esconder fluxos abaixo de", 0.0, 0.10, 0.01, 0.005,
                format="%.1f%%", help="Reduz a poluição visual nas fases finais.",
                key="sankey_thr",
            )
            st.plotly_chart(
                tournament_sankey(sk_teams, reach, n_stages, thr),
                width="stretch",
            )

