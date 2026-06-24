"""Preview do diagrama de Sankey do mata-mata (ideia visual).

Mostra como as seleções "fluem" pelas fases do torneio: a largura de cada faixa
é a probabilidade de a seleção alcançar aquela fase. As faixas afunilam à medida
que o torneio avança — quem chega à final/título mantém uma faixa larga.

Para não depender dos modelos pesados (que falham ao carregar localmente), as
probabilidades de avanço são calculadas analiticamente a partir do Elo
(propagação exata pelo chaveamento, sem Monte Carlo). É só um protótipo visual.

Rodar:
    .\\venv\\Scripts\\streamlit.exe run sankey_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

from feature_engineering import load_team_state  # noqa: E402
from simulate_tournament import default_bracket  # noqa: E402
from team_assets import team_color  # noqa: E402

st.set_page_config(page_title="Sankey — Mata-mata (preview)", page_icon="⚽", layout="wide")


# --------------------------- Probabilidade de avanço ------------------------ #
def elo_win_prob(elo_a: float, elo_b: float) -> float:
    """P(A vence B) pela curva logística de Elo (divisor 400)."""
    return 1.0 / (1.0 + 10.0 ** (-(elo_a - elo_b) / 400.0))


def analytic_reach(teams: list[str], elo: dict[str, float]) -> tuple[dict[str, list[float]], int]:
    """Probabilidade de cada seleção alcançar cada fase, propagada pelo chaveamento.

    Cada "bloco" guarda, para cada seleção, a probabilidade de ela ser a
    sobrevivente daquele trecho do chaveamento (soma 1 por bloco). Ao fundir dois
    blocos (uma rodada), a sobrevivente enfrenta uma adversária incerta -> média
    ponderada das probabilidades de vitória contra cada possível adversária.
    """
    n_stages = len(teams).bit_length()          # log2(n) + 1 fases
    reach = {t: [0.0] * n_stages for t in teams}
    for t in teams:
        reach[t][0] = 1.0                        # todas entram na 1ª rodada

    blocks: list[dict[str, float]] = [{t: 1.0} for t in teams]
    stage = 0
    while len(blocks) > 1:
        merged_blocks: list[dict[str, float]] = []
        for i in range(0, len(blocks), 2):
            a_blk, b_blk = blocks[i], blocks[i + 1]
            merged: dict[str, float] = {}
            for a, qa in a_blk.items():
                p_adv = sum(qb * elo_win_prob(elo[a], elo[b]) for b, qb in b_blk.items())
                merged[a] = qa * p_adv
            for b, qb in b_blk.items():
                p_adv = sum(qa * elo_win_prob(elo[b], elo[a]) for a, qa in a_blk.items())
                merged[b] = qb * p_adv
            merged_blocks.append(merged)
        blocks = merged_blocks
        stage += 1
        for blk in blocks:
            for t, q in blk.items():
                reach[t][stage] = q
    return reach, n_stages


def stage_name(remaining: int) -> str:
    names = {1: "Campeão", 2: "Final", 4: "Semifinal", 8: "Quartas",
             16: "Oitavas", 32: "Fase de 32", 64: "Fase de 64"}
    return names.get(remaining, f"{remaining} times")


def hex_to_rgba(hexc: str, alpha: float) -> str:
    hexc = hexc.lstrip("#")
    r, g, b = (int(hexc[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# ------------------------------ Sankey -------------------------------------- #
def build_sankey(teams: list[str], reach: dict[str, list[float]],
                 n_stages: int, threshold: float) -> go.Figure:
    n = len(teams)
    node_labels: list[str] = []
    node_colors: list[str] = []
    node_index: dict[tuple[str, int], int] = {}

    for s in range(n_stages):
        for t in teams:
            if reach[t][s] >= threshold:
                node_index[(t, s)] = len(node_labels)
                node_labels.append(t)
                node_colors.append(hex_to_rgba(team_color(t), 0.95))

    src, dst, val, link_colors, link_labels = [], [], [], [], []
    for s in range(n_stages - 1):
        remaining_next = n >> (s + 1)
        for t in teams:
            flow = reach[t][s + 1]               # fluxo que sobrevive p/ a próxima fase
            if flow >= threshold and (t, s) in node_index and (t, s + 1) in node_index:
                src.append(node_index[(t, s)])
                dst.append(node_index[(t, s + 1)])
                val.append(flow)
                link_colors.append(hex_to_rgba(team_color(t), 0.45))
                link_labels.append(f"{t} → {stage_name(remaining_next)}: {flow:.1%}")

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            label=node_labels,
            color=node_colors,
            pad=12,
            thickness=16,
            line=dict(width=0),
            hovertemplate="%{label}<extra></extra>",
        ),
        link=dict(
            source=src, target=dst, value=val,
            color=link_colors,
            customdata=link_labels,
            hovertemplate="%{customdata}<extra></extra>",
        ),
    ))

    # Rótulos das fases no topo de cada coluna.
    annotations = []
    for s in range(n_stages):
        x = s / (n_stages - 1) if n_stages > 1 else 0.5
        annotations.append(dict(
            x=x, y=1.06, xref="paper", yref="paper",
            text=f"<b>{stage_name(n >> s)}</b>", showarrow=False,
            font=dict(size=13, color="#334155"),
            xanchor="center",
        ))
    fig.update_layout(
        annotations=annotations,
        margin=dict(l=10, r=10, t=40, b=10),
        height=max(420, n * 26),
        font=dict(size=12),
    )
    return fig


# ------------------------------ UI ------------------------------------------ #
st.title("🏆 Mata-mata como diagrama de Sankey (preview)")
st.caption(
    "A largura de cada faixa = probabilidade de a seleção chegar àquela fase. "
    "As faixas afunilam a cada rodada. *(Avanço estimado pelo Elo — protótipo visual.)*"
)

state = load_team_state()
elo = {t: state[t]["elo"] for t in state}
names = sorted(state, key=lambda t: -state[t]["elo"])

c1, c2 = st.columns([1, 1])
with c1:
    size = st.select_slider("Tamanho do chaveamento", options=[4, 8, 16, 32], value=8)
with c2:
    threshold = st.slider("Esconder fluxos abaixo de", 0.0, 0.10, 0.01, 0.005,
                          format="%.1f%%", help="Reduz a poluição visual em fases tardias.")

default = [t for t in default_bracket(size) if t]
bracket = st.multiselect(
    "Seleções (ordem do chaveamento)", names, default=default,
    help="As melhores por Elo são pré-carregadas e semeadas (potência de 2).",
)

n = len(bracket)
if n < 2 or (n & (n - 1)) != 0:
    st.warning(f"O número de seleções deve ser potência de 2 (4, 8, 16, 32…). Você tem {n}.")
else:
    reach, n_stages = analytic_reach(bracket, elo)
    fig = build_sankey(bracket, reach, n_stages, threshold)
    st.plotly_chart(fig, use_container_width=True)

    champ = max(bracket, key=lambda t: reach[t][-1])
    st.markdown(f"Favorito ao título: **{champ}** ({reach[champ][-1]:.1%})")

    with st.expander("Tabela de probabilidades por fase"):
        import pandas as pd
        cols = [stage_name(n >> s) for s in range(n_stages)]
        df = pd.DataFrame(
            {stage_name(n >> s): [reach[t][s] for t in bracket] for s in range(n_stages)},
            index=bracket,
        )
        df = df.sort_values(cols[-1], ascending=False)
        st.dataframe(df.style.format("{:.1%}"), use_container_width=True)
