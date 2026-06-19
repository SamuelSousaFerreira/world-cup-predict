# Previsão de Resultados da Copa do Mundo

Sistema de Machine Learning que prevê o resultado de um confronto entre seleções
com **probabilidades** (vitória mandante / empate / vitória visitante), placar
provável e **detecção de divergência** entre modelos para decisão humana.

## Como funciona

Usa a base pública [martj42/international_results](https://github.com/martj42/international_results)
(CC0) — ~49 mil partidas oficiais de seleções desde 1872, atualizada de hora em
hora. A partir dela são derivadas, **sem vazamento de dados** (apenas informação
pré-jogo), as features pedidas:

| Feature | Descrição |
|---|---|
| **Forma** | Pontos nos **últimos 10 jogos** (3/1/0) normalizados |
| **Força de ataque** | Média de gols marcados nos últimos 10 jogos |
| **Força de defesa** | Média de gols sofridos nos últimos 10 jogos |
| **Estilo de jogo** | *Ritmo* (gols totais/jogo) e *agressividade* (saldo: ofensivo vs defensivo) |
| **Elo** | Rating dinâmico com vantagem de mando e peso por importância do torneio |
| **Mando / Importância** | Campo neutro e peso do torneio (Copa do Mundo > eliminatória > amistoso) |
| **Força de elenco** | Valor de mercado, idade média e ranking FIFA do elenco ([Transfermarkt](https://github.com/dcaribou/transfermarkt-datasets), CC0) |
| **Forma/contexto avançados** | EWMA da forma, ataque/defesa ajustados pela força do oponente, força de calendário (SoS), sequência, descanso e amplitude da janela |

## Seis modelos (com prós e contras)

Cada modelo gera probabilidades de forma **independente**. Quando discordam do
favorito, o sistema emite um alerta de **DIVERGÊNCIA** — cabe ao humano decidir.

| Modelo | Abordagem | Prós | Contras |
|---|---|---|---|
| **Elo** | Regressão logística sobre rating | Robusto, interpretável, ótimo p/ força relativa | Ignora forma e estilo; não dá placar |
| **Poisson** | Força ataque × defesa → gols esperados (com Dixon-Coles) | Modela ataque/defesa; gera placar provável | Assume independência; sensível a goleadas |
| **ML** | HistGradientBoosting calibrado (todas as features) | Captura interações não lineares; bem calibrado | Caixa-preta; precisa de dados |
| **CatBoost** | Gradient boosting calibrado | Robusto, bom com NaN; forte no ensemble | Caixa-preta; treino mais lento |
| **LightGBM** | Gradient boosting calibrado | Rápido; captura interações | Caixa-preta; pode superajustar |
| **XGBoost** | Gradient boosting calibrado | Forte e regularizado | Caixa-preta; ganho marginal |

Os três boostings (CatBoost, LightGBM, XGBoost) treinam sobre o mesmo conjunto
de features do ML e são todos **calibrados** (isotônica). O **Ensemble** combina
as seis probabilidades com **pesos otimizados** (em vez de média simples): os
pesos que minimizam o log loss numa janela de validação são aprendidos
automaticamente e salvos. Atualmente: CatBoost ~0.42, LightGBM ~0.23, Elo ~0.18,
Poisson ~0.17 (ML e XGBoost ficam ~0, pois suas previsões já são quase
redundantes com os demais).

### Melhorias aplicadas

- **Decaimento temporal** — jogos recentes pesam mais no treino (meia-vida de
  ~3 anos), via `sample_weight` exponencial. Reflete melhor a força atual.
- **Correção Dixon-Coles** no Poisson — ajusta a dependência entre os gols em
  placares baixos (0-0, 1-0, 0-1, 1-1), onde o Poisson puro erra mais.
- **Ensemble com pesos otimizados** — minimiza log loss numa janela de validação
  temporal separada (não usa o teste), evitando vazamento.
- **Mais modelos no ensemble** — CatBoost, LightGBM e XGBoost (calibrados) se
  somam ao Elo, Poisson e ML, reduzindo o log loss do ensemble.
- **Simulação de Monte Carlo de torneio** — estima a probabilidade de cada
  seleção ser campeã simulando milhares de chaveamentos (ver abaixo).
- **Força de elenco (Transfermarkt)** — valor de mercado, idade média e ranking
  FIFA do elenco entram como features dos modelos de árvore. É um *snapshot
  atual* aplicado a todos os jogos; o decaimento temporal faz só os recentes
  pesarem, onde o snapshot é uma boa aproximação. Seleções sem cobertura recebem
  `NaN`, tratado nativamente pelos modelos de árvore.
- **Features de contexto avançadas (v2)** — forma com EWMA, ataque/defesa
  ajustados pela força do oponente, força de calendário (SoS), sequência de
  resultados, dias de descanso e amplitude da janela de jogos.

### Desempenho (backtest temporal, ~4.800 jogos recentes)

| Modelo | Acurácia | Log Loss | Brier |
|---|---|---|---|
| Elo | 0.604 | 0.874 | 0.514 |
| Poisson | 0.595 | 0.948 | 0.541 |
| ML | 0.611 | 0.866 | 0.509 |
| CatBoost | 0.605 | 0.866 | 0.509 |
| LightGBM | 0.603 | 0.873 | 0.510 |
| XGBoost | 0.607 | 0.868 | 0.510 |
| **Ensemble (média)** | **0.608** | **0.862** | **0.507** |
| Ensemble (pesos) | 0.607 | 0.863 | 0.508 |
| Baseline (classe majoritária) | 0.477 | — | — |

Validação **temporal em três blocos** (treino → validação → teste): treina no
passado, ajusta os pesos do ensemble na validação e mede no futuro — simulando
o uso real, sem vazamento de dados.

## Instalação

```powershell
cd world-cup
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Uso

### 1. Treinar (baixa dados, gera features, avalia e salva os modelos)

```powershell
python src\train.py
```

### 2. Prever um confronto

```powershell
# Jogo com mando de campo
python src\predict.py "Brazil" "England"

# Final em campo neutro, peso de Copa do Mundo
python src\predict.py "Brazil" "Argentina" --neutral --importance 1.0
```

Saída de exemplo:

```
========================================================
  Brazil  x  Argentina   (campo neutro)
========================================================
|   Modelo | Brazil (V) | Empate | Argentina (V) |
|----------|-----------|--------|---------------|
|      Elo |     23.9% |  26.5% |         49.6% |
|  Poisson |     12.0% |  22.0% |         66.0% |
|       ML |     22.1% |  20.2% |         57.7% |
| CatBoost |     21.4% |  21.0% |         57.6% |
| LightGBM |     20.8% |  21.3% |         57.9% |
|  XGBoost |     22.0% |  20.5% |         57.5% |
| ENSEMBLE |     19.3% |  22.9% |         57.8% |

Placar mais provável (Poisson): Brazil 1 x 2 Argentina
Resultado favorito (ensemble): Argentina vencer (57.8%)
✓ Modelos concordam no favorito.
```

> Os nomes das seleções são em **inglês** (ex.: `Brazil`, `Germany`, `South Korea`).
> O sistema tolera erros leves de digitação e maiúsculas/minúsculas.

### 3. Simular um torneio (probabilidade de título)

```powershell
# 16 seleções de maior Elo (demonstração), 20 mil simulações
python src\simulate_tournament.py --n 20000 --size 16

# bracket explícito (potência de 2: 4, 8, 16 ou 32 times), em ordem de chaveamento
python src\simulate_tournament.py "Argentina" "Netherlands" "Croatia" "Brazil" `
  "England" "France" "Morocco" "Portugal"
```

Saída de exemplo:

```
|  # | Seleção   | P(título) |
|----|-----------|-----------|
|  1 | Argentina |     27.1% |
|  2 | Spain     |     18.3% |
|  3 | England   |      8.8% |
...
Favorito ao título: Argentina (27.1%)
```

Cada partida do mata-mata usa o ensemble (em campo neutro, com pênaltis no
empate) e o torneio é repetido milhares de vezes (Monte Carlo).

### 4. Interface web (Streamlit)

#### Métricas exibidas no dashboard

##### Cabeçalho (4 cards de resumo)

| Métrica | O que significa |
|---|---|
| **% vitória mandante / visitante** | Probabilidade do ensemble (média ponderada dos 6 modelos) de cada time vencer. |
| **% empate** | Probabilidade de empate no tempo normal, segundo o ensemble. |
| **Placar provável** | Placar de maior probabilidade individual segundo o modelo Poisson. O valor entre parênteses é a probabilidade daquele placar exato, e as siglas "xG" são os gols esperados de cada time (λ Poisson). |

##### Cards de perfil de cada seleção

| Métrica | O que significa |
|---|---|
| **Elo** | Rating dinâmico (escala ~1000–2100). A barra mostra a posição relativa entre todas as seleções da base. Jogos contra adversários mais fortes alteram o Elo mais do que jogos contra seleções fracas. |
| **Forma (últimos 10 jogos)** | Pontuação acumulada nos últimos 10 jogos (3 pts vitória, 1 empate, 0 derrota) normalizada para 0–100 %. Calculada com **EWMA** — jogos mais recentes pesam proporcionalmente mais. |
| **Ataque (aj. oponente)** | Média de gols marcados ajustada pela força defensiva dos adversários enfrentados. Um time que marcou 2 gols contra uma boa defesa terá um ajuste maior do que 2 gols contra uma defesa fraca. O valor "bruto" (sem ajuste) é exibido abaixo. |
| **Defesa (aj. oponente)** | Média de gols sofridos ajustada pela força ofensiva dos adversários. Quanto menor, melhor. O valor "bruto" é exibido abaixo. |
| **Valor de elenco** | Valor de mercado total do elenco em euros (fonte: Transfermarkt, CC0). Proxy de qualidade dos jogadores. Seleções sem cobertura exibem "—". |
| **Idade média / FIFA** | Idade média do elenco e posição atual no ranking FIFA (snapshot do último treino). |
| **Força de calendário (SoS)** | *Strength of Schedule* — Elo médio dos adversários que a seleção enfrentou nos jogos usados para calcular a forma, normalizado entre 0 e 1. **Alto** = jogou contra seleções difíceis; **baixo** = calendário mais fácil. Serve para contextualizar se a forma foi conquistada com mérito. |
| **Estilo de jogo** | Derivado da *agressividade* (saldo entre gols marcados e sofridos em relação à média): "Muito ofensivo / Ofensivo / Equilibrado / Defensivo". |
| **Ritmo** | Derivado do *ritmo* (total de gols por jogo): "Ritmo alto / médio / baixo". Indica se o time tende a jogos com muitos gols. |
| **Sequência** | Número de vitórias ou derrotas consecutivas até o último jogo da base. Neutro se não há sequência definida. |

##### Últimos 5 jogos

Cada card mostra resultado (**V** vitória / **E** empate / **D** derrota), placar, adversário com bandeira, data e competição. Ordenados do mais recente para o mais antigo.

##### Seção Resultado (1X2)

| Elemento | O que significa |
|---|---|
| **Tabela por modelo** | Probabilidades brutas de cada um dos 6 modelos (Elo, Poisson, ML, CatBoost, LightGBM, XGBoost) + linha do Ensemble. |
| **Gráfico de barras do ensemble** | Visualização das três probabilidades do ensemble com as cores das seleções. |
| **Alerta de divergência** | Aparece quando os modelos discordam sobre o favorito (ex.: Elo diz time A, Poisson diz time B). A dispersão em pp indica quão longe as probabilidades individuais estão umas das outras. |

##### Seção Probabilidades de gols

| Elemento | O que significa |
|---|---|
| **Mapa de calor (heatmap)** | Probabilidade de cada placar exato (mandante × visitante, até 5×5). Quanto mais escura a célula, mais provável aquele placar. |
| **Distribuição de gols** | Barras mostrando a probabilidade de cada time marcar 0, 1, 2, 3… gols, calculadas pela distribuição Poisson com λ = xG do modelo. |
| **Placares mais prováveis** | Top 10 placares individuais com maior probabilidade, em ordem decrescente. |
| **Mercados de gols** | Probabilidades acumuladas dos mercados clássicos de apostas: "Mais de 0.5 gols" (≈ jogo sem 0×0), "Mais de 1.5", "Mais de 2.5", "Mais de 3.5" e **Ambas marcam** (BTTS — ambos os times marcam pelo menos 1 gol). |

##### Comparar com o mercado (expansível)

Informe as odds decimais das casas de apostas. O sistema aplica **devig** (remove a margem da casa) e compara com as probabilidades do ensemble, mostrando a diferença em pontos percentuais. Uma diferença positiva indica uma potencial **vantagem de valor** (*value bet*).

---

```powershell
.\venv\Scripts\streamlit.exe run app.py
```

Abre uma interface no navegador com duas abas:

- **Previsão de jogo** — escolha mandante/visitante, campo neutro e importância;
  veja probabilidades por modelo e do ensemble, placares mais prováveis, forças
  atuais e comparação opcional com o mercado (devig de odds).
- **Simulação de mata-mata** — monte um bracket (potência de 2) e rode o Monte
  Carlo para ver a probabilidade de título de cada seleção.

## Estrutura

```
world-cup/
├── requirements.txt
├── README.md
├── app.py                      # interface web (Streamlit): jogo + mata-mata
├── experiment_squad.py         # experimento A/B das features de elenco
├── src/
│   ├── data_collection.py      # download + cache da base pública
│   ├── feature_engineering.py  # Elo, forma, ataque, defesa, estilo + features v2
│   ├── squad_data.py           # força de elenco (valor/idade/ranking) via Transfermarkt
│   ├── models.py               # Elo / Poisson(Dixon-Coles) / ML / CatBoost / LightGBM / XGBoost + ensemble ponderado
│   ├── train.py                # treino, decaimento temporal, backtest, pesos
│   ├── predict.py              # CLI de previsão de confronto
│   └── simulate_tournament.py  # simulação Monte Carlo do mata-mata
├── data/
│   ├── raw/                    # CSVs baixados
│   ├── processed/              # features + estado atual das seleções
│   └── squad/                  # tabela de elenco (Transfermarkt, cacheada)
└── models/                     # modelos treinados (.joblib) + pesos do ensemble
```

## Atualizar os dados

A base é atualizada continuamente. Para rebaixar e retreinar:

```powershell
python src\data_collection.py --force
python src\train.py
```

## Notas

- O "estado atual" de cada seleção (`data/processed/team_state.json`) reflete os
  últimos jogos disponíveis na base no momento do treino.
- O modelo prevê **resultado** (1X2). O placar exato vem do Poisson como
  referência, não como aposta de alta confiança.
