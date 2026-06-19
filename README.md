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

## Três modelos (com prós e contras)

Cada modelo gera probabilidades de forma **independente**. Quando discordam do
favorito, o sistema emite um alerta de **DIVERGÊNCIA** — cabe ao humano decidir.

| Modelo | Abordagem | Prós | Contras |
|---|---|---|---|
| **Elo** | Regressão logística sobre rating | Robusto, interpretável, ótimo p/ força relativa | Ignora forma e estilo; não dá placar |
| **Poisson** | Força ataque × defesa → gols esperados | Modela ataque/defesa; gera placar provável | Assume independência; sensível a goleadas |
| **ML** | Gradient Boosting calibrado (todas as features + força de elenco) | Captura interações não lineares; bem calibrado | Caixa-preta; precisa de dados |

O **Ensemble** combina as três probabilidades com **pesos otimizados** (em vez de
média simples): os pesos que minimizam o log loss numa janela de validação são
aprendidos automaticamente e salvos. Atualmente: ML ~0.56, Poisson ~0.23, Elo ~0.20.

### Melhorias aplicadas

- **Decaimento temporal** — jogos recentes pesam mais no treino (meia-vida de
  ~3 anos), via `sample_weight` exponencial. Reflete melhor a força atual.
- **Correção Dixon-Coles** no Poisson — ajusta a dependência entre os gols em
  placares baixos (0-0, 1-0, 0-1, 1-1), onde o Poisson puro erra mais.
- **Ensemble com pesos otimizados** — minimiza log loss numa janela de validação
  temporal separada (não usa o teste), evitando vazamento.
- **Simulação de Monte Carlo de torneio** — estima a probabilidade de cada
  seleção ser campeã simulando milhares de chaveamentos (ver abaixo).
- **Força de elenco (Transfermarkt)** — valor de mercado, idade média e ranking
  FIFA do elenco entram como features do ML. É um *snapshot atual* aplicado a
  todos os jogos; o decaimento temporal faz só os recentes pesarem, onde o
  snapshot é uma boa aproximação. Seleções sem cobertura recebem `NaN`, tratado
  nativamente pelo HistGradientBoosting. Ganho medido no teste: log loss
  0.876 → 0.869, acurácia +0.5pp.

### Desempenho (backtest temporal, ~4.800 jogos recentes)

| Modelo | Acurácia | Log Loss | Brier |
|---|---|---|---|
| Elo | 0.604 | 0.874 | 0.514 |
| Poisson | 0.595 | 0.948 | 0.541 |
| ML | 0.608 | 0.872 | 0.512 |
| Ensemble (média) | 0.605 | 0.872 | 0.513 |
| **Ensemble (pesos)** | **0.605** | **0.869** | **0.511** |
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
├── src/
│   ├── data_collection.py      # download + cache da base pública
│   ├── feature_engineering.py  # Elo, forma, ataque, defesa, estilo
│   ├── squad_data.py           # força de elenco (valor/idade/ranking) via Transfermarkt
│   ├── models.py               # Elo / Poisson(Dixon-Coles) / ML + ensemble ponderado
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
