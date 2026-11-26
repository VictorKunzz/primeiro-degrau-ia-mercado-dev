# O Primeiro Degrau — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Notebook-apresentação (CRISP-DM + ETL + EDA + Preditiva + Storytelling) que testa a tese "a IA removeu o primeiro degrau do mercado dev", com dados públicos, entregue por link de repositório GitHub.

**Architecture:** ETL em módulos `src/etl/*` que baixam para `data/raw/` (fora do git) e gravam CSVs limpos em `data/processed/`. Modelos em `src/models/*` gravam resultados em `data/results/`. O notebook `notebooks/primeiro_degrau.ipynb` só lê processed/results, faz gráficos e conta a história; `reports/primeiro_degrau.html` é o backup sem código.

**Tech Stack:** Python 3.14, pandas, numpy, requests, python-dotenv, statsmodels, scikit-learn, matplotlib, pyarrow, pytest, jupyter/nbconvert.

**Spec:** `docs/design.md` (ler inteiro antes de qualquer task, especialmente §5–§7.4).

## Global Constraints

- Raiz do projeto: `C:\Users\viche\OneDrive - Católica SC\Faculdade\6. Semestre\Ciência de Dados\IA e o Primeiro Degrau - Mercado Dev` (caminho tem espaços e acentos: sempre entre aspas).
- Chave da Stack Exchange API em `.env` como `SE_API_KEY` (lida com python-dotenv). Nunca imprimir, logar ou commitar a chave. Se vazia, o código funciona sem chave (quota 300/dia) e avisa.
- `data/raw/` nunca vai para o git. `data/processed/` e `data/results/` vão (cada arquivo < 50 MB).
- Todo texto visível (markdown do notebook, títulos de gráficos, README) em **português do Brasil**, revisado com a skill `humanizer-pt-br`. Gráficos seguem a skill `dataviz`.
- Linguagem não causal: "consistente com", sempre com IC. Nenhum número de literatura entra sem link verificado.
- Regras de veredito do §7.1 do design são fixas; não alterar thresholds depois de ver resultados.
- Seed global `SEED = 42`. Chave temporal mensal: coluna `mes` tipo `datetime64` no primeiro dia do mês.
- Corte pré-IA: treino até `2022-10-01` inclusive; intervenção `2022-11-30` (ChatGPT).
- Commits pequenos, mensagem em português.

## Review Focus

1. API do Stack Exchange devolvendo `backoff` ou erro 502/throttle no meio da coleta → coleta deve respeitar `backoff`, retomar do cache por mês e não sobrescrever meses já baixados. (Task 2)
2. Coluna do survey ausente num ano (ex.: `AIThreat` não existe em 2023; `DevType` vira `Role` em 2026) → harmonização preenche `NaN`, nunca quebra nem confunde colunas. (Task 4)
3. Meses parciais (out/nov 2026 incompletos) na série do SO ou da Indeed → mês corrente é descartado; último mês só entra se completo. (Tasks 2, 3)
4. Salários absurdos/nulos no survey (`ConvertedCompYearly` 0 ou > 1e6) → filtrados antes de medianas e da conta do gestor. (Task 4)
5. Notebook executado em máquina sem `data/raw/` e sem rede → roda a partir de processed/results sem erro. (Task 8)

---

### Task 1: Esqueleto do projeto

**Files:**
- Create: `requirements.txt`, `src/__init__.py`, `src/config.py`, `src/etl/__init__.py`, `src/models/__init__.py`, `tests/__init__.py`, `tests/test_config.py`, `README.md` (placeholder)

**Interfaces:**
- Produces: `src/config.py` com `ROOT: Path`, `RAW: Path`, `PROCESSED: Path`, `RESULTS: Path`, `SEED = 42`, `CORTE_PRE_IA = pd.Timestamp("2022-10-01")`, `DATA_CHATGPT = pd.Timestamp("2022-11-30")`, `def get_se_key() -> str | None`, `def ultimo_mes_completo(hoje: date | None = None) -> pd.Timestamp`.

- [ ] Step 1: `git init`; criar venv `.venv`; `requirements.txt` com os pacotes do Tech Stack (sem pin rígido; instalar e depois congelar versões que funcionaram com `pip freeze` filtrado).
- [ ] Step 2: Teste `test_ultimo_mes_completo`: `ultimo_mes_completo(date(2026,11,26)) == pd.Timestamp("2026-10-01")`; `ultimo_mes_completo(date(2026,12,1)) == pd.Timestamp("2026-11-01")`. Teste `test_get_se_key_vazia` com monkeypatch de env vazio → `None`.
- [ ] Step 3: Rodar `pytest tests/test_config.py -v` → FAIL; implementar; → PASS. Diretórios RAW/PROCESSED/RESULTS criados por `config` (mkdir exist_ok).
- [ ] Step 4: Commit.

### Task 2: ETL Stack Overflow

**Files:** Create `src/etl/stackoverflow.py`, `tests/test_stackoverflow.py`

**Interfaces:**
- Consumes: `config.RAW, PROCESSED, get_se_key, ultimo_mes_completo`
- Produces: `def contar_perguntas_mes(mes: pd.Timestamp, tag: str | None, key: str | None, session) -> int`; `def coletar_serie(inicio: str, tags: list[str | None]) -> pd.DataFrame` (colunas `mes, tag, perguntas`; `tag="__total__"` para total); arquivo `data/processed/so_mensal.csv`.

- [ ] Step 1: Testes com `requests` mockado (`responses` ou monkeypatch de session): (a) URL contém `fromdate`/`todate` epoch do 1º dia do mês e do 1º dia do mês seguinte, `site=stackoverflow`, `filter=total`, `tagged` só quando tag não é None; (b) resposta `{"total": 842}` → 842; (c) resposta com `"backoff": 2` faz `time.sleep(2)` antes da próxima chamada; (d) mês já em cache `data/raw/so/<tag>/<AAAA-MM>.json` não gera requisição.
- [ ] Step 2: FAIL → implementar → PASS.
- [ ] Step 3: Rodar coleta real: total de `2008-09` até `ultimo_mes_completo()`; tags `python`, `javascript`, `java`, `reactjs` de `2015-01` em diante. Conferir sanity: total de nov/2022 na casa de ~100 mil e out/2026 na casa de centenas a poucos milhares (valores observados na pesquisa: 109.119 e 842).
- [ ] Step 4: Commit (sem `data/raw`).

### Task 3: ETL Indeed, FRED e capacidade da IA

**Files:** Create `src/etl/indeed.py`, `src/etl/fred.py`, `src/etl/capacidade.py`, `tests/test_etl_outros.py`

**Interfaces:**
- Produces:
  - `indeed.carregar() -> DataFrame` colunas `mes, setor, indice` (média mensal do índice diário, só meses completos) → `data/processed/indeed_mensal.csv`. Fonte: `https://raw.githubusercontent.com/hiring-lab/job_postings_tracker/master/US/job_postings_by_sector_US.csv` (usar a variante sazonalmente ajustada se houver coluna para isso; registrar qual no README).
  - `fred.carregar(series: list[str]) -> DataFrame` colunas `mes, serie, valor` via `https://fred.stlouisfed.org/graph/fredgraph.csv?id=<ID>`, séries `FEDFUNDS`, `IHLIDXUS` → `data/processed/fred_mensal.csv`.
  - `capacidade.carregar() -> DataFrame` colunas `mes, swe_bench_melhor, metr_horizonte_min_melhor` (melhor valor acumulado até o mês, `cummax`, forward-fill, de 2019-01 até último mês completo) a partir de `https://epoch.ai/data/benchmark_data.zip` → `data/processed/capacidade_mensal.csv`. Também `data/processed/marcos.csv` (`data, marco, onda` com onda ∈ {chat, assistente, agente}) com os marcos do design §5, datas conferidas na fonte oficial de cada lançamento.
- [ ] Step 1: Testes com DataFrames sintéticos: agregação diária→mensal descarta mês incompleto; `cummax` nunca diminui; forward-fill preenche meses sem lançamento.
- [ ] Step 2: FAIL → implementar → PASS.
- [ ] Step 3: Rodar os três; inspecionar `head/tail` e colunas reais da Epoch (nomes podem diferir; adaptar o mapeamento e documentar no docstring).
- [ ] Step 4: Commit.

### Task 4: ETL Developer Survey 2023–2026

**Files:** Create `src/etl/survey.py`, `tests/test_survey.py`, `tests/fixtures/survey_mini_2024.csv`, `tests/fixtures/survey_mini_2026.csv`

**Interfaces:**
- Produces: `MAPA_COLUNAS: dict[int, dict[str, str]]` (ano → coluna original → nome harmonizado); `def harmonizar(df: DataFrame, ano: int) -> DataFrame`; `def carregar_todos() -> DataFrame` → `data/processed/survey_harmonizado.parquet` com colunas: `ano, anos_exp (float), faixa_exp ({"≤2","3–9","≥10"}), cargo, emprego, salario_usd, pais, ia_usa, ia_sentimento, ia_confianca, ia_ameaca, usa_agentes`. Colunas inexistentes num ano = NaN.
- [ ] Step 1: Baixar os 4 CSVs para `data/raw/survey/<ano>.csv` (URLs em `docs/pesquisa/pesquisa_dados.md`; via Git LFS media URL). Ler só cabeçalhos e montar `MAPA_COLUNAS` a partir do schema real de cada ano. `anos_exp`: preferir `WorkExp`/`YearsCodePro`; converter "Less than 1 year"→0.5, "More than 50 years"→51.
- [ ] Step 2: Fixtures de 5 linhas por ano (escritas à mão com colunas reais). Testes: `harmonizar(fixture_2026, 2026)` mapeia `Role`→`cargo`; ano sem `AIThreat` → `ia_ameaca` toda NaN; salário 0 e 2e6 viram NaN; "Less than 1 year" → faixa "≤2".
- [ ] Step 3: FAIL → implementar → PASS. Rodar `carregar_todos()`; imprimir n por ano e % de não-nulos por coluna; salvar em `data/results/survey_cobertura.csv`.
- [ ] Step 4: Commit.

### Task 5: Modelos de série temporal (Atos 1 e 2)

**Files:** Create `src/models/series.py`, `src/models/run_series.py`, `tests/test_series.py`

**Interfaces:**
- Consumes: `so_mensal.csv`, `indeed_mensal.csv`, `fred_mensal.csv`, `config.CORTE_PRE_IA`
- Produces:
  - `def contrafactual(y: pd.Series, corte: pd.Timestamp) -> pd.DataFrame` (index mes; colunas `real, previsto, lo95, hi95`; SARIMAX em `log(y)` ajustado até `corte`, previsão do mês seguinte até o fim, IC retransformado com `exp`).
  - `def efeito(cf: DataFrame) -> dict` com `efeito_pct_medio`, `meses_abaixo_ic_seguidos` (maior sequência), `sustentado: bool` (≥ 6 seguidos).
  - `def backtest(y, corte, horizontes=12, origens=6) -> DataFrame` com MAPE do modelo e do baseline sazonal ingênuo por origem.
  - `def placebos(y, datas: list[str]) -> DataFrame` (efeito com corte falso; usar `["2019-06-01", "2020-06-01"]`, sempre avaliando só até 12 meses e antes de 2022-10).
  - `def its_segmentado(y, intervencao) -> dict` (OLS em log y: tendência, nível pós, inclinação pós, dummies de mês; `cov_type="HAC", maxlags=12`; retorna coeficientes, IC, p).
  - `def vagas_vs_setores(indeed, fred) -> dict` (diferença software − mediana dos setores; regressão com FEDFUNDS como covariável e HAC; p-valor do termo pós-intervenção ou da capacidade).
  - `run_series.main()` grava `data/results/so_contrafactual.csv`, `so_backtest.csv`, `so_placebos.csv`, `so_its.json`, `so_tags_efeito.csv`, `vagas.json`, `vereditos_series.json` (aplicando §7.1 literalmente).
- [ ] Step 1: Testes em séries sintéticas (seed 42, 180 meses, tendência + sazonalidade + ruído): (a) com queda de 40% a partir do mês 150 → `efeito(...)["sustentado"] is True` e `efeito_pct_medio < -25`; (b) sem queda → `sustentado is False`; (c) `backtest` retorna colunas `mape_modelo, mape_ingenuo`; (d) placebo em série sem quebra → |efeito| < 10%.
- [ ] Step 2: FAIL → implementar → PASS. Ordem SARIMAX escolhida por AIC numa grade pequena (p,q ∈ {0,1,2}, d=1, sazonal (1,0,0,12) ou (0,1,1,12)) **somente no pré-período**.
- [ ] Step 3: `python -m src.models.run_series`; conferir que nenhum dado pós-corte entra no ajuste (assert no código: `y_treino.index.max() <= corte`).
- [ ] Step 4: Commit.

### Task 6: Modelos do survey e conta do gestor (Ato 3, H2, H3)

**Files:** Create `src/models/survey_models.py`, `src/models/break_even.py`, `src/models/run_survey.py`, `tests/test_break_even.py`, `tests/test_survey_models.py`

**Interfaces:**
- Consumes: `survey_harmonizado.parquet`
- Produces:
  - `def break_even(sal_jr: float, sal_sr: float, custo_ia_ano: float, prod_jr: float = 0.5) -> float` = ganho mínimo `u*` = `(sal_sr + custo_ia_ano) * (1 + prod_jr) / (sal_sr + sal_jr) - 1`.
  - `def cenarios(sal_jr, sal_sr, custo_ia_ano=2400.0) -> DataFrame` linhas pessimista 0.00 / central 0.15 / otimista 0.30 / estresse −0.19; colunas `cenario, ganho, custo_unidade_sr_ia, custo_unidade_sr_jr, compensa`; mais sensibilidade `prod_jr ∈ {0.3, 0.5, 0.7}`.
  - `def bootstrap_ic(serie, func, n=2000, seed=42) -> tuple[float, float, float]`.
  - `def classificador_ameaca(df) -> dict` (último ano com `ia_ameaca` não-nula; alvo binário "sim" vs resto, documentar mapeamento; features `faixa_exp, cargo, pais (top 10 + Outros), usa_agentes, ia_usa, emprego`; StratifiedKFold 5; Dummy, LogisticRegression, HistGradientBoosting; ROC-AUC e F1 médios ± dp; `interpretavel = auc_hgb - auc_logistica_simples >= 0.03 or auc_melhor - auc_dummy >= 0.03` conforme §7.1; permutation importance no hold-out 20%).
  - `run_survey.main()` grava `data/results/survey_emprego_iniciantes.csv` (por ano, % empregado em tempo integral entre `≤2` com IC bootstrap), `survey_gap_salarial.csv` (mediana sr/jr EUA por ano com IC), `survey_agentes_por_exp.csv`, `classificador.json`, `importancias.csv`, `break_even.csv`, `vereditos_survey.json`.
- [ ] Step 1: `test_break_even`: `break_even(50, 100, 0, 0.5) == pytest.approx(0.0)`; `break_even(50, 100, 50, 0.5) == pytest.approx(0.5)`; ganho ≥ `u*` ⇒ `compensa True`.
- [ ] Step 2: `test_survey_models`: `bootstrap_ic` de série constante 5 → (5,5,5); classificador num DataFrame sintético com alvo determinado por `faixa_exp` → AUC > 0.9 e `faixa_exp` no topo das importâncias.
- [ ] Step 3: FAIL → implementar → PASS → `python -m src.models.run_survey`.
- [ ] Step 4: Commit.

### Task 7: Fatos e referências verificados

**Files:** Create `docs/referencias.md`, `data/processed/fatos_literatura.csv`

**Interfaces:**
- Produces: `fatos_literatura.csv` colunas `id, afirmacao, numero, fonte, url, data_acesso, status (verificado|removido)`; o notebook só cita `status == verificado`.
- [ ] Step 1: Para cada número citado em `docs/pesquisa/pesquisa_dominio.md` que a narrativa usa (Canaries −19%/−20%; Hosseini & Lichtinger; Indeed 27,5% abaixo de 2020 e 71% sênior; PNAS Nexus −25% SO; Copilot +26%; METR −19% e follow-up; horizonte METR dobrando ~7 meses; SWE-bench ~97%), abrir a fonte primária e conferir número e data. Divergiu → corrigir; não achou → `removido`.
- [ ] Step 2: Commit.

### Task 8: Notebook-apresentação, HTML, README e GitHub

**Files:** Create `notebooks/primeiro_degrau.ipynb`, `src/viz.py`, `reports/primeiro_degrau.html`; Modify `README.md`

**Interfaces:**
- Consumes: tudo de `data/processed/` e `data/results/`, `fatos_literatura.csv`.
- Produces: `src/viz.py` com tema e helpers (`linha_com_marcos(ax, df, marcos)`, `faixa_ic(ax, cf)`), paleta definida via skill `dataviz` (uma cor de destaque).

- [ ] Step 1: Estrutura exata do design §6 (seções 0–7 + apêndice), rótulo "CRISP-DM fase X | componente" em cada seção, títulos que afirmam conclusão, bloco "o que vemos → o que significa → o que fazer" após cada gráfico principal, ≤ 8 gráficos principais. Vereditos lidos de `vereditos_*.json` (texto condicional ao resultado real, não pré-escrito).
- [ ] Step 2: Células de código com tag `hide_input` e metadata de colapso; primeira célula com `RECALCULAR = False` (True roda `src` inteiro).
- [ ] Step 3: Passar todo markdown pela skill `humanizer-pt-br`.
- [ ] Step 4: Verificar: `jupyter nbconvert --to notebook --execute --inplace notebooks/primeiro_degrau.ipynb` sem erro, com `data/raw` renomeado temporariamente (Review Focus 5); depois `jupyter nbconvert --to html --no-input notebooks/primeiro_degrau.ipynb --output-dir reports`.
- [ ] Step 5: README: resumo executivo (3–5 linhas com os vereditos), link para o HTML, mapa CRISP-DM, como reproduzir (`pip install -r requirements.txt`, `.env`, `python -m src.etl...`, `python -m src.models.run_series`, `run_survey`), equipe (nomes do README do projeto Azure PdM).
- [ ] Step 6: `gh repo create primeiro-degrau-ia-mercado-dev --private --source . --push` (confirmar com o Victor antes deste passo). Conferir no navegador que o notebook renderiza com saídas no GitHub.
- [ ] Step 7: Revisão final independente (agente verificador): roda do zero, confere vereditos vs. §7.1, procura vazamento temporal e afirmação causal, confere cada número citado com `fatos_literatura.csv`.

## Ordem e paralelismo

Task 1 → (Tasks 2, 3, 4 e 7 em paralelo; Task 2 precisa da chave em `.env`) → (Task 5 após 2+3; Task 6 após 4, em paralelo) → Task 8. Extra B (Streamlit sobre `break_even.csv`/`break_even()`) só se Task 8 terminar com folga.
