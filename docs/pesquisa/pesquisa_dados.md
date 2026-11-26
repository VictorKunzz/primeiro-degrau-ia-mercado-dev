# Pesquisa de fontes de dados: IA generativa e o mercado de desenvolvimento de software

Verificação feita em 25/11/2026 com requisições leves (HEAD, *range* nos primeiros bytes, chamadas de API com poucos itens). Tudo o que aparece como "testado" foi executado; o que é "não verificado" está marcado.

---

## 1. Stack Overflow: volume de perguntas e respostas (2008–2026)

### 1a. API api.stackexchange.com (RECOMENDADA)
- **Endpoint (testado, funciona sem login):**
  `https://api.stackexchange.com/2.3/questions?site=stackoverflow&fromdate=<unix>&todate=<unix>&filter=total`
  Retorna só `{"total": N}`. Com `&tagged=python` filtra por tag. O mesmo vale para `/2.3/answers` (sem filtro de tag).
- **Resultados de teste (perguntas no mês):**

  | Mês | Total |
  |---|---|
  | mar/2014 | 207.169 |
  | mar/2020 | 155.863 |
  | nov/2022 (lançamento do ChatGPT) | 109.119 |
  | nov/2023 | 50.271 |
  | jan/2024 | 47.570 (python: 6.128; respostas: 64.352) |
  | out/2025 | 5.260 |
  | out/2026 | 842 |

- **Formato:** JSON (gzip). Granularidade: o intervalo que você definir (mês, semana ou dia). Período: ago/2008 até hoje.
- **Quota (testada):** sem chave, **300 requisições/dia por IP** (`quota_max: 300`). Com uma *app key* gratuita (registro em stackapps.com/apps/oauth/register, que pede conta Stack Exchange) o limite sobe para 10.000/dia. Respeite o campo `backoff` quando ele vier na resposta.
- **Orçamento:** 2008-08 a 2026-11 são cerca de 220 meses. Para o total geral mais o de respostas são ~440 requisições: dois dias sem chave, ou um dia com chave. Cada tag extra custa mais 220. **Sem chave, dá para fazer o total mensal em um dia e, em outro, o total trimestral de umas 5 tags.** Com chave, dá para fazer tudo mensal (total + 8 a 10 tags) em minutos.
- **Ressalvas:** perguntas deletadas não entram na contagem, e o efeito é maior nos anos antigos. A API não agrupa por mês sozinha, então é uma requisição por mês e por tag.
- **Licença:** conteúdo CC BY-SA. As contagens agregadas podem ser usadas com citação.
- **Script Python:** sim, `requests` + laço de meses + `time.sleep(0.1)`. Já testado.

### 1b. Stack Exchange Data Explorer (SEDE)
- `https://data.stackexchange.com/` respondeu **403** para requisições de script (proteção anti-bot). Rodar *query* nova exige login. O CSV só sai pelo navegador, à mão.
- Não é reproduzível em notebook. Só vale como alternativa manual se não der para registrar a *app key*.

### 1c. Data dump (archive.org, `stackexchange_20251231`)
- São dezenas de GB em 7z. **Inviável em 2 dias.**

---

## 2. Stack Overflow Developer Survey 2022–2026

- O link "Data & files" do site agora aponta para o GitHub `StackExchange/Survey`. Os antigos links `.zip` em `survey.stackoverflow.co/datasets/...` dão **404** (testado).
- Os arquivos estão no Git LFS. Pelo `raw.githubusercontent.com` vem só o ponteiro de 134 bytes. **Use o domínio `media.githubusercontent.com`** (testado, HTTP 200):

| Ano | URL do CSV de respostas | Tamanho |
|---|---|---|
| 2022 | `https://media.githubusercontent.com/media/StackExchange/Survey/main/packages/archive/2022/results.csv` | 109 MB |
| 2023 | `.../archive/2023/results.csv` | 159 MB |
| 2024 | `.../archive/2024/results.csv` | 160 MB |
| 2025 | `.../archive/2025/results.csv` | 141 MB |
| 2026 | `.../archive/2026/survey_results_public.csv` | 122 MB |

  O dicionário de cada ano fica em `.../archive/<ano>/schema.csv` (2026: `survey_results_schema.csv`).
- **Formato:** CSV, uma linha por respondente. Número de respostas (aproximado, não recontado): 2022 ≈ 73 mil, 2023 ≈ 89 mil, 2024 ≈ 65 mil, 2025 ≈ 49 mil.
- **Licença:** ODbL 1.0 para os dados e DbCL 1.0 para o conteúdo. Exige atribuição.
- **Script:** `pd.read_csv(url, usecols=[...])` direto da URL. Recomendo `usecols` para economizar memória.

**Colunas relevantes (verificadas no cabeçalho de cada arquivo):**

| Tema | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|
| Usa IA | — (não há perguntas de IA) | AISelect, AIToolCurrently Using | AISelect, AIToolCurrently Using | AISelect, AIToolCurrently mostly/partially AI, AIAgents | AISelect, AIFreq_1..4, AIDaily, AIWorkCurrentlyUse |
| Sentimento e confiança | — | AISent, AIBen, AIAcc | AISent, AIAcc, AIBen, AIComplex | AISent, AIAcc, AIComplex, AIFrustration | AISentiment, AITrust, AIComplex |
| Ameaça ao emprego | — | — | **AIThreat** | **AIThreat**, NewRole | Disruption, AIImpact_1..7 (conferir no schema) |
| Experiência | YearsCode, YearsCodePro, WorkExp | igual | igual | YearsCode, WorkExp (sem YearsCodePro) | YearsCode, WorkExp |
| Cargo | DevType, MainBranch | DevType, MainBranch | DevType, MainBranch | DevType, MainBranch | **Role**, RoleFunctions |
| Salário | ConvertedCompYearly | igual | igual | igual | ConvertedCompYearly |
| País | Country | Country | Country | Country | Country |
| Outros úteis | EdLevel, Employment, Age, OrgSize | + Industry | + JobSat | + JobSat, LearnCodeAI | JobSat, JobEaseAI_*, JobDifficultyAI_*, LearnAI |

**Atenção na ETL:** os nomes das colunas mudam entre anos (DevType virou Role em 2026; YearsCodePro some em 2025). As respostas de múltipla escolha vêm separadas por `;`. Faixas como "Less than 1 year" e "More than 50 years" precisam de conversão. Para filtrar estudantes e juniores use `MainBranch`/`Employment` e `YearsCode`/`WorkExp` ≤ 2.

---

## 3. Indeed Hiring Lab: vagas de Software Development

### 3a. GitHub hiring-lab/job_postings_tracker (RECOMENDADA)
- Atualizado semanalmente. Último commit: 17/11/2026.
- **URL US setorial (testada):** `https://raw.githubusercontent.com/hiring-lab/job_postings_tracker/master/US/job_postings_by_sector_US.csv`, com 12 MB.
- Colunas: `date, jobcountry, indeed_job_postings_index, variable (total postings / new postings), display_name (setor)`.
- Inclui os setores "Software Development", "Data & Analytics", "IT Systems & Solutions" e outros ~45 para comparação (Nursing, Accounting...).
- **Período:** 01/02/2020 a 13/11/2026, **diário** (média móvel de 7 dias, dessazonalizada). Base 100 = 01/02/2020. Último valor de Software Development: 81,7 (total) e 88,2 (new).
- **Outros países (testados):** existe `job_postings_by_sector_{XX}.csv` para GB, DE, CA, FR e AU, e também ES, IT, NL e EA (sem setorial para IE). O GB tem Software Development. **Não há Brasil.**
- Também há o agregado nacional `aggregate_job_postings_US.csv` (200 KB) e versões por estado e por metrópole.
- **Licença:** CC BY 4.0. **Script:** `pd.read_csv(url)`.

### 3b. FRED IHLIDXUSTPSOFTDEVE
- `https://fred.stlouisfed.org/graph/fredgraph.csv?id=IHLIDXUSTPSOFTDEVE` funciona sem chave (testado). É a mesma série (US, diária, 2020-02-01 a 2026-11-13) em CSV de duas colunas. Bom como *backup* ou para ser citado no relatório.

---

## 4. Métricas de capacidade de IA com data

### 4a. Epoch AI: Benchmarking Hub (RECOMENDADA)
- **ZIP (testado, 2,3 MB):** `https://epoch.ai/data/benchmark_data.zip`. Tem um CSV por benchmark e foi atualizado em 26/11/2026.
  - `swe_bench_verified.csv`: 49 modelos, colunas `Model version, mean_score, Best score, Release date (2024-11 a 2026-06), Organization, Training compute`.
  - `metr_time_horizons_external.csv`: 100 linhas, colunas `Model version, Time horizon (min), Time Horizon (80%), CI_low/high, Release date (2019-11 a 2026-04), Organization`. **É o dado do METR em formato tabular.**
  - Outros: aider_polyglot, terminalbench, webdev_arena, frontierswe, cursorbench, gpqa_diamond, mmlu, arc_agi etc.
- **Modelos:** `https://epoch.ai/data/notable_ai_models.csv` (1.102 modelos, 1950 a nov/2026; datas, organização, parâmetros, compute, acessibilidade). Também há `all_ai_models.csv` e `large_scale_ai_models.csv`. Use `encoding_errors='replace'` no pandas porque há caracteres inválidos.
- **Licença:** CC BY 4.0, com citação "Epoch AI, Capabilities & benchmarking".

### 4b. SWE-bench (leaderboard oficial)
- `https://raw.githubusercontent.com/SWE-bench/swe-bench.github.io/master/data/leaderboards.json` (4 MB, testado). Traz os leaderboards Verified (180 entradas), Lite, Test, Multilingual e Multimodal.
- Campos: `name, date (2023-10 a 2026-02), resolved (% máx. 79,2), model_release_date, model_org, cost, os_model`. JSON, `pd.DataFrame(j['leaderboards'][i]['results'])`.
- Complementa a Epoch porque começa antes (out/2023). Mistura agentes e modelos, então o recomendado é pegar o máximo acumulado por data ("fronteira").

### 4c. METR
- O repositório `github.com/METR/eval-analysis-public` existe (dados em DVC). O caminho mais simples é a cópia tabular da Epoch (4a).

---

## 5. Fontes opcionais

### 5a. BLS (EUA): emprego
- **CES pela API v2 (testada, sem chave: 25 séries por consulta, 10 anos, 25 consultas/dia; com chave gratuita por e-mail o limite sobe):**
  POST `https://api.bls.gov/publicAPI/v2/timeseries/data/` com `{"seriesid":[...],"startyear":..,"endyear":..}`
  - `CES6054151101`: Custom computer programming services (dessazonalizada), mensal. Último valor: set/2026 = 1.058 mil.
  - `CEU6054150001`: Computer systems design and related services. Último valor: out/2026 = 2.369 mil.
  - Mensal desde 1990. Domínio público.
- **OEWS (ocupação 15-1252 Software Developers):** a API só devolveu 2025 (1.687.890 empregados; 15-1254 Web Developers = 148.100). Os arquivos históricos `https://www.bls.gov/oes/special-requests/oesmYYnat.zip` dão **403 por script** e precisam ser baixados à mão no navegador (alguns MB cada). Os dados são anuais.

### 5b. GitHub Innovation Graph
- `https://raw.githubusercontent.com/github/innovationgraph/main/data/developers.csv` (94 KB), `languages.csv` (6 MB), `git_pushes.csv`, `repositories.csv`, `topics.csv` e outros.
- **Trimestral por país (inclui BR)**, 2020T1 a 2026T2. Colunas: `developers, iso2_code, year, quarter` e, em languages, `num_pushers, language`. Licença CC0 (conferir o README do repositório). `pd.read_csv` direto.
- Serve de contraponto: a atividade no GitHub cresce enquanto o Stack Overflow cai.

### 5c. layoffs.fyi
- Os dados ficam num Airtable embutido (`airtable.com/embed/app1PaujS9zxVGUZ4/...`). Não há CSV oficial. Só dá para usar exportando à mão pelo Airtable, ou com cópias no Kaggle de origem e licença incertas. Os termos de uso não foram verificados.

### 5d. Novo CAGED (Brasil)
- Arquivos em FTP público: `ftp://ftp.mtps.gov.br/pdet/microdados/NOVO CAGED/<ano>/<aaaamm>/CAGEDMOV<aaaamm>.7z`. Testado: set/2025 tem 54 MB, e há dados até set/2026.
- São microdados de movimentações (admissões e desligamentos) com CBO, salário, UF e idade. Para TI, filtre CBO 2124-05 (analista de desenvolvimento de sistemas) e 3171-xx (técnicos de programação).
- Exige `py7zr`. De 2020 a 2026 são ~80 arquivos e ~4 GB comprimidos. Em 2 dias, só uma amostra (ex.: um mês por trimestre) ou a Base dos Dados (BigQuery, exige conta Google).

---

## Tabela-resumo

| # | Fonte | Granularidade / período | Acesso via script | Classificação |
|---|---|---|---|---|
| 1a | Stack Exchange API (`filter=total`) | mensal, 2008–2026, por tag | sim (300 req/dia sem chave) | **Pronta para usar** (planejar a quota) |
| 1b | SEDE (Data Explorer) | livre | não (403/login) | Usável com esforço (manual) |
| 1c | Data dump SO | evento | arquivos de dezenas de GB | Inviável em 2 dias |
| 2 | SO Developer Survey 2022–2026 | anual, por respondente | sim (`media.githubusercontent.com`) | **Pronta para usar** (ETL de colunas entre anos) |
| 3a | Indeed job_postings_tracker (setor) | diária, fev/2020–nov/2026, 9 países | sim | **Pronta para usar** |
| 3b | FRED IHLIDXUSTPSOFTDEVE | diária, US | sim | **Pronta para usar** |
| 4a | Epoch AI (benchmarks + modelos, inclui METR e SWE-bench) | por modelo/data | sim (zip de 2 MB) | **Pronta para usar** |
| 4b | SWE-bench leaderboards.json | por submissão, 2023–2026 | sim | **Pronta para usar** |
| 4c | METR repositório próprio | por modelo | DVC | Usável com esforço (use a 4a) |
| 5a | BLS CES API | mensal, 1990–2026 | sim | **Pronta para usar** |
| 5a | BLS OEWS histórico | anual | não (403, manual) | Usável com esforço |
| 5b | GitHub Innovation Graph | trimestral, por país | sim | **Pronta para usar** |
| 5c | layoffs.fyi | evento | não (Airtable) | Usável com esforço / licença incerta |
| 5d | Novo CAGED (TI) | mensal, microdados | sim, porém pesado (7z, ~54 MB/mês) | Inviável completo em 2 dias (amostra é usável com esforço) |
