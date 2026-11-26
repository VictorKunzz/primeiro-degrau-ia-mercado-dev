# Design — "O Primeiro Degrau": IA generativa e o mercado de desenvolvimento de software

Data: 26/11/2026 · Entrega: 27/11/2026 às 19:00 · Disciplina: Ciência de Dados (Católica SC)

## 1. Intenção e porquê

Atividade "Data Storytelling": escolher um problema de Ciência de Dados e aplicar as 6 fases do CRISP-DM, ETL, EDA, Análise Preditiva e Data Storytelling (fluxo dados → informação → conhecimento → decisão estratégica). Apresentação de 5–10 min só para o professor, **feita no próprio notebook** (sem slides).

A equipe já fez crédito (German Credit), sentimentos (KaBuM), C-MAPSS e Azure PdM. Queremos um tema diferente, ligado ao curso, que todos consigam defender e que gere uma tese útil para o setor. O critério central é o **storytelling**: tudo é construído em torno de uma história com decisão no fim.

**Tese (da equipe):** a IA não acabou com o emprego em software; ela **removeu o primeiro degrau**. Cada salto de autonomia da IA (chat → assistente na IDE → agente que executa na máquina) esvaziou primeiro o lugar onde o júnior aprendia (Stack Overflow) e depois as vagas onde ele começava. Para o gestor, um sênior com IA custa menos que sênior + júnior; o mercado fica mais qualificado e exigente.

**Tese rival (precisa ser testada com a mesma seriedade):** a queda de 2022–24 foi cíclica (juros, pós-boom); a IA muda a composição, não o volume.

**Decisores da história:** (1) o estudante escolhendo carreira; (2) o gestor reestruturando equipe. O final traz uma recomendação para cada um, com a tensão entre os dois ("cortar o júnior economiza hoje e seca a fonte de sêniores de amanhã").

## 2. Objetivos, não-objetivos, anti-objetivos

**Objetivos**
- Cobrir explicitamente as 6 fases do CRISP-DM e os 4 componentes (ETL, EDA, Preditiva, Storytelling), com rótulo em cada seção.
- Testar 3 hipóteses com dados públicos e uma conclusão honesta, mesmo que inconclusiva.
- Notebook que roda do zero, de cima a baixo, offline (lê dados já processados), em poucos segundos.
- Apresentável em 5–10 min com código recolhido.

**Não-objetivos**
- Provar causalidade. Linguagem: "consistente com", sempre com intervalo de confiança.
- Medir candidatos por vaga (não há dado público aberto); tratado só com fontes citadas.
- Dados do Brasil (Indeed não tem BR; CAGED inviável no prazo).

**Anti-objetivos** (falha mesmo que "funcione")
- Overfitting ou ajuste de modelo para "confirmar" a tese.
- Atribuir à IA o que é efeito de juros/ciclo sem controle.
- Número citado sem fonte verificada.
- Texto com cara de IA; gráficos sem título interpretativo; terminar sem decisão; estourar 10 min.

## 3. Restrições

- Prazo: ~1 dia. Dados públicos, gratuitos, baixáveis por script.
- Stack Exchange API: 10.000 req/dia com chave (chave em `.env`, fora do git); sem chave, 300/dia.
- Surveys ~110–160 MB cada: brutos fora do git; processados (só colunas usadas) no repo.
- Ambiente: Python 3 + Jupyter no VS Code; deve abrir no Colab.

## 4. Abordagens consideradas

| | Abordagem | Trade-off |
|---|---|---|
| **A (escolhida)** | Notebook-apresentação no VS Code + ETL em scripts `src/`; dados processados em arquivos leves; HTML sem código de backup | Mais robusto ao vivo e para a nota; conta júnior × sênior em cenários fixos |
| B (extra se sobrar tempo) | A + mini painel Streamlit com sliders para o ponto de equilíbrio | Impacto visual; mais uma peça que pode falhar e consome horas |
| C (descartada) | Databricks ou Power BI | Spark desnecessário para dados pequenos; BI fraco na parte preditiva |

Victor escolheu **A, com B como extra**.

## 5. Dados (verificados em 25–26/11/2026)

| Fonte | Acesso | Uso |
|---|---|---|
| Stack Exchange API `/2.3/questions?site=stackoverflow&fromdate&todate&filter=total[&tagged=]` | 1 req = 1 mês | Série mensal de perguntas 2008→out/2026, total e por tag (python, javascript, java + 1–2 a critério) |
| SO Developer Survey 2023–2026 (`media.githubusercontent.com/media/StackExchange/Survey/main/packages/archive/<ano>/...`) | CSV via Git LFS | Uso/confiança/ameaça de IA (AISelect, AISent, AIAcc, AIThreat…), YearsCode, WorkExp, DevType/Role, Employment, ConvertedCompYearly, Country. Nomes mudam por ano → mapeamento na ETL |
| Indeed Hiring Lab `hiring-lab/job_postings_tracker/master/US/job_postings_by_sector_US.csv` | CSV 12 MB | Índice diário de vagas por setor (fev/2020→nov/2026); Software Development vs. ~45 setores e vs. total |
| Epoch AI `epoch.ai/data/benchmark_data.zip` (swe_bench_verified, metr_time_horizons_external) | ZIP 2 MB | Índice de capacidade/autonomia por data → mensal (melhor valor acumulado) |
| Opcional: BLS CES (CES6054151101) | API | Emprego mensal em programação/serviços de TI EUA (robustez) |

Marcos (anotações): GitHub Copilot GA (jun/2022), ChatGPT (30/11/2022), GPT-4 (mar/2023), primeiros agentes de código (2024), agentes de terminal/IDE (2025). Datas exatas conferidas pelo verificador.

## 6. Estrutura do notebook (o "como" decidido)

Título de cada seção = conclusão (action title). Após cada gráfico: **o que vemos → o que significa → o que fazer**. No máximo ~8 gráficos principais; demais em apêndice. Uma cor de destaque; linha anotada nos marcos.

0. **Resumo executivo** (pirâmide/SCQA: resposta primeiro) + tabela de mapeamento CRISP-DM × componentes.
1. **Entendimento do Negócio** (CRISP 1): situação, decisores, H1–H3, tese rival, critérios de sucesso.
2. **Entendimento dos Dados / ETL-Extract** (CRISP 2): fontes, cobertura, qualidade.
3. **Preparação / ETL-Transform** (CRISP 3): agregação mensal, harmonização dos surveys, índice de capacidade, joins.
4. **EDA**: os três atos visualizados.
5. **Modelagem / Preditiva** (CRISP 4):
   - **Ato 1 — Stack Overflow contrafactual:** SARIMAX/ETS em log(y), treino até out/2022, previsão com IC 95%; efeito = real − previsto (%). Backtest rolling-origin no pré-período vs. baseline sazonal ingênuo. Placebos em datas falsas (ex.: 2019, 2020). Robustez: regressão segmentada (ITS) com erros HAC. Por tag como complemento.
   - **Ato 2 — Vagas:** software vs. outros setores e vs. total (razão / diferença), para separar IA de ciclo. Cruzar com índice de capacidade. Espera-se resultado possivelmente inconclusivo — reportar assim.
   - **Ato 3 — Survey:** classificador de percepção de ameaça (AIThreat), ano mais recente com a pergunta; ColumnTransformer, StratifiedKFold, baseline Dummy/logística → HistGradientBoosting; ROC-AUC, F1; permutation importance. Descritivo entre anos só com perguntas idênticas: situação de emprego dos iniciantes (≤2 anos), gap salarial sênior/júnior 2023→2026, uso de IA/agentes por experiência.
   - **Conta do gestor (H2):** ponto de equilíbrio sênior+IA vs. sênior+júnior com salários do survey (mediana, EUA), custo de ferramentas de IA e faixa de produtividade da literatura (cenários pessimista/central/otimista). Rotulada como simulação.
6. **Avaliação** (CRISP 5): backtest, placebos, limitações (autosseleção do survey, perguntas deletadas fora da API, confundidores), veredito por hipótese (confirmada / parcial / não sustentada).
7. **Implantação** (CRISP 6): recomendações para estudante e gestor, KPIs a monitorar, como atualizar (rodar `src/`), ciclo do CRISP.
- Apêndice: referências (estudos citados com link), detalhes técnicos.

## 7. Hipóteses → evidência

| Hipótese | Dado próprio | Apoio citado |
|---|---|---|
| H1 primeiro degrau sumiu; vagas concentradas e disputadas | Survey: emprego dos ≤2 anos por ano; Indeed software vs. setores; contrafactual SO | Brynjolfsson et al. "Canaries" (Stanford/ADP); Hosseini & Lichtinger (Harvard); Indeed Hiring Lab |
| H2 sênior+IA < sênior+júnior | Salários do survey + simulação | RCTs Copilot (+26%), METR 2025 (−19%) |
| H3 mercado mais qualificado/exigente | Gap salarial e uso de agentes por experiência; classificador | Indeed (ganho concentrado em sênior) |

## 7.1 Regras de veredito (definidas antes de rodar) — *decisão de projeto*

- **Ato 1 (SO):** "efeito sustentado" se o real ficar abaixo do IC 95% do contrafactual por ≥ 6 meses seguidos após nov/2022 **e** nenhum placebo produzir efeito de magnitude ≥ metade do efeito real; o backtest do pré-período precisa bater o baseline sazonal ingênuo (MAPE menor). Caso contrário: "inconclusivo".
- **Ato 2 (vagas):** "consistente com IA" só se software cair mais que a mediana dos setores **depois de** controlar juros (Fed Funds, FRED `FEDFUNDS`, como covariável) e a diferença for significativa (HAC, p < 0,05). Caso contrário: "consistente com ciclo".
- **Survey (H1/H3):** diferenças entre anos/grupos só contam com IC 95% por bootstrap que não cruze zero.
- **Classificador:** só é interpretado se ROC-AUC exceder o baseline Dummy/logística simples em ≥ 0,03 na validação cruzada.
- Veredito por hipótese: confirmada (todas as evidências próprias passam), parcial (algumas), não sustentada (nenhuma).

## 7.2 Premissas da conta do gestor — *decisão de projeto*

Parametrizada, mostrada em 3 cenários + 1 teste de estresse: produtividade do sênior com IA +0% (pessimista), +15% (central), +30% (otimista), −19% (estresse, METR 2025); produção do júnior = 50% da de um sênior (premissa explícita, com sensibilidade 30–70%); salários = mediana do survey mais recente (EUA, júnior ≤2 anos, sênior ≥10 anos); custo de IA = US$ 200/mês por dev (plano de agente). Saída: ganho de produtividade mínimo que torna sênior+IA mais barato que sênior+júnior por unidade de trabalho.

## 7.3 Papel do classificador na história — *decisão de projeto*

Testa H3: se experiência, cargo e uso avançado de IA (agentes) forem os fatores que mais **reduzem** a percepção de ameaça, isso sustenta "qualificação + domínio de IA é a proteção". Decisão que apoia: o que o estudante deve priorizar.

## 7.4 Execução — *decisão de projeto*

Cálculos pesados (backtest rolling, placebos, CV) rodam em `src/models/` e salvam resultados em `data/results/`; o notebook lê resultados e recalcula só o leve (ajuste final do SARIMAX, gráficos). Uma flag `RECALCULAR = False` no topo permite rodar tudo do zero.

## 8. Repositório e execução

Pasta: `IA e o Primeiro Degrau - Mercado Dev/`, repo GitHub **privado**.

**Entrega (decisão do Victor): link do repositório.** O repo precisa estar completo até 27/11 19:00: notebook executado (com saídas visíveis no GitHub), HTML em `reports/`, README com resumo executivo, link para o HTML e instruções de execução. Antes da entrega, o Victor dá acesso ao professor (convite de colaborador pelo usuário GitHub dele) ou torna o repo público — decidido na hora da entrega.

```
data/raw/        (gitignored)    data/processed/ (parquet/csv, versionado)
src/             etl_stackoverflow.py, etl_survey.py, etl_indeed.py, etl_capacidade.py, run_all.py
notebooks/       primeiro_degrau.ipynb (executado, código recolhido)
reports/         primeiro_degrau.html (nbconvert --no-input)
docs/            design.md, pesquisa/ (relatórios)
.env.example, requirements.txt, README.md
```

## 9. Divisão do trabalho

1. Agente ETL (primeiro): `src/` + `data/processed/`.
2. Agente séries temporais (paralelo ao 3): Atos 1 e 2, funções + figuras.
3. Agente survey (paralelo ao 2): Ato 3 + conta do gestor.
4. Orquestração: monta o notebook, narrativa (`dataviz`, `humanizer-pt-br`).
5. Agente verificador: confere citações/números, roda do zero, caça vazamento e afirmação causal.
Extra B (Streamlit) só se 1–5 terminarem com folga.

## 10. Deixado ao construtor

Hiperparâmetros e ordens do SARIMAX; tags exatas (2–4); escolha de ano do survey para o classificador conforme disponibilidade de AIThreat; paleta e biblioteca de gráficos (matplotlib/plotly estáticos que exportem para HTML); formato parquet vs. csv.
