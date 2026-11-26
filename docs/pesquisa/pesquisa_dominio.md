# Pesquisa de domínio: IA generativa e o mercado de trabalho de desenvolvimento de software (2021–2026)

*Levantamento feito em 25/11/2026 para o trabalho de Data Storytelling + CRISP-DM. Decisores: (1) estudante/júnior escolhendo carreira; (2) gestor/empregador reestruturando equipes.*

**Legenda de confiança**
- **[V]** verificado nesta pesquisa em fonte primária ou em página oficial
- **[S]** vem de fonte secundária ou agregadora; conferir antes de citar número exato
- **[M]** dado amplamente conhecido até 2025 (anúncios oficiais e papers), não reaberto hoje; vale conferir a data no link

---

## 0. Resumo executivo

1. **A capacidade cresceu de forma exponencial e mensurável.** O "horizonte de tarefa" do METR (duração de uma tarefa humana que um agente completa com 50% de sucesso) dobrou a cada ~7 meses entre 2019 e 2025, e mais rápido (~4 meses) em 2024–2025. A escala passou de segundos (GPT-2) para ~5 min (GPT-4, 2023), ~1 h (Claude 3.7, fev/2025), ~4h49 (Claude Opus 4.5, nov/2025) e para mais de 8 h nos modelos de 2026, já perto do teto da suíte. O SWE-bench Verified foi de ~2% (2023) a ~97% (set/2026) e está **saturado**. Os dois índices têm dados baixáveis.
2. **No mercado, o efeito aparece concentrado nos jovens.** No ADP/Stanford ("Canaries"), o emprego de 22–25 anos em ocupações muito expostas à IA está **~19% abaixo** do contrafactual (jun/2026). Nos desenvolvedores de software de 22–25 anos, o recuo foi de quase 20% entre o pico do fim de 2022 e jul/2025. Os trabalhadores experientes não tiveram queda. Hosseini & Lichtinger (Harvard, 62 milhões de currículos) chegam a padrão parecido: "mudança tecnológica enviesada por senioridade".
3. **Os dados agregados não mostram uma onda de desemprego.** Yale Budget Lab e Brookings não encontram ruptura macro. Juros, a correção pós-pandemia e o fim do boom de contratação de 2021–22 explicam boa parte da queda de vagas de software (ainda ~27,5% abaixo de fev/2020).
4. **Reviravolta de 2025–26.** As vagas de software no Indeed subiram ~15% desde o lançamento do Claude Code (fev/2025) enquanto o total de vagas caiu 7%. Cerca de **71% desse ganho foi em vagas sênior** e 37% em títulos com "AI". A demanda voltou, mas em outro formato: mais sênior e mais IA.
5. **A produtividade é contestada.** Os RCTs com Copilot mostram +26% de tarefas, com ganho maior para juniores. O RCT do METR (2025) mostrou devs experientes **19% mais lentos**, e o follow-up de 2026 sugere efeito perto de zero ou levemente positivo, com muita incerteza.

---

## 1. Linha do tempo de capacidade e autonomia da IA para código (2021–2026)

### 1.1 Marcos de produto e modelo

| Data | Marco | Por que importa | Fonte |
|---|---|---|---|
| 29/06/2021 | GitHub Copilot (technical preview), baseado no OpenAI Codex | Primeiro autocompletar de código com LLM em larga escala | [GitHub blog](https://github.blog/2021-06-29-introducing-github-copilot-ai-pair-programmer/) [M] |
| 07/2021 | Paper do Codex / benchmark HumanEval (Codex 12B ≈ 28,8% pass@1) | Primeira métrica padrão de geração de código | [arXiv 2107.03374](https://arxiv.org/abs/2107.03374) [M] |
| 21/06/2022 | Copilot GA (pago) | Começa a adoção comercial | [GitHub blog](https://github.blog/2022-06-21-github-copilot-is-generally-available-to-all-developers/) [M] |
| **30/11/2022** | **ChatGPT** | Choque de adoção; data de referência ("pós-ChatGPT") de quase todos os estudos de mercado | [OpenAI](https://openai.com/index/chatgpt/) [M] |
| 14/03/2023 | GPT-4 (HumanEval ~67%) | Salto de qualidade; "GPT-4 exposure" vira métrica de exposição ocupacional (Eloundou et al.) | [OpenAI](https://openai.com/research/gpt-4) · [Eloundou et al. 2023](https://arxiv.org/abs/2303.10130) [M] |
| 03/2023 | Claude (1ª versão pública) | Concorrência entre laboratórios | [Anthropic](https://www.anthropic.com/news/introducing-claude) [M] |
| 2023 | Cursor (IDE com IA, Anysphere) | IDE "AI-native" | [cursor.com](https://cursor.com) [M] |
| 10/2023 | Paper SWE-bench (issues reais do GitHub). Claude 2 resolvia ~2% | Benchmark de engenharia "real" | [arXiv 2310.06770](https://arxiv.org/abs/2310.06770) [M] |
| 12/03/2024 | **Devin** (Cognition), "primeiro engenheiro de software de IA", 13,9% no SWE-bench | Narrativa de agente autônomo | [Cognition](https://www.cognition.ai/blog/introducing-devin) [M] |
| 20/06/2024 | Claude 3.5 Sonnet | Modelo que popularizou o Cursor e o uso agêntico | [Anthropic](https://www.anthropic.com/news/claude-3-5-sonnet) [M] |
| 13/08/2024 | **SWE-bench Verified** (500 tarefas validadas por humanos). GPT-4o 33,2% | Série histórica principal do índice | [OpenAI](https://openai.com/index/introducing-swe-bench-verified/) [M] |
| 22/10/2024 | Claude 3.5 Sonnet (new) 49% Verified; "computer use" | Primeiro modelo perto de 50% | [Anthropic](https://www.anthropic.com/news/3-5-models-and-computer-use) [M] |
| **25/11/2024** | **Model Context Protocol (MCP)** | Padrão aberto para ligar agentes a ferramentas e dados. Adotado depois por OpenAI, Google e Microsoft | [Anthropic](https://www.anthropic.com/news/model-context-protocol) [M] |
| 12/2024 | OpenAI o3 anunciado (71,7% Verified) | Modelos de raciocínio | [OpenAI](https://openai.com/index/deliberative-alignment/) [M] |
| **24/02/2025** | Claude 3.7 Sonnet + **Claude Code** (research preview) | Agente de terminal. O Indeed usa esta data como marco | [Anthropic](https://www.anthropic.com/news/claude-3-7-sonnet) [M] · [Indeed](https://hiringlab.indeed.com/?p=20708) [V] |
| 16/04/2025 | OpenAI Codex CLI; 16/05/2025 Codex (agente na nuvem) | Agentes assíncronos que abrem PRs | [OpenAI](https://openai.com/index/introducing-codex/) [M] |
| 05/2025 | GitHub Copilot coding agent; Claude 4 (Opus/Sonnet ~72–73% Verified); Claude Code GA (22/05) | Tarefas delegadas (issue → PR) | [GitHub](https://github.blog/news-insights/product-news/github-copilot-meet-the-new-coding-agent/) · [Anthropic](https://www.anthropic.com/news/claude-4) [M] |
| 25/06/2025 | Gemini CLI (Google) | Todos os grandes laboratórios passam a ter agente de terminal | [Google](https://blog.google/technology/developers/introducing-gemini-cli-open-source-ai-agent/) [M] |
| 07/08/2025 | GPT-5 (~74,9% Verified; METR ≈ 2h17) | | [OpenAI](https://openai.com/index/introducing-gpt-5/) [M] |
| 29/09/2025 | Claude Sonnet 4.5 (~77% Verified) | | [Anthropic](https://www.anthropic.com/news/claude-sonnet-4-5) [M] |
| 18–24/11/2025 | Gemini 3 Pro; Claude Opus 4.5 (~80,9% Verified; METR ≈ 4h49) | | [Anthropic](https://www.anthropic.com/news/claude-opus-4-5) [M] · [METR](https://metr.org/time-horizons/) [S] |
| 29/01/2026 | METR Time Horizon 1.1 (suíte de 170 → 228 tarefas) | Recalibra a série | [METR](https://metr.org/blog/2026-01-29-time-horizon-1-1/) [V] |
| 2026 | Novas gerações (Claude Opus 4.6/4.7/5, GPT-5.x-Codex, Gemini 3.x etc.). Verified ~97% em set/2026 (Vals.ai), que deixou de rodar o benchmark por saturação | Fim da utilidade do Verified como métrica de fronteira | [Vals.ai](https://www.vals.ai/benchmarks/swebench) [S] |
| 2026 | SWE-bench Pro (Scale): topo ~60% no placar padronizado (set/2026) | Novo benchmark mais difícil | [Scale leaderboard](https://labs.scale.com/leaderboard/swe_bench_pro_public) [S] |

### 1.2 Métricas quantitativas para um "índice de capacidade" datado

**(a) METR: horizonte de tempo de tarefa (50%)**
- Paper: Kwa et al., "Measuring AI Ability to Complete Long Tasks", mar/2025 ([arXiv 2503.14499](https://arxiv.org/abs/2503.14499); [blog METR](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/)) [V]. Doubling de ~7 meses (2019–2025), com doubling mais rápido se o ajuste usar só 2024–25.
- TH1.1 (29/01/2026): 228 tarefas, 31 delas com 8 h ou mais. As estimativas ficam dentro dos ICs da TH1 ([METR](https://metr.org/blog/2026-01-29-time-horizon-1-1/)) [V]. Nota de 20/03/2026: com a suíte perto da saturação, escolhas de modelagem alteram o p50 em até −35% ([METR note](https://metr.org/notes/2026-03-20-impact-of-modelling-assumptions-on-time-horizon-results/)) [V]. O METR avisa que medidas acima de ~16 h não são confiáveis ([metr.org/time-horizons](https://metr.org/time-horizons/)) [S].
- Pontos aproximados: GPT-2 (2019) ~2 s; GPT-3.5 (2022) ~30 s a 1 min; GPT-4 (mar/2023) ~5 min; Claude 3.5 Sonnet (2024) ~20–30 min; Claude 3.7 (fev/2025) ~1 h; o3 (abr/2025) ~1,5 h; GPT-5 (ago/2025) ~2h17; Opus 4.5 (nov/2025) ~4h49; modelos de 2026 com 8–11 h em resumos secundários [S].
- **Dados baixáveis:** YAML com os resultados em `https://metr.org/assets/benchmark_results_1_1.yaml` e código em [github.com/METR/eval-analysis-public](https://github.com/METR/eval-analysis-public) [V]. A Epoch replica a série em [epoch.ai/benchmarks/metr-time-horizons](https://epoch.ai/benchmarks/metr-time-horizons).

**(b) SWE-bench Verified por data**
- Trajetória aproximada (melhor modelo da época): out/2023 ~2% (SWE-bench completo) → mar/2024 ~14% (Devin, completo) → ago/2024 33% → out/2024 49% → dez/2024 ~72% (o3) → fev/2025 ~70% → mai/2025 ~73% → set/2025 ~77% → nov/2025 ~81% → 2026 ~88% → set/2026 ~97% (Vals.ai) [M/S].
- **Dados baixáveis:** Epoch AI Benchmarking Hub, ZIP com CSVs: [epoch.ai/data/benchmark_data.zip](https://epoch.ai/data/benchmark_data.zip) (página "Use this data", atualizada 26/11/2026, licença CC-BY) ([epoch.ai/benchmarks/use-this-data](https://epoch.ai/benchmarks/use-this-data)) [V]. Também há o cliente Python `pip install epochai`. Página do benchmark: [epoch.ai/benchmarks/swe-bench-verified](https://epoch.ai/benchmarks/swe-bench-verified). A Epoch usa 484 das 500 amostras, e o scaffold mudou em fev/2026, o que quebra parcialmente a série [V]. O placar oficial fica em [swebench.com](https://www.swebench.com).
- O ZIP da Epoch é pequeno (dados tabulares) e pode ser baixado. Outras séries úteis da Epoch: [AI Benchmarking Dashboard](https://epoch.ai/data/ai-benchmarking-dashboard) e "Notable AI Models" (data de lançamento, compute).

**(c) Proposta de índice de capacidade (para cruzar com séries mensais de mercado)**
- Variável mensal = log2(horizonte METR p50 da fronteira em minutos), interpolada entre lançamentos (função degrau ou interpolação log-linear). Opcionalmente, z-score combinado com o SWE-bench Verified até 2025 (antes da saturação) e o SWE-bench Pro depois.
- Use dummies de evento para os marcos de adoção: ChatGPT (nov/2022), GPT-4 (mar/2023), Claude Code e agentes (fev–mai/2025), MCP (nov/2024).
- Cuidado: capacidade não é adoção. Para adoção, use o Stack Overflow Developer Survey (anual) e o Anthropic Economic Index (ver §2.6).

---

## 2. Evidências de impacto no mercado de trabalho

### 2.1 Emprego por idade e senioridade (evidência mais forte disponível, mas descritiva)

**Brynjolfsson, Chandar & Chen, "Canaries in the Coal Mine? Six Facts about the Recent Employment Effects of AI"** (Stanford Digital Economy Lab + ADP)
- Versão original: ago/2025 ([página](https://digitaleconomy.stanford.edu/publications/canaries-in-the-coal-mine)). Revisão de nov/2025 ([PDF](https://digitaleconomy.stanford.edu/app/uploads/2025/11/CanariesintheCoalMine_Nov25.pdf)). Revisão de ago/2026 ([notícia 12/08/2026](https://digitaleconomy.stanford.edu/news/canariesaug26/); [PDF](https://digitaleconomy.stanford.edu/app/uploads/2026/08/Canaries_August2026.pdf)) [V].
- Dados: folha de pagamento da ADP, ~25 mil firmas, 4,6 milhões de trabalhadores, mais de 730 ocupações. Exposição medida por GPT-4 β (Eloundou) e pelo Anthropic Economic Index [V].
- Achados:
  - O emprego de 22–25 anos nas ocupações mais expostas está **~19% abaixo** do contrafactual (jun/2026). Era ~15–16% na versão de 2025 [V].
  - Em nível absoluto, os dois quintis mais expostos caíram ~11% entre nov/2022 e jun/2026, enquanto os três menos expostos cresceram ~10% (mesma faixa etária) [S].
  - **Desenvolvedores de software de 22–25 anos:** queda de quase 20% do pico (fim de 2022) até jul/2025, na versão original [M].
  - Trabalhadores experientes nas mesmas ocupações ficaram estáveis ou cresceram [V].
  - O ajuste acontece pela **contratação** (menos admissões de jovens), não por demissões, e aparece no emprego, não no salário-base [V].
  - O efeito se concentra onde a IA **automatiza**. Onde ela **complementa**, o emprego fica estável ou cresce [V].
  - O padrão persiste quando se excluem firmas de tecnologia e ocupações de computação [V].
- Nota de 09/02/2026: juros não explicam bem o padrão. Com controles mais rígidos, a queda só fica significativa depois de 2024 ([link](https://digitaleconomy.stanford.edu/news/canaries-interest-rates-and-timinga-more-on-recent-drivers-of-employment-changes-for-young-workers)) [V].
- Limitações declaradas: descritivo, não causal; parte das tendências é anterior à IA generativa; o efeito é maior na amostra ADP do que nas pesquisas nacionais [V].
- **Dashboard mensal** ([Canaries Dashboard](https://digitaleconomy.stanford.edu/project/canaries-dashboard/)), atualizado em 21/10/2026. Cada gráfico tem link "Data" para um ZIP (ex.: `canaries_by_exposure_results.zip`). São arquivos agregados e pequenos, **utilizáveis no trabalho** [V].

**Hosseini & Lichtinger (Harvard), "Generative AI as Seniority-Biased Technological Change: Evidence from U.S. Résumé and Job Posting Data"**
- SSRN 5425555 (2025); rascunho de mai/2026 ([PDF](https://raw.githack.com/s-mahdihosseini/Website/b99fa2962dcc4c165bbc3c41dd283afa99c5ccfb/GenAI_SenBTC_May_2026.pdf)); [seminário Stanford](https://digitaleconomy.stanford.edu/event/seyed-m-hosseini-and-guy-lichtinger-generative-ai-as-seniority-biased-technological-change-evidence-from-u-s-resume-and-job-posting-data) [S].
- 62 milhões de trabalhadores em 285 mil firmas (2015–2025). A adoção é identificada por vagas de "GenAI integrator", com diferenças em diferenças.
- A partir de 2023T1, o emprego **júnior** cai nas firmas adotantes em relação às não adotantes, e o **sênior** continua subindo. O efeito vem principalmente de menos contratação de juniores [S].

**Contraponto macro (forte, mas com outra pergunta: "há choque agregado?")**
- Yale Budget Lab: a mudança no mix ocupacional é um pouco mais rápida que a histórica, mas não se alinha claramente à IA. Em "AI Is Probably Not (Yet) the Reason for Labor Market Weakening", ainda não há indicação clara de efeitos ([tracker abr/2026 PDF](https://budgetlab.yale.edu/node/1419/pdf); [o que sabemos e não sabemos](https://budgetlab.yale.edu/research/what-we-do-and-dont-know-about-how-ai-affecting-labor-market); [Fortune 02/02/2026](https://fortune.com/2026/02/02/ai-labor-market-yale-budget-lab-ai-washing/)) [S].
- Brookings, "New data show no AI jobs apocalypse — for now" ([link](https://www.brookings.edu/articles/new-data-show-no-ai-jobs-apocalypse-for-now/)) [S].
- EPI, "Class of 2026: what occupation data show about AI and young college graduates" ([link](https://www.epi.org/blog/class-of-2026-what-occupation-data-show-about-ai-and-the-young-college-graduate-workforce/)) [S].
- Humlum & Vestergaard (Dinamarca, 2025), "Large Language Models, Small Labor Market Effects": efeitos quase nulos sobre salários e horas, com dados administrativos ([NBER w33777](https://www.nber.org/papers/w33777)) [M].

### 2.2 Vagas (demanda) de software

**Indeed Hiring Lab**
- Guillermo Gallacher, "AI and Job Postings: From Destruction to Creation?" (08/07/2026) ([link](https://hiringlab.indeed.com/?p=20708)) [V]:
  - As vagas de software nos EUA subiram ~15% desde o lançamento do Claude Code (fev/2025), contra −7% no total de vagas.
  - Elas seguem ~27,5% abaixo de fev/2020, enquanto o total está no nível de fev/2020.
  - Entre mai/2022 e mai/2026, ocupações mais expostas à IA caíram mais (correlação sem controles). Entre mai/2025 e mai/2026 o sinal **inverte**: as mais expostas recuperaram mais.
  - **71% do ganho líquido (mai/25 → mai/26) veio de vagas sênior**, e 37% de títulos com AI (os grupos se sobrepõem). O artigo não traz número separado para júnior.
- Hiring Lab, jan/2026: vagas que mencionam IA crescem num mercado fraco ([link](https://www.hiringlab.org/2026/01/22/january-labor-market-update-jobs-mentioning-ai-are-growing-amid-broader-hiring-weakness)) [S]. Snapshot de abr/2026: software +14% a/a, com mais de 47% das vagas mencionando IA ([link](https://hiringlab.org/2026/05/14/us-labor-market-snapshot-april-2026)) [S]. Um número de +23,8% a/a (Q3/2026) circula em sites secundários e **não foi confirmado** [S].
- **Dados baixáveis (pequenos, ideais):**
  - Repositório do Indeed [github.com/hiring-lab/job_postings_tracker](https://github.com/hiring-lab/job_postings_tracker): CSV diário por setor (inclui "Software Development") e por país (US, UK, DE, FR, CA, AU etc.), índice com fev/2020 = 100 [M].
  - FRED, série `IHLIDXUSTPSOFTDEVE` ("Software Development Job Postings on Indeed in the United States") ([FRED](https://fred.stlouisfed.org/series/IHLIDXUSTPSOFTDEVE)) [M].
  - O Hiring Lab também tem um tracker de vagas que mencionam IA/GenAI no mesmo repositório ou no data portal ([data.indeed.com](https://data.indeed.com)) [M].

**LinkedIn Economic Graph, "US Software Engineer Talent Landscape 2026"** (fev/2026) ([PDF](https://economicgraph.linkedin.com/content/dam/me/economicgraph/en-us/PDF/us-software-engineer-talent-landscape-2026.pdf)) [S]: a contratação de SWE de entrada não se recuperou no fim de 2025. O LinkedIn diz que **não dá para concluir** que a IA (ou a expectativa dela) seja a causa.

**SignalFire, State of Tech Talent 2026** [S]: a contratação de recém-formados nas 12 "Tech Majors" está ~65% abaixo de 2019. A versão de 2025 falava em ~50% e em queda de ~25% em 2024 sobre 2023 ([SignalFire 2025](https://www.signalfire.com/blog/signalfire-state-of-talent-report-2025)) [M]. É fonte de VC: viés possível, mas usa dados próprios de perfis.

**NACE (Spring 2026):** mais de 1/3 das vagas de entrada exigem habilidades de IA, quase o triplo do outono de 2025 ([link](https://naceweb.org/job-market/trends-and-predictions/demand-for-ai-skills-in-entry-level-jobs-nearly-triples-since-fall-2025)) [S].

**NY Fed, "Labor Market for Recent College Graduates":** desemprego de recém-formados por curso. Ciência da computação e engenharia da computação aparecem entre os mais altos (~6–7,5% nas edições de 2025) ([NY Fed](https://www.newyorkfed.org/research/college-labor-market)) [M]. Tabela baixável (XLS pequeno).

### 2.3 Layoffs

- [Layoffs.fyi](https://layoffs.fyi): base colaborativa de demissões em tecnologia. Os totais de 2026 variam entre ~120 mil e ~154 mil conforme a data, e o número atual não foi verificado [S]. 2022: ~165 mil; 2023: ~264 mil (pico); 2024: ~152 mil; 2025: ~120 mil+ [M]. Há uma tabela exportável (Airtable) e versões no Kaggle.
- Casos citados como "IA": Amazon (14 mil em out/2025 e 16 mil em jan/2026; Andy Jassy escreveu em jun/2025 que a IA reduziria a força de trabalho corporativa), Microsoft (~4.800 em 2026), Salesforce (<1.000 em fev/2026). Lista do TechCrunch: [running list 2026](https://techcrunch.com/2026/06/22/the-running-list-major-tech-layoffs-in-2026-where-employers-cited-ai) [S].
- Challenger, Gray & Christmas: a IA foi o motivo mais citado nos cortes de tecnologia de mai/2026 ([relatórios mensais](https://www.challengergray.com/blog/category/job-cuts/)) [S].
- **Ressalva forte ("AI-washing"):** citar IA não prova causa. Superinvestimento em capex de IA, juros e correção pós-2021 são explicações concorrentes ([Fortune/Yale](https://fortune.com/2026/02/02/ai-labor-market-yale-budget-lab-ai-washing/); [SalesforceBen](https://www.salesforceben.com/can-we-finally-admit-these-tech-layoffs-arent-due-to-ai/)) [S].

### 2.4 Produtividade (o elo entre capacidade e demanda por trabalho)

| Estudo | Desenho | Resultado | Força |
|---|---|---|---|
| Peng et al. 2023, "The Impact of AI on Developer Productivity" ([arXiv 2302.06590](https://arxiv.org/abs/2302.06590)) [M] | RCT de laboratório, tarefa única (servidor HTTP) | 55,8% mais rápido com Copilot | Forte internamente, baixa validade externa |
| Cui, Demirer, Jaffe, Musolff, Peng & Salz, *Management Science* ([draft MIT](https://economics.mit.edu/sites/default/files/inline-files/draft_copilot_experiments.pdf); [DOI 10.1287/mnsc.2025.00535](https://pubsonline.informs.org/doi/10.1287/mnsc.2025.00535)) [S] | 3 RCTs de campo (Microsoft, Accenture, Fortune 100), ~4.800 devs | **+26% tarefas semanais concluídas**; **ganho maior para juniores e recém-contratados**; qualidade não medida; 2 experimentos sem pré-registro | Forte (campo + aleatorização), com limitações |
| METR 2025, "Measuring the Impact of Early-2025 AI on Experienced Open-Source Developer Productivity" ([blog](https://metr.org/blog/2025-07-10-early-2025-ai-experienced-os-dev-study/); [arXiv 2507.09089](https://arxiv.org/abs/2507.09089)) [V] | RCT, 16 devs experientes, 246 tarefas em repositórios maduros | **+19% de tempo (mais lentos)**. Antes previam −24%; depois ainda achavam que tinham sido 20% mais rápidos | Forte, mas amostra pequena e contexto específico |
| METR, follow-up (24/02/2026) ([link](https://metr.org/blog/2026-02-24-uplift-update/)) [V] | 10 devs originais + 47 novos, desde ago/2025 | Novos devs: −4% de tempo (IC −15% a +9%); provável ganho maior em 2026; viés de seleção nas tarefas | Inconclusivo |
| Google RCT interno 2024 (~96 engenheiros) ([arXiv 2410.12944](https://arxiv.org/abs/2410.12944)) [M] | RCT | ~21% mais rápido | Moderada |
| DORA / Google Cloud, State of DevOps 2024 e 2025 ([dora.dev](https://dora.dev/research/)) [M] | Survey grande | 2024: +25% de adoção associado a −1,5% de throughput e −7,2% de estabilidade de entrega. 2025: ~90% de adoção; a IA "amplifica" times bons e ruins | Correlacional |
| GitClear, "AI Copilot Code Quality" 2024/2025 ([gitclear.com](https://www.gitclear.com/ai_assistant_code_quality_2025_research)) [M] | Análise de repositórios | Mais churn e duplicação de código | Fonte comercial, correlacional |

Síntese: com tarefas bem delimitadas e devs menos experientes, há ganhos de 20–55%. Em bases de código maduras com especialistas, o ganho foi nulo ou negativo em 2025 e talvez positivo em 2026. A percepção de produtividade sistematicamente supera a medida, o que importa muito para o gestor.

### 2.5 Stack Overflow e bens públicos digitais

- del Rio-Chanona, Laurentsyeva & Wachs. Preprint: "Are Large Language Models a Threat to Digital Public Goods? Evidence from Activity on Stack Overflow" ([arXiv 2307.07367](https://arxiv.org/abs/2307.07367), jul/2023). Publicado como "Large language models reduce public knowledge sharing on online Q&A platforms", *PNAS Nexus* 3(9), set/2024, DOI [10.1093/pnasnexus/pgae400](https://doi.org/10.1093/pnasnexus/pgae400) ([PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11421660/)) [V].
  - Diferenças em diferenças contra plataformas sem acesso fácil ao ChatGPT (equivalentes russos e chineses, fóruns de matemática). Posts caíram **~25% em relação ao contrafactual** seis meses após o ChatGPT (o preprint dizia 16%). A qualidade (votos) não mudou, então não sumiram só as perguntas ruins.
- O volume mensal de perguntas no SO caiu para níveis de ~2009 em 2025, uma queda de mais de 75% desde o pico, embora o declínio já existisse antes de 2022 [M].
- **Dados baixáveis:** [Stack Exchange Data Explorer](https://data.stackexchange.com/stackoverflow/query/new) (uma query SQL de contagem mensal de posts por tag gera CSV pequeno), além do dump trimestral no archive.org (grande; **não baixar**).
- Por que entra na história: é um **proxy de adoção** com alta frequência (mensal) e um mecanismo de risco para juniores, que perdem o canal público de aprendizado.

### 2.6 Adoção e uso (para o eixo "adoção")

- **Stack Overflow Developer Survey:** em 2025, 84% usam ou planejam usar IA; 46% desconfiam da precisão contra 33% que confiam; 14% usam agentes diariamente ([resultados 2025](https://survey.stackoverflow.co/2025/); [blog](https://stackoverflow.blog/2025/07/29/developers-remain-willing-but-reluctant-to-use-ai-the-2025-developer-survey-results-are-here)) [S]. O pulse de abr/2026 mostra uso de agentes quase dobrando (59%) e Claude Code de 41% para 55% ([blog 27/05/2026](https://stackoverflow.blog/2026/05/27/agents-on-a-leash-agentic-ai-remains-mostly-monitored-at-work)) [S]. A pesquisa de 2026 saiu por volta de out/2026 [S]. **Microdados anuais em CSV** em [survey.stackoverflow.co](https://survey.stackoverflow.co) (~100 MB por ano descompactado; usar só se necessário, ou só as colunas de IA, anos de 2023 em diante).
- **Anthropic Economic Index:** relatórios de jan/2026 (dados de nov/2025, "economic primitives"), mar/2026 e jun/2026. Claude Code tem 79% de conversas de automação contra 49% no Claude.ai ([impacto no software](https://www.anthropic.com/research/impact-software-development); [jan/2026](https://www.anthropic.com/research/anthropic-economic-index-january-2026-report); [jun/2026](https://www.anthropic.com/research/economic-index-june-2026-report)) [S]. Dados por tarefa O*NET disponíveis no Hugging Face (`Anthropic/EconomicIndex`) [M]. É uma boa base para separar **automação vs. aumento**, o mesmo mecanismo usado no Canaries.
- **Código escrito por IA (declarações corporativas, não comparáveis entre si):** Google ~25% (out/2024) → ~50% (fim de 2025) → ~75% (abr/2026, Pichai) ([TechSpot](https://www.techspot.com/news/112152-google-ai-now-generates-75-new-code-up.html)) [S]. Microsoft 20–30% (Nadella, abr/2025) [S]. Anthropic 70–100% conforme a fonte ([Fortune 29/01/2026](https://fortune.com/2026/01/29/100-percent-of-code-at-anthropic-and-openai-is-now-ai-written-boris-cherny-roon/)) [S].

### 2.7 Classificação da evidência

| Nível | Afirmações |
|---|---|
| **Forte** | (1) A capacidade dos agentes em tarefas de software cresce de forma exponencial e mensurável (METR, SWE-bench). (2) Em dados de folha (ADP) e de currículos, o emprego de jovens e juniores em ocupações expostas, incluindo devs, divergiu negativamente desde 2022–23, enquanto o de seniores não caiu. (3) O ajuste se dá via **menos contratação** de entrada, não via demissão. (4) A atividade pública no Stack Overflow caiu de forma causal após o ChatGPT. (5) Copilot aumenta o output de tarefas em RCTs de campo, com mais ganho para juniores. |
| **Controverso** | (1) **Quanto** da queda de vagas e emprego júnior é IA e quanto é juros, correção pós-2021, offshoring ou incerteza (Yale/Brookings/LinkedIn vs. Stanford/Harvard). (2) Ganho real de produtividade para seniores em bases maduras (METR 2025 vs. 2026). (3) Se os layoffs "por IA" são de fato por IA ("AI-washing"). (4) Se a recuperação de vagas em 2025–26 é complementaridade (IA cria demanda) ou ciclo. |
| **Especulação** | "Fim do programador" e "IA escreve 90–100% do código" generalizado para a indústria; agentes substituindo times inteiros; extrapolação do doubling do METR para "tarefas de um mês" em 2027–2030; colapso do pipeline de seniores daqui a 5–10 anos por falta de juniores hoje (plausível, sem dados ainda). |

---

## 3. A visão do empregador

### 3.1 O que dizem e fazem (declarações e decisões datadas)

| Data | Empresa ou pessoa | Declaração ou ação | Fonte |
|---|---|---|---|
| dez/2024 e 2025 | Salesforce (Benioff) | Não contrataria novos engenheiros de software em 2025 por ganho de ~30% com IA. Em 2026, chamou layoffs atribuídos à IA de "saída preguiçosa" | [S] via [TechCrunch 2026](https://techcrunch.com/2026/06/22/the-running-list-major-tech-layoffs-in-2026-where-employers-cited-ai) |
| jan/2025 | Meta (Zuckerberg) | Previu IA no nível de "engenheiro mid-level" em 2025 | [M] (entrevista Joe Rogan) |
| abr/2025 | Shopify (Lütke) | Memo: provar que a IA não faz o trabalho antes de pedir headcount | [M] |
| abr/2025 | Duolingo | Memo "AI-first": reduzir terceirizados onde a IA faz o trabalho | [M] |
| abr/2025 | Microsoft (Nadella) | 20–30% do código escrito por IA | [S] |
| jun/2025 | Amazon (Jassy) | A IA reduzirá a força de trabalho corporativa nos próximos anos. Cortes de 14 mil (out/2025) e 16 mil (jan/2026) | [S] |
| ago/2025 | AWS (Matt Garman) | Substituir juniores por IA é "uma das coisas mais estúpidas", porque eles são o pipeline | [M] |
| início de 2026 | IBM | Vai **triplicar** contratações de entrada em 2026, incluindo SWE. Cognizant quadruplica o pipeline nos EUA | [S] via [IEEE Spectrum](https://spectrum.ieee.org/ai-effect-entry-level-jobs) |
| abr/2026 | Google (Pichai) | ~75% do código novo gerado por IA e revisado por humanos | [S] |

### 3.2 Pesquisas com empregadores

- **WEF Future of Jobs Report 2025** (jan/2025): ~40% dos empregadores esperam reduzir pessoal onde a IA automatiza tarefas. Ao mesmo tempo, desenvolvedores de software e especialistas em IA/big data aparecem entre os papéis que mais crescem até 2030 ([WEF](https://www.weforum.org/publications/the-future-of-jobs-report-2025/)) [M]. É um bom exemplo da tensão "substitui vs. complementa".
- **ResumeTemplates (1.000 gestores de contratação nos EUA, 2026, fonte comercial):** 55% já realocaram parte do orçamento de contratação de entrada para ferramentas de IA, e 30% dizem que a IA já reduziu a necessidade de recém-formados [S].
- **NACE Spring 2026:** 1/3 das vagas de entrada exigem IA [S].
- **LinkedIn (fev/2026):** a ausência de rebote em SWE de entrada preocupa, mas sem prova de causalidade [S].

### 3.3 Leitura para o gestor
- O padrão de comportamento é: congelar ou reduzir contratação júnior, manter e contratar sênior, exigir fluência em IA e reorganizar times menores com mais agentes ("menos pessoas, mais alavancagem").
- Há riscos documentados: (a) a produtividade percebida é maior que a medida (METR); (b) dívida técnica, churn e queda de estabilidade de entrega (DORA 2024, GitClear); (c) o pipeline de talentos (Garman, IBM), porque sem juniores hoje não haverá seniores amanhã; (d) risco reputacional e de "AI-washing" nos layoffs.

---

## 4. Teses candidatas (testáveis com dados públicos)

### Tese 1. "A IA está quebrando o primeiro degrau da escada": a demanda por devs se desloca do júnior para o sênior
- **Dados:** Canaries Dashboard (ZIP, mensal, por idade × exposição); Indeed Hiring Lab (vagas de software; o artigo de jul/2026 traz a decomposição sênior/júnior); opcionalmente o Stack Overflow Survey (anos de experiência × uso de IA); no Brasil, Novo CAGED (admissões por CBO 2124 "analistas de TI" × faixa etária, via [PDET/MTE](https://pdet.mte.gov.br/microdados-rais-e-caged), filtrando sem baixar o arquivo inteiro).
- **Confirmaria:** um gap de emprego 22–25 vs. 26+ em devs que se abre depois de 2022–23 e se amplia acompanhando o índice de capacidade; vagas sênior ganhando participação; o padrão repetido em ocupações expostas fora de tecnologia.
- **Refutaria:** gap igual em ocupações pouco expostas (efeito geral de juventude ou juros); gap que começa antes de nov/2022; recuperação de contratação júnior em 2026 junto com o avanço da capacidade.

### Tese 2. "Primeiro a correção, depois a IA": a queda de vagas de software de 2022–24 foi cíclica (juros e pós-boom); a IA muda a composição da demanda, mais que o volume
- **Dados:** Indeed `job_postings_tracker` / FRED `IHLIDXUSTPSOFTDEVE` (diário, 2020–2026); Fed Funds (FRED `FEDFUNDS`); índice de capacidade (METR/Epoch); comparação entre países pelo CSV do Indeed (Alemanha e França vs. EUA).
- **Confirmaria:** a queda de 2022–23 segue a alta de juros e o pico de 2021–22, e acontece também em países com adoção de IA mais lenta; o rebote de 2025–26 vem com mais vagas sênior ou com "AI" no título, não com volume agregado maior (consistente com Indeed jul/2026).
- **Refutaria:** queda de vagas de software maior que em outros setores e proporcional ao índice de capacidade mesmo controlando por juros; ausência de rebote após os agentes.

### Tese 3. "A IA ajuda mais quem sabe menos na tarefa e substitui quem sabe menos no mercado" (paradoxo da produtividade júnior)
- **Dados:** estimativas publicadas dos RCTs (Cui et al.: ganho maior para juniores; METR: seniores mais lentos) combinadas com os dados de emprego (Canaries/Hosseini). Uma meta-tabela de efeitos por senioridade é viável, já que os números estão nos papers.
- **Confirmaria:** efeitos de produtividade maiores para juniores em RCTs e, ao mesmo tempo, menos contratação júnior. A tarefa júnior (código "codificado", bem especificado) é exatamente o que os agentes automatizam (o Canaries mostra queda em conhecimento codificado e alta onde o conhecimento é tácito).
- **Refutaria:** RCTs que mostram ganho maior para seniores; firmas adotantes que contratam *mais* juniores (o caso IBM, se generalizado).

### Tese 4. "Os bens comuns de aprendizado estão secando": a IA substitui o Stack Overflow como canal de aprendizado, o que muda como juniores aprendem e se tornam visíveis
- **Dados:** Stack Exchange Data Explorer (posts e respostas mensais por tag, 2015–2026, CSV pequeno); índice de capacidade e marcos (ChatGPT, GPT-4, Claude Code); Stack Overflow Survey (fontes de aprendizado por faixa de experiência).
- **Confirmaria:** quebras estruturais no volume do SO nas datas dos marcos (replicando del Rio-Chanona com dados até 2026), com queda maior em tags e linguagens onde os LLMs são melhores (Python/JS) do que em nichos, e mais concentrada em perguntas de iniciante.
- **Refutaria:** a queda seguir só a tendência pré-2022, sem quebra; a queda ser uniforme entre tags (o que sugere declínio da plataforma, não substituição por LLM).

### Tese 5 (para o gestor). "Capacidade ≠ produtividade ≠ corte de pessoal": o salto de benchmarks não se traduz proporcionalmente em entrega
- **Dados:** índice METR/SWE-bench; métricas de produtividade publicadas (RCTs, DORA); headcount e receita por funcionário das big techs (10-K na SEC EDGAR, anual, pequeno); Layoffs.fyi.
- **Confirmaria:** o benchmark cresce 10–50× enquanto a produtividade medida cresce 0–30%; a receita por funcionário sobe mais por cortes do que por output; os layoffs "por IA" não se concentram em firmas com maior adoção.
- **Refutaria:** firmas com maior adoção mostrando ganho de output por engenheiro proporcional à capacidade e reduzindo headcount de engenharia sem perda de entrega.

---

## 5. Inventário de dados (para a fase Data Understanding do CRISP-DM)

| Dado | Granularidade | Período | Formato e tamanho | Link |
|---|---|---|---|---|
| Vagas Indeed por setor e país | diária | fev/2020–hoje | CSV pequeno (GitHub) | [hiring-lab/job_postings_tracker](https://github.com/hiring-lab/job_postings_tracker) |
| Vagas de software EUA (Indeed) | diária | 2020–hoje | CSV pelo FRED | [FRED IHLIDXUSTPSOFTDEVE](https://fred.stlouisfed.org/series/IHLIDXUSTPSOFTDEVE) |
| Emprego ADP por idade × exposição | mensal | 2021–2026 | ZIP agregado | [Canaries Dashboard](https://digitaleconomy.stanford.edu/project/canaries-dashboard/) |
| Horizonte METR | por modelo (data) | 2019–2026 | YAML pequeno | [metr.org/assets/benchmark_results_1_1.yaml](https://metr.org/assets/benchmark_results_1_1.yaml) · [GitHub](https://github.com/METR/eval-analysis-public) |
| Benchmarks Epoch (SWE-bench etc.) | por modelo (data) | 2023–2026 | ZIP/CSV pequeno | [epoch.ai/data/benchmark_data.zip](https://epoch.ai/data/benchmark_data.zip) |
| Atividade Stack Overflow | mensal, por tag | 2008–hoje | CSV via SQL | [data.stackexchange.com](https://data.stackexchange.com/stackoverflow/query/new) |
| SO Developer Survey | anual (microdados) | 2011–2026 | CSV ~50–100 MB/ano | [survey.stackoverflow.co](https://survey.stackoverflow.co) |
| Anthropic Economic Index | por tarefa O*NET | 2024–2026 | CSV (Hugging Face) | [Anthropic research](https://www.anthropic.com/research/impact-software-development) |
| Layoffs | por evento | 2020–hoje | tabela (Airtable/Kaggle) | [layoffs.fyi](https://layoffs.fyi) |
| BLS OEWS (15-1252 Software Developers; 15-1251 Programmers) | anual | 1997–2025 | XLS | [bls.gov/oes](https://www.bls.gov/oes/) |
| CS/CE recém-formados (desemprego) | anual | 2023–2026 | XLS | [NY Fed](https://www.newyorkfed.org/research/college-labor-market) |
| Brasil: Novo CAGED (CBO 2124xx, 3171xx) | mensal | 2020–hoje | microdados grandes; preferir painéis agregados do PDET | [PDET/MTE](https://pdet.mte.gov.br/microdados-rais-e-caged) |

**Cuidados metodológicos para a história**
1. Use nov/2022 como marco, mas teste placebo (2021) e marcos alternativos (GPT-4 em mar/2023; agentes em fev–mai/2025).
2. Compare sempre com um grupo de controle (ocupações pouco expostas, outros setores no Indeed, outros países).
3. Diga explicitamente que a correlação entre o índice de capacidade e o emprego **não é causal**, porque juros e o ciclo de 2021–22 confundem.
4. Os benchmarks saturam (Verified a partir de ~2025–26). Troque de série ou use o METR em escala log.
5. Declarações de CEOs são **dado qualitativo** (o "discurso" do empregador), não métrica.
