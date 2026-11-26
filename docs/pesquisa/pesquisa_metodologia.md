# Pesquisa de Metodologia: IA generativa e o mercado de desenvolvimento de software

Prazo: entrega em 27/11/2026. Restrições: notebook Jupyter como apresentação (5 a 10 min), CRISP-DM com 6 fases, ETL + EDA + Preditiva + Storytelling.

Legenda de confiança: **[verificado]** = conferido em fonte web nesta pesquisa; **[consolidado]** = conhecimento padrão da área/documentação, não reconferido agora.

---

## 1. Análise preditiva honesta com intervenção (séries temporais)

### 1.1 O problema central (o que o professor vai olhar)
- As perguntas do Stack Overflow **já caíam antes do ChatGPT** (pico ~2014, queda gradual desde então). Uma comparação "antes vs. depois" ingênua atribuiria à IA uma queda que já existia. O contrafactual precisa **modelar a tendência pré-existente**. [consolidado]
- As vagas de software do Indeed (FRED, série `IHLIDXUSTPSOFTDEVE`) **despencaram a partir de 2022 por causa de juros/fim do boom pós-pandemia**, antes e durante o lançamento do ChatGPT. Confundidor forte: sem série de controle, não dá para separar "IA" de "ciclo macro". [consolidado]
- Portanto: o resultado esperado e honesto é **"queda adicional ao contrafactual no SO = forte; efeito da IA nas vagas = inconclusivo/confundido"**. Isso é uma conclusão válida, não um fracasso.

### 1.2 Abordagem PRINCIPAL recomendada: previsão contrafactual com SARIMAX/ETS (statsmodels)
Motivo: robusta, rápida (segundos), sem TensorFlow, fácil de explicar em 1 minuto, com intervalos de previsão nativos.

Receita:
1. **Corte**: treinar só com dados até out/2022 (pré-ChatGPT, lançado 30/11/2022). Opcional: usar o período pré **até 2021 ou 2019** para evitar o choque da pandemia (ou incluir dummy da pandemia).
2. **Transformação**: `log(y)` (queda multiplicativa; intervalos ficam assimétricos e realistas).
3. **Modelo**: `statsmodels.tsa.holtwinters`/`ETSModel` (tendência amortecida, sazonalidade anual) **ou** `SARIMAX(order=(p,1,q), seasonal_order=(P,1,Q,12))` escolhido por AIC **apenas no pré-período**.
4. **Validação sem vazamento** dentro do pré-período: *rolling-origin* (backtest) com `sklearn.model_selection.TimeSeriesSplit` ou loop manual; horizonte = tamanho do pós-período (ex.: 24–36 meses); métricas MAPE/MASE; comparar com baseline ingênuo sazonal (`y[t-12]`) e drift. Se o modelo não bate o ingênuo, diga isso.
5. **Contrafactual**: `res.get_forecast(steps=h).summary_frame(alpha=0.05)` (SARIMAX) ou `ETSModel(...).fit().get_prediction(...).summary_frame()` para obter média e IC 95%.
6. **Efeito**: `efeito = real − previsto` (em %, `real/previsto − 1`), com faixa derivada dos limites do IC; efeito acumulado; momento em que o real sai do IC.
7. **Placebo (obrigatório para honestidade)**: repetir tudo com datas falsas de intervenção (ex.: nov/2019, nov/2020). Se o placebo também "mostra efeito", a conclusão enfraquece.
8. **Série de controle para as vagas**: comparar software vs. índice total do Indeed (`IHLIDXUS`) ou outra ocupação pouco exposta (ex.: enfermagem). Usar a **razão software/total** como série alvo, ou o total como covariável exógena. Isso parcialmente remove o ciclo macro.

### 1.3 Abordagem de ROBUSTEZ recomendada: ITS / regressão segmentada com erros HAC
Referência metodológica: Bernal, Cummins & Gasparrini (2017), tutorial de ITS no *Int. J. Epidemiology*. [verificado]

Modelo (OLS no log):
`log(y_t) = b0 + b1*t + b2*D_t + b3*(t − t0)*D_t + sazonalidade(mês) + e_t`
- `D_t` = 1 a partir de dez/2022; `b2` = mudança de nível; `b3` = mudança de inclinação.
- Autocorrelação: `sm.OLS(...).fit(cov_type="HAC", cov_kwds={"maxlags":12})` (Newey-West). Alternativas citadas na literatura: Prais-Winsten/GLSAR, ARIMA nos erros. [verificado que são as correções usuais; sintaxe statsmodels = consolidado]
- Marcos de agentes (ex.: 2024–2025, Copilot/agents, saltos no SWE-bench/METR): adicionar **no máximo 1 segundo ponto de quebra**; mais quebras = sobreajuste e "caça de p-valor". Melhor: usar a linha do tempo de capacidade como **anotação no gráfico**, não como regressor causal.

### 1.4 CausalImpact em Python: usar ou não?
| Pacote | Situação | Veredito |
|---|---|---|
| `tfcausalimpact` (W. Fuks) | Ativo; BSTS via TensorFlow Probability; Python 3.8–3.13; ajuste VI ~2–3 min, HMC pode levar >1h | Opcional, só se instalar TF sem dor |
| `tfp-causalimpact` (Google, "not officially supported") | v0.2.x | Alternativa |
| `pycausalimpact` | Parado; usa statsmodels/Kalman (frequentista) | Evitar |
| `causalimpact` (Senouci) | última versão jan/2023 | Evitar |
[verificado] Atenção: vários se importam como `import causalimpact`; não instalar mais de um. Resultados diferem do pacote R (o próprio autor mostra p-valores diferentes). Com 2 dias de prazo, **não vale o risco de instalação do TensorFlow**; o análogo frequentista é `statsmodels.tsa.UnobservedComponents` (tendência local + sazonal), que pode substituir o BSTS se quiser uma terceira opinião.

**Prophet**: aceitável como checagem extra, mas tende a perder para SARIMA/ETS e extrapola changepoints/gera intervalos estreitos demais no pós-período. Não usar como principal. [parcialmente verificado]

### 1.5 Como reportar resultado inconclusivo
- Sempre mostrar o **IC** no gráfico (faixa sombreada), não só a linha prevista.
- Linguagem: "Os dados são **consistentes com** uma queda adicional de X% (IC 95%: A%–B%) em relação à tendência pré-ChatGPT" — nunca "a IA causou".
- Para vagas: "O efeito não é distinguível do ciclo de juros; o placebo de 2021 também mostra queda" (se for o caso).
- Tabela final "o que sabemos / o que não sabemos / o que mudaria a conclusão".

---

## 2. Modelagem sobre o Stack Overflow Developer Survey

### 2.1 Escolha do alvo (recomendação)
- **Principal: classificação** em UM ano (2024 ou 2025): alvo binário de **percepção de ameaça** (`AIThreat`, 2024) ou **confiança na precisão da IA** (`AIAcc`, 2024+) ou **uso** (`AISelect`). Preditores: experiência (`YearsCodePro`), tipo de dev (`DevType`), país/região agregados, porte da empresa, educação, modo de trabalho. Conecta direto aos decisores (estudante: quem se sente ameaçado; gestor: quem adota/confia). [nomes de colunas: consolidado; conferir no schema de cada ano]
- **Secundário (opcional)**: regressão de `log(ConvertedCompYearly)` mostrando se usuários de IA ganham diferente controlando por experiência/país. Cuidado: país domina tudo; filtrar um país (ex.: EUA) ou usar efeitos fixos; cortar outliers (p1–p99).

### 2.2 Validação
- Train/test estratificado (80/20) + `StratifiedKFold(5)` no treino; **baseline obrigatório** (`DummyClassifier` e regressão logística) antes do `HistGradientBoostingClassifier`/LightGBM.
- Métricas: ROC-AUC, F1/Recall da classe minoritária, matriz de confusão; reportar **ganho sobre o baseline**. AUC ~0,6–0,7 em atitudes é normal; dizer isso.
- Pipeline `ColumnTransformer` + `OneHotEncoder(handle_unknown="ignore", min_frequency=...)` para evitar vazamento de pré-processamento.
- Nunca usar como preditor colunas que derivam do alvo (ex.: outras perguntas de sentimento de IA para prever ameaça = vazamento conceitual).

### 2.3 Interpretabilidade
- **Permutation importance** (`sklearn.inspection.permutation_importance`) no conjunto de teste: simples, agnóstico, robusto. Recomendado como principal.
- **SHAP** (`TreeExplainer`) para o gráfico beeswarm, com ressalvas: com variáveis colineares, o ranking do SHAP fica instável; agrupar dummies de uma mesma variável (somar SHAP por variável original) ou usar `PartitionExplainer` com clustering por correlação (ainda "alpha"). [verificado]
- SHAP/importância explicam **o modelo, não causalidade**. Escrever isso explicitamente.

### 2.4 Viés de survey e comparação entre anos
- **Autosseleção**: respondentes são usuários engajados do SO, majoritariamente EUA/Europa/Índia, homens; não representa "todos os devs". Mostrar a composição da amostra.
- **Viés de sobrevivência**: quem perdeu o emprego/saiu da área responde menos.
- **Perguntas mudam entre anos** (redação, opções, aparecimento de `AIThreat`/`AIAcc`): comparar **apenas perguntas idênticas** (ex.: `AISelect`, `AISent`) e harmonizar categorias num dicionário explícito; documentar no ETL. Trend entre anos = **descritivo** (EDA), não entrada do modelo.
- Respostas faltantes não são aleatórias (perguntas opcionais no fim do questionário): reportar % de NA e não imputar o alvo.
- Usar proporções com IC (ex.: Wilson) ao comparar grupos.

---

## 3. Data Storytelling

### 3.1 Frameworks (cite no notebook)
- **Cole Nussbaumer Knaflic — *Storytelling with Data* (2015)**: entender o contexto (quem, o quê, como); escolher o visual adequado; eliminar poluição visual; focar a atenção (cor/pré-atenção só no que importa); pensar como designer; contar uma história. Ferramentas: **"Big Idea"** (uma frase com ponto de vista + o que está em jogo) e **"história de 3 minutos"**. [consolidado]
- **Brent Dykes — *Effective Data Storytelling* (2019)**: história de dados = **dados + narrativa + visuais**; arco: situação atual (contexto) → insights que levam a um insight central ("aha") → recomendações/próximos passos; storyboard em 4 passos; não colocar todos os dados num gráfico. [verificado]
- **Barbara Minto — Princípio da Pirâmide**: resposta primeiro, depois argumentos agrupados (MECE); introdução **SCQA** (Situação, Complicação, Questão, Resposta). [consolidado]
- **Pirâmide DIKW** (dados → informação → conhecimento → sabedoria/decisão): exatamente o fluxo pedido pelo professor; usar como fio condutor visível.

### 3.2 Big Idea sugerida (rascunho)
"A IA generativa já mudou **como** desenvolvedores buscam ajuda (queda do SO além da tendência), mas os dados ainda **não provam** que ela reduziu vagas — o ciclo de juros explica boa parte; por isso estudantes devem investir em habilidades que a IA complementa e gestores devem reestruturar com cautela, medindo produtividade antes de cortar."

### 3.3 Regras práticas para os gráficos
- **Action titles**: o título afirma a conclusão ("Perguntas no SO caíram 40% abaixo do previsto após o ChatGPT"), não descreve ("Perguntas por mês").
- **Anotações**: linha vertical em nov/2022 (ChatGPT) e marcos de agentes; faixa sombreada do IC; rótulo direto nas linhas em vez de legenda; cinza para contexto, uma cor de destaque.
- Um gráfico = uma mensagem. Máximo ~6–8 gráficos na apresentação; o resto no apêndice.
- Cada gráfico seguido de uma célula Markdown curta: **"O que vemos → O que significa → O que fazer"** (informação → conhecimento → decisão).

### 3.4 Notebook como apresentação: opções
| Opção | Prós | Contras/risco em 2 dias |
|---|---|---|
| **Notebook pré-executado no JupyterLab, código recolhido** (`View > Collapse All Code` ou metadata `"jupyter": {"source_hidden": true}`) | É "o próprio notebook", zero instalação, mostra código se o professor pedir | Precisa de rolagem disciplinada |
| **nbconvert HTML sem código**: `jupyter nbconvert --to html --no-input --execute notebook.ipynb` ou por tags: `--TagRemovePreprocessor.remove_input_tags hide_input` | Backup offline perfeito, abre em qualquer navegador | Não é interativo |
| RISE (`jupyterlab-rise` 0.43, exige JupyterLab ≥4.1.2) | Slides dentro do notebook | Issues abertas de toolbar/sobreposição; risco no dia |
| Quarto (`quarto render nb.ipynb --to revealjs`) | Slides bonitos, `echo: false` | Instalação extra; vira "slides", que o professor não quer |
| Voilà | Remove código, bom para widgets | Feito para dashboard, não para narrativa linear |
[verificado]

**Recomendação**: apresentar no **JupyterLab com o notebook já executado, código recolhido e células de código marcadas com a tag `hide_input`**, e gerar **um HTML com `--no-input` como backup** (e para entregar). Não depender de RISE/Quarto. Ensaiar com cronômetro: ~1 min por seção principal.

---

## 4. Estrutura do notebook: CRISP-DM + ETL/EDA/Preditiva/Storytelling

Estrutura recomendada (pirâmide: resposta no topo, CRISP-DM no corpo, detalhes no apêndice):

| # | Seção (título = conclusão) | CRISP-DM | Componente | Tempo na apresentação |
|---|---|---|---|---|
| 0 | **Resumo executivo** (Big Idea, SCQA, 3 achados, 2 recomendações, mapa CRISP-DM em tabela) | — (Minto) | Storytelling | 1 min |
| 1 | Entendimento do negócio: decisores (estudante, gestor), perguntas, critérios de sucesso, hipóteses H1–H3 | 1. Business Understanding | Storytelling (contexto) | 1 min |
| 2 | Entendimento dos dados: fontes, período, granularidade, limitações, dicionário | 2. Data Understanding | ETL (Extract) + EDA inicial | 0,5 min |
| 3 | Preparação: limpeza, harmonização de anos do survey, merge de séries mensais, log, flags de intervenção; funções reutilizáveis; checagens (`assert`) | 3. Data Preparation | ETL (Transform/Load) | 0,5 min |
| 4 | EDA: 3–4 gráficos com action titles (tendência SO, vagas vs. total, adoção de IA por ano, capacidade dos modelos) | 2/3 (iterativo) | EDA → informação | 2 min |
| 5 | Modelagem: contrafactual SARIMAX/ETS + ITS segmentada; classificador do survey | 4. Modeling | Preditiva | 2 min |
| 6 | Avaliação: backtest vs. baseline, placebo, IC, métricas do classificador, importância de variáveis, limitações e ameaças à validade | 5. Evaluation | Preditiva → conhecimento | 1 min |
| 7 | Implantação/decisão: recomendações por decisor, o que monitorar (KPIs), próximo passo | 6. Deployment | Storytelling → decisão | 1 min |
| A | Apêndice: código auxiliar, tabelas completas, robustez extra | — | — | não apresentar |

Dicas de implementação:
- Primeira célula: imports, seed (`np.random.seed(42)`), versões das bibliotecas; caminhos relativos (`data/raw`, `data/processed`).
- Badge ou rótulo em cada seção: "CRISP-DM fase 3 | ETL". Torna a avaliação do requisito trivial para o professor.
- Mencionar que CRISP-DM é **cíclico**: mostrar onde voltou de uma fase (ex.: Avaliação → Preparação ao trocar a série alvo para razão software/total).

### Erros comuns que professores penalizam
1. Fases do CRISP-DM citadas só no título, sem conteúdo real (principalmente **Business Understanding** sem critério de sucesso e **Deployment** vazio).
2. **Vazamento temporal**: split aleatório em série temporal, escolher o modelo olhando o pós-período, normalizar usando a série inteira.
3. Afirmar **causalidade** a partir de correlação/antes-depois; ignorar a tendência pré-existente e o confundidor macro (juros).
4. Modelo sem **baseline** e sem intervalos; métrica única (acurácia em classe desbalanceada).
5. Gráficos sem título interpretativo, eixos sem unidade, legenda poluída, pizza/3D.
6. Notebook que não roda de cima para baixo (estado oculto); células fora de ordem; outputs gigantes; warnings visíveis.
7. Storytelling sem decisão: termina em "os dados mostram..." sem recomendação para o decisor.
8. Comparar anos do survey com perguntas diferentes como se fossem iguais; ignorar viés de amostra.
9. Estourar o tempo: mais de ~8 gráficos ou mostrar código na apresentação.
10. Não declarar limitações e não citar fontes dos dados.

---

## Fontes
- Bernal, Cummins & Gasparrini — Interrupted time series regression tutorial: https://pmc.ncbi.nlm.nih.gov/articles/PMC5407170/
- Avaliação de métodos para ITS (simulação; Newey-West, Prais-Winsten, ARIMA): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8403376/
- Interpretação de coeficientes de regressão segmentada: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10925407/
- Exemplo ITS em Python/statsmodels: https://dcacciarelli.github.io/my-causality-book/notebooks/interrupted_time_series.html
- tfcausalimpact: https://github.com/WillianFuks/tfcausalimpact | https://pypi.org/project/tfcausalimpact
- tfp-causalimpact: https://pypi.org/project/tfp-causalimpact/
- pycausalimpact: https://pypi.python.org/pypi/pycausalimpact | causalimpact: https://pypi.org/project/causalimpact/
- Prophet vs SARIMA: https://python.plainenglish.io/stop-guessing-trends-prophet-and-sarima-compared-so-you-dont-have-to-1b444c80e82a
- statsmodels (SARIMAX, ETSModel, OLS HAC, UnobservedComponents): https://www.statsmodels.org/stable/
- scikit-learn TimeSeriesSplit / permutation_importance: https://scikit-learn.org/stable/
- SHAP PartitionExplainer: https://shap.readthedocs.io/en/latest/generated/shap.PartitionExplainer.html | PR #3666: https://github.com/shap/shap/pull/3666
- SHAP e colinearidade: https://arxiv.org/pdf/2406.11524 | https://arxiv.org/html/2305.02012v3
- Dykes — Effective Data Storytelling: https://www.oreilly.com/library/view/effective-data-storytelling/9781119615712/ | https://www.datacamp.com/blog/telling-effective-data-stories-with-data-narrative-and-visuals
- Knaflic — Storytelling with Data (Wiley, 2015): https://www.storytellingwithdata.com/books
- Minto — The Pyramid Principle: https://www.barbaraminto.com/
- nbconvert (opções de exclusão de input): https://nbconvert.readthedocs.io/en/latest/config_options.html | https://mljar.com/blog/jupyter-notebook-hide-code/
- RISE: https://github.com/jupyterlab-contrib/rise | https://pypi.org/project/jupyterlab-rise/
- Quarto + Jupyter: https://quarto.org/docs/reference/cells/cells-jupyter | issue #13735: https://github.com/quarto-dev/quarto-cli/issues/13735
- Voilà: https://github.com/voila-dashboards/voila/releases/tag/v0.5.0
- CRISP-DM (Chapman et al., 2000, guia 1.0); FRED Indeed Software Development Postings: https://fred.stlouisfed.org/series/IHLIDXUSTPSOFTDEVE
