# Roteiro do notebook — "O Primeiro Degrau"

Guia da narrativa (para quem monta o notebook e para a equipe na apresentação de 5–10 min).
Os números NUNCA são digitados no texto: vêm dos arquivos em `data/results/` e `data/processed/` e são inseridos com f-strings / `display(Markdown(...))`. Os vereditos vêm de `vereditos_series.json` e `vereditos_survey.json`. Citações de literatura só de `fatos_literatura.csv` com `status == "verificado"`.

## Big Idea (3 minutos, Knaflic)
A IA não acabou com o emprego em software. Os dados sugerem que ela está tirando o **primeiro degrau**: o lugar onde o iniciante aprendia (Stack Overflow) esvaziou e quem tem até 2 anos de experiência perdeu emprego muito mais que o sênior. A tese rival (ciclo/juros) não é descartada: a gente mostra onde os dados conseguem separar as duas e onde não conseguem.

## Arco (SCQA, Minto)
- **Situação:** de 2022 a 2026 a IA passou de chatbot a agente que executa tarefas de horas sozinho (METR, SWE-bench).
- **Complicação:** juniores dizem que "não há vaga"; empresas dizem que "falta gente boa".
- **Pergunta:** a IA tirou o primeiro degrau da carreira, ou é só o ciclo econômico?
- **Resposta (primeiro):** resumo executivo com os 3 atos e os vereditos.

## Seções (rótulo "CRISP-DM fase X | componente" em cada uma; título = conclusão)

0. **Resumo executivo** — 4–5 linhas com a resposta; tabela de vereditos (H1, H2, H3, Ato 1, Ato 2) com ícone/texto (não só cor); tabela "CRISP-DM × componentes × seção".
1. **Entendimento do Negócio** (fase 1) — os dois decisores (estudante; gestor), H1/H2/H3, tese rival, regras de veredito definidas ANTES (design §7.1) — mostrar isso como diferencial de honestidade.
2. **Entendimento dos Dados | ETL-Extract** (fase 2) — tabela das 5 fontes (o quê, período, granularidade, link); gráfico 0 (contexto): **"A IA ficou ~X vezes mais autônoma em 3 anos"** — horizonte METR (escala log) com marcos coloridos por onda (chat/assistente/agente).
3. **Preparação | ETL-Transform** (fase 3) — o que foi tratado: mês completo, harmonização do survey (tabela COMPARAVEL resumida), filtro salarial, WorkExp só; mencionar armadilhas encontradas (seleção na pergunta de experiência) como aprendizado.
4. **EDA** (componente EDA) — gráfico 1: **"O Stack Overflow perdeu ~99% das perguntas desde o pico"** (série total 2008–2026, marcos). Gráfico 2: vagas de software vs mediana dos setores (índice, base fev/2020) com Fed Funds em painel separado (NUNCA dois eixos).
5. **Modelagem | Análise Preditiva** (fase 4)
   - **Ato 1** — gráfico 3: real vs contrafactual SARIMAX com faixa IC 95% (2014–2026). Título afirma o achado de 48 meses; texto explica que pela regra de 12 meses o veredito é inconclusivo por causa do placebo COVID. Gráfico 4 (pequeno): efeito por linguagem (barras horizontais, ordenadas).
   - **Ato 2** — resultado da regressão (tabela curta: dummy pós, FEDFUNDS, p) + o porém do boom 2021–22 (placebos pós-hoc).
   - **Ato 3 / H1** — gráfico 5: % empregado ≤2 vs ≥10, 2025→2026, três métodos (pontos com IC, small multiples ou dot plot). Título: "Quem tem até 2 anos perdeu entre ~9 e ~20 pontos de emprego em um ano; o sênior, ~2".
   - **H3** — gráfico 6: razão salarial sênior/júnior EUA 2023→2026 com IC. Classificador: AUC fraco (~0,61), importâncias; achado contraintuitivo: quem usa agentes relata MAIS ameaça.
   - **H2 (conta do gestor)** — gráfico 7: ganho de produtividade necessário (u*) vs cenários (0, +15, +30, −19%) para prod_jr 0,3/0,5/0,7. Rotular como simulação.
6. **Avaliação** (fase 5) — backtest vs ingênuo, placebos (tabela), limitações (lista do `vereditos_series.json["limitacoes"]` + survey: autosseleção, amostra menor em 2026, definição de agentes mudou), veredito por hipótese.
7. **Implantação | Data Storytelling → decisão** (fase 6) — duas recomendações:
   - **Estudante:** o degrau de entrada passou a exigir o que antes se aprendia nele → fundamentos fortes + domínio de agentes + portfólio que prove entrega; buscar entradas alternativas (estágio, open source, áreas adjacentes).
   - **Gestor:** a conta fecha hoje para "sênior + IA" em vários cenários, mas cortar o júnior seca a fonte de sêniores de amanhã → manter um pipeline júnior menor e intencional (aprendiz + IA + mentoria).
   - KPIs para monitorar (vagas júnior/sênior, emprego ≤2 no survey de 2027, razão salarial, horizonte METR) e como atualizar (`python -m src...`). Fechar mostrando o CRISP-DM como ciclo.
- **Apêndice:** referências (de `docs/referencias.md`), detalhes técnicos (ordem SARIMAX, AIC, ITS), reprodutibilidade.

## Regras de forma
- Após cada gráfico principal: bloco curto **O que vemos → O que significa → O que fazer** (3 linhas).
- ≤ 8 gráficos principais; `src/viz.py` (tema, `titulo`, `fonte`, `marcos`, `faixa_ic`, `rotulo_final`); uma cor de destaque; fonte embaixo de cada gráfico.
- Linguagem: "consistente com", "associado a"; nunca "causou/provou".
- Português do Brasil, frases curtas, sem jargão desnecessário; passar pela skill `humanizer-pt-br`.
