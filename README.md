# O Primeiro Degrau: a IA generativa e o mercado de desenvolvimento de software

Atividade de **Data Storytelling** da disciplina de Ciência de Dados (Católica SC, Joinville). O projeto segue as 6 fases do CRISP-DM e usa ETL, EDA, análise preditiva e storytelling para responder uma pergunta:

> **A IA tirou o primeiro degrau da carreira em software, ou a queda no mercado é só o ciclo econômico?**

**▶ Abra a apresentação:** [`notebooks/primeiro_degrau.ipynb`](notebooks/primeiro_degrau.ipynb). Ele já vem executado e o GitHub mostra os gráficos direto no navegador. A versão sem código está em [`reports/primeiro_degrau.html`](reports/primeiro_degrau.html) (baixe e abra no navegador).

## Resumo executivo

| Pergunta | O que os dados mostram | Veredito (regra definida antes de rodar) |
|---|---|---|
| **H1.** O primeiro degrau encolheu? | Entre 2025 e 2026, a fatia empregada de quem tem até 2 anos de experiência caiu de **9 a 20 pontos**, conforme o método de correção; a do sênior, ~2. | **Confirmada** nos três métodos |
| **Ato 1.** O lugar onde o júnior aprendia esvaziou? | O Stack Overflow ficou **64% abaixo** da projeção pela tendência anterior em 48 meses e 98% abaixo no último mês. No primeiro ano, porém, a queda não se distingue de um placebo da COVID. | **Inconclusivo** pela regra de 12 meses |
| **Ato 2.** As vagas de software caíram mais que as dos outros setores, descontando juros? | Software caiu mais que os outros 46 setores, mas, quando se modelam a inércia da série e os juros, a diferença não se separa do fim do boom de 2021–22. | **Consistente com ciclo** |
| **H2.** Sênior + IA custa menos que sênior + júnior? | Se o júnior rende metade de um sênior, basta a IA dar **+5,8%** de produtividade. Se rende 70%, são precisos +19,9%. | **Parcial** (depende da premissa) |
| **H3.** O mercado ficou mais exigente? | A razão salarial sênior/júnior nos EUA subiu de **1,8 para 2,3** (2023→2026). Quem usa agentes relata *mais* ameaça, não menos. | **Parcial** |

**Para o estudante:** o degrau de entrada agora cobra o que antes se aprendia nele. Vale chegar com fundamentos sólidos, domínio de agentes e um portfólio que prove entrega.
**Para o gestor:** a conta de "sênior + IA" fecha em vários cenários, mas cortar o júnior pode sair caro depois: tende a secar a fonte de sêniores, e a razão salarial já subiu para 2,3×. A sugestão é manter um pipeline júnior menor e intencional.

## CRISP-DM no notebook

| Fase | Seção | Componente |
|---|---|---|
| 1. Entendimento do negócio | 1 | Decisores, hipóteses e regras de veredito |
| 2. Entendimento dos dados | 2 | ETL (Extract) |
| 3. Preparação | 3 | ETL (Transform) |
| — | 4 | EDA |
| 4. Modelagem | 5 | Análise preditiva (SARIMAX contrafactual, ITS, regressão com erros AR(1), classificador, bootstrap) |
| 5. Avaliação | 6 | Backtest, placebos e limitações |
| 6. Implantação | 7 | Data Storytelling → decisões e KPIs |

## Dados (todos públicos)

- **Stack Exchange API:** perguntas mensais no Stack Overflow, de 2008 a 2026, no total e por linguagem.
- **Stack Overflow Developer Survey 2023–2026:** uso de IA, percepção de ameaça, experiência, emprego e salário.
- **Indeed Hiring Lab:** índice de vagas por setor nos EUA, de 2020 a 2026.
- **Federal Reserve H.15 (via DBnomics):** taxa de juros Fed Funds.
- **Epoch AI:** horizonte de tarefa do METR e SWE-bench Verified.

As referências da literatura, todas conferidas na fonte primária, estão em [`docs/referencias.md`](docs/referencias.md).

## Como reproduzir

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows; no Linux/Mac: .venv/bin/python
cp .env.example .env    # cole uma chave gratuita da Stack Exchange API (stackapps.com)
python -m src.etl.stackoverflow
python -m src.etl.indeed
python -m src.etl.fred
python -m src.etl.capacidade
python -m src.etl.survey          # baixa ~600 MB de CSVs do survey
python -m src.models.run_series
python -m src.models.run_survey
python -m src.build_notebook
jupyter nbconvert --to notebook --execute --inplace notebooks/primeiro_degrau.ipynb
python -m pytest
```

O notebook roda offline, a partir de `data/processed/` e `data/results/`, que já estão versionados.

## Estrutura

```
src/etl/        coleta e limpeza (um módulo por fonte)
src/models/     séries temporais, survey e conta do gestor
src/viz.py      tema visual dos gráficos
data/processed/ dados limpos     data/results/ resultados dos modelos
notebooks/      apresentação      reports/ HTML sem código
docs/           design, roteiro, pesquisa e referências
tests/          testes (pytest)
```

## Equipe

- Victor Henrique Kunz de Souza
- Lucas Rogério Mendonça
- Kauã Henrique Lucindo
- Henrique Cordeiro de Oliveira
- Vinicius Steuernagel
- Nicholas Scoz
