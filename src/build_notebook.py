"""Gera notebooks/primeiro_degrau.ipynb (notebook-apresentação) com nbformat.

Uso: python -m src.build_notebook
Depois: python -m jupyter nbconvert --to notebook --execute --inplace notebooks/primeiro_degrau.ipynb

O notebook lê só data/processed e data/results (sem rede, sem data/raw). Todo número
citado no texto vem desses arquivos via f-string; a literatura vem de
fatos_literatura.csv (status == "verificado"). Células de código ficam recolhidas
(tag hide_input + jupyter.source_hidden) para a apresentação.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SAIDA = ROOT / "notebooks" / "primeiro_degrau.ipynb"

cells: list = []


def md(texto: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(texto.strip("\n")))


def code(fonte: str) -> None:
    c = nbf.v4.new_code_cell(fonte.strip("\n"))
    c.metadata["tags"] = ["hide_input"]
    c.metadata["jupyter"] = {"source_hidden": True}
    cells.append(c)


# =====================================================================================
# Capa e configuração
# =====================================================================================
md("""
# O primeiro degrau

### A IA generativa está tirando o primeiro degrau da carreira em software?

Ciência de Dados · Católica SC · Data Storytelling com CRISP-DM, ETL, EDA e análise preditiva.

*Para apresentar, abra no JupyterLab (código recolhido) ou `reports/primeiro_degrau.html`; no GitHub o código aparece. Os números do texto são lidos de `data/results/` e `data/processed/` na execução; nada foi digitado à mão.*
""")

code(r'''
# Configuração: caminhos, tema e funções de formatação.
import json, math, os, sys, warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
PROC, RES = ROOT / "data" / "processed", ROOT / "data" / "results"

RECALCULAR = False  # True: refaz ETL (precisa de rede e data/raw) e modelos antes de ler os resultados
if RECALCULAR:
    from src.etl import stackoverflow, survey, indeed, fred, capacidade
    from src.models import run_series, run_survey
    for m in (stackoverflow, survey, indeed, fred, capacidade, run_series, run_survey):
        m.main()

from src.viz import (aplicar_tema, titulo, fonte, eixo_tempo, marcos, faixa_ic, rotulo_final,
                     DESTAQUE, CONTRASTE, CONTEXTO, TEXTO, TEXTO_2, TEXTO_3, GRADE, CORES_ONDA)
aplicar_tema()

def num(x, d=0):
    """Número no formato brasileiro (1.234,5)."""
    s = f"{x:,.{d}f}"
    s = s.replace(",", "§").replace(".", ",").replace("§", ".")
    return "−" + s[1:] if s.startswith("-") else s

def lit(i):
    """Número de fatos_literatura.csv sem sinal (o texto diz se é queda ou alta)."""
    return str(F.loc[i].numero).lstrip("+-−~ ")

def virgula(ax, eixo="y", d=1):
    fmt = plt.FuncFormatter(lambda v, _: num(v, d))
    (ax.yaxis if eixo == "y" else ax.xaxis).set_major_formatter(fmt)

def pct(x, d=0, sinal=False):
    """x já em pontos percentuais."""
    s = num(abs(x), d) + "%"
    if sinal:
        return ("+" if x > 0 else "−" if x < 0 else "") + s
    return ("−" + s) if x < 0 else s

def pp(x, d=1):
    """Diferença de proporções (0–1) em pontos percentuais com sinal."""
    v = x * 100
    return ("+" if v > 0 else "−") + num(abs(v), d) + " p.p."

def ler_json(nome):
    return json.loads((RES / nome).read_text(encoding="utf-8"))

def mes_br(ts):
    m = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]
    ts = pd.Timestamp(ts)
    return f"{m[ts.month - 1]}/{ts.year}"

def mostrar(texto):
    display(Markdown(texto))
''')

code(r'''
# Leitura de todos os resultados usados no notebook.
VS = ler_json("vereditos_series.json")
VSV = ler_json("vereditos_survey.json")
VAGAS = ler_json("vagas.json")
CLF = ler_json("classificador.json")
ITS = ler_json("so_its.json")
A1, A2 = VS["ato1_stackoverflow"], VS["ato2_vagas"]
H1, H2, H3 = VSV["H1_survey"], VSV["H2"], VSV["H3"]

so = pd.read_csv(PROC / "so_mensal.csv", parse_dates=["mes"])
so_total = so[so.tag == "__total__"].set_index("mes")["perguntas"].astype(float).sort_index()
cap = pd.read_csv(PROC / "capacidade_mensal.csv", parse_dates=["mes"]).set_index("mes")
indeed = pd.read_csv(PROC / "indeed_mensal.csv", parse_dates=["mes"])
fred = pd.read_csv(PROC / "fred_mensal.csv", parse_dates=["mes"])
mc = pd.read_csv(PROC / "marcos.csv", parse_dates=["data"])
fatos = pd.read_csv(PROC / "fatos_literatura.csv", encoding="utf-8-sig")
F = fatos[fatos.status == "verificado"].set_index("id")

cf = pd.read_csv(RES / "so_contrafactual.csv", parse_dates=["mes"]).set_index("mes")
tags = pd.read_csv(RES / "so_tags_efeito.csv")
placebos = pd.read_csv(RES / "so_placebos.csv")
backtest = pd.read_csv(RES / "so_backtest.csv")
vs_serie = pd.read_csv(RES / "vagas_serie.csv", parse_dates=["mes"]).set_index("mes")
gap = pd.read_csv(RES / "survey_gap_salarial.csv")
be = pd.read_csv(RES / "break_even.csv")
agentes = pd.read_csv(RES / "survey_agentes_por_exp.csv")
emp = pd.read_csv(RES / "survey_emprego_iniciantes.csv")
cobertura = pd.read_csv(RES / "survey_cobertura.csv")
comparab = pd.read_csv(RES / "survey_comparabilidade.csv")
imp = pd.read_csv(RES / "importancias.csv")

# Act 2: modelo principal (AR(1)) quando existir; o HAC original fica como modelo inicial substituído.
A2_PRINC = VAGAS.get("modelo_principal") or VAGAS["principal"]
A2_HAC = VAGAS.get("modelo_hac_original") if "modelo_principal" in VAGAS else None

def fato(i):
    """Linha verificada de fatos_literatura.csv (falha se o fato não estiver verificado)."""
    return F.loc[i]

def cita(i, texto=None):
    r = F.loc[i]
    url = str(r.url).split(" ; ")[0].strip()
    autor = texto or str(r.fonte).split(",")[0]
    return f"[{autor}]({url})"

# Rótulos curtos para os marcos (o CSV tem o nome completo e a fonte).
CURTO = {"GitHub Copilot disponível a todos (GA)": "Copilot", "Lançamento do ChatGPT": "ChatGPT",
         "GPT-4": "GPT-4", "Claude 3 (Opus/Sonnet/Haiku)": "Claude 3", "GPT-4o": "GPT-4o",
         "Devin, primeiro 'engenheiro de software IA' (Cognition)": "Devin",
         "Claude Code (prévia) + Claude 3.7 Sonnet": "Claude Code",
         "OpenAI Codex CLI (agente de terminal) + o3/o4-mini": "Codex CLI",
         "OpenAI Codex (agente de código na nuvem)": "Codex",
         "GitHub Copilot coding agent (Microsoft Build)": "Copilot agent",
         "Claude 4 e Claude Code disponível geral (GA)": "Claude 4", "GPT-5": "GPT-5",
         "Claude Opus 4.5": "Opus 4.5"}
mc_curto = mc.assign(marco=mc.marco.map(CURTO).fillna(mc.marco))

def sel_marcos(*nomes):
    return mc_curto[mc_curto.marco.isin(nomes)].reset_index(drop=True)

ICONE = {"confirmada": "● Confirmada", "parcial": "◐ Parcial", "não sustentada": "○ Não sustentada",
         "inconclusivo": "◌ Inconclusivo", "efeito sustentado": "● Efeito sustentado",
         "consistente com IA": "● Consistente com IA", "consistente com ciclo": "○ Consistente com ciclo"}
''')

code(r'''
# Números-chave usados em várias seções.
h = cap["metr_horizonte_min_melhor"].dropna()
h_fim = h.index[-1]
h_3a = h.loc[h_fim - pd.DateOffset(years=3)]
razao_3a = h.iloc[-1] / h_3a

so_pico_mes, so_pico = so_total.idxmax(), so_total.max()
so_ult_mes, so_ult = so_total.index[-1], so_total.iloc[-1]
so_queda = (1 - so_ult / so_pico) * 100

e48, e12 = A1["efeito"], A1["efeito_12m"]
emp_m = H1["numeros"]["emprego"]["metodos"]
jr_min = -H1["faixa_efeito_jr"]["min"] * 100   # maior queda (p.p.)
jr_max = -H1["faixa_efeito_jr"]["max"] * 100   # menor queda (p.p.)
sr_var = -emp_m["bruto"]["variacao_sr"]["estimativa"] * 100

g = gap.set_index("ano")
a0, a1 = int(g.index.min()), int(g.index.max())
be_p = be[be.principal]
u5 = be_p[be_p.prod_jr == 0.5].u_star.iloc[0] * 100
ano_sal = int(be_p.ano_salarios.iloc[0])

desc = VAGAS["descritivo"]
meses_pre_vagas = len(vs_serie.loc[:pd.Timestamp(VAGAS["intervencao"]) - pd.DateOffset(months=1)])
be_sem_estresse = be_p[be_p.cenario != "estresse_METR"]
n_comp, n_tot = int(be_sem_estresse.compensa.sum()), len(be_sem_estresse)
''')

# =====================================================================================
# 0. Resumo executivo
# =====================================================================================
md("""
## 0. Resumo executivo: o degrau de entrada encolheu; a causa ainda está em disputa
`CRISP-DM | visão geral` · `Data Storytelling`
""")

code(r'''
mostrar(f"""
**Resposta curta.** Não medimos o emprego total em software. Os dados mostram que o lugar onde o iniciante aprendia
esvaziou ({num(math.floor(so_queda))}% menos perguntas no Stack Overflow que no pico) e que a fatia empregada de quem tem até
2 anos caiu muito mais que a do sênior. {"Nas vagas, a tese rival (ciclo de juros) explica tão bem quanto a nossa." if "ciclo" in A2["veredito"] else "Nas vagas, o efeito resiste aos juros, com ressalvas."}
""")
''')

code(r'''
pos = A2_PRINC["coeficientes"]["pos"]
linhas = [
    ("Ato 1: o Stack Overflow esvaziou além da tendência?", A1["veredito"],
     f"{pct(e48['efeito_pct_medio'])} vs. tendência; o placebo da COVID empata com o 1º ano"),
    ("Ato 2: vagas de software caíram mais, descontando juros?", A2["veredito"],
     f"pós-ChatGPT {pct(pos['efeito_pct'], 1, True)} (p = {num(pos['p'], 2)}) com juros e inércia no modelo"),
    ("H1: o primeiro degrau encolheu?", H1["veredito"],
     f"≤2 anos: −{num(jr_max)} a −{num(jr_min)} pontos; ≥10 anos: ~−{num(sr_var)}"),
    ("H2: sênior + IA sai mais barato?", H2["veredito"],
     f"empata com {pct(u5, 1, True)} de ganho; depende de quanto o júnior produz"),
    ("H3: o mercado ficou mais exigente?", H3["veredito"],
     f"razão salarial {num(g.loc[a0,'razao_sr_jr'],2)} → {num(g.loc[a1,'razao_sr_jr'],2)}; agentes não protegem"),
]
tab = "| Pergunta | Veredito | Por quê |\n|---|---|---|\n" + "\n".join(
    f"| {p} | **{ICONE.get(v, v)}** | {e} |" for p, v, e in linhas)
mostrar("**Vereditos** (regras fixadas antes de rodar os modelos)\n\n" + tab)
''')

md("""
| Fase CRISP-DM | Componente | Seção |
|---|---|---|
| 1. Negócio | Storytelling (pergunta, decisores) | 1 |
| 2. Dados | ETL: Extract | 2 |
| 3. Preparação | ETL: Transform / Load | 3 |
| (exploração) | EDA | 4 |
| 4. Modelagem | Análise preditiva | 5 |
| 5. Avaliação | Validação | 6 |
| 6. Implantação | Storytelling → decisão | 7 |
""")

# =====================================================================================
# 1. Negócio
# =====================================================================================
md("""
## 1. Duas pessoas precisam decidir, e as regras foram escritas antes de olhar o resultado
`CRISP-DM fase 1 | Entendimento do negócio`

De 2022 a 2026 a IA foi do chatbot ao agente que executa tarefas de horas. Iniciantes dizem que "não tem vaga";
empresas, que "falta gente boa". **A IA tirou o primeiro degrau, ou é só o ciclo de juros pós-boom de 2021–22?**
Decidem com isso o **estudante** e o **gestor** que monta a equipe.

| | Hipótese | Teste |
|---|---|---|
| H1 | O primeiro degrau encolheu | Emprego ≤2 vs. ≥10 anos (survey); Stack Overflow vs. tendência |
| H2 | Sênior + IA custa menos que sênior + júnior | Simulação com salários do survey |
| H3 | O mercado ficou mais exigente | Razão salarial; classificador de ameaça |
| Rival | É o ciclo | Vagas de software vs. outros setores, com juros no modelo |

**Regras fixadas antes de rodar:** diferença só conta com IC 95% longe do zero; o Stack Overflow precisa vencer
placebos (datas falsas de corte); vagas só contam como IA se o efeito resistir aos juros (p < 0,05). O que saiu
"inconclusivo" fica assim.
""")

# =====================================================================================
# 2. Dados
# =====================================================================================
md("""
## 2. Cinco fontes públicas olham o mesmo problema por ângulos diferentes
`CRISP-DM fase 2 | Entendimento dos dados` · `ETL: Extract`
""")

code(r'''
ind_ini, ind_fim = indeed.mes.min(), indeed.mes.max()
ff = fred[fred.serie == "FEDFUNDS"]
cob = cobertura.set_index("ano")
fontes = [
    ("Stack Exchange API", "perguntas por mês no Stack Overflow",
     f"{mes_br(so_total.index.min())}–{mes_br(so_total.index.max())}", "https://api.stackexchange.com/docs/questions"),
    ("Stack Overflow Developer Survey", f"emprego, salário, uso de IA ({num(cob.n.sum())} respostas)",
     f"{cob.index.min()}–{cob.index.max()}", "https://survey.stackoverflow.co/"),
    ("Indeed Hiring Lab", f"vagas por setor nos EUA ({indeed.setor.nunique()} setores)",
     f"{mes_br(ind_ini)}–{mes_br(ind_fim)}", "https://github.com/hiring-lab/job_postings_tracker"),
    ("Federal Reserve H.15 (via DBnomics)", "juros básicos dos EUA (Fed Funds)",
     f"{mes_br(ff.mes.min())}–{mes_br(ff.mes.max())}", "https://db.nomics.world/FED/H15"),
    ("Epoch AI", "capacidade da IA (horizonte METR, SWE-bench)",
     f"{mes_br(h.index.min())}–{mes_br(h_fim)}", "https://epoch.ai/benchmarks"),
]
mostrar("| Fonte | O que mede | Período |\n|---|---|---|\n" +
        "\n".join(f"| [{a}]({d}) | {b} | {c} |" for a, b, c, d in fontes))
''')

code(r'''
# Gráfico 0: horizonte METR em escala log, com marcos por onda.
fig, ax = plt.subplots(figsize=(11, 5.4))
ax.step(h.index, h.values, where="post", color=DESTAQUE, lw=2)
ax.set_yscale("log")
ax.set_ylim(0.02, 6000)
ticks = [0.05, 1, 10, 60, 480, 960]
ax.set_yticks(ticks, ["3 s", "1 min", "10 min", "1 h", "8 h", "16 h"])
ax.minorticks_off()
ax.set_xlim(pd.Timestamp("2019-10-01"), pd.Timestamp("2027-03-01"))
eixo_tempo(ax, 1)
ax.axhline(960, color=TEXTO_3, lw=0.8)
ax.text(pd.Timestamp("2019-11-15"), 800, "acima de 16 h a medida deixa de ser confiável (METR)",
        fontsize=8.5, color=TEXTO_3, va="top")
marcos(ax, sel_marcos("Copilot", "ChatGPT", "GPT-4", "Devin", "Claude Code", "GPT-5"), y_frac=0.97)
ult = h.iloc[-1]
ax.annotate(f"{num(ult / 60, 1)} h", (h_fim, ult), xytext=(8, 0), textcoords="offset points",
            fontsize=10, color=TEXTO, va="center")
ax.annotate(f"{num(h_3a)} min", (h_fim - pd.DateOffset(years=3), h_3a), xytext=(6, -12),
            textcoords="offset points", fontsize=10, color=TEXTO)
ax.legend(handles=[Line2D([], [], color=CORES_ONDA[k], lw=1, ls=(0, (3, 3)), label=l) for k, l in
                   [("chat", "onda 1: chat"), ("assistente", "onda 2: assistente"), ("agente", "onda 3: agente")]],
          loc="lower right")
titulo(ax, f"Em 3 anos, a IA passou de tarefas de {num(h_3a)} minutos para perto de 16 horas,\no limite do que o METR mede",
       "Duração (para um humano) das tarefas de software que o melhor modelo conclui sozinho em 50% das vezes")
ax.text(pd.Timestamp("2019-11-15"), 0.2, "antes do GPT-4: só modelos antigos (GPT-2/GPT-3)",
        fontsize=8.5, color=TEXTO_3, va="bottom")
fonte(fig, "Epoch AI, benchmark_data.zip (METR time horizons), melhor valor acumulado no mês; marcos: anúncios oficiais.")
plt.show()
''')

code(r'''
md_ = fato("metr_doubling")
partes = [p.split("(")[0].strip() for p in str(md_.numero).split(";")]
urls = [u.strip() for u in str(md_.url).split(";")]
mostrar(f"""
**O que vemos →** de {num(h_3a)} minutos ({mes_br(h_fim - pd.DateOffset(years=3))}) para {num(h.iloc[-1] / 60, 1)} horas ({mes_br(h_fim)}); o horizonte dobra a cada {partes[0]} entre 2019 e 2025 e a cada {partes[1]} desde 2023 ([Kwa et al.]({urls[0]}); [METR TH1.1]({urls[-1]})).

**O que significa →** a IA já entrega as tarefas curtas que iam para o júnior. Acima de {fato('metr_16h').numero} a medida perde confiabilidade ({cita('metr_16h', 'METR')}).

**O que fazer →** treinar o que o agente ainda faz mal: revisar, integrar e decidir.
""")
''')

# =====================================================================================
# 3. Preparação
# =====================================================================================
md("""
## 3. Preparar os dados exigiu desarmar uma armadilha de seleção
`CRISP-DM fase 3 | Preparação dos dados` · `ETL: Transform / Load`
""")

code(r'''
c = comparab.set_index("coluna")["anos_comparaveis"]
nomes = {"faixa_exp": "faixa de experiência", "empregado": "está empregado", "salario_usd": "salário (US$)",
         "ia_usa": "usa IA", "ia_ameaca": "vê a IA como ameaça"}
sobrescreve = {"empregado": "2025, 2026 (2023–24 não comparáveis: seleção no WorkExp)"}
lin = "\n".join(f"| {nomes[k]} | {sobrescreve.get(k, c[k].replace(';', ', '))} |" for k in nomes
                if k in c.index and c[k][:1].isdigit())
e24 = emp[(emp.ano == 2024) & (emp.faixa_exp == "≤2") & (emp.metodo == "bruto")].iloc[0]
e25 = emp[(emp.ano == 2025) & (emp.faixa_exp == "≤2") & (emp.metodo == "bruto")].iloc[0]
mostrar(f"""
| Variável do survey | Anos comparáveis |
|---|---|
{lin}

Usamos só meses completos, médias mensais das vagas (fev/2020 = 100) e salários dos EUA entre US$ 1 mil e US$ 1 milhão;
experiência é anos de trabalho profissional (WorkExp). A armadilha: em 2024, {pct(e24.pct_empregado * 100)} de quem tinha
até 2 anos aparecia empregado, porque quase só quem trabalhava respondeu a pergunta de experiência. Em 2025 eram
{pct(e25.pct_empregado * 100)}. Por isso a comparação de emprego usa só 2025 e 2026.
""")
''')

# =====================================================================================
# 4. EDA
# =====================================================================================
md("""
## 4. Primeiro esvaziou o lugar onde se aprendia; depois, as vagas
`EDA | análise exploratória`
""")

code(r'''
# Gráfico 1: Stack Overflow, série total.
fig, ax = plt.subplots()
ax.plot(so_total.index, so_total / 1000, color=DESTAQUE)
ax.set_ylim(0, so_pico / 1000 * 1.15)
ax.set_ylabel("perguntas por mês (mil)")
eixo_tempo(ax, 2)
ax.set_xlim(so_total.index.min(), so_total.index.max() + pd.DateOffset(months=14))
marcos(ax, sel_marcos("ChatGPT", "Claude Code"), y_frac=0.97)
ax.annotate(f"pico: {num(so_pico / 1000)} mil\n({mes_br(so_pico_mes)})", (so_pico_mes, so_pico / 1000),
            xytext=(12, 4), textcoords="offset points", fontsize=10, color=TEXTO, va="bottom")
ax.annotate(f"{num(so_ult)}\n({mes_br(so_ult_mes)})", (so_ult_mes, so_ult / 1000), xytext=(8, 10),
            textcoords="offset points", fontsize=10, color=TEXTO, va="bottom")
titulo(ax, f"O Stack Overflow perdeu ~{math.floor(so_queda)}% das perguntas desde o pico",
       "Perguntas novas por mês no Stack Overflow, todas as linguagens")
fonte(fig, "Stack Exchange API (/2.3/questions, filtro total), meses completos.")
plt.show()
''')

code(r'''
p23 = so_total.loc["2022-11-01"]
mostrar(f"""
**O que vemos →** a queda começou antes da IA e acelerou depois do ChatGPT: de {num(p23 / 1000)} mil perguntas em nov/2022 para {num(so_ult)} em {mes_br(so_ult_mes)}.

**O que significa →** esvaziou o fórum onde o iniciante aprendia; com plataformas-controle, um estudo mediu queda de {lit('pnas_25')} só nos 6 meses após o ChatGPT ({cita('pnas_25', 'PNAS Nexus')}).

**O que fazer →** separar a tendência que já existia do que veio depois: é o teste do Ato 1.
""")
''')

code(r'''
# Gráfico 2: vagas (índice fev/2020 = 100) e juros em painel separado.
fig, (a1x, a2x) = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True,
                               gridspec_kw={"height_ratios": [3, 1.1], "hspace": 0.12})
a1x.plot(vs_serie.index, vs_serie.mediana_setores, color=CONTEXTO, label=f"mediana de {VAGAS['n_setores_outros']} outros setores")
a1x.plot(vs_serie.index, vs_serie.software, color=DESTAQUE, label="desenvolvimento de software")
a1x.axhline(100, color=TEXTO_3, lw=0.8)
a1x.set_ylabel("índice (fev/2020 = 100)")
a1x.set_ylim(0, vs_serie.software.max() * 1.18)
a1x.set_xlim(vs_serie.index.min(), vs_serie.index.max() + pd.DateOffset(months=9))
eixo_tempo(a1x, 1)
pk = pd.Timestamp(VAGAS["pico_software"]["mes"])
a1x.annotate(f"pico: {num(VAGAS['pico_software']['indice'])}", (pk, VAGAS["pico_software"]["indice"]),
             xytext=(8, 2), textcoords="offset points", fontsize=10, color=TEXTO)
rotulo_final(a1x, vs_serie.software, num(desc["software_ultimo"]), cor=TEXTO, dx_dias=25)
rotulo_final(a1x, vs_serie.mediana_setores, num(desc["mediana_ultimo"]), cor=TEXTO_2, dx_dias=25)
a1x.legend(loc="upper right", bbox_to_anchor=(0.93, 1.0))
marcos(a1x, sel_marcos("ChatGPT"), y_frac=0.97)
fed = vs_serie.FEDFUNDS
a2x.step(fed.index, fed, where="post", color=TEXTO_2, lw=1.6)
a2x.set_ylabel("juros EUA\n(% ao ano)")
a2x.set_ylim(0, fed.max() * 1.3)
marcos(a2x, sel_marcos("ChatGPT"), rotulos=False)
a2x.text(0.005, 0.92, "Fed Funds (taxa básica de juros dos EUA)", transform=a2x.transAxes, fontsize=9,
         color=TEXTO_2, va="top")
sw_var, md_var = desc["software_ultimo"] - 100, desc["mediana_ultimo"] - 100
qual = lambda v: "menos" if v < 0 else "mais"
titulo(a1x, f"Software tem {pct(abs(sw_var))} {qual(sw_var)} vagas que em 2020; os demais setores, {pct(abs(md_var))} {qual(md_var)}",
       "Índice de vagas anunciadas no Indeed (EUA), média mensal; juros no painel de baixo")
fonte(fig, "Indeed Hiring Lab (job_postings_by_sector_US); Fed Funds: Federal Reserve H.15 via DBnomics.")
plt.show()
''')

code(r'''
mostrar(f"""
**O que vemos →** software subiu mais no boom (pico de {num(VAGAS['pico_software']['indice'])} em {mes_br(VAGAS['pico_software']['mes'])}) e caiu mais depois: hoje {num(desc['software_ultimo'])}, contra {num(desc['mediana_ultimo'])} dos outros setores, junto com a alta dos juros.

**O que significa →** o desenho serve às duas histórias, IA ou fim do boom; o Indeed também registra software {lit('indeed_275')} abaixo de fev/2020 ({cita('indeed_275', 'Indeed Hiring Lab')}).

**O que fazer →** pôr os juros no modelo antes de concluir (Ato 2).
""")
''')

# =====================================================================================
# 5. Modelagem
# =====================================================================================
md("""
## 5. Os modelos confirmam o júnior encolhendo; o motivo nas vagas continua aberto
`CRISP-DM fase 4 | Modelagem` · `Análise preditiva`

### Ato 1: o Stack Overflow ficou muito abaixo da própria tendência
Um SARIMAX treinado de 2014 a out/2022 projeta a tendência anterior; a distância até o real é o efeito estimado.
""")

code(r'''
# Gráfico 3: real vs. projeção pela tendência anterior (SARIMAX).
real = so_total.loc["2014-01-01":]
fig, ax = plt.subplots()
faixa_ic(ax, cf / 1000, rotulo="Tendência anterior (projeção, IC 95%)")
ax.plot(real.index, real / 1000, color=DESTAQUE, label="perguntas reais")
ax.axvline(pd.Timestamp(A1["modelo"]["treino_fim"]), color=TEXTO_3, lw=0.8)
ax.text(pd.Timestamp(A1["modelo"]["treino_fim"]) - pd.Timedelta(days=30), 3, "fim do treino ",
        fontsize=8.5, color=TEXTO_3, ha="right", va="bottom")
ax.set_ylim(0, real.max() / 1000 * 1.12)
ax.set_xlim(real.index.min(), real.index.max() + pd.DateOffset(months=12))
ax.set_ylabel("perguntas por mês (mil)")
eixo_tempo(ax, 2)
marcos(ax, sel_marcos("ChatGPT"), y_frac=0.97)
lm = cf.index[-1]
ax.annotate(f"{pct(e48['efeito_pct_ultimo_mes'])}\nno último mês", (lm, cf.previsto.iloc[-1] / 1000),
            xytext=(10, 0), textcoords="offset points", fontsize=10, color=TEXTO, va="center")
ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.86))
titulo(ax, f"Pós-ChatGPT, o site ficou {pct(-e48['efeito_pct_medio'])} abaixo da projeção pela tendência anterior",
       f"Média de {e48['n_meses']} meses; abaixo do IC 95% por {e48['meses_abaixo_ic_seguidos']} meses seguidos")
fonte(fig, "Stack Exchange API; projeção SARIMAX em log treinada em 2014-01 a 2022-10 (ordem por AIC). Faixa = IC 95%.")
plt.show()
''')

code(r'''
pl = A1["comparacao_placebo_principal"]
if A1["veredito"] == "efeito sustentado":
    leitura = "Pela regra fixada antes, o efeito é **sustentado**: nenhum placebo chega perto."
else:
    leitura = (f"Pela regra fixada antes, **inconclusivo**: no 1º ano a queda ({pct(pl['efeito_real_12m_pct'])}) empata com o "
               f"placebo de jun/2020, quando o pico da COVID se desfez ({pct(-pl['max_abs_placebo_pct'])}). Em {e48['n_meses']} "
               f"meses nenhum placebo chega perto, mas isso não estava na regra.")
mostrar(f"""
**O que vemos →** o modelo acerta o passado melhor que um ingênuo sazonal (erro de {num(A1['backtest']['mape_modelo_medio'], 1)}% contra {num(A1['backtest']['mape_ingenuo_medio'], 1)}%); depois de nov/2022 o real descola e não volta.

**O que significa →** {leitura}

**O que fazer →** tratar como sinal forte, não como prova: o site mede perguntas, não empregos.
""")
''')

code(r'''
# Gráfico 4 (pequeno): efeito por linguagem.
nomes_tag = {"python": "Python", "javascript": "JavaScript", "java": "Java", "reactjs": "React"}
t = tags.assign(nome=tags.tag.map(nomes_tag)).sort_values("efeito_pct_medio", ascending=False)
fig, ax = plt.subplots(figsize=(10, 3.6))
fig.subplots_adjust(bottom=0.2)
ax.set_axisbelow(True)
y = np.arange(len(t))
ax.barh(y, t.efeito_pct_medio, color=DESTAQUE, height=0.55)
ax.errorbar(t.efeito_pct_medio, y, xerr=[t.efeito_pct_medio - t.efeito_pct_lo, t.efeito_pct_hi - t.efeito_pct_medio],
            fmt="none", ecolor=TEXTO_2, elinewidth=1, capsize=3)
for yi, v in zip(y, t.efeito_pct_medio):
    ax.text(2, yi, pct(v), va="center", ha="left", fontsize=10, color=TEXTO)
ax.axvline(e48["efeito_pct_medio"], color=TEXTO_3, lw=1)
ax.text(e48["efeito_pct_medio"] + 1, len(t) - 0.45, f"total: {pct(e48['efeito_pct_medio'])}", fontsize=9, color=TEXTO_3)
ax.set_yticks(y, t.nome)
ax.set_xlim(-95, 14)
ax.axvline(0, color=TEXTO_2, lw=0.8)
ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
ax.set_xlabel("diferença média vs. projeção pela tendência anterior (%), com IC 95%")
titulo(ax, f"A queda aparece em todas as linguagens, de {pct(t.efeito_pct_medio.max())} a {pct(t.efeito_pct_medio.min())}")
fonte(fig, "Stack Exchange API por tag; um SARIMAX por linguagem, treino 2015-01 a 2022-10.")
plt.show()
mostrar(f"""
**O que vemos →** todas as linguagens caem; a que menos cai é {t.nome.iloc[0]} ({pct(t.efeito_pct_medio.iloc[0])}).

**O que significa →** mudou o hábito de perguntar em fórum, não uma tecnologia.

**O que fazer →** quem aprende com IA deve conferir a resposta na documentação oficial; o fórum que corrigia erros esvaziou.
""")
''')

code(r'''
pos, fed_c = A2_PRINC["coeficientes"]["pos"], A2_PRINC["coeficientes"]["FEDFUNDS"]
if "ciclo" in A2["veredito"]:
    t2 = "software caiu mais que os outros setores, mas o modelo não separa IA do fim do boom"
else:
    t2 = "software caiu mais que os outros setores mesmo descontando os juros"
p_txt = lambda p: "< 0,001" if p < 0.001 else num(p, 2)
modelo_txt = "regressão com erros AR(1)" if A2_PRINC.get("ar1") is not None else "regressão com erros HAC"
mostrar(f"""### Ato 2: {t2}

**Veredito: {ICONE.get(A2['veredito'], A2['veredito'])}.** Com juros e a inércia da série no modelo ({modelo_txt},
{A2_PRINC.get('n', VAGAS['n_meses'])} meses), software ÷ outros setores fica em {pct(pos['efeito_pct'], 1, True)} depois do
ChatGPT. A versão anterior, substituída, está no Apêndice.

| Termo | Efeito | IC 95% (log) | p |
|---|---|---|---|
| depois do ChatGPT | {pct(pos['efeito_pct'], 1, True)} | {num(pos['ic95'][0], 2)} a {num(pos['ic95'][1], 2)} | {p_txt(pos['p'])} |
| Fed Funds (+1 ponto) | {pct(fed_c['efeito_pct'], 1, True)} | {num(fed_c['ic95'][0], 2)} a {num(fed_c['ic95'][1], 2)} | {p_txt(fed_c['p'])} |
""")
''')

code(r'''
h1_tit = ("a fatia empregada de quem tem até 2 anos caiu muito mais que a do sênior" if H1["veredito"] != "não sustentada"
          else "a queda do júnior não se separa da do sênior")
mostrar(f"""### Ato 3 (H1): {h1_tit}
Três métodos (bruto, reponderado e só quem tem 2 anos), porque quem responde à pergunta de experiência mudou entre 2025 e 2026.""")
''')

code(r'''
# Gráfico 5: variação da % empregada 2025 → 2026, ≤2 vs ≥10, três métodos, com IC 95%.
rot_met = {"bruto": "Bruto", "ipw": "Reponderado (IPW)", "workexp_igual_2": "Só quem tem 2 anos"}
fig, ax = plt.subplots(figsize=(11, 4.6))
for i, (k, rot) in enumerate(rot_met.items()):
    yy = len(rot_met) - 1 - i
    for chave, dy, cor, lab in [("variacao_jr", 0.14, DESTAQUE, "até 2 anos de experiência"),
                                ("variacao_sr", -0.14, TEXTO_2, "10 anos ou mais")]:
        v = emp_m[k][chave]
        est, lo, hi = v["estimativa"] * 100, v["ic_inf"] * 100, v["ic_sup"] * 100
        ax.plot([lo, hi], [yy + dy] * 2, color=cor, lw=2, solid_capstyle="round")
        ax.plot(est, yy + dy, "o", color=cor, ms=8, mec="white", mew=2, label=lab if i == 0 else None)
        if chave == "variacao_jr":
            ax.text(lo - 0.6, yy + dy, pp(v["estimativa"]), ha="right", va="center", fontsize=10, color=TEXTO)
ax.set_yticks(range(len(rot_met)), list(rot_met.values())[::-1])
ax.axvline(0, color=TEXTO_2, lw=0.8)
ax.set_xlim(-28, 3)
ax.set_ylim(-0.6, len(rot_met) - 0.4)
ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
ax.set_xlabel("variação na fatia empregada, 2025 → 2026 (pontos percentuais); linha = IC 95% bootstrap")
ax.legend(loc="lower left", ncols=2)
titulo(ax, f"A fatia empregada de quem tem até 2 anos caiu {num(jr_max)} a {num(jr_min)} pontos; a do sênior, ~{num(sr_var)}",
       "Quem tem até 2 anos de experiência vs. quem tem 10 ou mais, por três métodos de correção")
fonte(fig, "Stack Overflow Developer Survey 2025 e 2026; experiência = WorkExp; 2.000 reamostragens bootstrap.")
plt.show()
''')

code(r'''
rb = H1["numeros"]["emprego"]["robustez_jr_sem_estudantes"]
e26 = emp[(emp.ano == 2026) & (emp.faixa_exp == "≤2") & (emp.metodo == "bruto")].iloc[0]
e25s = emp[(emp.ano == 2025) & (emp.faixa_exp == "≤2") & (emp.metodo == "bruto")].iloc[0]
mostrar(f"""
**O que vemos →** nos três métodos o júnior cai com IC longe do zero; sem estudantes, {pp(rb['variacao']['estimativa'])} H1: **{ICONE.get(H1['veredito'], H1['veredito'])}**.

**O que significa →** bate com estudos nos EUA: jovens de 22–25 anos em ocupações expostas à IA estão {lit('canaries_19')} abaixo dos pares ({cita('canaries_19', 'Brynjolfsson et al.')}); em firmas que adotaram IA, o emprego júnior cai ~{lit('hl_efeito').split(' ')[0]} em seis trimestres e o sênior segue sem quebra de tendência ({cita('hl_efeito', 'Hosseini e Lichtinger')}). É associação: o survey se autosseleciona e 2026 tem mais estudantes ({pct(e26.pct_so_estudante * 100)} contra {pct(e25s.pct_so_estudante * 100)}).

**O que fazer →** refazer a conta com o survey de 2027 antes de tratar a queda como tendência.
""")
''')

md("""
### H3: o salário de entrada perdeu terreno, e usar agentes não protege
""")

code(r'''
# Gráfico 6: razão salarial sênior/júnior nos EUA, com IC 95%.
fig, ax = plt.subplots(figsize=(10, 4.6))
anos = g.index.values
ax.errorbar(anos, g.razao_sr_jr, yerr=[g.razao_sr_jr - g.razao_ic_inf, g.razao_ic_sup - g.razao_sr_jr],
            fmt="o-", color=DESTAQUE, ecolor=DESTAQUE, elinewidth=1.2, capsize=4, ms=8, mec="white", mew=2, lw=2)
for a, dx, ha in ((a0, -12, "right"), (a1, 12, "left")):
    ax.annotate(f"{num(g.loc[a, 'razao_sr_jr'], 2)}×", (a, g.loc[a, "razao_sr_jr"]), xytext=(dx, 0),
                textcoords="offset points", fontsize=11, color=TEXTO, va="center", ha=ha)
virgula(ax, "y", 1)
ax.set_xticks(anos, [str(a) for a in anos])
ax.set_xlim(a0 - 0.6, a1 + 0.5)
ax.set_ylim(1.5, max(2.9, g.razao_ic_sup.max() * 1.05))
ax.set_ylabel("salário sênior ÷ salário júnior")
titulo(ax, f"Nos EUA, o sênior ganha {num(g.loc[a1, 'razao_sr_jr'], 1)} vezes o júnior; em {a0} eram {num(g.loc[a0, 'razao_sr_jr'], 1)}",
       "Mediana do salário anual de quem tem ≥10 anos ÷ ≤2 anos de experiência; barras = IC 95% bootstrap")
fonte(fig, "Stack Overflow Developer Survey 2023–2026, respondentes dos EUA com salário informado.")
plt.show()
''')

code(r'''
tx = CLF["taxas_descritivas"]
m = CLF["metricas"]
vr = H3['numeros']['gap_salarial']['variacao_razao']
mostrar(f"""
**O que vemos →** júnior de US$ {num(g.loc[a0, 'mediana_jr'] / 1000)} mil para US$ {num(g.loc[a1, 'mediana_jr'] / 1000, 1)} mil; sênior de US$ {num(g.loc[a0, 'mediana_sr'] / 1000)} mil para US$ {num(g.loc[a1, 'mediana_sr'] / 1000)} mil. No classificador (AUC {num(m[CLF['melhor_modelo']]['auc_media'], 2)}, fraco), quem usa agentes relata mais ameaça ({pct(tx['por_usa_agentes']['sim']['estimativa'] * 100)} contra {pct(tx['por_usa_agentes']['nao']['estimativa'] * 100)}).

**O que significa →** H3: **{ICONE.get(H3['veredito'], H3['veredito'])}**. O mercado paga mais pela experiência; usar agentes não protege.

**O que fazer →** usar agentes no dia a dia, mas investir em experiência comprovada, que é o que o salário premia.
""")
''')

code(r'''
mostrar(f"""### H2: a conta do gestor fecha para sênior + IA em {n_comp} de {n_tot} cenários
**Simulação**, não observação: custo por entrega de sênior + júnior vs. sênior + agente (US$ {num(be_p.custo_ia_ano.iloc[0] / 12)}/mês), salários medianos dos EUA em {ano_sal}.""")
''')

code(r'''
# Gráfico 7: ganho necessário (u*) vs. cenários de produtividade.
cen = be_p.drop_duplicates("cenario").set_index("cenario").ganho * 100
sens = H2["numeros"]["u_star_por_prod_jr"][str(ano_sal)]
pj = sorted(sens, key=float)
fig, ax = plt.subplots(figsize=(10.5, 5))
rot_cen = {"otimista": "otimista", "central": "central", "pessimista": "pessimista (sem ganho)",
           "estresse_METR": "estresse (RCT do METR)"}
for nome, v in cen.items():
    ax.axhline(v, color=CONTEXTO, lw=1)
    ax.text(len(pj) - 0.4, v + 0.6, f"{rot_cen.get(nome, nome)}: {pct(v, 0, True)}", fontsize=9.5, color=TEXTO_2,
            va="bottom")
for i, p in enumerate(pj):
    s = sens[p]
    est, lo, hi = s["estimativa"] * 100, s["ic_inf"] * 100, s["ic_sup"] * 100
    ax.plot([i, i], [lo, hi], color=DESTAQUE, lw=2, solid_capstyle="round")
    ax.plot(i, est, "o", color=DESTAQUE, ms=9, mec="white", mew=2)
    ax.text(i + 0.08, est, pct(est, 1, True), fontsize=10.5, color=TEXTO, va="center")
ax.set_xticks(range(len(pj)), [f"júnior rende {pct(float(p) * 100)}\nde um sênior" for p in pj])
ax.set_xlim(-0.5, len(pj) + 0.9)
ax.set_ylim(-25, 38)
ax.set_ylabel("ganho de produtividade necessário (%)")
ax.grid(axis="y", visible=False)
titulo(ax, f"Se o júnior rende metade de um sênior, basta a IA render {pct(u5, 1, True)}",
       f"Simulação: salários medianos EUA {ano_sal}, IA a US$ {num(be_p.custo_ia_ano.iloc[0] / 12)}/mês; ponto = ganho de empate (IC 95%)")
fonte(fig, "Salários: Stack Overflow Developer Survey; cenários: Cui et al. (RCTs Copilot) e METR (2025); preço: planos de agente.")
plt.show()
''')

code(r'''
abaixo_central = [pct(float(p) * 100) for p in pj if sens[p]["estimativa"] * 100 < cen["central"]]
acima_central = [pct(float(p) * 100) for p in pj if sens[p]["estimativa"] * 100 >= cen["central"]]
mostrar(f"""
**O que vemos →** com o júnior rendendo {' ou '.join(abaixo_central)} de um sênior, o empate fica abaixo do cenário central ({pct(cen['central'], 0, True)}); com {' ou '.join(acima_central) or 'nenhum'}, acima. No estresse (IA atrasa o trabalho), nunca compensa.

**O que significa →** os cenários vêm de RCTs: {lit('copilot_26')} mais tarefas com assistente ({cita('copilot_26', 'Cui et al.')}), {lit('metr_rct_19').split(' ')[0]} mais tempo para devs experientes ({cita('metr_rct_19', 'METR')}). H2: **{ICONE.get(H2['veredito'], H2['veredito'])}**: depende de quanto o júnior produz.

**O que fazer →** gestor: antes de cortar a vaga, medir quanto o júnior da própria equipe entrega com IA.
""")
''')

# =====================================================================================
# 6. Avaliação
# =====================================================================================
md("""
## 6. O SARIMAX prevê melhor que o ingênuo, mas o placebo da COVID impede um veredito; a dúvida que sobra é de causa
`CRISP-DM fase 5 | Avaliação`
""")

code(r'''
ver = [("Ato 1", A1["veredito"]), ("Ato 2", A2["veredito"]), ("H1", H1["veredito"]), ("H2", H2["veredito"]),
       ("H3", H3["veredito"])]
mostrar(f"""
O SARIMAX errou menos que o ingênuo em {A1['backtest']['origens_modelo_melhor']} de {A1['backtest']['n_origens']} janelas do passado
({num(A1['backtest']['mape_modelo_medio'], 1)}% contra {num(A1['backtest']['mape_ingenuo_medio'], 1)}%), mas o placebo de jun/2020
({pct(-A1['placebos']['max_abs_efeito_pct'])}) tem o tamanho do 1º ano real ({pct(e12['efeito_pct_medio'])}). Tabelas no Apêndice.

**Limitações**
- O Stack Overflow mede perguntas, não empregos; parte da queda é o chatbot substituindo o fórum.
- Quem responde o survey se autosseleciona, e a amostra de 2026 é menor ({num(cob.loc[2026, 'n'])} contra {num(cob.loc[2025, 'n'])}) e tem mais estudantes.
- Vagas só existem desde fev/2020, e juros e ChatGPT andam juntos: não dá para separar IA de ciclo.
- A conta do gestor usa só salário e uma premissa sobre quanto o júnior produz.

**Vereditos:** {' · '.join(f"{a} {ICONE.get(b, b)}" for a, b in ver)}
""")
''')

# =====================================================================================
# 7. Implantação
# =====================================================================================
md("""
## 7. Para quem está entrando, o degrau ficou mais alto; para quem contrata, cortá-lo pode sair caro depois
`CRISP-DM fase 6 | Implantação` · `Data Storytelling → decisão`
""")

code(r'''
tx = CLF["taxas_descritivas"]
mostrar(f"""
**Para o estudante**
- A fatia empregada de quem tem até 2 anos caiu {num(jr_max)} a {num(jr_min)} pontos: comece a buscar entrada antes de se formar (estágio, projeto com usuário real).
- A razão salarial subiu para {num(g.loc[a1, 'razao_sr_jr'], 1)}×: acumule experiência que dê para provar (PR aceito em código aberto, freela, monitoria).
- Quem usa agentes relata mais ameaça: agente é requisito, não proteção. O diferencial é revisar o que ele entrega, e isso exige fundamentos.

**Para o gestor.** A conta fecha para sênior + IA em {n_comp} de {n_tot} cenários. Mas cortar a entrada tende a secar a fonte de sêniores, e a razão salarial já subiu para {num(g.loc[a1, 'razao_sr_jr'], 1)}×.
- Manter um pipeline júnior menor e intencional (aprendiz + IA + mentoria), avaliado pelo que entrega com IA.

| KPI | Fonte | A tese se fortalece se... |
|---|---|---|
| Vagas de software vs. outros setores | Indeed | software não se recuperar quando os juros caírem |
| % empregado ≤2 vs. ≥10 anos | Survey 2027 | a distância persistir |
| Razão salarial sênior/júnior | Survey 2027 | continuar subindo |
| Horizonte de tarefa | Epoch AI / METR | continuar dobrando |

**Próxima rodada:** refazer tudo com o survey de 2027 (`RECALCULAR = True` na primeira célula).
""")
''')

# =====================================================================================
# Apêndice
# =====================================================================================
md("""
---
## Apêndice
### A. Detalhes técnicos
""")

code(r'''
mo = A1["modelo"]; s12 = A1["sensibilidade_inicio_2012"]
ordem = lambda m: f"SARIMAX{tuple(m['order'])}×{tuple(m['seasonal_order'])} com tendência linear"
its = ITS
bt = backtest.copy()
bt["melhor"] = np.where(bt.mape_modelo < bt.mape_ingenuo, "SARIMAX", "ingênuo")
tb = "| Origem | Erro SARIMAX (MAPE) | Erro ingênuo sazonal | Melhor |\n|---|---|---|---|\n" + "\n".join(
    f"| {mes_br(r.origem)} | {num(r.mape_modelo, 1)}% | {num(r.mape_ingenuo, 1)}% | {r.melhor} |" for r in bt.itertuples())
tp = "| Corte falso | Efeito médio em 12 meses | Meses seguidos abaixo do IC |\n|---|---|---|\n" + "\n".join(
    f"| {mes_br(r.corte_falso)} | {pct(r.efeito_pct_medio, 1, True)} | {r.meses_abaixo_ic_seguidos} |" for r in placebos.itertuples())
tp += f"\n| **ChatGPT (real), 12 meses** | **{pct(e12['efeito_pct_medio'], 1, True)}** | **{e12['meses_abaixo_ic_seguidos']}** |"
mostrar(f"""
**Ato 1, SARIMAX.** Log das perguntas mensais; treino {mes_br(mo['treino_inicio'])} a {mes_br(mo['treino_fim'])}
({mo['n_treino']} meses). Ordem escolhida por AIC: {ordem(mo)} (AIC = {num(mo['aic'], 1)}).
Sensibilidade com treino desde 2012: {ordem(s12['modelo'])}, efeito médio {pct(s12['efeito']['efeito_pct_medio'], 1)}
(o principal dá {pct(e48['efeito_pct_medio'], 1)}).

**Backtest** (previsão de 12 meses a partir de cada origem, só com dados anteriores):

{tb}

**Placebos** (o mesmo modelo, fingindo que a IA chegou em outra data):

{tp}

**Ato 1, regressão segmentada (ITS, erros {its['cov_type']}).** {its['n']} meses, R² = {num(its['r2'], 3)}.
Mudança de nível em dez/2022: {pct(its['nivel_pos']['efeito_pct'], 1, True)} (p = {num(its['nivel_pos']['p'], 2)}, não significativa).
Mudança de inclinação: {pct(its['inclinacao_pos']['efeito_pct'], 1, True)} ao mês (p < 0,001): a série passa a cair mais rápido todo mês.
""")
''')

code(r'''
ar1 = A2_PRINC.get("ar1")
txt = ""
if A2_HAC is not None:
    txt += f"""
**Ato 2, modelo inicial, substituído.** A primeira versão era uma regressão comum com erros HAC. Ela dava o marcador
"depois do ChatGPT" em {pct(A2_HAC['coeficientes']['pos']['efeito_pct'], 0, True)} com p < 0,001, o que passaria na regra.
A revisão achou os resíduos muito autocorrelacionados (Durbin-Watson {num(A2_HAC['durbin_watson'], 2)}; correlação de um mês
com o seguinte de {num(A2_HAC['autocorr_lag1'], 2)}), e com tão poucos meses o HAC não corrige isso: o p-valor saía otimista
demais. No modelo AR(1) cada mês carrega {pct(ar1 * 100)} do desvio do mês anterior; com essa inércia na conta, o efeito
encolhe para {pct(A2_PRINC['coeficientes']['pos']['efeito_pct'], 1, True)} e deixa de ser significativo.
"""
ph = VAGAS["sensibilidade_pos_hoc"]["placebos_dummy"]
txt += f"""
**Ato 2, o porém do boom.** Juros e marcador pós-ChatGPT têm correlação de {num(VAGAS.get('corr_fedfunds_pos', np.nan), 2)},
e o período antes do ChatGPT tem só {meses_pre_vagas} meses, quase todos de boom. No modelo antigo, pôr o marcador em
datas falsas ({', '.join(mes_br(p['dummy_desde']) for p in ph)}) também gerava saltos grandes
({', '.join(pct(p['efeito_pct'], 0, True) for p in ph)}). Checagem feita depois de ver o resultado; fora do veredito.

**Ato 2, robustez.** Diferença em pontos de índice, software vs. índice total do Indeed e capacidade da IA no lugar do
marcador: p = {num(VAGAS['diferenca']['p_termo'], 4)}, {num(VAGAS['vs_total']['p_termo'], 6)} e
{num(VAGAS['capacidade']['p_termo'], 3)}. Foram estimadas com erros HAC, sem a correção AR(1), por isso não valem como teste.
"""
traduz = {"a dummy pós-ChatGPT": "o marcador pós-ChatGPT", "dummy": "marcador", "trend='t'": "tendência linear",
          "`previsto`": "A projeção", "os limites lo95/hi95": "os limites do IC 95%", "lo95/hi95": "limites do IC 95%",
          "OLS+HAC": "regressão comum com erros HAC", "do OLS": "da regressão comum", " (D=0 vs. D=1)": "",
          "o Q&A": "o fórum de perguntas e respostas", "lag-1": "correlação mês a mês", " — ": ": ",
          "FEDFUNDS": "Fed Funds", "2020-02": "fev/2020", "2020-06": "jun/2020"}
lim = []
for l in VS["limitacoes"]:
    for a, b in traduz.items():
        l = l.replace(a, b)
    lim.append(l)
txt += "\n**Limitações técnicas**\n\n" + "\n".join(f"- {l}" for l in lim)
mostrar(txt)
''')

code(r'''
mt = CLF["metricas"]
nm = {"dummy": "Trivial (sempre a classe mais comum)", "logistica_so_faixa_exp": "Logística só com experiência",
      "logistica": "Logística com todas as variáveis", "hgb": "HistGradientBoosting"}
tc = "| Modelo | AUC (CV, média ± dp) | F1 |\n|---|---|---|\n" + "\n".join(
    f"| {nm.get(k, k)} | {num(v['auc_media'], 3)} ± {num(v['auc_dp'], 3)} | {num(v['f1_media'], 3)} |" for k, v in mt.items())
ag = agentes.set_index(["ano", "faixa_exp"])
tx = CLF["taxas_descritivas"]
mostrar(f"""
**Classificador de ameaça (H3).** Survey {CLF['ano']}, {num(CLF['n'])} respostas, {pct(CLF['prevalencia_ameaca'] * 100)} dizem que a IA ameaça o emprego.
Validação: {CLF['cv']}. Variáveis: {', '.join(CLF['features'])}. AUC no teste: {num(CLF['auc_holdout'], 3)}.
Taxa de ameaça: {pct(tx['por_faixa_exp']['≤2']['estimativa'] * 100)} entre juniores e {pct(tx['por_faixa_exp']['≥10']['estimativa'] * 100)} entre sêniores.
Uso de agentes em 2026: {pct(ag.loc[(2026, '≥10'), 'pct_usa_agentes'] * 100)} dos sêniores e {pct(ag.loc[(2026, '≤2'), 'pct_usa_agentes'] * 100)} dos juniores
(a pergunta mudou entre 2025 e 2026, então não comparamos anos).

{tc}
""")
fig, ax = plt.subplots(figsize=(9, 3.6))
fig.subplots_adjust(bottom=0.2)
ax.set_axisbelow(True)
nomes_f = {"usa_agentes": "usa agentes", "emprego": "situação de emprego", "ia_usa": "usa IA", "pais": "país",
           "cargo": "cargo", "faixa_exp": "experiência"}
ii = imp.sort_values("importancia_media")
ax.barh(ii.feature.map(nomes_f).fillna(ii.feature), ii.importancia_media, xerr=ii.importancia_dp, color=CONTEXTO,
        ecolor=TEXTO_2, height=0.55)
virgula(ax, "x", 3)
ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
ax.set_xlabel("queda da AUC ao embaralhar a variável (permutation importance)")
top, low = ii.feature.iloc[-1], ii.feature.iloc[0]
titulo(ax, f"No classificador, '{nomes_f.get(top, top)}' é o que mais pesa; '{nomes_f.get(low, low)}', o que menos pesa")
fonte(fig, f"Stack Overflow Developer Survey {CLF['ano']}; conjunto de teste (20%).")
plt.show()
''')

code(r'''
mostrar("**Reprodutibilidade.** `pip install -r requirements.txt`; chave da Stack Exchange API no `.env` (veja `.env.example`); "
        "`python -m src.etl.stackoverflow` (e `survey`, `indeed`, `fred`, `capacidade`), `python -m src.models.run_series`, "
        "`python -m src.models.run_survey`, `python -m src.build_notebook` e executar o notebook. Ele lê só `data/processed/` e "
        "`data/results/` e não acessa a rede.")
''')

# Referências: texto estático vindo de docs/referencias.md (inserido na construção do notebook).
ref = (ROOT / "docs" / "referencias.md").read_text(encoding="utf-8")
ref = ref.replace("# Referências", "### B. Referências", 1).replace("\n## ", "\n#### ")
md(ref)


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, SAIDA)
    print(f"{SAIDA} ({len(cells)} células)")


if __name__ == "__main__":
    main()
