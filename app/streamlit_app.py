"""Painel interativo de "O Primeiro Degrau".

Rode na raiz do projeto:  streamlit run app/streamlit_app.py
Lê só data/processed e data/results (sem internet, sem dados brutos).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.break_even import break_even  # noqa: E402

PROC, RES = ROOT / "data" / "processed", ROOT / "data" / "results"
AZUL, LARANJA, CINZA, TEXTO2 = "#2a78d6", "#eb6834", "#b9b8b2", "#52514e"
CORES_ONDA = {"chat": "#8a8984", "assistente": "#6250d6", "agente": "#e34948"}
REPO = "https://github.com/VictorKunzz/primeiro-degrau-ia-mercado-dev"

st.set_page_config(page_title="O Primeiro Degrau", page_icon=":material/stairs:", layout="wide")


def brl(v: float, casas: int = 0) -> str:
    s = f"{v:,.{casas}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def nota(texto: str) -> None:
    """Destaque neutro: caixa com borda fina, sem cor de status."""
    with st.container(border=True):
        st.markdown(texto)


def pct(v: float, casas: int = 1, sinal: bool = False) -> str:
    return (f"{v * 100:+.{casas}f}" if sinal else f"{v * 100:.{casas}f}").replace(".", ",") + "%"


@st.cache_data
def carregar():
    so = pd.read_csv(PROC / "so_mensal.csv", parse_dates=["mes"])
    cf = pd.read_csv(RES / "so_contrafactual.csv", parse_dates=["mes"])
    tags = pd.read_csv(RES / "so_tags_efeito.csv")
    marcos = pd.read_csv(PROC / "marcos.csv", parse_dates=["data"])
    emp = pd.read_csv(RES / "survey_emprego_iniciantes.csv")
    gap = pd.read_csv(RES / "survey_gap_salarial.csv")
    be = pd.read_csv(RES / "break_even.csv")
    vagas = pd.read_csv(RES / "vagas_serie.csv", parse_dates=["mes"])
    vs = json.loads((RES / "vereditos_series.json").read_text(encoding="utf-8"))
    vv = json.loads((RES / "vereditos_survey.json").read_text(encoding="utf-8"))
    return so, cf, tags, marcos, emp, gap, be, vagas, vs, vv


so, cf, tags, marcos, emp, gap, be, vagas, vs, vv = carregar()
base_be = be[be["principal"]].iloc[0]

st.title("O Primeiro Degrau")
st.caption("A IA generativa está tirando o primeiro degrau da carreira em software? "
           f"Painel interativo do trabalho de Data Storytelling · [repositório]({REPO})")

aba_gestor, aba_so, aba_degrau, aba_vagas = st.tabs(
    [":material/calculate: Conta do gestor", ":material/forum: Stack Overflow", ":material/school: Primeiro degrau", ":material/work: Vagas e juros"])

# ---------------------------------------------------------------- gestor
with aba_gestor:
    st.subheader("Sênior + IA sai mais barato que sênior + júnior?")
    st.markdown("Mexa nas premissas. A conta compara o **custo por unidade de trabalho** das duas equipes "
                "(1 unidade = o que um sênior sem IA produz em um ano). É uma simulação de cenários, "
                "não uma medição.")
    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        sal_jr = st.slider("Salário anual do júnior (US$)", 30_000, 150_000, int(base_be.sal_jr), 2_500,
                           help=f"Padrão: mediana EUA {int(base_be.ano_salarios)}, até 2 anos de experiência (Stack Overflow Survey).")
        sal_sr = st.slider("Salário anual do sênior (US$)", 80_000, 300_000, int(base_be.sal_sr), 2_500,
                           help=f"Padrão: mediana EUA {int(base_be.ano_salarios)}, 10+ anos de experiência.")
        custo_mes = st.slider("Ferramenta de IA por dev (US$/mês)", 0, 1_000, 200, 20,
                              help="Planos de agente custam de US$ 100 a 200/mês (ver referências).")
        prod_jr = st.slider("Quanto o júnior produz, em relação a um sênior", 0.1, 1.0, 0.5, 0.05,
                            format="%.2f", help="0,5 = metade de um sênior. Premissa, não medida.")
        ganho = st.slider("Ganho de produtividade do sênior com IA", -0.30, 0.60, 0.15, 0.01, format="%.2f",
                          help="Estudos: +26% (RCTs do Copilot) a −19% (RCT do METR, 2025).")
    custo_ia_ano = custo_mes * 12.0
    u_star = break_even(sal_jr, sal_sr, custo_ia_ano, prod_jr)
    custo_a = (sal_sr + sal_jr) / (1 + prod_jr)
    custo_b = (sal_sr + custo_ia_ano) / (1 + ganho) if ganho > -1 else float("inf")
    compensa = ganho >= u_star
    with c2:
        m1, m2, m3 = st.columns(3)
        m1.metric("Sênior + júnior", f"US$ {brl(custo_a)}", help="custo por unidade de trabalho")
        m2.metric("Sênior + IA", f"US$ {brl(custo_b)}", delta=f"{pct(custo_b / custo_a - 1, sinal=True)} vs. com júnior",
                  delta_color="off")
        m3.metric("Ganho que empata", pct(u_star, sinal=True), help="Abaixo disso, manter o júnior sai mais barato.")
        if compensa:
            nota(f"Com +{pct(ganho)} de ganho, **sênior + IA compensa**: o empate está em {pct(u_star, sinal=True)}.")
        else:
            nota(f"Com {pct(ganho, sinal=True)} de ganho, **manter o júnior ainda compensa**: "
                       f"a IA precisaria render {pct(u_star, sinal=True)}.")
        xs = pd.DataFrame({"ganho": [x / 100 for x in range(-30, 61)]})
        xs["Sênior + IA"] = (sal_sr + custo_ia_ano) / (1 + xs["ganho"])
        xs["Sênior + júnior"] = custo_a
        longo = xs.melt("ganho", var_name="equipe", value_name="custo")
        linhas = alt.Chart(longo).mark_line(strokeWidth=2.5).encode(
            x=alt.X("ganho:Q", title="ganho de produtividade do sênior com IA", axis=alt.Axis(format="+%")),
            y=alt.Y("custo:Q", title="custo por unidade (US$)", scale=alt.Scale(zero=False)),
            color=alt.Color("equipe:N", scale=alt.Scale(domain=["Sênior + IA", "Sênior + júnior"], range=[AZUL, CINZA]),
                            legend=alt.Legend(orient="top", title=None, labelLimit=0)),
            tooltip=[alt.Tooltip("ganho:Q", format="+.0%"), "equipe:N", alt.Tooltip("custo:Q", format=",.0f")])
        regra = alt.Chart(pd.DataFrame({"x": [u_star]})).mark_rule(color=LARANJA, strokeDash=[4, 3]).encode(x="x:Q")
        ponto = alt.Chart(pd.DataFrame({"ganho": [ganho], "custo": [custo_b]})).mark_point(
            size=120, filled=True, color=AZUL).encode(x="ganho:Q", y="custo:Q")
        cen = pd.DataFrame({"ganho": [-0.19, 0.0, 0.15, 0.30],
                            "nome": ["estresse (METR)", "pessimista", "central", "otimista"]})
        rot = alt.Chart(cen).mark_text(angle=270, dx=-60, color=TEXTO2, fontSize=11).encode(x="ganho:Q", text="nome:N")
        tick = alt.Chart(cen).mark_rule(color="#e6e5e0").encode(x="ganho:Q")
        st.altair_chart((tick + rot + linhas + regra + ponto).properties(height=340), use_container_width=True)
        st.caption("Linha laranja = ganho de empate. Fora da conta: encargos, tempo de mentoria, rotatividade "
                   "e o fato de que o júnior de hoje é quem vira sênior.")

# ---------------------------------------------------------------- stack overflow
with aba_so:
    st.subheader("Onde o iniciante aprendia esvaziou")
    nomes = {"__total__": "Total", "python": "Python", "javascript": "JavaScript", "java": "Java", "reactjs": "React"}
    c1, c2, c3 = st.columns([2, 1, 1])
    escolha = c1.multiselect("Séries", list(nomes), default=["__total__"], format_func=nomes.get)
    log = c2.toggle("Escala logarítmica", value=False)
    mostrar_cf = c3.toggle("Projeção pela tendência anterior (só Total)", value=True)
    ondas = st.multiselect("Marcos da IA", ["chat", "assistente", "agente"], default=["chat", "agente"])
    st.caption("Linhas tracejadas: cinza = chat · roxo = assistente na IDE · vermelho = agente. Passe o mouse para ver o lançamento.")
    if escolha:
        d = so[so.tag.isin(escolha) & (so.mes >= "2014-01-01")].assign(serie=lambda x: x.tag.map(nomes))
        escala = alt.Scale(type="log") if log else alt.Scale(zero=True)
        cores = alt.Scale(domain=[nomes[t] for t in nomes], range=[AZUL, "#1baf7a", "#eda100", "#6250d6", "#e87ba4"])
        base = alt.Chart(d).mark_line(strokeWidth=2).encode(
            x=alt.X("mes:T", title=None, axis=alt.Axis(format="%Y", tickCount="year")), y=alt.Y("perguntas:Q", title="perguntas por mês", scale=escala),
            color=alt.Color("serie:N", scale=cores, legend=alt.Legend(orient="top", title=None, labelLimit=0)),
            tooltip=[alt.Tooltip("mes:T", format="%b/%Y"), "serie:N", alt.Tooltip("perguntas:Q", format=",")])
        camadas = [base]
        if mostrar_cf and "__total__" in escolha:
            banda = alt.Chart(cf).mark_area(color=LARANJA, opacity=0.15).encode(x="mes:T", y="lo95:Q", y2="hi95:Q")
            proj = alt.Chart(cf).mark_line(color=LARANJA, strokeDash=[6, 4], strokeWidth=2).encode(
                x="mes:T", y="previsto:Q", tooltip=[alt.Tooltip("mes:T", format="%b/%Y"),
                                                     alt.Tooltip("previsto:Q", title="projeção", format=",.0f")])
            camadas = [banda, proj, base]
        for onda in ondas:
            m = marcos[marcos.onda == onda]
            if len(m):
                camadas.append(alt.Chart(m).mark_rule(strokeDash=[3, 3], color=CORES_ONDA[onda], strokeWidth=1.5).encode(
                    x="data:T", tooltip=["marco:N", "onda:N", alt.Tooltip("data:T", format="%d/%m/%Y")]))
        st.altair_chart(alt.layer(*camadas).properties(height=380), use_container_width=True)
    a1 = vs["ato1_stackoverflow"]
    nota(f"**Veredito pela regra pré-definida: {a1['veredito']}.** {a1.get('leitura', '')}")
    t = tags.assign(serie=tags.tag.map(nomes))[["serie", "efeito_pct_12m", "efeito_pct_medio", "efeito_pct_ultimo_mes"]]
    t.columns = ["Linguagem", "Efeito 1º ano (%)", "Efeito médio 48 meses (%)", "Último mês (%)"]
    st.dataframe(t.round(1), hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- primeiro degrau
with aba_degrau:
    st.subheader("A fatia empregada de quem está começando caiu muito mais")
    metodos = {"bruto": "Bruto", "ipw": "Reponderado (corrige quem não respondeu)", "workexp_igual_2": "Só quem tem 2 anos"}
    met = st.radio("Método", list(metodos), format_func=metodos.get, horizontal=True,
                   help="Os três métodos existem porque em 2026 mais estudantes passaram a responder a pergunta de experiência.")
    e = emp[(emp.metodo == met) & emp.ano.isin([2025, 2026])].copy()
    e["grupo"] = e.faixa_exp.map({"≤2": "até 2 anos", "≥10": "10 anos ou mais"})
    e["ano"] = e.ano.astype(str)
    barras = alt.Chart(e).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("ano:N", title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("grupo:N", sort=["até 2 anos", "10 anos ou mais"]),
        y=alt.Y("pct_empregado:Q", title="fatia empregada", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("grupo:N", scale=alt.Scale(domain=["até 2 anos", "10 anos ou mais"], range=[AZUL, CINZA]),
                        legend=alt.Legend(orient="top", title=None, labelLimit=0)),
        tooltip=["grupo:N", "ano:N", alt.Tooltip("pct_empregado:Q", format=".1%"), "n:Q"])
    erro = alt.Chart(e).mark_errorbar(color="#0b0b0b").encode(
        x="ano:N", xOffset=alt.XOffset("grupo:N", sort=["até 2 anos", "10 anos ou mais"]),
        y=alt.Y("ic_inf:Q", title="fatia empregada"), y2="ic_sup:Q")
    c1, c2 = st.columns([3, 2], gap="large")
    c1.altair_chart((barras + erro).properties(height=340), use_container_width=True)
    jr = e[e.grupo == "até 2 anos"].set_index("ano").pct_empregado
    sr = e[e.grupo == "10 anos ou mais"].set_index("ano").pct_empregado
    c2.metric("Até 2 anos, 2025 → 2026", pct(jr["2026"]), delta=f"{(jr['2026'] - jr['2025']) * 100:+.1f} p.p.".replace(".", ","), delta_color="off")
    c2.metric("10 anos ou mais, 2025 → 2026", pct(sr["2026"]), delta=f"{(sr['2026'] - sr['2025']) * 100:+.1f} p.p.".replace(".", ","), delta_color="off")
    c2.caption("Pessoas diferentes em cada ano (pesquisa voluntária): é associação, não acompanhamento de indivíduos. "
               "2023 e 2024 ficam de fora por viés de seleção na pergunta de experiência.")
    st.markdown("#### E quem fica, ganha cada vez mais que o iniciante")
    g = gap.copy()
    g["ano"] = g.ano.astype(str)
    linha = alt.Chart(g).mark_line(color=AZUL, strokeWidth=2.5, point=alt.OverlayMarkDef(size=80, filled=True)).encode(
        x=alt.X("ano:N", title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("razao_sr_jr:Q", title="salário sênior ÷ júnior (EUA)", scale=alt.Scale(zero=False)),
        tooltip=["ano:N", alt.Tooltip("razao_sr_jr:Q", format=".2f"), alt.Tooltip("mediana_jr:Q", format=",.0f"),
                 alt.Tooltip("mediana_sr:Q", format=",.0f")])
    faixa = alt.Chart(g).mark_errorbar(color=AZUL).encode(x="ano:N", y=alt.Y("razao_ic_inf:Q", title="salário sênior ÷ júnior (EUA)"), y2="razao_ic_sup:Q")
    st.altair_chart((faixa + linha).properties(height=260), use_container_width=True)

# ---------------------------------------------------------------- vagas
with aba_vagas:
    st.subheader("Vagas: software caiu mais, mas os juros explicam boa parte")
    v = vagas.melt("mes", ["software", "mediana_setores"], var_name="serie", value_name="indice")
    v["serie"] = v.serie.map({"software": "Desenvolvimento de software", "mediana_setores": "Mediana de 46 outros setores"})
    ch = alt.Chart(v).mark_line(strokeWidth=2.5).encode(
        x=alt.X("mes:T", title=None, axis=alt.Axis(format="%Y", tickCount="year")), y=alt.Y("indice:Q", title="índice de vagas (fev/2020 = 100)"),
        color=alt.Color("serie:N", scale=alt.Scale(range=[AZUL, CINZA]), legend=alt.Legend(orient="top", title=None, labelLimit=0)),
        tooltip=[alt.Tooltip("mes:T", format="%b/%Y"), "serie:N", alt.Tooltip("indice:Q", format=".1f")])
    juros = alt.Chart(vagas).mark_line(color="#52514e", interpolate="step-after").encode(
        x=alt.X("mes:T", title=None, axis=alt.Axis(format="%Y", tickCount="year")), y=alt.Y("FEDFUNDS:Q", title="juros EUA (% a.a.)"),
        tooltip=[alt.Tooltip("mes:T", format="%b/%Y"), "FEDFUNDS:Q"])
    st.altair_chart(alt.vconcat(ch.properties(height=300), juros.properties(height=120)).resolve_scale(x="shared"),
                    use_container_width=True)
    a2 = vs["ato2_vagas"]
    nota(f"**Veredito: {a2['veredito']}.** {a2.get('explicacao', '')}")

st.divider()
st.caption("Fontes: Stack Exchange API · Stack Overflow Developer Survey 2023–2026 · Indeed Hiring Lab · "
           "Federal Reserve H.15 (via DBnomics) · Epoch AI / METR. Análise completa no notebook do repositório.")
