"""Roda as análises do survey (Ato 3: H1-parte survey, H2, H3) e grava data/results/.

Uso: python -m src.models.run_survey

Regras de veredito (design §7.1), fixadas aqui antes de ler os resultados finais:
- Diferença entre anos/grupos só conta com IC 95% bootstrap que não cruza zero.
- Veredito: confirmada (todas as evidências próprias passam), parcial (algumas),
  não sustentada (nenhuma).
- Nunca comparamos uma coluna entre anos fora de `COMPARAVEL` (src/etl/survey.py).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import PROCESSED, RESULTS, SEED
from src.etl.survey import COMPARAVEL
from src.models.break_even import break_even, cenarios
from src.models.survey_models import (bootstrap_amostras, bootstrap_ic, classificador_ameaca,
                                      ic_de, preparar_X)

N_BOOT = 2000
CUSTO_IA_ANO = 2400.0  # US$ 200/mês (fatos_literatura.csv, preço verificado)
MIN_N = 30
JR, SR = "≤2", "≥10"

# Anos em que `anos_exp` (WorkExp) foi respondido também por quem não trabalha.
# Em 2023/2024 WorkExp está preenchido para ~60% dos empregados e <10% dos não
# empregados -> % empregado entre ≤2 anos fica ~97% por seleção, não por mercado.
ANOS_EMPREGO_VALIDOS = [2025, 2026]
NOTA_EMPREGO = {
    2024: ("Fora da inferência: em 2024 WorkExp foi respondido quase só por quem trabalha "
           "(58% dos empregados vs 9% dos não empregados), então a faixa ≤2 é selecionada "
           "por emprego. 2023 excluído pelo mesmo motivo."),
    2025: "Válido: WorkExp respondido por 73% dos não empregados e 95% dos empregados.",
    2026: ("Válido: WorkExp respondido por 70% dos não empregados e 85% dos empregados. "
           "Amostra 2026 tem mais estudantes entre ≤2 (composição muda)."),
}


def _ic(est, lo, hi):
    return {"estimativa": est, "ic_inf": lo, "ic_sup": hi, "cruza_zero": bool(lo <= 0 <= hi)}


def _diff(a, b, func=np.mean, seed=SEED):
    """IC bootstrap de func(a) − func(b), grupos independentes."""
    ba = bootstrap_amostras(a, func, N_BOOT, seed)
    bb = bootstrap_amostras(b, func, N_BOOT, seed + 1)
    return ba, bb, ic_de(ba - bb, func(np.asarray(a, float)) - func(np.asarray(b, float)))


METODOS_H1 = {
    "bruto": "% empregado entre quem respondeu WorkExp na faixa (sem correção).",
    "ipw": ("Reponderado por 1/P(responder WorkExp | ano, categoria de Employment): corrige a "
            "diferença de taxa de resposta (ex.: Employed 95,5%->84,9%, Student 41,3%->51,6%)."),
    "workexp_igual_2": ("Só WorkExp == 2 na faixa júnior (em 2026 as respostas se amontoam em 1 ano); "
                        "comparação com ≥10 bruto."),
}


def _boot_wmean(y, w, seed, n=N_BOOT):
    y, w = np.asarray(y, float), np.asarray(w, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), (n, len(y))) if len(y) * n <= 1e7 else None
    if idx is not None:
        return (y[idx] * w[idx]).sum(1) / w[idx].sum(1)
    out = np.empty(n)
    for b in range(n):
        k = rng.integers(0, len(y), len(y))
        out[b] = (y[k] * w[k]).sum() / w[k].sum()
    return out


def _pesos_ipw(df):
    """1/P(anos_exp não nulo | ano, emprego), estimado sobre todos os respondentes do ano."""
    grupo = df["emprego"].fillna("NA")
    p = df["anos_exp"].notna().groupby([df["ano"], grupo]).transform("mean")
    return 1.0 / p


def emprego_iniciantes(df):
    df = df.assign(_w=_pesos_ipw(df), _emp=df["emprego"].fillna("NA"))
    linhas, boots, est = [], {}, {}

    def sub(ano, faixa, metodo):
        d = df[(df.ano == ano) & (df.faixa_exp == faixa) & df.empregado.notna()]
        if metodo == "workexp_igual_2":
            d = d[d.anos_exp == 2] if faixa == JR else d
        w = d["_w"] if metodo == "ipw" else pd.Series(1.0, index=d.index)
        return d, w

    for ano in (2024, 2025, 2026):
        assert ano in COMPARAVEL["empregado"]
        metodos = ["bruto"] if ano not in ANOS_EMPREGO_VALIDOS else list(METODOS_H1)
        for metodo in metodos:
            for faixa in (JR, SR):
                d, w = sub(ano, faixa, metodo)
                y = d.empregado.astype(float)
                b = _boot_wmean(y, w, SEED + ano + (0 if faixa == JR else 7))
                e = float((y * w).sum() / w.sum())
                boots[(metodo, ano, faixa)], est[(metodo, ano, faixa)] = b, e
                pct_est = float(((d._emp == "Student") * w).sum() / w.sum()) if ano >= 2025 else np.nan
                linhas.append({"ano": ano, "faixa_exp": faixa, "metodo": metodo, "n": len(d),
                               "pct_empregado": e, "ic_inf": np.percentile(b, 2.5), "ic_sup": np.percentile(b, 97.5),
                               "pct_so_estudante": pct_est, "usar_na_inferencia": ano in ANOS_EMPREGO_VALIDOS,
                               "nota": NOTA_EMPREGO[ano] + " Método: " + METODOS_H1[metodo]})
    tab = pd.DataFrame(linhas)
    a0, a1 = ANOS_EMPREGO_VALIDOS
    res = {"anos": [a0, a1], "metodos": {}}
    for metodo in METODOS_H1:
        dj = boots[(metodo, a1, JR)] - boots[(metodo, a0, JR)]
        ds = boots[(metodo, a1, SR)] - boots[(metodo, a0, SR)]
        ej = est[(metodo, a1, JR)] - est[(metodo, a0, JR)]
        es = est[(metodo, a1, SR)] - est[(metodo, a0, SR)]
        res["metodos"][metodo] = {
            "descricao": METODOS_H1[metodo],
            "pct_jr": {str(a0): est[(metodo, a0, JR)], str(a1): est[(metodo, a1, JR)]},
            "variacao_jr": _ic(*ic_de(dj, ej)), "variacao_sr": _ic(*ic_de(ds, es)),
            "dif_em_dif_jr_menos_sr": _ic(*ic_de(dj - ds, ej - es)),
        }
    # Robustez extra (não entra no veredito): exclui quem se declara só "Student".
    sem_est = df[(df.faixa_exp == JR) & (df.emprego != "Student")]
    s0 = sem_est.loc[sem_est.ano == a0, "empregado"].dropna().astype(float)
    s1 = sem_est.loc[sem_est.ano == a1, "empregado"].dropna().astype(float)
    res["robustez_jr_sem_estudantes"] = {
        "pct_" + str(a0): float(s0.mean()), "pct_" + str(a1): float(s1.mean()),
        "n_" + str(a0): len(s0), "n_" + str(a1): len(s1),
        "variacao": _ic(*_diff(s1, s0)[2]),
    }
    return tab, res


def gap_salarial(df):
    us = df[df.pais.fillna("").str.contains("United States") & df.salario_usd.notna()]
    linhas, boots = [], {}
    for ano in sorted(us.ano.unique()):
        assert ano in COMPARAVEL["salario_usd"]
        jr = us.loc[(us.ano == ano) & (us.faixa_exp == JR), "salario_usd"]
        sr = us.loc[(us.ano == ano) & (us.faixa_exp == SR), "salario_usd"]
        if len(jr) < MIN_N or len(sr) < MIN_N:
            continue
        bj = bootstrap_amostras(jr, np.median, N_BOOT, SEED + ano)
        bs = bootstrap_amostras(sr, np.median, N_BOOT, SEED + ano + 1000)
        boots[ano] = (bj, bs)
        mj, ms = float(jr.median()), float(sr.median())
        linhas.append({
            "ano": int(ano), "n_jr": len(jr), "n_sr": len(sr),
            "mediana_jr": mj, "mediana_jr_ic_inf": np.percentile(bj, 2.5), "mediana_jr_ic_sup": np.percentile(bj, 97.5),
            "mediana_sr": ms, "mediana_sr_ic_inf": np.percentile(bs, 2.5), "mediana_sr_ic_sup": np.percentile(bs, 97.5),
            "razao_sr_jr": ms / mj, "razao_ic_inf": np.percentile(bs / bj, 2.5), "razao_ic_sup": np.percentile(bs / bj, 97.5),
        })
    tab = pd.DataFrame(linhas)
    a0, a1 = int(tab.ano.min()), int(tab.ano.max())
    r = lambda a: boots[a][1] / boots[a][0]
    est = tab.set_index("ano").loc[a1, "razao_sr_jr"] - tab.set_index("ano").loc[a0, "razao_sr_jr"]
    res = {"anos": [a0, a1], "variacao_razao": _ic(*ic_de(r(a1) - r(a0), est))}
    return tab, res, boots


def agentes_por_exp(df):
    # usa_agentes: COMPARAVEL vazio -> só comparações DENTRO do ano (≥10 vs ≤2).
    linhas, res = [], {}
    for ano in sorted(df.loc[df.usa_agentes.notna(), "ano"].unique()):
        d = df[df.ano == ano]
        for faixa in (JR, "3–9", SR):
            s = d.loc[d.faixa_exp == faixa, "usa_agentes"].dropna().astype(float)
            est, lo, hi = bootstrap_ic(s, np.mean, N_BOOT, SEED)
            linhas.append({"ano": int(ano), "faixa_exp": faixa, "n": len(s), "pct_usa_agentes": est,
                           "ic_inf": lo, "ic_sup": hi,
                           "nota": "Definição de usa_agentes muda entre 2025 e 2026: não comparar anos."})
        s_sr = d.loc[d.faixa_exp == SR, "usa_agentes"].dropna().astype(float)
        s_jr = d.loc[d.faixa_exp == JR, "usa_agentes"].dropna().astype(float)
        res[int(ano)] = _ic(*_diff(s_sr, s_jr)[2])
    return pd.DataFrame(linhas), res


def taxas_ameaca(df, ano):
    d = df[(df.ano == ano) & df.ia_ameaca.isin(["Yes", "No", "I'm not sure"])]
    y = d.ia_ameaca.eq("Yes").astype(float)
    out = {"por_faixa_exp": {}, "por_usa_agentes": {}}
    for f in (JR, "3–9", SR):
        out["por_faixa_exp"][f] = dict(zip(("estimativa", "ic_inf", "ic_sup"), bootstrap_ic(y[d.faixa_exp == f])))
    ag = d.usa_agentes.astype("object")
    for k, v in (("sim", True), ("nao", False)):
        out["por_usa_agentes"][k] = dict(zip(("estimativa", "ic_inf", "ic_sup"), bootstrap_ic(y[ag == v])))
    out["dif_sr_menos_jr"] = _ic(*_diff(y[d.faixa_exp == SR], y[d.faixa_exp == JR])[2])
    out["dif_agentes_sim_menos_nao"] = _ic(*_diff(y[ag == True], y[ag == False])[2])  # noqa: E712
    return out


def conta_gestor(gap_tab, boots):
    linhas, sens = [], {}
    ano_ult = int(gap_tab.ano.max())
    for ano in sorted({ano_ult, 2025}):
        row = gap_tab.set_index("ano").loc[ano]
        t = cenarios(row.mediana_jr, row.mediana_sr, CUSTO_IA_ANO)
        t.insert(0, "ano_salarios", ano)
        t["principal"] = ano == ano_ult
        linhas.append(t)
        bj, bs = boots[ano]
        sens[ano] = {}
        for p in (0.3, 0.5, 0.7):
            u = break_even(bj, bs, CUSTO_IA_ANO, p)
            est = break_even(row.mediana_jr, row.mediana_sr, CUSTO_IA_ANO, p)
            sens[ano][str(p)] = dict(zip(("estimativa", "ic_inf", "ic_sup"), ic_de(u, est)))
    return pd.concat(linhas, ignore_index=True), sens, ano_ult


def veredito(passa: list[bool]) -> str:
    if all(passa):
        return "confirmada"
    return "parcial" if any(passa) else "não sustentada"


def main():
    df = pd.read_parquet(PROCESSED / "survey_harmonizado.parquet")
    RESULTS.mkdir(parents=True, exist_ok=True)

    emp_tab, emp = emprego_iniciantes(df)
    emp_tab.to_csv(RESULTS / "survey_emprego_iniciantes.csv", index=False)

    gap_tab, gap, boots = gap_salarial(df)
    gap_tab.to_csv(RESULTS / "survey_gap_salarial.csv", index=False)

    ag_tab, ag = agentes_por_exp(df)
    ag_tab.to_csv(RESULTS / "survey_agentes_por_exp.csv", index=False)

    clf = classificador_ameaca(df)
    assert clf["ano"] in COMPARAVEL["cargo"]
    clf["taxas_descritivas"] = taxas_ameaca(df, clf["ano"])
    pd.DataFrame(clf["importancias_holdout"]).to_csv(RESULTS / "importancias.csv", index=False)
    (RESULTS / "classificador.json").write_text(json.dumps(clf, ensure_ascii=False, indent=2, default=float),
                                                encoding="utf-8")

    be_tab, be_sens, ano_be = conta_gestor(gap_tab, boots)
    be_tab.to_csv(RESULTS / "break_even.csv", index=False)

    # ---------- vereditos ----------
    # H1: confirmada só se, nos 3 métodos, a queda do júnior E a dif-em-dif tiverem IC < 0.
    h1_ev = {}
    for metodo, m in emp["metodos"].items():
        h1_ev[f"queda_jr_{metodo}"] = m["variacao_jr"]["ic_sup"] < 0
        h1_ev[f"dif_em_dif_{metodo}"] = m["dif_em_dif_jr_menos_sr"]["ic_sup"] < 0
    efeitos = [m["variacao_jr"]["estimativa"] for m in emp["metodos"].values()]
    h1_ok = [all(h1_ev[f"{k}_{metodo}"] for k in ("queda_jr", "dif_em_dif")) for metodo in emp["metodos"]]
    central = be_tab[be_tab.principal & (be_tab.cenario == "central")]
    h2_ev = {
        # u* (IC bootstrap dos salários) abaixo do ganho central de 15% com prod_jr = 0,5
        "central_compensa_com_ic": be_sens[ano_be]["0.5"]["ic_sup"] < 0.15,
        # robusto à premissa do júnior: compensa no central para prod_jr 0,3/0,5/0,7
        "central_compensa_em_toda_sensibilidade": bool(central.compensa.all()),
    }
    td = clf["taxas_descritivas"]
    h3_ev = {
        "gap_salarial_cresce": gap["variacao_razao"]["ic_inf"] > 0,
        "extra_fora_do_design_senior_usa_mais_agentes_em_cada_ano": all(v["ic_inf"] > 0 for v in ag.values()),
        "classificador_interpretavel_e_exp_agentes_reduzem_ameaca": bool(
            clf["interpretavel"] and td["dif_sr_menos_jr"]["ic_sup"] < 0
            and td["dif_agentes_sim_menos_nao"]["ic_sup"] < 0),
    }
    vereditos = {
        "regras": "design §7.1: IC 95% bootstrap não cruza zero; confirmada = todas passam, parcial = algumas, não sustentada = nenhuma.",
        "H1_survey": {
            "veredito": veredito(h1_ok), "evidencias": h1_ev,
            "regra": "confirmada se nos 3 métodos (bruto, ipw, workexp_igual_2) a queda do júnior e a dif-em-dif têm IC 95% < 0; parcial se em alguns.",
            "faixa_efeito_jr": {"min": min(efeitos), "max": max(efeitos)},
            "numeros": {"emprego": emp, "tabela": emp_tab.drop(columns="nota").to_dict(orient="records")},
            "explicacao": (f"Entre quem tem ≤2 anos, a fração empregada caiu de {emp['anos'][0]} para {emp['anos'][1]} "
                           f"entre {min(efeitos):+.1%} e {max(efeitos):+.1%} conforme o método (bruto, reponderado pela taxa de "
                           "resposta do WorkExp, só WorkExp = 2); como quem responde a experiência difere por situação de emprego "
                           "e a amostra de 2026 tem mais estudantes, há viés de seleção possível e o resultado é associação, não causa."),
        },
        "H2": {
            "veredito": veredito(list(h2_ev.values())), "evidencias": h2_ev,
            "numeros": {"ano_salarios": ano_be, "u_star_por_prod_jr": be_sens,
                        "custo_ia_ano": CUSTO_IA_ANO,
                        "cenarios_principais": central[["prod_jr", "u_star", "compensa"]].to_dict(orient="records")},
            "explicacao": (f"Com salários medianos dos EUA em {ano_be}, sênior+IA empata com sênior+júnior se a IA render "
                           f"+{be_sens[ano_be]['0.5']['estimativa']:.1%} (júnior = 50% de um sênior); o resultado depende "
                           "da premissa sobre o júnior e é uma conta de cenários, não observação de empresas."),
        },
        "H3": {
            "veredito": veredito(list(h3_ev.values())), "evidencias": h3_ev,
            "notas": [
                "O critério 'extra_fora_do_design_senior_usa_mais_agentes_em_cada_ano' NÃO está no design (§7); foi "
                "acrescentado nesta análise. Sem ele, o veredito continua 'parcial' (gap passa, classificador não).",
                "usa_agentes muda de definição: 2025 = pergunta AIAgents; 2026 = opção 'AI agents or automated workflows' "
                "dentro do AISelect multi-seleção. Por isso só se compara ≥10 vs ≤2 dentro de cada ano, nunca 2025 vs 2026.",
            ],
            "numeros": {"gap_salarial": gap, "agentes_sr_menos_jr_por_ano": ag,
                        "classificador": {k: clf[k] for k in ("ano", "n", "melhor_modelo", "delta_auc_melhor_vs_dummy",
                                                              "interpretavel", "auc_holdout")},
                        "auc_cv": {k: v["auc_media"] for k, v in clf["metricas"].items()},
                        "taxa_ameaca": td},
            "explicacao": (f"A razão salário sênior/júnior nos EUA variou {gap['variacao_razao']['estimativa']:+.2f} "
                           f"entre {gap['anos'][0]} e {gap['anos'][1]}; uso de agentes e percepção de ameaça por experiência "
                           "são comparados só dentro do mesmo ano."),
        },
    }
    (RESULTS / "vereditos_survey.json").write_text(
        json.dumps(vereditos, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
        encoding="utf-8")
    return vereditos


if __name__ == "__main__":
    v = main()
    for h in ("H1_survey", "H2", "H3"):
        print(h, v[h]["veredito"], v[h]["evidencias"])
        print("  ", v[h]["explicacao"])
