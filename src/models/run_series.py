"""Roda os Atos 1 (Stack Overflow) e 2 (vagas Indeed) e grava data/results/.

Uso: python -m src.models.run_series

Regras de veredito (design §7.1), fixadas antes de ler os resultados:
- Ato 1: "efeito sustentado" se o real ficar abaixo do IC 95% do contrafactual por ≥ 6
  meses seguidos após nov/2022 E nenhum placebo produzir efeito de magnitude ≥ metade do
  efeito real E o backtest do pré-período bater o baseline sazonal ingênuo (MAPE menor).
  Senão "inconclusivo".
  Leitura usada para "efeito real" na comparação com placebos (que avaliam só 12 meses):
  o efeito médio dos primeiros 12 meses pós-corte (mesmo horizonte; fixada antes de rodar e
  confirmada pelo controlador). A comparação com o período todo vai no JSON como leitura
  secundária, não usada no veredito.
  Backtest: MAPE médio do modelo < MAPE médio do ingênuo nas origens.
- Ato 2: "consistente com IA" só se software cair mais que a mediana dos setores depois de
  controlar FEDFUNDS (coeficiente da dummy pós-ChatGPT < 0) e p < 0,05 (HAC). Senão
  "consistente com ciclo". Especificação principal decidida antes: log(software/mediana)
  ~ FEDFUNDS + pós; as demais (diferença, vs. total, capacidade) são só robustez.
  Correção metodológica (revisão estatística, decisão do controlador): os resíduos do OLS
  têm DW ≈ 0,2 (autocorrelação lag-1 ≈ 0,9) e HAC com n≈80 não torna o p-valor válido;
  o teste principal passa a ser a regressão com erros AR(1) (SARIMAX (1,0,0) + exog).
  A correção vai CONTRA a tese; o resultado HAC original fica em vagas.json.

Escolhas do Ato 1 (decisão do controlador):
- Série principal = total mensal de perguntas; janela de treino 2014-01 → CORTE_PRE_IA
  (a fase de crescimento 2008–2013 distorceria a tendência); sensibilidade com 2012-01.
- O pré-período inclui o pico da COVID (2020); mantido e tratado como limitação.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import CORTE_PRE_IA, DATA_CHATGPT, PROCESSED, RESULTS
from src.models.series import (_ols_hac, backtest, contrafactual, efeito, its_segmentado, placebos,
                               vagas_vs_setores)

INICIO_TOTAL = pd.Timestamp("2014-01-01")
INICIO_SENSIBILIDADE = pd.Timestamp("2012-01-01")
INICIO_TAGS = pd.Timestamp("2015-01-01")
# primeiro mês inteiro após o ChatGPT (30/11/2022) → 2022-12-01
INTERVENCAO = (DATA_CHATGPT.to_period("M") + 1).to_timestamp()
DATAS_PLACEBO = ["2019-06-01", "2020-06-01"]
LIMITACOES = [
    "O Stack Overflow mede demanda por perguntas no site, não o 'primeiro degrau' da carreira; a queda também "
    "tem causas fora do mercado de trabalho (chatbots substituindo o Q&A, política de moderação, perguntas "
    "apagadas que não aparecem na API).",
    "O pré-período inclui o pico da COVID (2020): ele gera o placebo de 2020-06 (reversão do pico) e a pior "
    "origem do backtest.",
    "A ordem do SARIMAX é escolhida por AIC entre modelos com e sem diferença sazonal (D=0 vs. D=1); AICs com "
    "diferenciações diferentes não são estritamente comparáveis.",
    "Vagas (Indeed) só existem desde 2020-02: o pré-período tem 34 meses e é dominado pelo boom de software de "
    "2021–22; FEDFUNDS e a dummy pós-ChatGPT são colineares (correlação ≈ 0,93) e um único juro é controle "
    "grosseiro do ciclo.",
    "Ato 2: os resíduos do OLS são fortemente autocorrelacionados (Durbin-Watson ≈ 0,2; lag-1 ≈ 0,9). Erros HAC "
    "com ~80 meses não corrigem isso; o teste usado no veredito é a regressão com erros AR(1). As regressões de "
    "robustez (diferença, vs. total, capacidade) e as checagens pós-hoc seguem em OLS+HAC e têm o mesmo problema.",
    "Escolha de método: na variante sazonal sem diferença sazonal (1,0,0,12) o SARIMAX leva deriva (trend='t', "
    "crescimento % constante com d=1); sem ela a previsão ficaria plana numa série que já caía, inflando o efeito.",
    "`previsto` no contrafactual é exp da média em log, ou seja, a mediana da previsão em nível (não a média); os "
    "limites lo95/hi95 são exp dos limites em log.",
    "ITS: o termo de mudança de nível (+27%, não significativo) é um desajuste de forma funcional — a queda "
    "acelera em termos de log, e a mudança de inclinação absorve a quebra; não ler o nível como 'alta'.",
]


def _serie(so: pd.DataFrame, tag: str) -> pd.Series:
    s = so[so.tag == tag].set_index("mes")["perguntas"].astype(float)
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def _limpo(d):
    if isinstance(d, dict):
        return {k: _limpo(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_limpo(v) for v in d]
    if isinstance(d, (np.floating, float)):
        return None if not np.isfinite(d) else round(float(d), 6)
    if isinstance(d, np.integer):
        return int(d)
    if isinstance(d, np.bool_):
        return bool(d)
    return d


def ato1(so: pd.DataFrame) -> dict:
    y = _serie(so, "__total__")
    corte = CORTE_PRE_IA

    cf = contrafactual(y, corte, inicio=INICIO_TOTAL)
    assert cf.index.min() > corte
    cf.reset_index().to_csv(RESULTS / "so_contrafactual.csv", index=False)
    e = efeito(cf)
    e12 = efeito(cf.iloc[:12])

    cf_s = contrafactual(y, corte, inicio=INICIO_SENSIBILIDADE)
    e_s = efeito(cf_s)

    bt = backtest(y, corte, horizontes=12, origens=6, inicio=INICIO_TOTAL)
    bt.to_csv(RESULTS / "so_backtest.csv", index=False)

    pl = placebos(y, DATAS_PLACEBO, meses=12, inicio=INICIO_TOTAL)
    pl.to_csv(RESULTS / "so_placebos.csv", index=False)

    its = its_segmentado(y, INTERVENCAO, inicio=INICIO_TOTAL)
    its["sensibilidade_inicio_2012"] = its_segmentado(y, INTERVENCAO, inicio=INICIO_SENSIBILIDADE)
    its["nota_nivel"] = ("A mudança de nível é desajuste de forma funcional: a queda acelera em log, então a "
                         "inclinação pós absorve a quebra e o nível fica positivo/não significativo; não ler como alta.")
    (RESULTS / "so_its.json").write_text(json.dumps(_limpo(its), ensure_ascii=False, indent=2),
                                         encoding="utf-8")

    tags = []
    for tag in sorted(t for t in so.tag.unique() if t != "__total__"):
        cft = contrafactual(_serie(so, tag), corte, inicio=INICIO_TAGS)
        et = efeito(cft)
        tags.append({"tag": tag, **{k: v for k, v in et.items() if k != "efeito_pct_faixa_ic"},
                     "efeito_pct_lo": et["efeito_pct_faixa_ic"][0], "efeito_pct_hi": et["efeito_pct_faixa_ic"][1],
                     "efeito_pct_12m": efeito(cft.iloc[:12])["efeito_pct_medio"],
                     "modelo": str(cft.attrs["modelo"]["order"]) + str(cft.attrs["modelo"]["seasonal_order"]),
                     "aic": cft.attrs["modelo"]["aic"]})
    pd.DataFrame(tags).to_csv(RESULTS / "so_tags_efeito.csv", index=False)

    # ---- veredito §7.1 (literal)
    mape_m, mape_i = float(bt.mape_modelo.mean()), float(bt.mape_ingenuo.mean())
    max_placebo = float(pl.efeito_pct_medio.abs().max())
    c_sust = e["sustentado"] and e["efeito_pct_medio"] < 0
    c_plac = max_placebo < 0.5 * abs(e12["efeito_pct_medio"])  # mesmo horizonte (12 meses) dos placebos
    c_bt = mape_m < mape_i
    ok = bool(c_sust and c_plac and c_bt)
    veredito = "efeito sustentado" if ok else "inconclusivo"
    if ok:
        expl = (f"As perguntas ficaram abaixo do IC 95% do contrafactual por {e['meses_abaixo_ic_seguidos']} meses "
                f"seguidos (efeito médio {e['efeito_pct_medio']:.0f}%); placebos e backtest passaram. "
                "Isso é consistente com uma queda adicional à tendência pré-ChatGPT, não prova causa.")
    else:
        falhas = [n for n, c in [("sequência abaixo do IC", c_sust), ("placebos", c_plac),
                                 ("backtest vs. ingênuo", c_bt)] if not c]
        expl = f"Inconclusivo: falhou(aram) {', '.join(falhas)} (regra §7.1)."
    leitura = (
        f"Nos primeiros 12 meses a queda ({e12['efeito_pct_medio']:.0f}%) não se distingue do placebo de "
        f"reversão da COVID (corte falso 2020-06: {pl.efeito_pct_medio.iloc[pl.corte_falso.tolist().index('2020-06-01')]:.0f}%). "
        f"Em 48 meses a divergência ({e['efeito_pct_medio']:.0f}% em média, {e['meses_abaixo_ic_seguidos']} meses "
        f"seguidos abaixo do IC 95%, {e['efeito_pct_ultimo_mes']:.0f}% no último mês) não tem análogo placebo de "
        "duração parecida, mas a regra de 12 meses não testa isso.")
    secundaria = {
        "rotulo": "secundária, não usada no veredito",
        "descricao": "placebo comparado ao efeito médio do período pós-corte inteiro (48 meses)",
        "efeito_real_periodo_todo_pct": e["efeito_pct_medio"],
        "metade_efeito_real_pct": 0.5 * abs(e["efeito_pct_medio"]),
        "max_abs_placebo_pct": max_placebo,
        "passaria": bool(max_placebo < 0.5 * abs(e["efeito_pct_medio"])),
    }
    principal = {
        "descricao": "placebo (12 meses) comparado ao efeito real dos primeiros 12 meses — leitura usada no veredito",
        "efeito_real_12m_pct": e12["efeito_pct_medio"], "metade_efeito_real_pct": 0.5 * abs(e12["efeito_pct_medio"]),
        "max_abs_placebo_pct": max_placebo, "passa": bool(max_placebo < 0.5 * abs(e12["efeito_pct_medio"])),
    }
    return {
        "veredito": veredito, "explicacao": expl, "leitura": leitura,
        "comparacao_placebo_principal": principal, "comparacao_placebo_secundaria": secundaria,
        "criterios": {"sustentado_6m": bool(c_sust), "placebos_menores_que_metade": bool(c_plac),
                      "backtest_bate_ingenuo": bool(c_bt)},
        "modelo": cf.attrs["modelo"], "efeito": e, "efeito_12m": e12,
        "sensibilidade_inicio_2012": {"modelo": cf_s.attrs["modelo"], "efeito": e_s},
        "backtest": {"mape_modelo_medio": mape_m, "mape_ingenuo_medio": mape_i,
                     "origens_modelo_melhor": int((bt.mape_modelo < bt.mape_ingenuo).sum()),
                     "n_origens": int(len(bt))},
        "placebos": {"max_abs_efeito_pct": max_placebo,
                     "por_data": pl[["corte_falso", "efeito_pct_medio", "meses_abaixo_ic_seguidos"]].to_dict("records")},
        "its": {"nivel_pos_pct": its["nivel_pos"]["efeito_pct"], "nivel_pos_p": its["nivel_pos"]["p"],
                "inclinacao_pos_pct_mes": its["inclinacao_pos"]["efeito_pct"],
                "inclinacao_pos_p": its["inclinacao_pos"]["p"]},
    }


def _pos_hoc(serie: pd.DataFrame) -> dict:
    """Checagens feitas DEPOIS de ver o resultado principal; não entram no veredito."""
    d = serie.set_index("mes")
    y = np.log(d["razao"])
    out = {"rotulo": "pós-hoc (feito depois de ver o resultado), não usado no veredito",
           "alvo": "log(software / mediana dos outros setores)", "placebos_dummy": []}
    for dt in ["2021-06-01", "2022-03-01", "2022-06-01"]:
        X = pd.DataFrame({"FEDFUNDS": d["FEDFUNDS"], "pos": (d.index >= dt).astype(float)})
        r = _ols_hac(y, X, "pos")
        out["placebos_dummy"].append({"dummy_desde": dt, "coef": r["coef_termo"], "p": r["p_termo"],
                                      "efeito_pct": r["coeficientes"]["pos"]["efeito_pct"]})
    X = pd.DataFrame({"FEDFUNDS": d["FEDFUNDS"], "pos": (d.index >= INTERVENCAO).astype(float),
                      "tendencia": np.arange(len(d), dtype=float)})
    out["com_tendencia_linear"] = _ols_hac(y, X, "pos")
    return out


def ato2(indeed: pd.DataFrame, fred: pd.DataFrame, cap: pd.DataFrame) -> dict:
    out = vagas_vs_setores(indeed, fred, cap, intervencao=INTERVENCAO)
    serie = out.pop("serie")
    serie.to_csv(RESULTS / "vagas_serie.csv", index=False)
    out["sensibilidade_pos_hoc"] = _pos_hoc(serie)
    (RESULTS / "vagas.json").write_text(json.dumps(_limpo(out), ensure_ascii=False, indent=2), encoding="utf-8")
    p = out["modelo_principal"]
    h = out["modelo_hac_original"]
    coef, pval = p["coef_termo"], p["p_termo"]
    ok = coef < 0 and pval < 0.05
    veredito = "consistente com IA" if ok else "consistente com ciclo"
    pct = (np.exp(coef) - 1) * 100
    if ok:
        expl = ("consistente com IA pela regra (erros AR(1)), mas sem separar o fim do boom de 2021–22")
    else:
        expl = ("Software caiu mais que os outros setores depois de 2022, mas quando se modela a inércia da série "
                "e os juros, a diferença não se separa do fim do boom de 2021–22; os dados não permitem atribuir à IA.")
    return {"veredito": veredito, "explicacao": expl,
            "numeros": {"teste": "regressão com erros AR(1), alvo log(software/mediana) ~ FEDFUNDS + pós",
                        "coef_pos_log": coef, "ic95_pos_log": p["ic95_termo"], "efeito_pos_pct": float(pct),
                        "p": pval, "ar1": p["ar1"],
                        "coef_fedfunds": p["coeficientes"]["FEDFUNDS"]["coef"],
                        "p_fedfunds": p["coeficientes"]["FEDFUNDS"]["p"], "n_meses": p["n"],
                        "corr_fedfunds_pos": out["corr_fedfunds_pos"],
                        "hac_original_substituido": {"coef_pos_log": h["coef_termo"], "p_hac": h["p_termo"],
                                                     "durbin_watson": h["durbin_watson"],
                                                     "autocorr_lag1": h["autocorr_lag1"],
                                                     "nota": h["nota"]},
                        "robustez_hac_p": {k: out[k]["p_termo"] for k in ("diferenca", "vs_total", "capacidade") if k in out},
                        "robustez_hac_coef": {k: out[k]["coef_termo"] for k in ("diferenca", "vs_total", "capacidade") if k in out},
                        "nota_robustez": "robustez estimada com OLS+HAC, sujeita ao mesmo problema de autocorrelação"}}


def main() -> dict:
    so = pd.read_csv(PROCESSED / "so_mensal.csv", parse_dates=["mes"])
    indeed = pd.read_csv(PROCESSED / "indeed_mensal.csv", parse_dates=["mes"])
    fred = pd.read_csv(PROCESSED / "fred_mensal.csv", parse_dates=["mes"])
    cap = pd.read_csv(PROCESSED / "capacidade_mensal.csv", parse_dates=["mes"])
    vereditos = {"ato1_stackoverflow": ato1(so), "ato2_vagas": ato2(indeed, fred, cap),
                 "limitacoes": LIMITACOES,
                 "regras": "design.md §7.1, aplicadas literalmente (ver docstring de src/models/run_series.py)"}
    (RESULTS / "vereditos_series.json").write_text(json.dumps(_limpo(vereditos), ensure_ascii=False, indent=2),
                                                   encoding="utf-8")
    return vereditos


if __name__ == "__main__":
    v = main()
    for k in ("ato1_stackoverflow", "ato2_vagas"):
        print(k, "->", v[k]["veredito"], "|", v[k]["explicacao"])
