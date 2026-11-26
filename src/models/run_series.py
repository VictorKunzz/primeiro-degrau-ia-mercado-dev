"""Roda os Atos 1 (Stack Overflow) e 2 (vagas Indeed) e grava data/results/.

Uso: python -m src.models.run_series

Regras de veredito (design §7.1), fixadas antes de ler os resultados:
- Ato 1: "efeito sustentado" se o real ficar abaixo do IC 95% do contrafactual por ≥ 6
  meses seguidos após nov/2022 E nenhum placebo produzir efeito de magnitude ≥ metade do
  efeito real E o backtest do pré-período bater o baseline sazonal ingênuo (MAPE menor).
  Senão "inconclusivo".
  Leitura usada para "efeito real" na comparação com placebos (que avaliam só 12 meses):
  exigimos as duas comparações — contra o efeito médio dos primeiros 12 meses pós-corte
  (comparação de mesmo horizonte, a mais exigente) e contra o efeito médio do período todo.
  Backtest: MAPE médio do modelo < MAPE médio do ingênuo nas origens.
- Ato 2: "consistente com IA" só se software cair mais que a mediana dos setores depois de
  controlar FEDFUNDS (coeficiente da dummy pós-ChatGPT < 0) e p < 0,05 (HAC). Senão
  "consistente com ciclo". Especificação principal decidida antes: log(software/mediana)
  ~ FEDFUNDS + pós; as demais (diferença, vs. total, capacidade) são só robustez.

Escolhas do Ato 1 (decisão do controlador):
- Série principal = total mensal de perguntas; janela de treino 2014-01 → CORTE_PRE_IA
  (a fase de crescimento 2008–2013 distorceria a tendência); sensibilidade com 2012-01.
- O pré-período inclui o pico da COVID (2020); mantido e tratado como limitação.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import CORTE_PRE_IA, PROCESSED, RESULTS
from src.models.series import (backtest, contrafactual, efeito, its_segmentado, placebos,
                               vagas_vs_setores)

INICIO_TOTAL = pd.Timestamp("2014-01-01")
INICIO_SENSIBILIDADE = pd.Timestamp("2012-01-01")
INICIO_TAGS = pd.Timestamp("2015-01-01")
INTERVENCAO = pd.Timestamp("2022-12-01")  # primeiro mês inteiro após o ChatGPT (30/11/2022)
DATAS_PLACEBO = ["2019-06-01", "2020-06-01"]


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
    c_plac = max_placebo < 0.5 * abs(e12["efeito_pct_medio"]) and max_placebo < 0.5 * abs(e["efeito_pct_medio"])
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
    return {
        "veredito": veredito, "explicacao": expl,
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


def ato2(indeed: pd.DataFrame, fred: pd.DataFrame, cap: pd.DataFrame) -> dict:
    out = vagas_vs_setores(indeed, fred, cap, intervencao=INTERVENCAO)
    serie = out.pop("serie")
    serie.to_csv(RESULTS / "vagas_serie.csv", index=False)
    (RESULTS / "vagas.json").write_text(json.dumps(_limpo(out), ensure_ascii=False, indent=2), encoding="utf-8")
    p = out["principal"]
    coef, pval = p["coef_termo"], p["p_termo"]
    ok = coef < 0 and pval < 0.05
    veredito = "consistente com IA" if ok else "consistente com ciclo"
    pct = (np.exp(coef) - 1) * 100
    if ok:
        expl = (f"Mesmo controlando a Fed Funds, software ficou {abs(pct):.0f}% mais baixo que a mediana dos setores "
                f"após dez/2022 (p HAC = {pval:.3f}); consistente com IA, mas sem prova de causa.")
    else:
        expl = (f"Controlando a Fed Funds, a diferença pós-dez/2022 de software vs. mediana dos setores "
                f"({pct:+.0f}%, p HAC = {pval:.3f}) não passa o critério; não dá para separar do ciclo de juros.")
    return {"veredito": veredito, "explicacao": expl,
            "numeros": {"coef_pos_log": coef, "efeito_pos_pct": float(pct), "p_hac": pval,
                        "coef_fedfunds": p["coeficientes"]["FEDFUNDS"]["coef"],
                        "p_fedfunds": p["coeficientes"]["FEDFUNDS"]["p"], "n_meses": p["n"],
                        "robustez_p": {k: out[k]["p_termo"] for k in ("diferenca", "vs_total", "capacidade") if k in out},
                        "robustez_coef": {k: out[k]["coef_termo"] for k in ("diferenca", "vs_total", "capacidade") if k in out}}}


def main() -> dict:
    so = pd.read_csv(PROCESSED / "so_mensal.csv", parse_dates=["mes"])
    indeed = pd.read_csv(PROCESSED / "indeed_mensal.csv", parse_dates=["mes"])
    fred = pd.read_csv(PROCESSED / "fred_mensal.csv", parse_dates=["mes"])
    cap = pd.read_csv(PROCESSED / "capacidade_mensal.csv", parse_dates=["mes"])
    vereditos = {"ato1_stackoverflow": ato1(so), "ato2_vagas": ato2(indeed, fred, cap),
                 "regras": "design.md §7.1, aplicadas literalmente (ver docstring de src/models/run_series.py)"}
    (RESULTS / "vereditos_series.json").write_text(json.dumps(_limpo(vereditos), ensure_ascii=False, indent=2),
                                                   encoding="utf-8")
    return vereditos


if __name__ == "__main__":
    v = main()
    for k in ("ato1_stackoverflow", "ato2_vagas"):
        print(k, "->", v[k]["veredito"], "|", v[k]["explicacao"])
