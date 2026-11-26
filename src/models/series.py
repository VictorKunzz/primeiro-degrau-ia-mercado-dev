"""Modelos de série temporal dos Atos 1 (Stack Overflow) e 2 (vagas Indeed).

Princípios (design §6 e §7.1; pesquisa_metodologia §1):
- Nada pós-corte entra no ajuste: `assert y_treino.index.max() <= corte` em todo ajuste.
- Ordem do SARIMAX escolhida por AIC numa grade pequena, só no período de treino.
- Linguagem: "consistente com", nunca "causou".

Nota sobre a grade: com d=1 e sem diferença sazonal (D=0), um SARIMAX sem termo de
tendência prevê nível constante; numa série que já caía antes de 2022 isso inflaria o
"efeito". Por isso a variante sazonal (1,0,0,12) leva deriva (`trend="t"`, que com d=1
equivale a um crescimento % constante). A variante (0,1,1,12) com d=1 e D=1 já
extrapola a tendência local e vai sem termo de tendência.
"""
from __future__ import annotations

import warnings
from itertools import product

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

GRADE_PQ = (0, 1, 2)
GRADE_SAZONAL = (((1, 0, 0, 12), "t"), ((0, 1, 1, 12), "n"))
LIMITE_PLACEBO = pd.Timestamp("2022-10-01")


def _ajustar(y_log: pd.Series, order, seasonal, trend):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = SARIMAX(y_log, order=order, seasonal_order=seasonal, trend=trend,
                      enforce_stationarity=False, enforce_invertibility=False)
        return mod.fit(disp=False, maxiter=200)


def escolher_ordem(y_log_treino: pd.Series) -> dict:
    """Grade p,q ∈ {0,1,2}, d=1, sazonal (1,0,0,12)+deriva ou (0,1,1,12); menor AIC."""
    melhor = None
    for (p, q), (seas, trend) in product(product(GRADE_PQ, GRADE_PQ), GRADE_SAZONAL):
        try:
            res = _ajustar(y_log_treino, (p, 1, q), seas, trend)
        except Exception:  # noqa: BLE001 - combinação que não converge é descartada
            continue
        if not np.isfinite(res.aic):
            continue
        if melhor is None or res.aic < melhor["aic"]:
            melhor = {"order": (p, 1, q), "seasonal_order": seas, "trend": trend,
                      "aic": float(res.aic), "res": res}
    if melhor is None:
        raise RuntimeError("nenhum SARIMAX da grade convergiu")
    return melhor


def _preparar(y: pd.Series) -> pd.Series:
    y = y.astype(float).sort_index()
    y.index = pd.DatetimeIndex(y.index).to_period("M").to_timestamp()
    y = y.asfreq("MS")
    if (y <= 0).any() or y.isna().any():
        raise ValueError("série precisa ser positiva e sem buracos para log")
    return y


def contrafactual(y: pd.Series, corte: pd.Timestamp, inicio: pd.Timestamp | None = None,
                  fim: pd.Timestamp | None = None) -> pd.DataFrame:
    """SARIMAX em log(y) ajustado em [inicio, corte]; prevê do mês seguinte até `fim`.

    Retorna DataFrame (index mes) com real, previsto, lo95, hi95 (exp do IC em log).
    `previsto` = exp(média prevista em log) = mediana da previsão em nível (não a média).
    A ordem escolhida fica em `df.attrs["modelo"]`.
    """
    y = _preparar(y)
    corte = pd.Timestamp(corte)
    if inicio is not None:
        y = y[y.index >= pd.Timestamp(inicio)]
    if fim is not None:
        y = y[y.index <= pd.Timestamp(fim)]
    y_treino = y[y.index <= corte]
    assert y_treino.index.max() <= corte, "vazamento: dado pós-corte no treino"
    y_pos = y[y.index > corte]
    if y_pos.empty:
        raise ValueError("não há período pós-corte para prever")
    sel = escolher_ordem(np.log(y_treino))
    fc = sel["res"].get_forecast(steps=len(y_pos)).summary_frame(alpha=0.05)
    df = pd.DataFrame({
        "real": y_pos.values,
        "previsto": np.exp(fc["mean"].values),
        "lo95": np.exp(fc["mean_ci_lower"].values),
        "hi95": np.exp(fc["mean_ci_upper"].values),
    }, index=pd.Index(y_pos.index, name="mes"))
    df.attrs["modelo"] = {"order": list(sel["order"]), "seasonal_order": list(sel["seasonal_order"]),
                          "trend": sel["trend"], "aic": sel["aic"],
                          "treino_inicio": str(y_treino.index.min().date()),
                          "treino_fim": str(y_treino.index.max().date()), "n_treino": len(y_treino)}
    return df


def _maior_sequencia(mask) -> int:
    melhor = atual = 0
    for v in mask:
        atual = atual + 1 if v else 0
        melhor = max(melhor, atual)
    return melhor


def efeito(cf: pd.DataFrame, min_seguidos: int = 6) -> dict:
    """Efeito % médio (real/previsto − 1), faixa pelo IC e maior sequência abaixo do IC."""
    pct = (cf["real"] / cf["previsto"] - 1) * 100
    # faixa: real contra os limites do IC (hi95 → efeito mais negativo)
    pct_lo = (cf["real"] / cf["hi95"] - 1) * 100
    pct_hi = (cf["real"] / cf["lo95"] - 1) * 100
    seguidos = _maior_sequencia((cf["real"] < cf["lo95"]).to_numpy())
    return {
        "efeito_pct_medio": float(pct.mean()),
        "efeito_pct_faixa_ic": [float(pct_lo.mean()), float(pct_hi.mean())],
        "efeito_pct_ultimo_mes": float(pct.iloc[-1]),
        "meses_abaixo_ic_seguidos": int(seguidos),
        "meses_acima_ic_seguidos": int(_maior_sequencia((cf["real"] > cf["hi95"]).to_numpy())),
        "sustentado": bool(seguidos >= min_seguidos),
        "n_meses": int(len(cf)),
    }


def backtest(y: pd.Series, corte: pd.Timestamp, horizontes: int = 12, origens: int = 6,
             inicio: pd.Timestamp | None = None, passo: int = 6) -> pd.DataFrame:
    """Rolling-origin só no pré-período: origens espaçadas `passo` meses, a última
    terminando a avaliação em `corte`. Cada origem reescolhe a ordem por AIC no seu treino.
    Baseline: sazonal ingênuo (y[t-12], só com dados até a origem)."""
    y = _preparar(y)
    corte = pd.Timestamp(corte)
    if inicio is not None:
        y = y[y.index >= pd.Timestamp(inicio)]
    y = y[y.index <= corte]  # nada pós-corte
    linhas = []
    for k in range(origens):
        fim_aval = corte - pd.DateOffset(months=passo * k)
        origem = fim_aval - pd.DateOffset(months=horizontes)
        y_tr = y[y.index <= origem]
        y_te = y[(y.index > origem) & (y.index <= fim_aval)]
        assert y_tr.index.max() <= origem and y_te.index.max() <= corte
        if len(y_tr) < 36 or len(y_te) < horizontes:
            continue
        sel = escolher_ordem(np.log(y_tr))
        prev = np.exp(sel["res"].get_forecast(steps=len(y_te)).predicted_mean.values)
        ingenuo = np.array([y_tr.loc[m - pd.DateOffset(months=12 * ((h // 12) + 1))]
                            for h, m in enumerate(y_te.index)])
        real = y_te.values
        linhas.append({
            "origem": origem.date().isoformat(), "fim_avaliacao": fim_aval.date().isoformat(),
            "ordem": str(sel["order"]) + str(sel["seasonal_order"]) + sel["trend"],
            "mape_modelo": float(np.mean(np.abs(prev / real - 1)) * 100),
            "mape_ingenuo": float(np.mean(np.abs(ingenuo / real - 1)) * 100),
        })
    return pd.DataFrame(linhas)


def placebos(y: pd.Series, datas: list[str], meses: int = 12, inicio: pd.Timestamp | None = None,
             limite: pd.Timestamp = LIMITE_PLACEBO) -> pd.DataFrame:
    """Efeito com corte falso: ajusta até a data falsa e avalia só `meses` meses depois,
    sempre antes de `limite` (nunca toca dado pós-out/2022)."""
    y = _preparar(y)
    linhas = []
    for d in datas:
        corte = pd.Timestamp(d)
        fim = corte + pd.DateOffset(months=meses)
        assert fim <= limite, f"placebo {d} avançaria além de {limite.date()}"
        cf = contrafactual(y[y.index <= fim], corte, inicio=inicio)
        assert cf.index.max() <= limite
        e = efeito(cf)
        linhas.append({"corte_falso": corte.date().isoformat(), "fim_avaliacao": fim.date().isoformat(),
                       "modelo": str(cf.attrs["modelo"]["order"]) + str(cf.attrs["modelo"]["seasonal_order"]),
                       **{k: v for k, v in e.items() if k != "efeito_pct_faixa_ic"},
                       "efeito_pct_lo": e["efeito_pct_faixa_ic"][0], "efeito_pct_hi": e["efeito_pct_faixa_ic"][1]})
    return pd.DataFrame(linhas)


def _coef(res, nome: str) -> dict:
    ci = res.conf_int().loc[nome]
    return {"coef": float(res.params[nome]), "ic95": [float(ci[0]), float(ci[1])],
            "p": float(res.pvalues[nome]), "efeito_pct": float((np.exp(res.params[nome]) - 1) * 100)}


def its_segmentado(y: pd.Series, intervencao: pd.Timestamp, inicio: pd.Timestamp | None = None) -> dict:
    """ITS (Bernal et al. 2017): log y = b0 + b1 t + b2 D + b3 (t−t0) D + mês + e; HAC(12)."""
    y = _preparar(y)
    if inicio is not None:
        y = y[y.index >= pd.Timestamp(inicio)]
    t0 = pd.Timestamp(intervencao)
    t = np.arange(len(y), dtype=float)
    D = (y.index >= t0).astype(float)
    i0 = int(np.argmax(D)) if D.any() else len(y)
    X = pd.DataFrame({"tendencia": t, "nivel_pos": D, "inclinacao_pos": np.where(D > 0, t - i0, 0.0)},
                     index=y.index)
    meses = pd.get_dummies(y.index.month, prefix="m", drop_first=True, dtype=float)
    meses.index = y.index
    X = sm.add_constant(pd.concat([X, meses], axis=1))
    res = sm.OLS(np.log(y), X).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
    return {
        "intervencao": t0.date().isoformat(), "inicio": y.index.min().date().isoformat(),
        "fim": y.index.max().date().isoformat(), "n": int(len(y)), "r2": float(res.rsquared),
        "cov_type": "HAC (Newey-West), maxlags=12",
        "tendencia_mensal": _coef(res, "tendencia"),
        "nivel_pos": _coef(res, "nivel_pos"),
        "inclinacao_pos": _coef(res, "inclinacao_pos"),
    }


# ----------------------------------------------------------------------------- Ato 2
SETOR_SOFTWARE = "Software Development"


def _ols_hac(y: pd.Series, X: pd.DataFrame, termo: str) -> dict:
    res = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
    out = {"n": int(res.nobs), "r2": float(res.rsquared), "cov_type": "HAC (Newey-West), maxlags=12",
           "coeficientes": {k: _coef(res, k) for k in X.columns}}
    out["termo_teste"] = termo
    out["p_termo"] = out["coeficientes"][termo]["p"]
    out["coef_termo"] = out["coeficientes"][termo]["coef"]
    return out


def regressao_erros_ar1(y: pd.Series, X: pd.DataFrame, termo: str) -> dict:
    """Regressão com erros AR(1): SARIMAX(order=(1,0,0), exog=X, trend="c").

    Usada como teste principal do Ato 2 porque os resíduos do OLS são fortemente
    autocorrelacionados (DW ≈ 0,2) e HAC com n≈80 não corrige isso.
    """
    y = pd.Series(np.asarray(y, dtype=float), index=pd.RangeIndex(len(y)))
    Xr = pd.DataFrame(np.asarray(X, dtype=float), columns=list(X.columns), index=y.index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SARIMAX(y, exog=Xr, order=(1, 0, 0), trend="c").fit(disp=False, maxiter=500)
    ci = res.conf_int()
    coefs = {k: {"coef": float(res.params[k]), "ic95": [float(ci.loc[k, 0]), float(ci.loc[k, 1])],
                 "p": float(res.pvalues[k]), "efeito_pct": float((np.exp(res.params[k]) - 1) * 100)}
             for k in Xr.columns}
    return {"modelo": "regressão com erros AR(1): SARIMAX order=(1,0,0), exog, trend='c'",
            "n": int(res.nobs), "aic": float(res.aic), "coeficientes": coefs,
            "ar1": float(res.params["ar.L1"]), "ar1_p": float(res.pvalues["ar.L1"]),
            "termo_teste": termo, "coef_termo": coefs[termo]["coef"], "ic95_termo": coefs[termo]["ic95"],
            "p_termo": coefs[termo]["p"]}


def durbin_watson_ols(y: pd.Series, X: pd.DataFrame) -> dict:
    from statsmodels.stats.stattools import durbin_watson
    r = sm.OLS(y, sm.add_constant(X)).fit().resid
    return {"durbin_watson": float(durbin_watson(r)), "autocorr_lag1": float(pd.Series(r).autocorr(1))}


def vagas_vs_setores(indeed: pd.DataFrame, fred: pd.DataFrame, capacidade: pd.DataFrame | None = None,
                     intervencao: pd.Timestamp = pd.Timestamp("2022-12-01")) -> dict:
    """Software vs. mediana dos outros setores (e vs. total Indeed), com FEDFUNDS + HAC.

    Alvo principal: log(software / mediana dos outros setores) — negativo = software pior.
    Termo testado: dummy pós-ChatGPT (a partir de `intervencao`), controlando FEDFUNDS.
    Teste principal: regressão com erros AR(1) (`modelo_principal`); o OLS+HAC original fica
    em `modelo_hac_original` com Durbin-Watson. Robustez (diferença, vs. total, capacidade) segue HAC.
    Alternativa: capacidade da IA (log(1 + horizonte METR em min)) no lugar da dummy.
    """
    ind = indeed.copy()
    ind["mes"] = pd.to_datetime(ind["mes"])
    wide = ind.pivot(index="mes", columns="setor", values="indice").sort_index()
    sw = wide[SETOR_SOFTWARE]
    mediana = wide.drop(columns=SETOR_SOFTWARE).median(axis=1)
    fr = fred.copy()
    fr["mes"] = pd.to_datetime(fr["mes"])
    fr = fr.pivot(index="mes", columns="serie", values="valor")
    df = pd.DataFrame({"software": sw, "mediana_setores": mediana}).join(fr, how="inner").dropna(
        subset=["software", "mediana_setores", "FEDFUNDS"])
    df["diferenca"] = df["software"] - df["mediana_setores"]
    df["razao"] = df["software"] / df["mediana_setores"]
    df["log_razao"] = np.log(df["razao"])
    if "IHLIDXUS" in df:
        df["razao_total"] = df["software"] / df["IHLIDXUS"]
    df["pos"] = (df.index >= pd.Timestamp(intervencao)).astype(float)

    out = {"inicio": df.index.min().date().isoformat(), "fim": df.index.max().date().isoformat(),
           "n_meses": int(len(df)), "intervencao": pd.Timestamp(intervencao).date().isoformat(),
           "n_setores_outros": int(wide.shape[1] - 1),
           "pico_software": {"mes": sw.idxmax().date().isoformat(), "indice": float(sw.max())}}
    pre, pos = df[df.pos == 0], df[df.pos == 1]
    out["descritivo"] = {
        "razao_media_pre": float(pre.razao.mean()), "razao_media_pos": float(pos.razao.mean()),
        "diferenca_media_pre": float(pre.diferenca.mean()), "diferenca_media_pos": float(pos.diferenca.mean()),
        "software_ultimo": float(df.software.iloc[-1]), "mediana_ultimo": float(df.mediana_setores.iloc[-1]),
        "razao_ultimo": float(df.razao.iloc[-1]),
    }
    alvo = "log(software / mediana dos outros setores)"
    out["corr_fedfunds_pos"] = float(df["FEDFUNDS"].corr(df["pos"]))
    out["modelo_principal"] = regressao_erros_ar1(df["log_razao"], df[["FEDFUNDS", "pos"]], "pos")
    out["modelo_principal"]["alvo"] = alvo
    out["modelo_hac_original"] = _ols_hac(df["log_razao"], df[["FEDFUNDS", "pos"]], "pos")
    out["modelo_hac_original"]["alvo"] = alvo
    out["modelo_hac_original"].update(durbin_watson_ols(df["log_razao"], df[["FEDFUNDS", "pos"]]))
    out["modelo_hac_original"]["nota"] = ("substituído: resíduos fortemente autocorrelacionados; HAC com n≈80 "
                                          "não corrige isso, então o p-valor não é um teste válido")
    out["diferenca"] = _ols_hac(df["diferenca"], df[["FEDFUNDS", "pos"]], "pos")
    out["diferenca"]["alvo"] = "software − mediana dos outros setores (pontos de índice)"
    if "razao_total" in df:
        out["vs_total"] = _ols_hac(np.log(df["razao_total"]), df[["FEDFUNDS", "pos"]], "pos")
        out["vs_total"]["alvo"] = "log(software / índice total Indeed IHLIDXUS)"
    if capacidade is not None:
        cap = capacidade.copy()
        cap["mes"] = pd.to_datetime(cap["mes"])
        cap = cap.set_index("mes")["metr_horizonte_min_melhor"]
        d2 = df.join(cap.rename("metr"), how="left").dropna(subset=["metr"])
        d2["log_capacidade"] = np.log1p(d2["metr"])
        out["capacidade"] = _ols_hac(d2["log_razao"], d2[["FEDFUNDS", "log_capacidade"]], "log_capacidade")
        out["capacidade"]["alvo"] = "log(software / mediana dos outros setores)"
        out["capacidade"]["nota"] = "capacidade = log(1 + melhor horizonte METR em minutos); meses sem medição excluídos"
    out["serie"] = df[["software", "mediana_setores", "diferenca", "razao", "FEDFUNDS"]].reset_index()
    return out
