import numpy as np
import pandas as pd
import pytest

from src.models.series import (backtest, contrafactual, efeito, its_segmentado, placebos,
                               regressao_erros_ar1, vagas_vs_setores)

N = 180
IDX = pd.date_range("2005-01-01", periods=N, freq="MS")
CORTE = IDX[149]  # último mês de treino; queda a partir do mês 150


def _serie(queda: bool) -> pd.Series:
    rng = np.random.default_rng(42)
    t = np.arange(N)
    log_y = np.log(10_000) - 0.004 * t + 0.08 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.02, N)
    y = np.exp(log_y)
    if queda:
        y[150:] *= 0.6
    return pd.Series(y, index=IDX)


@pytest.fixture(scope="module")
def cf_queda():
    return contrafactual(_serie(True), CORTE)


def test_queda_detectada(cf_queda):
    e = efeito(cf_queda)
    assert e["sustentado"] is True
    assert e["efeito_pct_medio"] < -25
    assert cf_queda.index.min() == IDX[150]
    assert list(cf_queda.columns) == ["real", "previsto", "lo95", "hi95"]
    assert (cf_queda["lo95"] <= cf_queda["previsto"]).all() and (cf_queda["previsto"] <= cf_queda["hi95"]).all()


def test_sem_queda_nao_sustentado():
    e = efeito(contrafactual(_serie(False), CORTE))
    assert e["sustentado"] is False
    assert abs(e["efeito_pct_medio"]) < 10


def test_backtest_colunas():
    bt = backtest(_serie(False), CORTE, horizontes=12, origens=2)
    assert {"mape_modelo", "mape_ingenuo"} <= set(bt.columns)
    assert len(bt) == 2
    assert pd.Timestamp(bt["fim_avaliacao"].max()) <= CORTE


def test_placebo_sem_quebra():
    pl = placebos(_serie(False), ["2015-06-01"])
    assert len(pl) == 1
    assert abs(pl["efeito_pct_medio"].iloc[0]) < 10


def test_placebo_nao_passa_limite():
    with pytest.raises(AssertionError):
        placebos(_serie(False), ["2022-06-01"])


def test_its_detecta_nivel():
    r = its_segmentado(_serie(True), IDX[150])
    assert r["nivel_pos"]["coef"] < -0.3
    assert r["nivel_pos"]["p"] < 0.05


def test_vagas_estrutura():
    meses = pd.date_range("2020-02-01", periods=60, freq="MS")
    rng = np.random.default_rng(42)
    linhas = []
    for s in ["Software Development", "A", "B", "C"]:
        base = 100 + rng.normal(0, 2, 60).cumsum()
        if s == "Software Development":
            base = base * np.where(meses >= "2022-12-01", 0.6, 1.0)
        linhas += [{"mes": m, "setor": s, "indice": v} for m, v in zip(meses, base)]
    indeed = pd.DataFrame(linhas)
    fred = pd.DataFrame({"mes": meses, "serie": "FEDFUNDS", "valor": np.linspace(0, 5, 60)})
    out = vagas_vs_setores(indeed, fred)
    assert out["modelo_principal"]["coef_termo"] < 0
    assert 0 <= out["modelo_principal"]["p_termo"] <= 1
    assert "durbin_watson" in out["modelo_hac_original"]


def _ar1_sintetico(efeito_pos: float, n: int = 120, phi: float = 0.9):
    rng = np.random.default_rng(42)
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = phi * e[i - 1] + rng.normal(0, 0.05)
    pos = (np.arange(n) >= 60).astype(float)
    x = rng.normal(0, 1, n)
    y = 0.5 + 0.1 * x + efeito_pos * pos + e
    return pd.Series(y), pd.DataFrame({"x": x, "pos": pos})


def test_ar1_recupera_phi_e_efeito():
    y, X = _ar1_sintetico(-0.5, phi=0.6)
    r = regressao_erros_ar1(y, X, "pos")
    assert 0.4 < r["ar1"] < 0.8
    assert r["coef_termo"] < -0.3 and r["p_termo"] < 0.05
    assert r["ic95_termo"][0] < r["coef_termo"] < r["ic95_termo"][1]


def test_ar1_sem_efeito_nao_significativo():
    y, X = _ar1_sintetico(0.0)
    r = regressao_erros_ar1(y, X, "pos")
    assert r["p_termo"] > 0.05
