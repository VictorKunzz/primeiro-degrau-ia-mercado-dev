import pandas as pd

from src.etl import capacidade, fred, indeed


def test_indeed_descarta_mes_incompleto():
    dias = pd.date_range("2024-01-01", "2024-02-15")
    df = pd.DataFrame({"date": dias, "setor": "SW", "indice": 100.0})
    out = indeed.agregar_mensal(df, ate=pd.Timestamp("2024-12-01"))
    assert list(out["mes"]) == [pd.Timestamp("2024-01-01")]
    assert out["indice"].iloc[0] == 100.0


def test_indeed_descarta_mes_apos_ultimo_completo():
    dias = pd.date_range("2024-01-01", "2024-02-29")
    df = pd.DataFrame({"date": dias, "setor": "SW", "indice": 1.0})
    out = indeed.agregar_mensal(df, ate=pd.Timestamp("2024-01-01"))
    assert list(out["mes"]) == [pd.Timestamp("2024-01-01")]


def test_fred_mensal_e_diaria():
    m = pd.DataFrame({"date": ["2024-01-01", "2024-02-01"], "valor": [5.0, "."]})
    out = fred.agregar_mensal(m, "X", ate=pd.Timestamp("2024-12-01"))
    assert len(out) == 1 and out["valor"].iloc[0] == 5.0
    d = pd.DataFrame({"date": pd.date_range("2024-01-01", "2024-02-10"), "valor": 1.0})
    out = fred.agregar_mensal(d, "Y", ate=pd.Timestamp("2024-12-01"))
    assert list(out["mes"]) == [pd.Timestamp("2024-01-01")]


def test_capacidade_cummax_e_ffill():
    df = pd.DataFrame({"d": ["2024-01-15", "2024-03-10", "2024-03-20", "2024-06-01"],
                       "v": [50.0, 40.0, 60.0, 55.0]})
    s = capacidade.melhor_acumulado(df, "d", "v", "x", inicio=pd.Timestamp("2023-12-01"),
                                    fim=pd.Timestamp("2024-07-01"))
    assert pd.isna(s.loc["2023-12-01"])          # antes da 1ª medição
    assert s.loc["2024-02-01"] == 50.0           # ffill sem lançamento
    assert s.loc["2024-03-01"] == 60.0
    assert s.loc["2024-06-01"] == 60.0           # cummax não diminui
    assert s.dropna().is_monotonic_increasing
