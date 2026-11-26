import pandas as pd

from src.etl.survey import COLUNAS_FINAIS, MAPA_COLUNAS, harmonizar

FIX = "tests/fixtures"


def _ler(ano):
    return pd.read_csv(f"{FIX}/survey_mini_{ano}.csv")


def test_2026_mapeia_role_para_cargo():
    assert MAPA_COLUNAS[2026]["Role"] == "cargo"
    h = harmonizar(_ler(2026), 2026)
    assert list(h.columns) == COLUNAS_FINAIS
    assert h.loc[0, "cargo"] == "Developer, full-stack"


def test_ano_sem_aithreat_tem_ia_ameaca_nan():
    h = harmonizar(_ler(2026), 2026)
    assert h["ia_ameaca"].isna().all()
    assert harmonizar(_ler(2024), 2024)["ia_ameaca"].notna().any()


def test_salario_zero_e_2e6_viram_nan():
    for ano in (2024, 2026):
        h = harmonizar(_ler(ano), ano)
        assert h["salario_usd"].notna().sum() == 2
        assert h["salario_usd"].max() <= 1e6
        assert h["salario_usd"].min() >= 1_000


def test_less_than_1_year_vira_faixa_ate_2():
    h = harmonizar(_ler(2024), 2024)
    assert h.loc[0, "anos_exp"] == 0.5
    assert h.loc[0, "faixa_exp"] == "≤2"
    assert h.loc[1, "faixa_exp"] == "3–9"
    assert h.loc[3, "faixa_exp"] == "≥10"


def test_empregado_integral_e_agentes():
    h24 = harmonizar(_ler(2024), 2024)
    assert h24.loc[0, "empregado_integral"]
    assert not h24.loc[3, "empregado_integral"]  # part-time
    assert pd.isna(h24.loc[4, "empregado_integral"])
    assert h24["usa_agentes"].isna().all()
    h26 = harmonizar(_ler(2026), 2026)
    assert list(h26["usa_agentes"].astype(object)) == [True, False, False, pd.NA, True]
    assert h26.loc[0, "empregado_integral"]
    assert h26.loc[0, "ia_sentimento"] == "Favorable"


def test_2025_aiagents_e_workexp_unico():
    df = pd.DataFrame({
        "Employment": ["Employed", "Employed", None],
        "WorkExp": [60, 2, None], "AISelect": ["Yes, I use AI tools daily", "No, and I don't plan to", None],
        "AIAgents": ["Yes, I use AI agents at work daily", "No, but I plan to", None],
    })
    h = harmonizar(df, 2025)
    assert list(h["usa_agentes"].astype(object)) == [True, False, pd.NA]
    assert h["anos_exp"].max() == 51.0
    assert list(h["anos_exp_fonte"].fillna("x")) == ["WorkExp", "WorkExp", "x"]


def test_comparavel_cobre_todas_as_colunas():
    from src.etl.survey import COMPARAVEL
    assert set(COMPARAVEL) == set(COLUNAS_FINAIS) - {"ano"}
    assert {2025, 2026} <= set(COMPARAVEL["emprego"])
