import numpy as np
import pandas as pd

from src.models.survey_models import bootstrap_ic, classificador_ameaca


def test_bootstrap_constante():
    assert bootstrap_ic(pd.Series([5.0] * 50), np.median) == (5.0, 5.0, 5.0)


def test_classificador_sinal_em_faixa_exp():
    rng = np.random.default_rng(0)
    n = 800
    faixa = rng.choice(["≤2", "3–9", "≥10"], n)
    df = pd.DataFrame({
        "ano": 2025,
        "faixa_exp": faixa,
        "cargo": rng.choice(["Developer, back-end", "Developer, front-end", "Student"], n),
        "pais": rng.choice(["Brazil", "United States of America", "Germany"], n),
        "usa_agentes": pd.array(rng.choice([True, False], n), dtype="boolean"),
        "ia_usa": pd.array(rng.choice([True, False], n), dtype="boolean"),
        "emprego": rng.choice(["Employed", "Student"], n),
        "ia_ameaca": np.where(faixa == "≤2", "Yes", rng.choice(["No", "I'm not sure"], n)),
    })
    r = classificador_ameaca(df)
    assert r["metricas"][r["melhor_modelo"]]["auc_media"] > 0.9
    assert r["importancias_holdout"][0]["feature"] == "faixa_exp"
    assert r["interpretavel"]
