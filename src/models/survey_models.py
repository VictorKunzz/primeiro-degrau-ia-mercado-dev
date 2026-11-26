"""Estatística do survey: bootstrap e classificador de percepção de ameaça (H3, §7.1/§7.3).

Decisões documentadas:
- Alvo binário de `ia_ameaca`: "Yes" -> 1; "No" e "I'm not sure" -> 0 (quem não
  afirma ameaça). NaN (não respondeu) é excluído.
- `cargo`: em 2025 o campo é de resposta única no parquet (nenhum ";"), mas por
  segurança usamos o primeiro item listado; depois top 10 + "Outros". O mesmo
  top 10 + "Outros" vale para `pais`.
- Booleanos nulos (`usa_agentes`, `ia_usa`) e `faixa_exp` nula viram categoria "NA"
  (não respondeu é informação, e excluir enviesaria a amostra).
- Baseline: DummyClassifier(prior). Regra §7.1 aplicada: interpretável só se a
  melhor AUC (logística ou HGB) superar a do Dummy em >= 0,03 na CV. Também
  reportamos HGB − logística e a logística só com `faixa_exp` como contexto.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import SEED

FEATURES = ["faixa_exp", "cargo", "pais", "usa_agentes", "ia_usa", "emprego"]
MAPEAMENTO_AMEACA = {"Yes": 1, "No": 0, "I'm not sure": 0}
LIMIAR_INTERPRETAVEL = 0.03


def bootstrap_ic(serie, func: Callable = np.mean, n: int = 2000, seed: int = SEED
                 ) -> tuple[float, float, float]:
    """(estimativa, IC 2,5%, IC 97,5%) por bootstrap percentil."""
    x = np.asarray(pd.Series(serie).dropna(), dtype=float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    est = float(func(x))
    boots = np.array([func(x[rng.integers(0, len(x), len(x))]) for _ in range(n)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return (est, float(lo), float(hi))


def bootstrap_amostras(serie, func: Callable = np.mean, n: int = 2000, seed: int = SEED
                       ) -> np.ndarray:
    """Distribuição bootstrap de func(serie) (para combinar grupos independentes)."""
    x = np.asarray(pd.Series(serie).dropna(), dtype=float)
    rng = np.random.default_rng(seed)
    return np.array([func(x[rng.integers(0, len(x), len(x))]) for _ in range(n)])


def ic_de(boots: np.ndarray, est: float) -> tuple[float, float, float]:
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return (float(est), float(lo), float(hi))


def _top_n(s: pd.Series, n: int = 10) -> pd.Series:
    s = s.fillna("NA").astype(str)
    top = s.value_counts().index[:n]
    return s.where(s.isin(top), "Outros")


def preparar_X(df: pd.DataFrame) -> pd.DataFrame:
    """Matriz de features categóricas (strings) a partir do survey harmonizado."""
    X = pd.DataFrame(index=df.index)
    X["faixa_exp"] = df["faixa_exp"].fillna("NA").astype(str)
    X["cargo"] = _top_n(df["cargo"].astype("string").str.split(";").str[0].str.strip())
    X["pais"] = _top_n(df["pais"])
    for c in ("usa_agentes", "ia_usa"):
        X[c] = df[c].astype("object").map({True: "sim", False: "nao"}).fillna("NA")
    X["emprego"] = _top_n(df["emprego"])
    return X[FEATURES]


def _pipe(modelo, cols) -> Pipeline:
    enc = ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cols)])
    return Pipeline([("enc", enc), ("m", modelo)])


def classificador_ameaca(df: pd.DataFrame, seed: int = SEED) -> dict:
    """CV 5 folds de Dummy / logística / HGB para prever 'IA ameaça meu emprego'."""
    anos = df.loc[df["ia_ameaca"].notna(), "ano"]
    ano = int(anos.max())
    d = df[(df["ano"] == ano) & df["ia_ameaca"].isin(list(MAPEAMENTO_AMEACA))].copy()
    y = d["ia_ameaca"].map(MAPEAMENTO_AMEACA).astype(int).to_numpy()
    X = preparar_X(d)

    modelos = {
        "dummy": _pipe(DummyClassifier(strategy="prior"), FEATURES),
        "logistica_so_faixa_exp": _pipe(LogisticRegression(max_iter=2000, class_weight="balanced"), ["faixa_exp"]),
        "logistica": _pipe(LogisticRegression(max_iter=2000, class_weight="balanced"), FEATURES),
        "hgb": _pipe(HistGradientBoostingClassifier(random_state=seed, class_weight="balanced"), FEATURES),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    metricas = {}
    for nome, m in modelos.items():
        r = cross_validate(m, X, y, cv=cv, scoring=["roc_auc", "f1"])
        metricas[nome] = {
            "auc_media": float(r["test_roc_auc"].mean()), "auc_dp": float(r["test_roc_auc"].std()),
            "f1_media": float(r["test_f1"].mean()), "f1_dp": float(r["test_f1"].std()),
        }
    auc = {k: v["auc_media"] for k, v in metricas.items()}
    melhor = max(("logistica", "hgb"), key=lambda k: auc[k])
    delta_dummy = auc[melhor] - auc["dummy"]
    interpretavel = bool(delta_dummy >= LIMIAR_INTERPRETAVEL)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    final = modelos[melhor].fit(X_tr, y_tr)
    pi = permutation_importance(final, X_te, y_te, scoring="roc_auc", n_repeats=10, random_state=seed)
    imp = (pd.DataFrame({"feature": FEATURES, "importancia_media": pi.importances_mean,
                         "importancia_dp": pi.importances_std})
           .sort_values("importancia_media", ascending=False).reset_index(drop=True))

    return {
        "ano": ano, "n": int(len(y)), "prevalencia_ameaca": float(y.mean()),
        "mapeamento_alvo": MAPEAMENTO_AMEACA, "features": FEATURES,
        "tratamento_features": "categóricas one-hot; cargo = 1º item listado, cargo/pais/emprego top 10 + Outros; nulos = 'NA'",
        "cv": "StratifiedKFold(5, shuffle, seed)", "metricas": metricas, "melhor_modelo": melhor,
        "delta_auc_melhor_vs_dummy": float(delta_dummy),
        "delta_auc_hgb_vs_logistica": float(auc["hgb"] - auc["logistica"]),
        "delta_auc_melhor_vs_logistica_so_faixa_exp": float(auc[melhor] - auc["logistica_so_faixa_exp"]),
        "regra": "interpretável se AUC(melhor) − AUC(dummy) >= 0,03 (design §7.1)",
        "interpretavel": interpretavel,
        "importancias_holdout": imp.to_dict(orient="records"),
        "auc_holdout": float(roc_auc_score(y_te, final.predict_proba(X_te)[:, 1])),
    }
