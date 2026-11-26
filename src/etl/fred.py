"""ETL macro (séries com nomes FRED): juros e vagas totais, saída longa mes, serie, valor.

O fred.stlouisfed.org está inacessível nesta máquina, então as fontes reais são:
- ``FEDFUNDS``: taxa de fundos federais (efetiva, mensal) do release H.15 do Federal Reserve,
  espelhado pelo DBnomics (FED/H15/129.FF.O). ``period`` é fim de mês -> ``mes`` = 1º dia do mês.
- ``IHLIDXUS``: total de vagas Indeed (EUA, agregado), Hiring Lab
  ``aggregate_job_postings_US.csv``, ``variable == "total postings"``, coluna
  ``indeed_job_postings_index_SA`` (dessazonalizada); média mensal só de meses completos.
O nome da série segue o ID FRED para manter o contrato com o restante do projeto.
"""
from __future__ import annotations

import requests

import pandas as pd

from src.config import PROCESSED, RAW, ultimo_mes_completo

URL_FEDFUNDS = "https://api.db.nomics.world/v22/series/FED/H15/129.FF.O?observations=1&format=csv"
URL_INDEED_AGG = "https://raw.githubusercontent.com/hiring-lab/job_postings_tracker/master/US/aggregate_job_postings_US.csv"
SERIES = ["FEDFUNDS", "IHLIDXUS"]


def agregar_mensal(df: pd.DataFrame, serie: str, ate: pd.Timestamp | None = None,
                   diaria: bool = True) -> pd.DataFrame:
    """df com colunas date, valor -> mes, serie, valor.

    diaria=True exige o mês inteiro observado; diaria=False aceita 1 obs/mês."""
    ate = ate if ate is not None else ultimo_mes_completo()
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["valor"] = pd.to_numeric(d["valor"], errors="coerce")
    d = d.dropna(subset=["valor"])
    d["mes"] = d["date"].dt.to_period("M").dt.to_timestamp()
    g = d.groupby("mes").agg(valor=("valor", "mean"), n=("date", "nunique")).reset_index()
    if diaria:
        g = g[g["n"] == g["mes"].dt.days_in_month]
    g = g[g["mes"] <= ate].copy()
    g["serie"] = serie
    return g[["mes", "serie", "valor"]]


def fedfunds_mensal(bruto: pd.DataFrame, ate: pd.Timestamp | None = None) -> pd.DataFrame:
    """CSV DBnomics (colunas period e a coluna de valor) -> mes, serie, valor."""
    ate = ate if ate is not None else ultimo_mes_completo()
    col_valor = next(c for c in bruto.columns if c != "period")  # nome longo varia
    d = pd.DataFrame({"date": pd.to_datetime(bruto["period"]),
                      "valor": pd.to_numeric(bruto[col_valor], errors="coerce")}).dropna()
    d["mes"] = d["date"].dt.to_period("M").dt.to_timestamp()
    d["serie"] = "FEDFUNDS"
    return d[d["mes"] <= ate][["mes", "serie", "valor"]].reset_index(drop=True)


def indeed_total(bruto: pd.DataFrame, ate: pd.Timestamp | None = None) -> pd.DataFrame:
    """CSV agregado Indeed -> mes, serie='IHLIDXUS', valor (índice SA, meses completos)."""
    b = bruto[bruto["variable"] == "total postings"]
    b = b.rename(columns={"indeed_job_postings_index_SA": "valor"})[["date", "valor"]]
    return agregar_mensal(b, "IHLIDXUS", ate)


def _baixar(url: str, arq) -> None:
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    r.raise_for_status()
    arq.write_bytes(r.content)


def carregar(series: list[str] | None = None, baixar: bool = True) -> pd.DataFrame:
    series = series or SERIES
    pasta = RAW / "fred"
    pasta.mkdir(parents=True, exist_ok=True)
    partes = []
    if "FEDFUNDS" in series:
        arq = pasta / "FEDFUNDS_dbnomics.csv"
        if baixar or not arq.exists():
            _baixar(URL_FEDFUNDS, arq)
        partes.append(fedfunds_mensal(pd.read_csv(arq)))
    if "IHLIDXUS" in series:
        arq = pasta / "aggregate_job_postings_US.csv"
        if baixar or not arq.exists():
            _baixar(URL_INDEED_AGG, arq)
        partes.append(indeed_total(pd.read_csv(arq)))
    return pd.concat(partes, ignore_index=True)


def main() -> None:
    df = carregar()
    df.to_csv(PROCESSED / "fred_mensal.csv", index=False)
    for s, g in df.groupby("serie"):
        print(s, g.head(2), g.tail(2), sep="\n")


if __name__ == "__main__":
    main()
