"""ETL Indeed Hiring Lab: índice diário de vagas por setor (EUA) -> média mensal.

Variante usada: o CSV tem as colunas ``date, jobcountry, indeed_job_postings_index,
variable, display_name``. Não existe coluna sazonalmente ajustada separada: o índice
publicado já é média móvel de 7 dias e dessazonalizado (base 100 = 01/02/2020).
Usamos ``variable == "total postings"`` (a outra variante é "new postings").
Só meses completos entram (dias observados == dias do mês e mês <= último mês completo).
Fonte: https://github.com/hiring-lab/job_postings_tracker (CC BY 4.0).
"""
from __future__ import annotations

import requests

import pandas as pd

from src.config import PROCESSED, RAW, ultimo_mes_completo

URL = "https://raw.githubusercontent.com/hiring-lab/job_postings_tracker/master/US/job_postings_by_sector_US.csv"
VARIAVEL = "total postings"


def agregar_mensal(df: pd.DataFrame, ate: pd.Timestamp | None = None) -> pd.DataFrame:
    """df com colunas date, setor, indice (diário) -> mes, setor, indice mensal (só meses completos)."""
    ate = ate if ate is not None else ultimo_mes_completo()
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["mes"] = d["date"].dt.to_period("M").dt.to_timestamp()
    g = d.groupby(["mes", "setor"]).agg(indice=("indice", "mean"), n=("date", "nunique")).reset_index()
    g = g[(g["n"] == g["mes"].dt.days_in_month) & (g["mes"] <= ate)]
    return g[["mes", "setor", "indice"]].sort_values(["setor", "mes"]).reset_index(drop=True)


def carregar(baixar: bool = True) -> pd.DataFrame:
    pasta = RAW / "indeed"
    pasta.mkdir(parents=True, exist_ok=True)
    arq = pasta / "job_postings_by_sector_US.csv"
    if baixar or not arq.exists():
        r = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
        r.raise_for_status()
        arq.write_bytes(r.content)
    bruto = pd.read_csv(arq)
    bruto = bruto[bruto["variable"] == VARIAVEL].rename(
        columns={"display_name": "setor", "indeed_job_postings_index": "indice"}
    )
    return agregar_mensal(bruto[["date", "setor", "indice"]])


def main() -> None:
    df = carregar()
    df.to_csv(PROCESSED / "indeed_mensal.csv", index=False)
    print(df.head(), df.tail(), sep="\n")


if __name__ == "__main__":
    main()
