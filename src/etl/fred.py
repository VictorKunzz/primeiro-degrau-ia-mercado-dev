"""ETL FRED: séries mensais/diárias via fredgraph.csv (sem chave) -> média mensal.

Séries padrão: FEDFUNDS (juros, mensal) e IHLIDXUS (índice Indeed de vagas, diário).
Saída longa: mes, serie, valor. Meses incompletos de séries diárias são descartados.
"""
from __future__ import annotations

import io
import urllib.request

import pandas as pd

from src.config import PROCESSED, RAW, ultimo_mes_completo

URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"
SERIES = ["FEDFUNDS", "IHLIDXUS"]


def agregar_mensal(df: pd.DataFrame, serie: str, ate: pd.Timestamp | None = None) -> pd.DataFrame:
    """df com colunas date, valor -> mes, serie, valor (só meses completos)."""
    ate = ate if ate is not None else ultimo_mes_completo()
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["valor"] = pd.to_numeric(d["valor"], errors="coerce")
    d = d.dropna(subset=["valor"])
    d["mes"] = d["date"].dt.to_period("M").dt.to_timestamp()
    g = d.groupby("mes").agg(valor=("valor", "mean"), n=("date", "nunique")).reset_index()
    # séries mensais têm 1 obs/mês; diárias precisam cobrir o mês todo
    if (g["n"] > 1).any():
        g = g[g["n"] == g["mes"].dt.days_in_month]
    g = g[g["mes"] <= ate].copy()
    g["serie"] = serie
    return g[["mes", "serie", "valor"]]


def carregar(series: list[str] | None = None, baixar: bool = True) -> pd.DataFrame:
    series = series or SERIES
    pasta = RAW / "fred"
    pasta.mkdir(parents=True, exist_ok=True)
    partes = []
    for s in series:
        arq = pasta / f"{s}.csv"
        if baixar or not arq.exists():
            req = urllib.request.Request(URL.format(id=s), headers={"User-Agent": "Mozilla/5.0"})
            arq.write_bytes(urllib.request.urlopen(req, timeout=60).read())
        bruto = pd.read_csv(io.BytesIO(arq.read_bytes()))
        bruto.columns = ["date", "valor"]
        partes.append(agregar_mensal(bruto, s))
    return pd.concat(partes, ignore_index=True)


def main() -> None:
    df = carregar()
    df.to_csv(PROCESSED / "fred_mensal.csv", index=False)
    for s, g in df.groupby("serie"):
        print(s, g.head(2), g.tail(2), sep="\n")


if __name__ == "__main__":
    main()
