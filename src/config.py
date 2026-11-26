"""Configuração central: caminhos, datas-chave e chave da Stack Exchange API."""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "data" / "results"
for _p in (RAW, PROCESSED, RESULTS):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 42
CORTE_PRE_IA = pd.Timestamp("2022-10-01")  # último mês usado no treino
DATA_CHATGPT = pd.Timestamp("2022-11-30")  # lançamento do ChatGPT

load_dotenv(ROOT / ".env")


def get_se_key() -> str | None:
    """Chave da Stack Exchange API lida do .env (nunca imprimir)."""
    key = os.environ.get("SE_API_KEY", "").strip()
    return key or None


def ultimo_mes_completo(hoje: date | None = None) -> pd.Timestamp:
    """Primeiro dia do último mês que já terminou."""
    hoje = hoje or date.today()
    inicio_mes_atual = pd.Timestamp(hoje.year, hoje.month, 1)
    return inicio_mes_atual - pd.DateOffset(months=1)
