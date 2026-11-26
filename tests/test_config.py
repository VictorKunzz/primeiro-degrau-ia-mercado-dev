from datetime import date

import pandas as pd

from src import config


def test_ultimo_mes_completo():
    assert config.ultimo_mes_completo(date(2026, 11, 26)) == pd.Timestamp("2026-10-01")
    assert config.ultimo_mes_completo(date(2026, 12, 1)) == pd.Timestamp("2026-11-01")
    assert config.ultimo_mes_completo(date(2026, 1, 15)) == pd.Timestamp("2025-12-01")


def test_get_se_key_vazia(monkeypatch):
    monkeypatch.setenv("SE_API_KEY", "")
    assert config.get_se_key() is None
