import pandas as pd
import pytest

from src import config
from src.etl import stackoverflow as so


class FakeResp:
    status_code = 200

    def __init__(self, data):
        self._d = data

    def json(self):
        return self._d


class FakeSession:
    def __init__(self, data):
        self.data = data
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(params)
        return FakeResp(self.data)


@pytest.fixture(autouse=True)
def raw_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW", tmp_path)
    monkeypatch.setattr(so, "_estado", {"quota": None})


def test_params_total():
    s = FakeSession({"total": 842})
    n = so.contar_perguntas_mes(pd.Timestamp("2022-11-01"), None, "k", s)
    p = s.calls[0]
    assert n == 842
    assert p["fromdate"] == 1667260800 and p["todate"] == 1669852800
    assert p["site"] == "stackoverflow" and p["filter"] == "total"
    assert "tagged" not in p


def test_params_tag():
    s = FakeSession({"total": 5})
    so.contar_perguntas_mes(pd.Timestamp("2022-12-01"), "python", None, s)
    assert s.calls[0]["tagged"] == "python" and "key" not in s.calls[0]


def test_backoff(monkeypatch):
    dormidas = []
    monkeypatch.setattr(so.time, "sleep", dormidas.append)
    so.contar_perguntas_mes(pd.Timestamp("2022-11-01"), None, None, FakeSession({"total": 1, "backoff": 2}))
    assert dormidas == [2]


def test_cache():
    s = FakeSession({"total": 7})
    m = pd.Timestamp("2022-11-01")
    assert so.contar_perguntas_mes(m, "java", None, s) == 7
    assert so.contar_perguntas_mes(m, "java", None, s) == 7
    assert len(s.calls) == 1
