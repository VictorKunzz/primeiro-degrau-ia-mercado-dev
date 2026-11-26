import pandas as pd
import pytest

from src import config
from src.etl import stackoverflow as so


class FakeResp:
    def __init__(self, data, status_code=200):
        self._d = data
        self.status_code = status_code

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


def test_throttle_400_error_id_502(monkeypatch):
    dormidas = []
    monkeypatch.setattr(so.time, "sleep", dormidas.append)
    respostas = [FakeResp({"error_id": 502, "error_name": "throttle_violation"}, 400), FakeResp({"total": 5})]

    class S:
        def get(self, url, params=None, timeout=None):
            return respostas.pop(0)

    assert so.contar_perguntas_mes(pd.Timestamp("2022-11-01"), None, None, S()) == 5
    assert dormidas


def test_erro_de_rede_sem_url(monkeypatch):
    import requests

    class S:
        def get(self, url, params=None, timeout=None):
            raise requests.ConnectionError("https://x?key=SEGREDO")

    with pytest.raises(RuntimeError) as e:
        so.contar_perguntas_mes(pd.Timestamp("2022-11-01"), None, "SEGREDO", S())
    assert "SEGREDO" not in str(e.value) and e.value.__cause__ is None
