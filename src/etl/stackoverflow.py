"""ETL Stack Overflow: série mensal de perguntas via Stack Exchange API."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

from src import config

API_URL = "https://api.stackexchange.com/2.3/questions"
USER_AGENT = "primeiro-degrau-projeto-academico/1.0 (analise de series mensais)"
QUOTA_MINIMA = 50
TOTAL = "__total__"
MAX_TENTATIVAS = 5

_estado = {"quota": None}


class QuotaBaixa(RuntimeError):
    """Quota da API quase esgotada; coleta interrompida."""


def _epoch(ts: pd.Timestamp) -> int:
    return int(pd.Timestamp(ts).tz_localize("UTC").timestamp())


def _cache_path(mes: pd.Timestamp, tag: str | None) -> Path:
    return config.RAW / "so" / (tag or TOTAL) / f"{pd.Timestamp(mes):%Y-%m}.json"


def contar_perguntas_mes(mes: pd.Timestamp, tag: str | None, key: str | None, session) -> int:
    """Número de perguntas criadas no mês (1 requisição, com cache em disco)."""
    mes = pd.Timestamp(mes)
    cache = _cache_path(mes, tag)
    if cache.exists():
        return int(json.loads(cache.read_text(encoding="utf-8"))["total"])

    params = {
        "site": "stackoverflow",
        "filter": "total",
        "fromdate": _epoch(mes),
        "todate": _epoch(mes + pd.DateOffset(months=1)),
    }
    if tag is not None:
        params["tagged"] = tag
    if key:
        params["key"] = key

    for tentativa in range(MAX_TENTATIVAS):
        resp = session.get(API_URL, params=params, timeout=30)
        if resp.status_code in (429, 502, 503):
            time.sleep(2 ** (tentativa + 1))
            continue
        if resp.status_code != 200:
            # não usar raise_for_status: a mensagem incluiria a URL com a chave
            raise RuntimeError(f"Stack Exchange API retornou HTTP {resp.status_code}")
        dados = resp.json()
        break
    else:
        raise RuntimeError("Stack Exchange API indisponível após várias tentativas")

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(dados), encoding="utf-8")
    if "quota_remaining" in dados:
        _estado["quota"] = dados["quota_remaining"]
    if dados.get("backoff"):
        time.sleep(dados["backoff"])
    return int(dados["total"])


def coletar_serie(inicio: str, tags: list[str | None]) -> pd.DataFrame:
    """Série mensal para cada tag (None = total) de `inicio` até o último mês completo."""
    key = config.get_se_key()
    meses = pd.date_range(inicio, config.ultimo_mes_completo(), freq="MS")
    linhas = []
    with requests.Session() as session:
        session.headers["User-Agent"] = USER_AGENT
        for tag in tags:
            for mes in meses:
                if _estado["quota"] is not None and _estado["quota"] < QUOTA_MINIMA:
                    print(f"Quota baixa ({_estado['quota']}); interrompendo coleta.")
                    return pd.DataFrame(linhas, columns=["mes", "tag", "perguntas"])
                n = contar_perguntas_mes(mes, tag, key, session)
                linhas.append({"mes": mes, "tag": tag or TOTAL, "perguntas": n})
    return pd.DataFrame(linhas, columns=["mes", "tag", "perguntas"])


def main() -> pd.DataFrame:
    total = coletar_serie("2008-09-01", [None])
    tags = coletar_serie("2015-01-01", ["python", "javascript", "java", "reactjs"])
    df = pd.concat([total, tags], ignore_index=True)
    df.to_csv(config.PROCESSED / "so_mensal.csv", index=False)
    print(f"{len(df)} linhas; quota restante: {_estado['quota']}")
    return df


if __name__ == "__main__":
    main()
