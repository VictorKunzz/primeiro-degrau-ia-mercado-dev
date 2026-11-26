"""ETL capacidade da IA (Epoch AI benchmark_data.zip).

Mapeamento de colunas reais (verificadas em 26/11/2026):
- swe_bench_verified.csv: ``Release date`` + ``Best score (across scorers)``
  (fallback ``mean_score``); fração 0-1, convertida para % (0-100).
- metr_time_horizons_external.csv: ``Release date`` + ``Time horizon`` (minutos,
  horizonte a 50%). A coluna ``Time Horizon (80%)`` não é usada.
Saída: mes, swe_bench_melhor, metr_horizonte_min_melhor = melhor valor acumulado
(cummax) até o mês, com forward-fill; de 2019-01 até o último mês completo.
Meses antes da primeira medição ficam NaN (sem dado, não zero).
Os marcos (data/processed/marcos.csv) são mantidos à mão, com fonte por linha.
Fonte: Epoch AI, Capabilities & benchmarking (CC BY 4.0).
"""
from __future__ import annotations

import io
import urllib.request
import zipfile

import pandas as pd

from src.config import PROCESSED, RAW, ultimo_mes_completo

URL = "https://epoch.ai/data/benchmark_data.zip"
INICIO = pd.Timestamp("2019-01-01")

# (data, marco, onda, fonte) - datas conferidas em 26/11/2026 nas páginas oficiais.
MARCOS = [
    ("2022-06-21", "GitHub Copilot disponível a todos (GA)", "assistente", "https://github.blog/2022/06/21/github-copilot-is-generally-available-to-all-developers/"),
    ("2022-11-30", "Lançamento do ChatGPT", "chat", "https://openai.com/index/chatgpt/"),
    ("2023-03-14", "GPT-4", "chat", "https://openai.com/index/gpt-4-research/"),
    ("2024-03-04", "Claude 3 (Opus/Sonnet/Haiku)", "assistente", "https://www.anthropic.com/news/claude-3-family"),
    ("2024-03-12", "Devin, primeiro 'engenheiro de software IA' (Cognition)", "agente", "https://cognition.ai/introducing-devin"),
    ("2024-05-13", "GPT-4o", "assistente", "https://openai.com/index/hello-gpt-4o/"),
    ("2025-02-24", "Claude Code (prévia) + Claude 3.7 Sonnet", "agente", "https://www.anthropic.com/news/claude-3-7-sonnet"),
    ("2025-04-16", "OpenAI Codex CLI (agente de terminal) + o3/o4-mini", "agente", "https://techcrunch.com/2025/04/16/openai-debuts-codex-cli-an-open-source-coding-tool-for-terminals"),
    ("2025-05-16", "OpenAI Codex (agente de código na nuvem)", "agente", "https://openai.com/index/introducing-codex/"),
    ("2025-05-19", "GitHub Copilot coding agent (Microsoft Build)", "agente", "https://github.com/newsroom/press-releases/coding-agent-for-github-copilot"),
    ("2025-05-22", "Claude 4 e Claude Code disponível geral (GA)", "agente", "https://www.anthropic.com/news/claude-4"),
    ("2025-08-07", "GPT-5", "agente", "https://openai.com/index/introducing-gpt-5/"),
    ("2025-11-24", "Claude Opus 4.5", "agente", "https://www.anthropic.com/news/claude-opus-4-5"),
]


def marcos() -> pd.DataFrame:
    df = pd.DataFrame(MARCOS, columns=["data", "marco", "onda", "fonte"])
    df["data"] = pd.to_datetime(df["data"])
    return df.sort_values("data").reset_index(drop=True)


def melhor_acumulado(df: pd.DataFrame, col_data: str, col_valor: str, nome: str,
                     inicio: pd.Timestamp = INICIO, fim: pd.Timestamp | None = None) -> pd.Series:
    """Série mensal do melhor valor acumulado (cummax + ffill)."""
    fim = fim if fim is not None else ultimo_mes_completo()
    d = df[[col_data, col_valor]].dropna().copy()
    d["mes"] = pd.to_datetime(d[col_data]).dt.to_period("M").dt.to_timestamp()
    por_mes = d.groupby("mes")[col_valor].max()
    idx = pd.date_range(inicio, fim, freq="MS")
    s = por_mes.reindex(idx).ffill().cummax()  # ffill só após a 1ª medição
    s.index.name = "mes"
    return s.rename(nome)


def carregar(baixar: bool = True) -> pd.DataFrame:
    pasta = RAW / "epoch"
    pasta.mkdir(parents=True, exist_ok=True)
    arq = pasta / "benchmark_data.zip"
    if baixar or not arq.exists():
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        arq.write_bytes(urllib.request.urlopen(req, timeout=120).read())
    z = zipfile.ZipFile(io.BytesIO(arq.read_bytes()))
    swe = pd.read_csv(z.open("swe_bench_verified.csv"), encoding_errors="replace")
    col = "Best score (across scorers)" if "Best score (across scorers)" in swe.columns else "mean_score"
    swe["pct"] = swe[col] * 100
    metr = pd.read_csv(z.open("metr_time_horizons_external.csv"), encoding_errors="replace")
    a = melhor_acumulado(swe, "Release date", "pct", "swe_bench_melhor")
    b = melhor_acumulado(metr, "Release date", "Time horizon", "metr_horizonte_min_melhor")
    return pd.concat([a, b], axis=1).reset_index()


def main() -> None:
    df = carregar()
    df.to_csv(PROCESSED / "capacidade_mensal.csv", index=False)
    print(df.head(), df.tail(), sep="\n")
    m = marcos()
    m.to_csv(PROCESSED / "marcos.csv", index=False)
    print(m[["data", "marco", "onda"]])


if __name__ == "__main__":
    main()
