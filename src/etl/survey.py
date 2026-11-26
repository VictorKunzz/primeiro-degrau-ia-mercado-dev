"""ETL do Stack Overflow Developer Survey 2023-2026.

Saída: data/processed/survey_harmonizado.parquet com as colunas
ano, anos_exp, anos_exp_fonte, faixa_exp, cargo, emprego, empregado, empregado_integral,
salario_usd, pais, ia_usa, ia_sentimento, ia_confianca, ia_ameaca, usa_agentes.

Decisões de harmonização (schema real conferido nos cabeçalhos de cada ano):
- anos_exp: somente WorkExp (existe nos 4 anos; sem reserva por YearsCodePro, para
  manter a mesma definição). "Less than 1 year" -> 0.5; "More than 50 years" -> 51;
  teto 51. `anos_exp_fonte` = "WorkExp" quando há valor, NaN caso contrário.
- cargo: DevType (2023-2025, texto multi-seleção separado por ";") e Role (2026).
  Não são estritamente comparáveis; o texto original é mantido.
- emprego: texto original de Employment. Em 2023/24 há "Employed, full-time" e
  "Employed, part-time"; em 2025/26 a categoria foi fundida em "Employed".
  `empregado`: qualquer vínculo de emprego (comparável entre anos).
  `empregado_integral`: True se algum item é "Employed, full-time" (não conta
  "Student, full-time"); em 2025/26, "Employed" (que
  mistura integral e parcial) conta como True. NaN se Employment ausente.
  Cuidado: essa flag NÃO é estritamente comparável entre 2024 e 2025.
  O texto de `emprego` é comparável dentro de 2023-24 e dentro de 2025-26 (6 opções
  idênticas de seleção única), não entre os dois blocos.
- salario_usd: ConvertedCompYearly; fora de [1.000; 1.000.000] USD vira NaN (piso de
  1.000 USD aprovado: valores menores são ruído).
- COMPARAVEL: por coluna, anos em que o significado é idêntico (ver dicionário abaixo).
- ia_usa (bool): AISelect. 2023-2025: "Yes..." = True. 2026: AISelect virou
  multi-seleção de tipos de ferramenta; qualquer ferramenta = True, "I don't use" = False.
- ia_sentimento: AISent (2023-25) / AISentiment (2026), na escala
  Very favorable / Favorable / Indifferent / Unfavorable / Very unfavorable / Unsure.
- ia_confianca: AIAcc (2024, 2025; confiança na precisão) e AITrust (2026, ESCALA
  DIFERENTE, mantida como texto original). 2023: AIAcc era "benefícios" -> NaN.
- ia_ameaca: AIThreat (só 2024 e 2025); NaN nos demais.
- usa_agentes (bool): 2025: AIAgents ("Yes, I use AI agents ..." = True; demais
  respostas = False). 2026: entre quem respondeu AISelect, True se inclui
  "AI agents or automated workflows". 2023/2024: NaN (pergunta não existia).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.config import PROCESSED, RAW, RESULTS

BASE = "https://media.githubusercontent.com/media/StackExchange/Survey/main/packages/archive"
URLS = {
    2023: f"{BASE}/2023/results.csv",
    2024: f"{BASE}/2024/results.csv",
    2025: f"{BASE}/2025/results.csv",
    2026: f"{BASE}/2026/survey_results_public.csv",
}
RAW_SURVEY = RAW / "survey"
SAIDA = PROCESSED / "survey_harmonizado.parquet"

COLUNAS_FINAIS = [
    "ano", "anos_exp", "anos_exp_fonte", "faixa_exp", "cargo", "emprego", "empregado",
    "empregado_integral", "salario_usd", "pais", "ia_usa", "ia_sentimento",
    "ia_confianca", "ia_ameaca", "usa_agentes",
]
SALARIO_MIN, SALARIO_MAX = 1_000, 1_000_000

# ano -> coluna original -> nome harmonizado (conferido no cabeçalho de cada CSV).
# "ia_select" e "agentes_raw" são intermediários usados em harmonizar().
_COMUM = {"Employment": "emprego", "Country": "pais", "ConvertedCompYearly": "salario_usd",
          "WorkExp": "anos_exp", "AISelect": "ia_select"}
MAPA_COLUNAS: dict[int, dict[str, str]] = {
    2023: {**_COMUM, "DevType": "cargo", "AISent": "ia_sentimento"},
    2024: {**_COMUM, "DevType": "cargo", "AISent": "ia_sentimento",
           "AIAcc": "ia_confianca", "AIThreat": "ia_ameaca"},
    2025: {**_COMUM, "DevType": "cargo", "AISent": "ia_sentimento", "AIAcc": "ia_confianca",
           "AIThreat": "ia_ameaca", "AIAgents": "agentes_raw"},
    2026: {**_COMUM, "Role": "cargo", "AISentiment": "ia_sentimento", "AITrust": "ia_confianca"},
}

# Anos em que o significado de cada coluna harmonizada é idêntico (conferido nos schemas):
# - ia_usa: AISelect é sim/não em 2023-25 (2025 só adiciona frequência); em 2026 virou
#   multi-seleção de tipos de ferramenta.
# - usa_agentes: só existe em 2025 (AIAgents) e 2026 (item de AISelect): perguntas
#   diferentes, logo uso de ano único.
# - empregado_integral: 2025+ funde integral/parcial em "Employed".
# - ia_ameaca: AIThreat só em 2024-25. ia_confianca: AIAcc 2024-25 (2023 = benefícios,
#   2026 = AITrust, outra escala). ia_sentimento: 2026 foi recodificado/colapsado.
# - cargo: DevType (2023-25); Role (2026) é outra pergunta.
_TODOS = [2023, 2024, 2025, 2026]
COMPARAVEL: dict[str, list[int]] = {
    "anos_exp": _TODOS, "anos_exp_fonte": _TODOS, "faixa_exp": _TODOS,
    "cargo": [2023, 2024, 2025],
    # emprego: comparável DENTRO de 2023-24 (multi-seleção) e DENTRO de 2025-26 (6 opções
    # de seleção única, idênticas), mas NÃO entre 2024 e 2025.
    "emprego": [2023, 2024, 2025, 2026],
    "empregado": _TODOS,
    "empregado_integral": [2023, 2024],
    "salario_usd": _TODOS, "pais": _TODOS,
    "ia_usa": [2023, 2024, 2025],
    "ia_sentimento": [2023, 2024, 2025],
    "ia_confianca": [2024, 2025],
    "ia_ameaca": [2024, 2025],
    "usa_agentes": [],
}

_SENTIMENTO_2026 = {"Somewhat favorable": "Favorable", "Somewhat unfavorable": "Unfavorable",
                    "Neutral": "Indifferent", "Not sure": "Unsure"}


def baixar(ano: int) -> Path:
    """Baixa o CSV do ano (streaming) para data/raw/survey/<ano>.csv; pula se existir."""
    RAW_SURVEY.mkdir(parents=True, exist_ok=True)
    dest = RAW_SURVEY / f"{ano}.csv"
    if dest.exists():
        return dest
    tmp = dest.with_suffix(".part")
    with requests.get(URLS[ano], stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.replace(dest)
    return dest


def _anos(valor) -> float:
    """Converte texto/número de experiência para float."""
    if pd.isna(valor):
        return np.nan
    s = str(valor).strip()
    if s == "Less than 1 year":
        return 0.5
    if s == "More than 50 years":
        return 51.0
    try:
        return float(s)
    except ValueError:
        return np.nan


def _faixa(x: float):
    if pd.isna(x):
        return np.nan
    return "≤2" if x <= 2 else ("3–9" if x < 10 else "≥10")


def _col(df: pd.DataFrame, nome: str) -> pd.Series:
    return df[nome] if nome in df.columns else pd.Series(np.nan, index=df.index, dtype=object)


def _bool(mask: pd.Series, valido: pd.Series) -> pd.Series:
    return pd.Series(pd.array(mask.where(valido), dtype="boolean"), index=mask.index)


def harmonizar(df: pd.DataFrame, ano: int) -> pd.DataFrame:
    """Renomeia colunas do ano para o esquema comum e deriva as colunas padronizadas."""
    d = df.rename(columns=MAPA_COLUNAS[ano])
    out = pd.DataFrame(index=d.index)
    out["ano"] = ano

    exp = _col(d, "anos_exp").map(_anos)
    exp = exp.astype(float).clip(upper=51.0)
    out["anos_exp"] = exp
    out["anos_exp_fonte"] = pd.Series(np.where(exp.notna(), "WorkExp", None), index=d.index, dtype=object)
    out["faixa_exp"] = exp.map(_faixa)

    out["cargo"] = _col(d, "cargo")
    emp = _col(d, "emprego")
    out["emprego"] = emp
    tem = emp.notna()
    # Employment é multi-seleção ("a;b"): avalia cada item (evita contar "Student, full-time").
    itens = emp.fillna("").astype(object).map(lambda v: str(v).split(";"))
    out["empregado"] = _bool(itens.map(lambda L: any(i.strip().startswith("Employed") for i in L)), tem)
    integral = itens.map(lambda L: any(i.strip() in ("Employed, full-time", "Employed") for i in L))
    out["empregado_integral"] = _bool(integral, tem)

    sal = pd.to_numeric(_col(d, "salario_usd"), errors="coerce")
    out["salario_usd"] = sal.where((sal >= SALARIO_MIN) & (sal <= SALARIO_MAX))
    out["pais"] = _col(d, "pais")

    sel = _col(d, "ia_select")
    s = sel.astype(str)
    if ano == 2026:
        usa = ~s.str.startswith("I don")
        out["ia_usa"] = _bool(usa, sel.notna())
        out["usa_agentes"] = _bool(s.str.contains("AI agents or automated workflows", regex=False),
                                   sel.notna())
    else:
        out["ia_usa"] = _bool(s.str.startswith("Yes"), sel.notna())
        if ano == 2025:
            ag = _col(d, "agentes_raw")
            out["usa_agentes"] = _bool(ag.astype(str).str.startswith("Yes"), ag.notna())
        else:
            out["usa_agentes"] = pd.Series(pd.array([pd.NA] * len(d), dtype="boolean"), index=d.index)

    sent = _col(d, "ia_sentimento")
    out["ia_sentimento"] = sent.replace(_SENTIMENTO_2026) if ano == 2026 else sent
    out["ia_confianca"] = _col(d, "ia_confianca")
    out["ia_ameaca"] = _col(d, "ia_ameaca")
    return out[COLUNAS_FINAIS]


def carregar_ano(ano: int) -> pd.DataFrame:
    caminho = baixar(ano)
    return harmonizar(pd.read_csv(caminho, usecols=list(MAPA_COLUNAS[ano]), low_memory=False), ano)


def carregar_todos() -> pd.DataFrame:
    """Harmoniza 2023-2026, grava parquet e cobertura (% não-nulos por coluna e ano)."""
    df = pd.concat([carregar_ano(a) for a in sorted(URLS)], ignore_index=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SAIDA, index=False)
    cob = (df.drop(columns="ano").notna().groupby(df["ano"]).mean() * 100).round(1)
    cob.insert(0, "n", df.groupby("ano").size())
    cob.to_csv(RESULTS / "survey_cobertura.csv")
    pd.DataFrame(
        [{"coluna": c, "anos_comparaveis": ";".join(map(str, a)) or "nenhum (ano unico)"}
         for c, a in COMPARAVEL.items()]
    ).to_csv(RESULTS / "survey_comparabilidade.csv", index=False)
    return df


def main() -> None:
    df = carregar_todos()
    cob = pd.read_csv(RESULTS / "survey_cobertura.csv")
    print(cob.to_string(index=False))
    print(f"Gravado: {SAIDA} ({SAIDA.stat().st_size / 1e6:.1f} MB), {len(df)} linhas")


if __name__ == "__main__":
    sys.exit(main())
