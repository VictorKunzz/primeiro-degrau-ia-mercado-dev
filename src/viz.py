"""Tema visual e helpers de gráfico do notebook.

Regras (skill dataviz): uma cor de destaque, contexto em cinza, linhas de 2px,
grade recessiva, um eixo só, título que afirma a conclusão, rótulo direto
seletivo. Paleta categórica (ordem fixa) da paleta de referência validada.
"""
from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

SUPERFICIE = "#fcfcfb"
TEXTO = "#0b0b0b"
TEXTO_2 = "#52514e"
TEXTO_3 = "#8a8984"
GRADE = "#e6e5e0"

DESTAQUE = "#2a78d6"   # série principal (azul, slot 1)
CONTRASTE = "#eb6834"  # contrafactual / era da IA (laranja, slot 2)
TERCEIRA = "#1baf7a"   # aqua, slot 3
CONTEXTO = "#b9b8b2"   # séries de contexto (cinza)
CATEGORICA = [DESTAQUE, CONTRASTE, TERCEIRA, "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

CORES_ONDA = {"chat": "#8a8984", "assistente": "#6250d6", "agente": "#e34948"}


def aplicar_tema() -> None:
    plt.rcParams.update({
        "figure.figsize": (11, 5.2),
        "figure.dpi": 110,
        "figure.facecolor": SUPERFICIE,
        "axes.facecolor": SUPERFICIE,
        "axes.edgecolor": GRADE,
        "axes.labelcolor": TEXTO_2,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.titlecolor": TEXTO,
        "axes.titlelocation": "left",
        "axes.titlepad": 26,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRADE,
        "grid.linewidth": 0.8,
        "xtick.color": TEXTO_2,
        "ytick.color": TEXTO_2,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "font.family": "DejaVu Sans",
    })


def titulo(ax, afirmacao: str, subtitulo: str | None = None) -> None:
    """Título que afirma a conclusão + subtítulo explicativo em cinza."""
    ax.set_title(afirmacao)
    if subtitulo:
        ax.text(0, 1.02, subtitulo, transform=ax.transAxes, fontsize=10.5,
                color=TEXTO_2, va="bottom", ha="left")


def fonte(fig, texto: str) -> None:
    fig.text(0.01, -0.02, f"Fonte: {texto}", fontsize=8.5, color=TEXTO_3, ha="left", va="top")


def eixo_tempo(ax, anos: int = 2) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator(anos))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def marcos(ax, df_marcos: pd.DataFrame, rotulos: bool = True, y_frac: float = 0.97,
           apenas: list[str] | None = None) -> None:
    """Linhas verticais finas nos marcos de IA, coloridas pela onda (chat/assistente/agente)."""
    xmin, xmax = ax.get_xlim()
    for i, r in df_marcos.reset_index(drop=True).iterrows():
        d = pd.Timestamp(r["data"])
        if not (xmin <= mdates.date2num(d) <= xmax):
            continue
        if apenas is not None and r["marco"] not in apenas:
            continue
        cor = CORES_ONDA.get(r.get("onda", "chat"), TEXTO_3)
        ax.axvline(d, color=cor, lw=1, ls=(0, (3, 3)), alpha=0.85, zorder=1)
        if rotulos:
            ax.text(d, y_frac - 0.07 * (i % 3), f" {r['marco']}", transform=ax.get_xaxis_transform(),
                    fontsize=8.5, color=cor, va="top", ha="left")


def faixa_ic(ax, cf: pd.DataFrame, rotulo: str = "Sem IA (projeção, IC 95%)") -> None:
    """Contrafactual: linha tracejada + faixa do IC 95%."""
    ax.fill_between(cf.index, cf["lo95"], cf["hi95"], color=CONTRASTE, alpha=0.15, lw=0)
    ax.plot(cf.index, cf["previsto"], color=CONTRASTE, ls="--", lw=2, label=rotulo)


def rotulo_final(ax, serie: pd.Series, texto: str, cor: str = TEXTO, dx_dias: int = 40) -> None:
    """Rótulo direto no último ponto da série."""
    x, y = serie.index[-1], serie.iloc[-1]
    ax.annotate(texto, (x, y), xytext=(x + pd.Timedelta(days=dx_dias), y), fontsize=10,
                color=cor, va="center", annotation_clip=False)
