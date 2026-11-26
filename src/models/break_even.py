"""Conta do gestor (design §7.2): quando sênior+IA sai mais barato que sênior+júnior?

Modelo por unidade de trabalho (1 unidade = produção anual de um sênior sem IA):
- Equipe A (sênior + júnior): custo sal_sr + sal_jr, produz 1 + prod_jr unidades.
- Equipe B (sênior + IA): custo sal_sr + custo_ia_ano, produz 1 + u unidades.
B compensa quando (sal_sr + c) / (1 + u) <= (sal_sr + sal_jr) / (1 + prod_jr),
ou seja, quando u >= u* = (sal_sr + c)(1 + prod_jr) / (sal_sr + sal_jr) - 1.

Simplificações explícitas: só salário (sem encargos, mentoria, rotatividade),
júnior sem custo de supervisão, IA sem curva de aprendizado. É uma conta de
cenários, não uma estimativa causal.
"""
from __future__ import annotations

import pandas as pd

CENARIOS = {"pessimista": 0.00, "central": 0.15, "otimista": 0.30, "estresse_METR": -0.19}
PROD_JR_SENS = (0.3, 0.5, 0.7)


def break_even(sal_jr: float, sal_sr: float, custo_ia_ano: float, prod_jr: float = 0.5) -> float:
    """Ganho mínimo de produtividade u* do sênior com IA para empatar com sênior+júnior."""
    return (sal_sr + custo_ia_ano) * (1 + prod_jr) / (sal_sr + sal_jr) - 1


def cenarios(sal_jr: float, sal_sr: float, custo_ia_ano: float = 2400.0) -> pd.DataFrame:
    """Tabela cenário × prod_jr com custo por unidade e se sênior+IA compensa."""
    linhas = []
    for prod_jr in PROD_JR_SENS:
        u_star = break_even(sal_jr, sal_sr, custo_ia_ano, prod_jr)
        custo_jr = (sal_sr + sal_jr) / (1 + prod_jr)
        for nome, ganho in CENARIOS.items():
            custo_ia = (sal_sr + custo_ia_ano) / (1 + ganho)
            linhas.append({
                "cenario": nome, "ganho": ganho, "prod_jr": prod_jr,
                "custo_unidade_sr_ia": custo_ia, "custo_unidade_sr_jr": custo_jr,
                "compensa": bool(ganho >= u_star - 1e-12), "u_star": u_star,
                "sal_jr": sal_jr, "sal_sr": sal_sr, "custo_ia_ano": custo_ia_ano,
            })
    return pd.DataFrame(linhas)
