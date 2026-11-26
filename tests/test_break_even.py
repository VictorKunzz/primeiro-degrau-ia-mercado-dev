import pytest

from src.models.break_even import break_even, cenarios


def test_break_even_valores():
    assert break_even(50, 100, 0, 0.5) == pytest.approx(0.0)
    assert break_even(50, 100, 50, 0.5) == pytest.approx(0.5)


def test_compensa_quando_ganho_supera_u_star():
    tab = cenarios(50, 100, custo_ia_ano=0.0)
    for _, r in tab.iterrows():
        assert r["compensa"] == (r["ganho"] >= r["u_star"] - 1e-12)
        # consistência: compensa <=> custo por unidade com IA <= com júnior
        assert r["compensa"] == (r["custo_unidade_sr_ia"] <= r["custo_unidade_sr_jr"] + 1e-9)
    central = tab[(tab.cenario == "central") & (tab.prod_jr == 0.5)].iloc[0]
    assert central["compensa"]  # u* = 0, ganho 0,15


def test_cenarios_estrutura():
    tab = cenarios(80000, 166000)
    assert set(tab.cenario) == {"pessimista", "central", "otimista", "estresse_METR"}
    assert set(tab.prod_jr) == {0.3, 0.5, 0.7}
    assert len(tab) == 12
