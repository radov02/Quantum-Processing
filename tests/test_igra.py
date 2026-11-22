"""Igra GHZ: pravila, klasična meja, kvantna strategija na idealnem simulatorju, analiza in shema čipa."""

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import ghz
import igra


def test_cilj_in_veljavnost():
    assert igra.cilj([0, 0, 0]) == 0
    assert igra.cilj([1, 1, 0]) == 1
    assert igra.cilj([1, 1, 1, 1]) == 0
    with pytest.raises(ValueError):
        igra.cilj([1, 0, 0])  # liha vsota ni veljaven vhod
    with pytest.raises(ValueError):
        igra.cilj([2, 0, 0])


def test_delez_zmag_na_rocnih_stetjih():
    # vhod 110: zmaga liha pariteta odgovorov
    assert igra.delez_zmag({"001": 30, "111": 20, "000": 50}, [1, 1, 0]) == pytest.approx(0.5)
    assert igra.delez_zmag({"000": 10}, [1, 1, 0]) == 0.0
    assert igra.delez_zmag({"000": 10}, [0, 0, 0]) == 1.0


@pytest.mark.parametrize("n", [3, 4, 5, 6, 7, 8])
def test_klasicna_meja_se_ujema_z_izcrpnim_pregledom(n):
    assert igra.klasicna_meja_izcrpno(n) == pytest.approx(igra.klasicna_meja(n))


def test_klasicna_meja_zavrne_manj_kot_tri_igralce():
    with pytest.raises(ValueError):
        igra.klasicna_meja(2)


def test_izcrpni_pregled_zavrne_prevelik_n():
    with pytest.raises(ValueError, match="le do N = 16"):
        igra.klasicna_meja_izcrpno(igra.NAJVEC_IZCRPNO + 1)


def test_vhodi_so_veljavni_in_razlicni():
    rng = np.random.default_rng(1)
    majhni = igra.vhodi(3, 8, rng)
    assert len(majhni) == 8 and {tuple(x) for x in majhni} == {(0, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1)}
    veliki = igra.vhodi(20, 8, rng)
    assert len({tuple(x) for x in veliki}) == 8 and all(igra.veljaven(x) and len(x) == 20 for x in veliki)


@pytest.mark.parametrize("zasnova", [
    ghz.ghz_drevo(list(range(3)), [None, 0, 1]),
    ghz.ghz_drevo(list(range(6)), [None, 0, 1, 2, 3, 4]),
    ghz.ghz_drevo(list(range(5)), [None, 0, 0, 1, 2]),
])
def test_kvantna_strategija_idealno_vedno_zmaga(zasnova):
    n = zasnova.priprava.num_qubits
    xs = igra.vhodi(n, 2 ** (n - 1), np.random.default_rng(0))  # vsi veljavni vhodi
    vezja = [igra.vezje_igre(zasnova.priprava, x) for x in xs]
    stetja = ghz.simulator().pozeni(vezja, 200)
    assert all(igra.delez_zmag(s, x) == 1.0 for s, x in zip(stetja, xs))


def test_brez_vrat_s_strategija_ne_zmaga_vedno():
    # brez vrat S igralci pri vhodu 110 vedno izgubijo
    z = ghz.ghz_drevo(list(range(3)), [None, 0, 1])
    napacno = igra.vezje_igre(z.priprava, [0, 0, 0])
    stetje = ghz.simulator().pozeni([napacno], 200)[0]
    assert igra.delez_zmag(stetje, [1, 1, 0]) == 0.0


def test_vezje_zavrne_neveljaven_vhod():
    z = ghz.ghz_drevo(list(range(3)), [None, 0, 1])
    with pytest.raises(ValueError):
        igra.vezje_igre(z.priprava, [1, 0, 0])
    with pytest.raises(ValueError):
        igra.vezje_igre(z.priprava, [1, 1])


def test_statistika_na_rocnih_vrednostih():
    s = igra.statistika([0.6, 0.8], 100)
    assert s["povprecje"] == pytest.approx(0.7)
    assert s["sigma"] == pytest.approx(0.1)  # prevlada razpršenost med vhodoma
    enaka = igra.statistika([0.5, 0.5], 100)
    assert enaka["sigma"] == pytest.approx(math.sqrt(0.25 / 200))  # brez razpršenosti ostane binomska napaka


def _rocni_posel(stetja, shots=10):
    return {"shots": shots, "posli": [{"job_id": "x", "stetja": stetja, "poskusi": [
        {"n": 3, "zacetek": 0, "st_vezij": 2, "vhodi": [[0, 0, 0], [1, 1, 0]], "postavitev": [0, 1, 2],
         "lastnosti_pred": {}, "lastnosti_po": {}},
        {"n": 4, "zacetek": 2, "st_vezij": 1, "vhodi": [[1, 1, 1, 1]], "postavitev": [0, 1, 2, 3],
         "lastnosti_pred": {}, "lastnosti_po": {}}]}]}


def test_analiza_qpu_razreze_stetja_po_poskusih():
    vrstice = igra.analiza_qpu(_rocni_posel([{"000": 10}, {"000": 10}, {"0011": 10}]))
    assert [v["n"] for v in vrstice] == [3, 4]
    assert vrstice[0]["zmage"] == [1.0, 0.0] and vrstice[0]["povprecje"] == pytest.approx(0.5)
    assert vrstice[1]["zmage"] == [1.0]


def test_analiza_qpu_zavrne_manjkajoce_izide_in_napacno_stevilo_posnetkov():
    with pytest.raises(ValueError):
        igra.analiza_qpu(_rocni_posel([{"000": 10}, {"000": 10}]))  # zadnjemu N manjka slovar izidov
    with pytest.raises(ValueError):
        igra.analiza_qpu(_rocni_posel([{"000": 10}, {"000": 9}, {"0011": 10}]))  # 9 namesto 10 posnetkov


@pytest.mark.parametrize("n", [3, 4])
def test_klasicna_meja_neodvisno_po_vseh_strategijah(n):
    # pregled vseh 4^n kombinacij štirih funkcij enega bita, brez funkcij iz igra.py
    vhodi_ = [x for x in itertools.product((0, 1), repeat=n) if sum(x) % 2 == 0]
    funkcije = [(0, 0), (0, 1), (1, 0), (1, 1)]  # (odgovor pri x=0, odgovor pri x=1)
    najboljsa = 0.0
    for strategija in itertools.product(funkcije, repeat=n):
        zmag = sum(sum(f[b] for f, b in zip(strategija, x)) % 2 == (sum(x) // 2) % 2 for x in vhodi_)
        najboljsa = max(najboljsa, zmag / len(vhodi_))
    assert najboljsa == pytest.approx(igra.klasicna_meja(n))


def test_vsi_odgovorijo_1_zmagajo_pri_treh_od_stirih_vhodov():
    """Strategija "vsak odgovori 1" pri N = 3 izgubi le pri vhodu 000 in doseže klasično mejo 3/4."""
    xs = igra.vhodi(3, 4, np.random.default_rng(0))  # vsi štirje veljavni vhodi
    zmage = [igra.delez_zmag({"111": 1}, x) for x in xs]
    assert sorted(map(tuple, xs)) == [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
    assert sum(zmage) == 3 and zmage[xs.index([0, 0, 0])] == 0.0
    assert sum(zmage) / 4 == igra.klasicna_meja(3)


def test_zaokrozevanje_tabele_polovica_navzgor():
    """0,53125 -> 0,5313 in 0,8115 -> 0,812."""
    assert igra._st(igra.klasicna_meja(10), 4) == "0{,}5313"
    assert igra._st(0.8115) == "0{,}812"
    assert igra._z_napako(0.8115, 0.0085) == "0{,}812 $\\pm$ 0{,}009"
    assert igra._z_napako(0.94875, 0.0030) == "0{,}949 $\\pm$ 0{,}003"


@pytest.fixture(scope="module")
def fez():
    from qiskit_ibm_runtime.fake_provider import FakeFez

    return FakeFez()


@pytest.mark.parametrize("n", [5, 20, 75])
def test_shema_cipa_obnovi_drevo_iz_postavitve(fez, n):
    """Povezave FakeFez ustrezajo shemi čipa; drevo, obnovljeno iz postavitve in uporabnih povezav, je enako drevesu
    iz `najdi_drevo`."""
    povezave = sorted({tuple(sorted(p)) for p in fez.coupling_map.get_edges()})
    xy = igra.koordinate_cipa(povezave, fez.num_qubits)
    assert len(set(xy.values())) == fez.num_qubits == 156
    sosedi = ghz.napake(fez)[2]
    uporabne = sorted({tuple(sorted((a, b))) for a in sosedi for b in sosedi[a]})
    postavitev, starsi = ghz.najdi_drevo(fez, n)
    assert igra.drevo_iz_postavitve(postavitev, uporabne) == starsi


def test_shema_cipa_zavrne_povezavo_med_nesosedoma():
    with pytest.raises(ValueError):
        igra.koordinate_cipa([(0, 2)])
