"""Posel igre GHZ in proračun procesorskega časa v qpu.poslji brez stika z IBM (posel, storitev in Sampler so
nadomestki).

Glavni scenarij: obračun posla ostane "pending" (job.usage() = 0), zato skupna rezervacija ne sme pasti.
"""

import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
from qiskit_ibm_runtime.fake_provider import FakeFez

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import ghz  # noqa: E402
import ibm_skupno  # noqa: E402
import igra  # noqa: E402
import qpu  # noqa: E402

BACKEND = FakeFez()
PREOSTANEK = {"usage_remaining_seconds": 598}

class _Pub:
    def __init__(self, n, shots):
        self.n, self.shots = n, shots

    def join_data(self):
        n, shots = self.n, self.shots

        class _D:
            def get_counts(self_inner):
                return {"0" * n: shots // 2, "1" * n: shots - shots // 2}
        return _D()

class _Job:
    def __init__(self, vezja, napaka=None, shots=1000):
        self.vezja, self.shots = vezja, shots
        self.napaka = napaka
        self.preklici = 0

    def job_id(self):
        return f"job{id(self)}"

    def result(self):
        if self.napaka is not None:
            raise self.napaka
        return [_Pub(qc.num_clbits, self.shots) for qc in self.vezja]

    def usage(self):
        return 0  # obračun še "pending"

    def metrics(self):
        return {"usage": {"status": "pending"}}

    def cancel(self):
        self.preklici += 1

@pytest.fixture
def okolje(monkeypatch, tmp_path):
    """Nadomestni Sampler zapiše omejitev vsakega oddanega posla (seznam) in posel (okolje.posli); dnevnik in obračun
    sta izklopljena. okolje.napaka: izjema, ki jo vrže job.result() (neuspel posel)."""
    class _Seznam(list):
        pass

    omejitve = _Seznam()
    omejitve.posnetki, omejitve.posli, omejitve.napaka = [], [], None

    class _Sampler:
        def __init__(self, mode, options):
            omejitve.append(options["max_execution_time"])

        def run(self, pubi, shots=None):
            assert shots is not None and not any(isinstance(p, tuple) for p in pubi)  # eno število posnetkov za vsa vezja
            omejitve.posnetki.append([shots for _ in pubi])
            omejitve.posli.append(_Job(pubi, omejitve.napaka, shots))
            return omejitve.posli[-1]

    import qiskit_ibm_runtime
    monkeypatch.setattr(qiskit_ibm_runtime, "SamplerV2", _Sampler)
    monkeypatch.setattr(qpu, "ocena_ibm", lambda *a, **k: None)
    monkeypatch.setattr(qpu, "dejanska_poraba", lambda *a, **k: None)
    monkeypatch.setattr(qpu, "dodaj_v_dnevnik", lambda *a, **k: None)
    monkeypatch.setattr(qpu, "dopolni_zapis", lambda *a, **k: None)
    monkeypatch.setattr(qpu, "IZHOD", tmp_path)
    monkeypatch.setattr(qpu, "beri_dnevnik", lambda: [])
    return omejitve

@pytest.fixture
def dnevnik(okolje, monkeypatch, tmp_path):
    """Pravi dnevnik iz ibm_skupno (brez nadomestkov iz `okolje`) v začasni mapi; vrne pot do naloge.jsonl."""
    monkeypatch.setattr(ibm_skupno, "REZULTATI", tmp_path)
    monkeypatch.setattr(ibm_skupno, "DNEVNIK", tmp_path / "naloge.jsonl")
    for ime in ("beri_dnevnik", "dodaj_v_dnevnik", "dopolni_zapis"):
        monkeypatch.setattr(qpu, ime, getattr(ibm_skupno, ime))
    return tmp_path / "naloge.jsonl"

def _argumenti(meja, **k):
    nastavitve = {"max_sekund": meja, "fake": False, "ime": "t", "shots": 1000, "k": 2, "seme": 5,
                  "max_poslov": None, "faktor_omejitve": 2.0, "brez_kubitov": []}
    nastavitve.update(k)
    return Namespace(**nastavitve)

@pytest.fixture(scope="module")
def posel_3_5():
    return qpu.zgradi_posel_igre(BACKEND, [3, 5], 2, 5)

def test_posel_igre(posel_3_5):
    """N = 3 in 5, 2 vhoda: zaporedni odmiki, 4 vezja brez kalibracijskih, vhodi iz generatorja s semenom 5 + N,
    N - 1 dvokubitnih vrat in meritve na izbranih kubitih."""
    opis, vezja = posel_3_5
    p3, p5 = opis["poskusi"]
    assert (p3["n"], p5["n"]) == (3, 5)
    assert (p3["zacetek"], p3["st_vezij"], p5["zacetek"], p5["st_vezij"]) == (0, 2, 2, 2)
    assert len(vezja) == 4
    assert not any("kal" in qc.name for qc in vezja)
    for p in (p3, p5):
        n = p["n"]
        assert p["vhodi"] == igra.vhodi(n, 2, np.random.default_rng(5 + n))
        for qc in vezja[p["zacetek"]:p["zacetek"] + p["st_vezij"]]:
            assert qc.num_clbits == n
            assert qpu.st_dvokubitnih(qc) == n - 1
            assert qpu.merjeni_kubiti(qc) == dict(enumerate(p["postavitev"]))

def _trije_kubiti_v_vrsti():
    """Trije zaporedno povezani kubiti čipa a - b - c in kubit, ki ni sosed kubita b."""
    sosedi = {}
    for p, q in BACKEND.coupling_map.get_edges():
        sosedi.setdefault(p, set()).add(q)
        sosedi.setdefault(q, set()).add(p)
    b = next(q for q in sorted(sosedi) if len(sosedi[q]) >= 2)
    a, c = sorted(sosedi[b])[:2]
    dalec = next(q for q in sorted(sosedi) if q not in sosedi[b] | {a, b, c})
    return [a, b, c], dalec

def test_posel_zavrne_vezje_z_dodanimi_vrati(monkeypatch):
    """Drevo s povezavo, ki je na čipu ni: prevajalnik doda SWAP, gradnja posla se ustavi pred oddajo."""
    (a, b, _), dalec = _trije_kubiti_v_vrsti()
    monkeypatch.setattr(ghz, "najdi_drevo", lambda backend, n: ([a, b, dalec], [None, 0, 1]))
    with pytest.raises(RuntimeError, match="dodal vrata"):
        qpu.zgradi_posel_igre(BACKEND, [3], 1, 5)

def test_posel_zavrne_meritve_na_drugih_kubitih(monkeypatch):
    """Prevajanje na obrnjen vrstni red kubitov ohrani 2 dvokubitni vrati, meri pa druge kubite, kot jih je izbrala izbira."""
    vrsta, _ = _trije_kubiti_v_vrsti()
    pravi = ghz.prevedi
    monkeypatch.setattr(ghz, "najdi_drevo", lambda backend, n: (vrsta, [None, 0, 1]))
    monkeypatch.setattr(ghz, "prevedi", lambda vezja, backend, postavitev: pravi(vezja, backend, postavitev[::-1]))
    with pytest.raises(RuntimeError, match="meritve niso"):
        qpu.zgradi_posel_igre(BACKEND, [3], 1, 5)

def test_neobracunan_prejsnji_posel_ustavi_oddajo(okolje, monkeypatch, posel_3_5):
    monkeypatch.setattr(qpu, "beri_dnevnik", lambda: [{"job_id": "stari", "ime_poskusa": "prejsnji", "fake": False,
                                                       "porabljen_cas_s": None}])
    with pytest.raises(SystemExit, match="brez obračuna"):
        qpu.poslji(_argumenti(100), None, BACKEND, PREOSTANEK, [posel_3_5])
    assert okolje == []

def test_dnevnik_od_oddaje_do_obracuna(okolje, dnevnik, posel_3_5):
    """Brez nadomestkov dnevnika: oddaja zapiše posel brez obračuna, zato se naslednja oddaja ustavi; po obračunu
    (dopolni_zapis) gre, enak obračun drugič pa ne doda vrstice."""
    job_id = qpu.poslji(_argumenti(100, ime="a"), None, BACKEND, PREOSTANEK, [posel_3_5])["posli"][0]["job_id"]
    [z] = ibm_skupno.beri_dnevnik()
    assert (z["job_id"], z["ime_poskusa"], z["fake"], z["porabljen_cas_s"], z["st_vezij"]) == (job_id, "a", False,
                                                                                                None, 4)
    with pytest.raises(SystemExit, match="brez obračuna"):
        qpu.poslji(_argumenti(100, ime="b"), None, BACKEND, PREOSTANEK, [posel_3_5])
    assert len(okolje) == 1
    ibm_skupno.dopolni_zapis(job_id, 21.0)
    ibm_skupno.dopolni_zapis(job_id, 21.0)
    assert len(dnevnik.read_text().splitlines()) == 2
    assert ibm_skupno.beri_dnevnik()[0]["porabljen_cas_s"] == 21.0
    qpu.poslji(_argumenti(100, ime="b"), None, BACKEND, PREOSTANEK, [posel_3_5])
    assert len(okolje) == 2

def test_prevzemi_dopolni_zapis_in_ne_prepise_popolnega(okolje, dnevnik, monkeypatch, tmp_path, posel_3_5):
    """Prekinjeno čakanje (ni izidov ne obračuna): prevzemi ju zapiše v datoteko in dnevnik; drugič je zapis popoln,
    zato ga ne prepiše in IBM ne kliče."""
    job_id = qpu.poslji(_argumenti(100, ime="a"), None, BACKEND, PREOSTANEK, [posel_3_5])["posli"][0]["job_id"]
    pot = tmp_path / "a.json"
    zapis = json.loads(pot.read_text())
    zapis["posli"][0]["stetja"] = None
    pot.write_text(json.dumps(zapis))
    klici = []

    class _Storitev:
        def job(self, jid):
            klici.append(jid)
            return okolje.posli[0]

    monkeypatch.setattr(qpu, "dejanska_poraba", lambda *a, **k: 21.0)
    qpu.prevzemi(Namespace(ime="a"), _Storitev())
    shranjeno = json.loads(pot.read_text())["posli"][0]
    assert klici == [job_id] and shranjeno["poraba_s"] == 21.0 and len(shranjeno["stetja"]) == 4
    assert ibm_skupno.beri_dnevnik()[0]["porabljen_cas_s"] == 21.0
    monkeypatch.setattr(qpu, "shrani", lambda *a, **k: pytest.fail("popoln zapis je bil prepisan"))
    qpu.prevzemi(Namespace(ime="a"), _Storitev())
    assert klici == [job_id]

def test_omejitev_ni_nikoli_pod_oceno():
    with pytest.raises(SystemExit, match="manjša od ocene"):
        qpu.omejitev_posla(Namespace(faktor_omejitve=0.1), 20.0)
    assert qpu.omejitev_posla(Namespace(faktor_omejitve=1.0), 20.0) >= 20

def test_vsota_omejitev_ne_preseze_meje_ko_obracun_caka(okolje, posel_3_5):
    """Vsak posel oceni 3,4 s in rezervira 10 s; z mejo 15 s gre le prvi, ker obračun ostane neznan, čeprav bi
    po ocenah (2 x 3,4 s) šla oba."""
    posli = [posel_3_5, posel_3_5]
    a = _argumenti(15)
    assert qpu.ocena_casa(2, sum(qpu.posnetkov_posla(v, a) for _, v in posli)) < 15
    podatki = qpu.poslji(a, None, BACKEND, PREOSTANEK, posli)
    assert okolje == [10]  # drugi posel ne gre, ker prvi še drži celotno omejitev
    assert podatki["rezervirano_s"] <= 15
    assert okolje.posnetki == [[1000] * 4]  # vsa vezja z enakim številom posnetkov

def test_prevzemi_ohrani_razlog_preklica(okolje, tmp_path):
    """Posel, ki ga je varovalo preklicalo (ocena IBM nad omejitvijo): poznejši prevzemi ne prepiše razloga."""
    zapis = {"ime": "t", "backend": "x", "shots": 10, "posli": [{"job_id": "j1", "stetja": None, "poraba_s": None,
                                                                 "stanje": "CANCELLED (ocena IBM nad omejitvijo)",
                                                                 "poskusi": []}]}
    (tmp_path / "t.json").write_text(json.dumps(zapis))

    class _Storitev:
        def job(self, jid):
            return _Job([], RuntimeError("job cancelled"))

    podatki = qpu.prevzemi(Namespace(ime="t"), _Storitev())
    assert podatki["posli"][0]["stanje"] == "CANCELLED (ocena IBM nad omejitvijo)"

def test_ocena_ibm_nad_omejitvijo_preklice_posel(okolje, monkeypatch, tmp_path, posel_3_5):
    monkeypatch.setattr(qpu, "ocena_ibm", lambda *a, **k: okolje[-1] + 1)
    with pytest.raises(SystemExit, match="posel preklican"):
        qpu.poslji(_argumenti(100), None, BACKEND, PREOSTANEK, [posel_3_5, posel_3_5])
    assert len(okolje) == 1 and okolje.posli[0].preklici == 1  # drugi posel ni bil poslan
    zapis = json.loads((tmp_path / "t.json").read_text())["posli"]
    assert len(zapis) == 1 and zapis[0]["stanje"] == "CANCELLED (ocena IBM nad omejitvijo)"

def test_neuspel_posel_ustavi_oddajo(okolje, tmp_path, posel_3_5):
    okolje.napaka = RuntimeError("napaka na procesorju")
    with pytest.raises(SystemExit, match="ni uspel"):
        qpu.poslji(_argumenti(100), None, BACKEND, PREOSTANEK, [posel_3_5, posel_3_5])
    assert len(okolje) == 1  # drugi posel ni bil poslan
    zapis = json.loads((tmp_path / "t.json").read_text())["posli"]
    assert len(zapis) == 1 and zapis[0]["stanje"].startswith("NAPAKA") and zapis[0]["stetja"] is None

def test_zavrne_brez_meje_in_ob_dosezeni_porabi(okolje, posel_3_5):
    with pytest.raises(SystemExit, match="max-sekund"):
        qpu.poslji(_argumenti(None), None, BACKEND, PREOSTANEK, [posel_3_5])
    with pytest.raises(SystemExit, match="dosežena"):
        qpu.poslji(_argumenti(100), None, BACKEND, {**PREOSTANEK, "usage_limit_reached": True}, [posel_3_5])
    with pytest.raises(SystemExit, match="dosežena"):  # preostanka ni mogoče prebrati
        qpu.poslji(_argumenti(100), None, BACKEND, {}, [posel_3_5])
    assert okolje == []

def test_zavrne_obstojeco_datoteko_in_prevec_poslov(okolje, tmp_path, posel_3_5):
    posli = [posel_3_5, posel_3_5]
    (tmp_path / "t.json").write_text("{}")
    with pytest.raises(SystemExit, match="že obstaja"):
        qpu.poslji(_argumenti(100), None, BACKEND, PREOSTANEK, posli)
    (tmp_path / "t.json").unlink()
    with pytest.raises(SystemExit, match="odobrenih je 1"):
        qpu.poslji(_argumenti(100, max_poslov=1), None, BACKEND, PREOSTANEK, posli)
    with pytest.raises(SystemExit, match="presega preostanek"):  # meja nad preostankom instance
        qpu.poslji(_argumenti(100), None, BACKEND, {"usage_remaining_seconds": 50}, posli)
    assert okolje == []  # nič ni bilo poslano

def test_lokalni_zagon_ne_prepise_placanega(okolje, tmp_path, posel_3_5):
    placani = tmp_path / "t.json"
    placani.write_text(json.dumps({"fake": False, "posli": []}))
    with pytest.raises(SystemExit, match="lokalni preizkus"):
        qpu.poslji(_argumenti(100, fake=True), None, BACKEND, {}, [posel_3_5])
    assert json.loads(placani.read_text()) == {"fake": False, "posli": []}

def test_lokalna_pot_od_oddaje_do_analize(okolje, tmp_path, posel_3_5):
    """--fake: šumni simulator namesto procesorja, zapis v rezultati/qpu/<ime>.json in analiza deleža zmag."""
    podatki = qpu.poslji(_argumenti(100, fake=True, shots=50), None, BACKEND, {}, [posel_3_5])
    assert okolje == []  # Sampler ni bil uporabljen
    shranjeno = json.loads((tmp_path / "t.json").read_text())
    assert shranjeno["fake"] is True and shranjeno["posli"][0]["job_id"] == "lokalno-1"
    assert not list(tmp_path.glob("*.tmp"))  # shrani ne pusti začasne datoteke
    vrstice = igra.analiza_qpu(podatki)
    assert [v["n"] for v in vrstice] == [3, 5]
    for v in vrstice:
        assert len(v["zmage"]) == 2
        assert v["povprecje"] > v["klasicna_meja"]  # šumni simulator pri N = 3 in 5 zmaguje daleč nad mejo
