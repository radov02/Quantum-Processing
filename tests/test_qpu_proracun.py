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
    nastavitve = {"max_sekund": meja, "fake": False, "ime": "t", "shots": 1000, "kopije_od": None, "k": 2, "seme": 5,
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


def test_shranjeni_dnevnik_ima_obe_obracunani_nalogi_igre():
    """rezultati/naloge.jsonl: obe nalogi igre, obe obračunani (sicer bi varovalo ustavilo vsako novo oddajo):
    igra 21 s (11. 11.) in igra2 34 s (17. 11.)."""
    zapisi = ibm_skupno.beri_dnevnik()
    assert len(zapisi) == 2
    assert all(z["porabljen_cas_s"] is not None for z in zapisi)
    assert {z["ime_poskusa"]: z["porabljen_cas_s"] for z in zapisi} == {"igra": 21, "igra2": 34}


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


@pytest.fixture(scope="module")
def posel_s_kalibracijo():
    """N = 3 in 5, 2 vhoda, vezja z N >= 5 trikrat, na koncu kalibracijski vezji: 2 + 6 + 2 = 10 vezij."""
    return qpu.zgradi_posel_igre(BACKEND, [3, 5], 2, 5, kalibracija=True, kopije_od=(5, 3))


def test_kopije_vezij_so_zapored_in_enake(posel_s_kalibracijo, posel_3_5):
    """--kopije-od 5 3: pri N = 5 je vsako od 2 vezij v poslu trikrat zapored (kopije istega vhoda skupaj), pri
    N = 3 enkrat; kopije so enaka vezja kot v poslu brez kopij."""
    opis, vezja = posel_s_kalibracijo
    p3, p5 = opis["poskusi"]
    assert (p3["zacetek"], p3["st_vezij"], p3["kopij"]) == (0, 2, 1)
    assert (p5["zacetek"], p5["st_vezij"], p5["kopij"]) == (2, 6, 3)
    assert len(p5["vhodi"]) == 2
    imena = [qc.name for qc in vezja[2:8]]
    assert [i.rsplit("_p", 1)[1] for i in imena] == ["0", "1", "2", "0", "1", "2"]
    assert len({i.rsplit("_p", 1)[0] for i in imena[:3]}) == 1 and imena[0].rsplit("_p", 1)[0] != imena[3].rsplit("_p", 1)[0]
    brez_kopij = posel_3_5[1][2:4]
    for j, izvirnik in enumerate(brez_kopij):
        for c in range(3):
            kopija = vezja[2 + 3 * j + c]
            assert kopija.count_ops() == izvirnik.count_ops()
            assert qpu.merjeni_kubiti(kopija) == qpu.merjeni_kubiti(izvirnik) == dict(enumerate(p5["postavitev"]))


def test_kalibracijski_vezji_merita_vse_izbrane_kubite(posel_s_kalibracijo):
    """Z --kalibracija sta na koncu posla dve vezji brez dvokubitnih vrat, ki merita unijo izbranih kubitov; prvo
    nima vrat X, drugo ima X na vsakem kubitu. Zapisana je objavljena napaka branja vsakega kubita."""
    opis, vezja = posel_s_kalibracijo
    kal = opis["kalibracija"]
    assert kal["kubiti"] == sorted({q for p in opis["poskusi"] for q in p["postavitev"]})
    assert (kal["zacetek"], kal["st_vezij"], len(vezja)) == (8, 2, 10)
    nic, ena = vezja[8:]
    assert (nic.name, ena.name) == ("kal_0", "kal_1")
    for qc in (nic, ena):
        assert qpu.st_dvokubitnih(qc) == 0
        assert qpu.merjeni_kubiti(qc) == dict(enumerate(kal["kubiti"]))
    assert nic.count_ops().get("x", 0) == 0 and ena.count_ops().get("x", 0) == len(kal["kubiti"])
    assert len(kal["objavljena_napaka"]) == len(kal["kubiti"]) and all(0 < e < 0.5 for e in kal["objavljena_napaka"])


def test_kalibracijsko_vezje_brez_vrat_x_se_zavrne(monkeypatch):
    """Če bi prevajanje izgubilo vrata X (ali jih dodalo vezju kal_0), se gradnja posla ustavi pred oddajo."""
    pravi = ghz.prevedi

    def brez_x(vezja, backend, postavitev):
        prevedena = pravi(vezja, backend, postavitev)
        for qc in prevedena:
            qc.data = [i for i in qc.data if i.operation.name != "x"]
        return prevedena

    monkeypatch.setattr(ghz, "prevedi", brez_x)
    with pytest.raises(RuntimeError, match="vrata niso na pričakovanih kubitih"):
        qpu.kalibracijski_vezji(BACKEND, [0, 1, 2])


def test_posel_s_kopijami_ocena_in_oddaja(okolje, posel_s_kalibracijo):
    """Vseh 10 vezij gre v en posel z enakim številom posnetkov; ocena in omejitev sledita številu vezij. Analiza
    sešteje kopije: pri N = 5 ima vsak vhod 3000 posnetkov."""
    opis, vezja = posel_s_kalibracijo
    a = _argumenti(100, kopije_od=[5, 3])
    assert qpu.posnetkov_posla(vezja, a) == 10000
    podatki = qpu.poslji(a, None, BACKEND, PREOSTANEK, [(opis, vezja)])
    assert okolje.posnetki == [[1000] * 10]
    zapis = podatki["posli"][0]
    assert zapis["ocena_s"] == round(qpu.ocena_casa(1, 10000), 2)
    assert okolje == [qpu.omejitev_posla(a, qpu.ocena_casa(1, 10000))]
    assert podatki["kopije_od"] == [5, 3] and len(zapis["imena_vezij"]) == 10
    vrstice = igra.analiza_qpu(podatki)
    assert [(v["n"], v["shots"], v["kopij"], len(v["zmage"])) for v in vrstice] == [(3, 1000, 1, 2), (5, 3000, 3, 2)]


def test_zavrne_posel_ko_se_stevilo_vezij_ne_ujema(okolje, posel_s_kalibracijo):
    opis, vezja = posel_s_kalibracijo
    with pytest.raises(SystemExit, match="opis pa jih navaja 10"):
        qpu.poslji(_argumenti(100), None, BACKEND, PREOSTANEK, [(opis, vezja[:-1])])
    assert okolje == []


def test_zavrne_oddajo_ko_zapis_ni_mogoc(okolje, monkeypatch, tmp_path, posel_3_5):
    """Mapa za zapise ni zapisljiva (tu: na njenem mestu je datoteka): oddaje ni, ker ID in izidi ne bi bili shranjeni."""
    ovira = tmp_path / "ovira"
    ovira.write_text("x")
    monkeypatch.setattr(qpu, "IZHOD", ovira)
    with pytest.raises(SystemExit, match="ni mogoče pisati"):
        qpu.poslji(_argumenti(100), None, BACKEND, PREOSTANEK, [posel_3_5])
    assert okolje == []


def test_lokalna_pot_s_kalibracijo_in_kopijami(okolje, tmp_path, posel_s_kalibracijo):
    """--fake s --kopije-od in --kalibracija: vsako vezje ima 200 posnetkov, analiza sešteje kopije, analiza branja
    vrne napako vsakega kubita (majhno, a ne nič, ker šumni simulator doda napako branja), shrani_analizo pa oboje
    zapiše v <ime>_analiza.json."""
    opis, vezja = posel_s_kalibracijo
    podatki = qpu.poslji(_argumenti(100, fake=True, shots=200, kopije_od=[5, 3]), None, BACKEND, {}, [(opis, vezja)])
    assert [sum(s.values()) for s in podatki["posli"][0]["stetja"]] == [200] * 10
    assert [v["shots"] for v in igra.analiza_qpu(podatki)] == [200, 600]
    branje = igra.analiza_branja(podatki)
    assert branje["kubiti"] == opis["kalibracija"]["kubiti"]
    assert len(branje["p10"]) == len(branje["p01"]) == len(branje["kubiti"])
    assert all(0 <= p < 0.2 for p in branje["p10"] + branje["p01"])
    assert 0 < branje["povprecje"] < 0.1 and branje["najslabsi_kubit"] in branje["kubiti"]
    pot = qpu.shrani_analizo("t", podatki)
    shranjeno = json.loads(pot.read_text())
    assert pot == tmp_path / "t_analiza.json" and shranjeno["branje"]["kubiti"] == branje["kubiti"]
    assert [(v["n"], v["shots"], v["kopij"]) for v in shranjeno["rezultati"]] == [(3, 200, 1), (5, 600, 3)]


def test_analiza_branja_na_rocnih_izidih():
    """Trije kubiti (čip 7, 9 in 4), 10 posnetkov: kubit 7 je klasični bit 0 (desni znak), kubit 4 bit 2 (levi znak).
    V stanju 0: kubit 7 dvakrat preberemo kot 1, kubit 4 enkrat. V stanju 1: kubit 9 trikrat preberemo kot 0."""
    kal = {"kubiti": [7, 9, 4], "zacetek": 0, "st_vezij": 2, "objavljena_napaka": [0.01, 0.02, 0.03]}
    podatki = {"shots": 10, "posli": [{"job_id": "x", "poskusi": [], "kalibracija": kal, "imena_vezij": ["kal_0", "kal_1"],
                                       "stetja": [{"000": 7, "001": 2, "100": 1}, {"111": 7, "101": 3}]}]}
    branje = igra.analiza_branja(podatki)
    assert branje["p10"] == [0.2, 0.0, 0.1] and branje["p01"] == pytest.approx([0.0, 0.3, 0.0])
    assert branje["najslabsi_kubit"] == 9 and branje["najvecja"] == pytest.approx(0.15)
    assert branje["objavljena_napaka"] == [0.01, 0.02, 0.03]
    assert igra.analiza_branja({"shots": 10, "posli": [{"job_id": "x", "poskusi": [], "stetja": []}]}) is None
    posel = podatki["posli"][0]
    posel["stetja"][1] = {"111": 9}
    with pytest.raises(ValueError, match="nimata 10 posnetkov"):
        igra.analiza_branja(podatki)
    posel["stetja"][1] = {"1111": 10}
    with pytest.raises(ValueError, match="nima 3 bitov"):
        igra.analiza_branja(podatki)
    posel["stetja"][1] = {"111": 10}
    posel["imena_vezij"] = ["kal_1", "kal_0"]
    with pytest.raises(ValueError, match="nista na pričakovanem mestu"):
        igra.analiza_branja(podatki)


def test_analiza_qpu_sesteje_kopije_istega_vhoda():
    """N = 3, vhoda 000 (zmaga: sodo število enic) in 110 (zmaga: liho), vsak v dveh kopijah ZAPORED. Prvi vhod:
    kopiji {000: 10} in {000: 10}; drugi vhod: {001: 10} in {111: 10}. Pravilno združevanje da delež zmag 1,0 pri
    obeh vhodih; združevanje s korakom (vezji 0 in 2, 1 in 3) bi dalo 0,5 in 0,5."""
    posel = {"job_id": "x", "stetja": [{"000": 10}, {"000": 10}, {"001": 10}, {"111": 10}],
             "imena_vezij": ["v_0_p0", "v_0_p1", "v_1_p0", "v_1_p1"],
             "poskusi": [{"n": 3, "postavitev": [0, 1, 2], "vhodi": [[0, 0, 0], [1, 1, 0]], "zacetek": 0, "st_vezij": 4,
                          "kopij": 2, "lastnosti_pred": {}, "lastnosti_po": {}}]}
    podatki = {"shots": 10, "posli": [posel]}
    [v] = igra.analiza_qpu(podatki)
    assert v["zmage"] == [1.0, 1.0] and v["shots"] == 20 and v["kopij"] == 2
    posel["imena_vezij"] = ["v_0_p0", "v_1_p0", "v_0_p1", "v_1_p1"]  # kopije niso zapored po vhodih
    with pytest.raises(ValueError, match="niso zapored po vhodih"):
        igra.analiza_qpu(podatki)
    posel["imena_vezij"] = ["v_0_p0", "v_0_p1", "v_1_p0", "v_1_p1"]
    posel["poskusi"][0]["st_vezij"] = 3
    with pytest.raises(ValueError, match="slovarjev izidov"):
        igra.analiza_qpu(podatki)


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
