"""Igra GHZ na pravem procesorju IBM (privzeto ibm_fez) s strogim nadzorom porabe procesorskega časa.

En posel vsebuje za vsak N iz --n K veljavnih vhodov (--k, seme vhodov --seme + N), vsak vhod je eno vezje: priprava
GHZ na drevesu (src/ghz.py), nato poteza igralcev (src/igra.py). Delež zmag poročamo brez ublažitve branja (igralci
dobijo, kar preberejo). Vsako vezje dobi --shots posnetkov; z `--kopije-od N0 R` je vsako vezje z N >= N0 v poslu
R-krat (analiza izide kopij sešteje), tako da večji N dobijo več posnetkov brez različnih nastavitev v istem poslu.
Z `--kalibracija` posel dobi še dve vezji za preverbo branja (vsi izbrani kubiti v stanju 0 in vsi v stanju 1); z
njima le izmerimo napako branja vsakega kubita, deleža zmag ne popravljamo.

Ukazi (nič se ne pošlje brez `poslji`; `poslji` brez --max-sekund se ne izvede):
  nacrt     zgradi in prevede vezja, izpiše tabelo, oceno časa in preostanek na instanci
  poslji    isto + oddaja (en posel na ukaz). Posel dobi max_execution_time (faktor x ocena + 3 s), ki ga pred oddajo
            rezervira v meji --max-sekund tega ukaza; rezervacija se sprosti šele po obračunu. Ukaz zavrne oddajo,
            dokler kateri od prejšnjih poslov ni obračunan, zato naj bo --max-sekund dogovorjena meja minus že
            porabljeni čas.
  prevzemi  izide in obračun že oddanih poslov (če je bilo čakanje prekinjeno)
  analiza   delež zmag po N iz rezultati/qpu/<ime>.json -> rezultati/qpu/<ime>_analiza.json

Žeton in instanca: spremenljivki QISKIT_IBM_TOKEN in QISKIT_IBM_INSTANCE morata biti v okolju (skripta datoteke .env ne
bere), npr.
  .venv/bin/python src/qpu.py nacrt --ime igra --k 8 --n 3 5 10 20 30 40 50 60 75
Lokalni preizkus celotne poti brez procesorja: --fake (FakeFez in šumni simulator).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ghz  # noqa: E402
import igra  # noqa: E402
from ibm_skupno import (KOREN, beri_dnevnik, dodaj_v_dnevnik, dopolni_zapis, povezi, st_dvokubitnih,  # noqa: E402
                        stetja_iz_rezultata)

# model porabe: fiksni čas na posel + čas na posnetek. Zagon igre je uporabil --model-posnetek-ms 0.284: ocena
# 22,5 s, obračunano 21 s (72 vezij x 1000 posnetkov, rezultati/naloge.jsonl).
CAS_POSEL_S = 2.0
CAS_POSNETEK_S = 0.35e-3
OZNAKA = "skupina20"
IZHOD = KOREN / "rezultati" / "qpu"


def ocena_casa(st_poslov, posnetkov):
    """Model porabe: fiksni del na posel + čas na posnetek (posnetkov = vsota posnetkov vseh vezij)."""
    return CAS_POSEL_S * st_poslov + CAS_POSNETEK_S * posnetkov


def povezava(ime, fake):
    """(service, backend, poraba). Pri --fake brez omrežja in brez žetona."""
    if fake:
        from qiskit_ibm_runtime.fake_provider import FakeFez
        return None, FakeFez(), {}
    service = povezi()
    try:
        poraba = service.usage()
    except Exception as e:  # noqa: BLE001
        sys.exit(f"Porabe instance ni bilo mogoče prebrati ({e}); brez nje ne nadaljujem.")
    return service, service.backend(ime), poraba


def merjeni_kubiti(qc):
    """Slovar klasični bit -> fizični kubit za meritve v prevedenem vezju."""
    return {qc.find_bit(i.clbits[0]).index: qc.find_bit(i.qubits[0]).index for i in qc.data if i.operation.name == "measure"}


def kalibracijski_vezji(backend, kubiti):
    """Dve vezji za preverbo branja na podanih kubitih čipa: vsi v stanju 0 (brez vrat) in vsi v stanju 1 (vrata X),
    nato meritev. Preveri, da vezji nimata dvokubitnih vrat in merita podane kubite."""
    n = len(kubiti)
    nic, ena = QuantumCircuit(n, name="kal_0"), QuantumCircuit(n, name="kal_1")
    ena.x(range(n))
    for qc in (nic, ena):
        qc.measure_all()
    isa = ghz.prevedi([nic, ena], backend, kubiti)
    for qc, z_vrati in zip(isa, (set(), set(kubiti))):
        if st_dvokubitnih(qc) != 0:
            raise RuntimeError(f"{qc.name}: kalibracijsko vezje ima dvokubitna vrata")
        if merjeni_kubiti(qc) != dict(enumerate(kubiti)):
            raise RuntimeError(f"{qc.name}: meritve niso na izbranih kubitih")
        z_x = {qc.find_bit(q).index for i in qc.data if i.operation.name not in ("barrier", "delay", "measure")
               for q in i.qubits}
        if z_x != z_vrati:
            raise RuntimeError(f"{qc.name}: vrata niso na pričakovanih kubitih (kal_0 brez vrat, kal_1 X na vseh)")
    return isa


def zgradi_posel_igre(backend, ns, k, seme, kalibracija=False, kopije_od=None):
    """Vezja enega posla za igro GHZ na drevesu: za vsak N k veljavnih vhodov (seme + N); s `kalibracija` na koncu še
    dve vezji za preverbo branja na vseh kubitih, ki jih posel uporablja. Z `kopije_od = (N0, R)` je vsako vezje z
    N >= N0 v poslu R-krat zapored (kopije istega vhoda so skupaj).

    Pred oddajo preveri, da ima vsako prevedeno vezje N - 1 dvokubitnih vrat in da meri izbrane kubite."""
    poskusi, vezja = [], []
    for n in ns:
        z = ghz.zasnova_za(backend, n)
        xs = igra.vhodi(n, k, np.random.default_rng(seme + n))
        logicna = [igra.vezje_igre(z.priprava, x, f"_{j}") for j, x in enumerate(xs)]
        isa = ghz.prevedi(logicna, backend, z.postavitev)
        for qc in isa:
            if st_dvokubitnih(qc) != n - 1:
                raise RuntimeError(f"{qc.name}: prevajalnik je dodal vrata ({st_dvokubitnih(qc)} namesto {n - 1})")
            if merjeni_kubiti(qc) != dict(enumerate(z.postavitev)):
                raise RuntimeError(f"{qc.name}: meritve niso na izbranih kubitih")
        j = igra.najglobje(logicna)
        kopij = kopije_od[1] if kopije_od and n >= kopije_od[0] else 1
        if kopij > 1:  # kopije istega vhoda zapored: vezje j, kopija c je na mestu zacetek + j * kopij + c
            isa_vsa = [qc.copy(name=f"{qc.name}_p{c}") for qc in isa for c in range(kopij)]
        else:
            isa_vsa = isa
        poskusi.append({"n": n, "postavitev": z.postavitev, "vhodi": xs, "zacetek": len(vezja),
                        "st_vezij": len(isa_vsa), "kopij": kopij,
                        "lastnosti_pred": ghz.lastnosti(logicna[j]),
                        "lastnosti_po": ghz.lastnosti(isa[j], backend.target)})
        vezja += isa_vsa
    opis = {"poskusi": poskusi}
    if kalibracija:
        kubiti = sorted({q for p in poskusi for q in p["postavitev"]})
        isa = kalibracijski_vezji(backend, kubiti)
        # objavljena napaka branja istih kubitov ob oddaji, za primerjavo z izmerjeno
        objavljena = [getattr(backend.target["measure"].get((q,)), "error", None) for q in kubiti]
        opis["kalibracija"] = {"kubiti": kubiti, "zacetek": len(vezja), "st_vezij": len(isa),
                               "objavljena_napaka": objavljena}
        vezja += isa
    return opis, vezja


def posnetkov_posla(vezja, a):
    """Vsota posnetkov vseh vezij posla (iz nje sledi ocena porabe): vsa vezja imajo a.shots posnetkov."""
    return len(vezja) * a.shots


def izpisi_nacrt(posli, a, poraba):
    print(f"{'N':>4}  {'globina':>8} {'2q':>4} {'trajanje':>10}  vezij x posnetkov")
    posnetkov = 0
    for opis, vezja in posli:
        for p in opis["poskusi"]:
            lp = p["lastnosti_po"]
            kopije = f" ({p['kopij']} kopije vsakega vhoda)" if p.get("kopij", 1) > 1 else ""
            print(f"{p['n']:>4}  {lp['globina']:>8} {lp['dvokubitna']:>4} {(lp['trajanje_us'] or 0):>8.2f}us "
                  f" {p['st_vezij']:>3} x {a.shots}{kopije}")
        kal = opis.get("kalibracija")
        if kal:
            print(f" kal  preverba branja na {len(kal['kubiti'])} kubitih: {kal['st_vezij']} x {a.shots}")
        posnetkov += posnetkov_posla(vezja, a)
    st_vezij = sum(len(v) for _, v in posli)
    ocena = ocena_casa(len(posli), posnetkov)
    preostalo = poraba.get("usage_remaining_seconds")
    print(f"\nposli: {len(posli)}, vezja: {st_vezij}, posnetkov skupaj: {posnetkov}")
    print(f"ocena porabe: {ocena:.1f} s  (= {CAS_POSEL_S:g} s x {len(posli)} + {CAS_POSNETEK_S * 1e3:g} ms x {posnetkov} posnetkov)")
    print(f"preostanek na instanci: {preostalo if preostalo is not None else '? (lokalno)'} s;  meja: {a.max_sekund or '-'} s")
    return ocena


def shrani(ime, podatki):
    """Zapiše rezultati/qpu/<ime>.json najprej v začasno datoteko in jo nato preimenuje, zato izpad elektrike med
    pisanjem ne pusti napol zapisanega zapisa posla."""
    IZHOD.mkdir(parents=True, exist_ok=True)
    pot = IZHOD / f"{ime}.json"
    zacasna = pot.with_suffix(".tmp")
    zacasna.write_text(json.dumps(podatki, indent=1, ensure_ascii=False))
    zacasna.replace(pot)
    return pot


def shrani_analizo(ime, podatki):
    """Delež zmag po N (in napake branja, če ima posel kalibracijski vezji) -> rezultati/qpu/<ime>_analiza.json."""
    analiza = {"ime": ime, "backend": podatki["backend"], "shots": podatki["shots"],
               "rezultati": igra.analiza_qpu(podatki)}
    branje = igra.analiza_branja(podatki)
    if branje:
        analiza["branje"] = branje
    return shrani(f"{ime}_analiza", analiza)


def beri(ime):
    pot = IZHOD / f"{ime}.json"
    if not pot.exists():
        sys.exit(f"Ni datoteke {pot}.")
    return json.loads(pot.read_text())


def omejitev_posla(a, ocena_posla):
    """Najdaljši dovoljeni čas posla (max_execution_time): faktor x ocena + 3 s. Tolikšen del meje posel rezervira."""
    c = int(math.ceil(a.faktor_omejitve * ocena_posla + 3))
    if c < max(1, ocena_posla):
        sys.exit(f"Omejitev posla {c} s je manjša od ocene {ocena_posla:.1f} s: preveri --faktor-omejitve in model.")
    return c


def ocena_ibm(service, job_id, cakaj_s=90, korak_s=5):
    """IBM-ova ocena trajanja posla (quantum_seconds); vsakič nov objekt posla, ker se prva prazna ocena ne osveži."""
    konec = time.time() + cakaj_s
    while time.time() < konec:
        try:
            q = service.job(job_id).usage_estimation.get("quantum_seconds")
            if q is not None:
                return float(q)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(korak_s)
    return None


def dejanska_poraba(job, cakaj_s=240, korak_s=10):
    """Obračunani čas posla. Takoj po koncu je obračun še "pending" in job.usage() vrne 0, zato počakamo."""
    konec = time.time() + cakaj_s
    while time.time() < konec:
        try:
            if job.metrics().get("usage", {}).get("status", "pending") != "pending":
                return float(job.usage())
        except Exception:  # noqa: BLE001
            pass
        time.sleep(korak_s)
    return None


def poslji(a, service, backend, poraba, posli):
    """Zaporedna oddaja poslov znotraj meje a.max_sekund.

    Pravilo rezervacije: pred oddajo posel rezervira svojo celotno omejitev (max_execution_time), vsota rezervacij
    nikoli ne preseže meje. Neporabljeni del se sprosti šele, ko je dejanska poraba obračunana; če obračuna ni,
    ostane rezervirana cela omejitev. Tako skupna poraba ne more preseči meje, ne glede na oceno."""
    if not a.max_sekund:
        sys.exit("Za oddajo je obvezna meja --max-sekund (dogovorjeni skupni procesorski čas).")
    if not a.fake:
        preostalo = poraba.get("usage_remaining_seconds")
        if poraba.get("usage_limit_reached") or preostalo is None:
            sys.exit("Omejitev porabe je dosežena ali preostanka ni mogoče prebrati: nič ni poslano.")
        if a.max_sekund > preostalo:
            sys.exit(f"Meja {a.max_sekund} s presega preostanek {preostalo} s: nič ni poslano.")
        neobracunani = [z["job_id"] for z in beri_dnevnik() if z.get("ime_poskusa") and not z.get("fake")
                        and z.get("porabljen_cas_s") is None]
        if neobracunani:
            sys.exit(f"Posli brez obračuna ({', '.join(neobracunani)}): najprej `prevzemi`, da se poraba zapiše; nič ni poslano.")
        if (IZHOD / f"{a.ime}.json").exists():
            sys.exit(f"rezultati/qpu/{a.ime}.json že obstaja (zapis plačanega zagona): izberi drugo --ime.")
    obstojeca = IZHOD / f"{a.ime}.json"
    if a.fake and obstojeca.exists() and not json.loads(obstojeca.read_text()).get("fake"):
        sys.exit(f"rezultati/qpu/{a.ime}.json je zapis plačanega zagona: za lokalni preizkus izberi drugo --ime.")
    if a.max_poslov and len(posli) > a.max_poslov:
        sys.exit(f"Načrt ima {len(posli)} poslov, odobrenih je {a.max_poslov}: nič ni poslano.")

    if not a.fake:  # zapis mora biti mogoč, preden karkoli plačamo (disk samo za branje ali poln)
        poskusna = IZHOD / f"{a.ime}.zapis-test"
        try:
            IZHOD.mkdir(parents=True, exist_ok=True)
            poskusna.write_text("ok")
            poskusna.unlink()
        except OSError as e:
            sys.exit(f"V {IZHOD} ni mogoče pisati ({e}): nič ni poslano.")

    podatki = {"ime": a.ime, "backend": backend.name, "fake": a.fake, "shots": a.shots,
               "kopije_od": getattr(a, "kopije_od", None), "k": a.k, "seme": a.seme,
               "brez_kubitov": a.brez_kubitov, "max_sekund": a.max_sekund,
               "model_posnetek_ms": CAS_POSNETEK_S * 1e3, "faktor_omejitve": a.faktor_omejitve, "posli": []}
    rezervirano = 0.0
    for opis, vezja in posli:
        pricakovanih = sum(p["st_vezij"] for p in opis["poskusi"]) + (opis.get("kalibracija") or {}).get("st_vezij", 0)
        if pricakovanih != len(vezja):
            sys.exit(f"Posel ima {len(vezja)} vezij, opis pa jih navaja {pricakovanih}: nič ni poslano.")
        ocena_posla = ocena_casa(1, posnetkov_posla(vezja, a))
        omejitev = omejitev_posla(a, ocena_posla)
        if rezervirano + omejitev > a.max_sekund:
            print(f"Ustavljeno pred poslom: rezervirano {rezervirano:.0f} s + omejitev {omejitev} s > meja {a.max_sekund} s.")
            break
        zapis = {**opis, "ocena_s": round(ocena_posla, 2), "max_execution_time": omejitev, "job_id": None,
                 "ocena_ibm_s": None, "poraba_s": None, "stanje": None, "imena_vezij": [qc.name for qc in vezja],
                 "stetja": None, "cas_oddaje": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        podatki["posli"].append(zapis)
        if a.fake:
            sim = ghz.simulator(backend)
            zapis["stetja"] = sim.pozeni(vezja, a.shots)
            zapis["job_id"], zapis["stanje"] = f"lokalno-{len(podatki['posli'])}", "DONE"
            shrani(a.ime, podatki)
            continue

        from qiskit_ibm_runtime import SamplerV2 as Sampler
        moznosti = {"default_shots": a.shots, "max_execution_time": omejitev,
                    "environment": {"job_tags": [OZNAKA, a.ime]}}
        rezervirano += omejitev
        job = Sampler(mode=backend, options=moznosti).run(vezja, shots=a.shots)
        zapis["job_id"] = job.job_id()
        print(f"oddan posel {job.job_id()}", flush=True)  # ID najprej na zaslon, če bi zapis na disk spodletel
        # takoj zapišemo job ID (dnevnik in datoteka), šele nato karkoli drugega
        dodaj_v_dnevnik({"cas": zapis["cas_oddaje"], "job_id": job.job_id(), "backend": backend.name, "fake": False,
                         "shots": a.shots, "posnetkov": posnetkov_posla(vezja, a), "ime_poskusa": a.ime,
                         "max_sekund": omejitev,
                         "ocena_cas_s": zapis["ocena_s"], "ocena_ibm_s": None, "st_vezij": len(vezja),
                         "porabljen_cas_s": None})
        shrani(a.ime, podatki)
        print(f"posel {job.job_id()}, omejitev {omejitev} s, "
              f"rezervirano {rezervirano:.0f} s od {a.max_sekund} s", flush=True)

        zapis["ocena_ibm_s"] = ocena_ibm(service, job.job_id())
        shrani(a.ime, podatki)
        if zapis["ocena_ibm_s"] is not None and zapis["ocena_ibm_s"] > omejitev:
            job.cancel()
            zapis["stanje"] = "CANCELLED (ocena IBM nad omejitvijo)"
            shrani(a.ime, podatki)
            sys.exit(f"Ocena IBM {zapis['ocena_ibm_s']} s presega omejitev {omejitev} s: posel preklican, ustavljeno.")
        try:
            rezultat = job.result()
        except Exception as e:  # noqa: BLE001  (neuspel ali preklican posel: omejitev ostane rezervirana)
            zapis["stanje"] = f"NAPAKA: {e}"
            shrani(a.ime, podatki)
            sys.exit(f"Posel {job.job_id()} ni uspel ({e}); ustavljeno, nič več ni poslano.")
        zapis["stetja"], zapis["stanje"] = stetja_iz_rezultata(rezultat), "DONE"
        shrani(a.ime, podatki)
        zapis["poraba_s"] = dejanska_poraba(job)
        if zapis["poraba_s"] is not None:
            dopolni_zapis(job.job_id(), zapis["poraba_s"])
            rezervirano += zapis["poraba_s"] - omejitev  # tudi poraba nad omejitvijo se šteje
        print(f"  končano: obračunano {zapis['poraba_s']} s (ocena {ocena_posla:.1f} s, IBM {zapis['ocena_ibm_s']} s); "
              f"rezervirano {rezervirano:.0f} s od {a.max_sekund} s", flush=True)
        shrani(a.ime, podatki)
    podatki["rezervirano_s"] = rezervirano
    print(f"shranjeno: {shrani(a.ime, podatki)}")
    return podatki


def prevzemi(a, service):
    """Izidi in obračun za posle brez izidov ali brez obračuna; izidi gredo v rezultati/qpu/<ime>.json, obračun v
    dnevnik. Neuspel posel zapiše in nadaljuje z ostalimi. Popolnega zapisa (izidi in obračun) ne prepiše."""
    podatki = beri(a.ime)
    for zapis in podatki["posli"]:
        if not zapis["job_id"] or str(zapis["job_id"]).startswith("lokalno"):
            continue
        if zapis["stetja"] is not None and zapis.get("poraba_s") is not None:
            dopolni_zapis(zapis["job_id"], zapis["poraba_s"])  # le dnevnik, brez IBM in brez pisanja zapisa
            continue
        job = service.job(zapis["job_id"])
        if zapis["stetja"] is None:
            try:
                zapis["stetja"], zapis["stanje"] = stetja_iz_rezultata(job.result()), "DONE"
            except Exception as e:  # noqa: BLE001
                if not str(zapis.get("stanje") or "").startswith("CANCELLED"):  # razlog preklica ostane zapisan
                    zapis["stanje"] = f"NAPAKA: {e}"
        if zapis.get("poraba_s") is None:
            zapis["poraba_s"] = dejanska_poraba(job, cakaj_s=30)
        if zapis["poraba_s"] is not None:
            dopolni_zapis(zapis["job_id"], zapis["poraba_s"])
        shrani(a.ime, podatki)
    return podatki


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ukaz", choices=["nacrt", "poslji", "prevzemi", "analiza"])
    ap.add_argument("--ime", required=True, help="ime poskusa (rezultati/qpu/<ime>.json)")
    ap.add_argument("--backend", default="ibm_fez")
    ap.add_argument("--n", type=int, nargs="+", default=[3, 5, 10], help="števila igralcev N")
    ap.add_argument("--k", type=int, default=8, help="število vhodov na N")
    ap.add_argument("--seme", type=int, default=5, help="seme vhodov: za N se uporabi seme + N")
    ap.add_argument("--brez-kubitov", type=int, nargs="+", default=[],
                    help="kubiti, ki jih izbira ne sme uporabiti (privzeto nobeden)")
    ap.add_argument("--shots", type=int, default=1000)
    ap.add_argument("--kopije-od", type=int, nargs=2, default=None, metavar=("N0", "R"),
                    help="vsako vezje z N >= N0 je v poslu R-krat (R x --shots posnetkov na vhod)")
    ap.add_argument("--kalibracija", action="store_true",
                    help="dodaj dve vezji za preverbo branja (vsi izbrani kubiti v stanju 0 in v stanju 1)")
    ap.add_argument("--max-sekund", type=float, default=None, help="skupna meja porabe, dogovorjena pred zagonom")
    ap.add_argument("--max-poslov", type=int, default=None, help="največ poslov, dogovorjeno pred zagonom")
    ap.add_argument("--faktor-omejitve", type=float, default=2.0,
                    help="omejitev posla = faktor x ocena + 3 s")
    ap.add_argument("--model-posnetek-ms", type=float, default=None,
                    help="čas na posnetek [ms], izmerjen na prejšnjih poslih, namesto privzetih 0,35 ms")
    ap.add_argument("--fake", action="store_true", help="lokalno: FakeFez in šumni simulator, brez procesorja")
    a = ap.parse_args()
    ghz.IZKLJUCENI_KUBITI = set(a.brez_kubitov)
    if a.faktor_omejitve < 1:
        ap.error("--faktor-omejitve mora biti vsaj 1")
    if a.model_posnetek_ms is not None and a.model_posnetek_ms <= 0:
        ap.error("--model-posnetek-ms mora biti pozitiven")
    if a.shots <= 0 or (a.kopije_od and (a.kopije_od[0] < 3 or a.kopije_od[1] < 1)):
        ap.error("--shots mora biti pozitiven, pri --kopije-od N0 R pa N0 vsaj 3 in R vsaj 1")
    if a.model_posnetek_ms:
        global CAS_POSNETEK_S
        CAS_POSNETEK_S = a.model_posnetek_ms / 1e3

    if a.ukaz == "analiza":
        print(f"shranjeno: {shrani_analizo(a.ime, beri(a.ime)).relative_to(KOREN)}")
        return
    service, backend, poraba = povezava(a.backend, a.fake)
    if a.ukaz == "prevzemi":
        podatki = prevzemi(a, service)
        if any(z.get("stetja") for z in podatki["posli"]):
            shrani_analizo(a.ime, podatki)
        return
    if not a.fake:
        st = backend.status()
        print(f"procesor {backend.name}: deluje={st.operational}, v vrsti={st.pending_jobs}")
        if not st.operational:
            sys.exit("Procesor ne deluje: nič ni poslano.")
    posli = [zgradi_posel_igre(backend, a.n, a.k, a.seme, a.kalibracija, a.kopije_od)]
    izpisi_nacrt(posli, a, poraba)
    if a.ukaz == "nacrt":
        print("\nnacrt: nič ni bilo poslano.")
        return
    t0 = time.time()
    podatki = poslji(a, service, backend, poraba, posli)
    print(f"čas: {time.time() - t0:.0f} s")
    shrani_analizo(a.ime, podatki)


if __name__ == "__main__":
    main()
