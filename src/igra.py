"""Igra GHZ (Merminova paritetna igra) na stanjih GHZ: vezja, delež zmag, klasična meja, simulacije, tabela in graf.

Pravila (Brassard, Broadbent in Tapp, 2005): N >= 3 igralcev, igralec i dobi bit x_i, vsota vseh x_i je soda.
Igralci se med igro ne smejo pogovarjati; vsak odgovori z bitom y_i. Zmagajo, če velja
    y_1 xor ... xor y_N = (x_1 + ... + x_N) / 2 mod 2.
Kvantna strategija: igralci si vnaprej razdelijo stanje GHZ. Igralec z x_i = 1 na svojem kubitu izvede vrata S, nato
vsi izvedejo vrata H in kubit izmerijo; izid je odgovor y_i. Z idealnim stanjem GHZ zmagajo vedno.
Klasična meja: brez prepletenosti nobena strategija ne zmaga z verjetnostjo, večjo od 1/2 + 2^(-ceil(N/2)), če so
vhodi enakomerno naključni med veljavnimi. `klasicna_meja_izcrpno` to za majhne N preveri s pregledom vseh strategij.

Ukazi:
  meja     primerja formulo klasične meje z izčrpnim pregledom strategij
  sim      idealni in šumni simulator (FakeFez) -> rezultati/sim/<ime>.json
  tabela   vrstice tabel za poročilo iz simulacije in zagona na procesorju
  graf     porocilo-seminarska-naloga/img/ghz_igra.pdf
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ghz  # noqa: E402

KOREN = Path(__file__).resolve().parent.parent
SIM = KOREN / "rezultati" / "sim"
QPU = KOREN / "rezultati" / "qpu"
SLIKA = KOREN / "porocilo-seminarska-naloga" / "img" / "ghz_igra.pdf"
# seme obeh simulatorjev; klic za i-ti N v istem zagonu dobi seme SEME_SIMULATORJA + 1000 i (glej ghz.py)
SEME_SIMULATORJA = 11
# izčrpni pregled ima 2^N korakov nad 2^(N-1) vhodi: N = 16 traja nekaj sekund, N = 30 bi porabil gigabajte pomnilnika
NAJVEC_IZCRPNO = 16


# ------------------------------------------------------------------------------------------ pravila


def veljaven(x):
    """Ali je x niz bitov s sodo vsoto (pogoj igre za vhod)."""
    return all(b in (0, 1) for b in x) and sum(x) % 2 == 0


def cilj(x):
    """Pariteta odgovorov, s katero igralci zmagajo pri vhodu x."""
    if not veljaven(x):
        raise ValueError("vhod igre mora biti niz bitov s sodo vsoto")
    return (sum(x) // 2) % 2


def klasicna_meja(n):
    """Največja verjetnost zmage brez prepletenosti pri enakomerno naključnih veljavnih vhodih."""
    if n < 3:
        raise ValueError("igra je določena za vsaj 3 igralce")
    return 0.5 + 2.0 ** (-math.ceil(n / 2))


def klasicna_meja_izcrpno(n):
    """Najboljša deterministična strategija, najdena s pregledom vseh.

    Igralec pozna le svoj bit, zato so njegove strategije štiri: y_i = a_i x_i xor b_i. Pariteta odgovorov je tedaj
    (a . x) xor b, kjer je b = xor vseh b_i; pregledamo vseh 2^n vektorjev a in oba b. Naključna strategija je
    mešanica determinističnih, zato ne more biti boljša od najboljše med njimi."""
    if n < 3:
        raise ValueError("igra je določena za vsaj 3 igralce")
    if n > NAJVEC_IZCRPNO:
        raise ValueError(f"izčrpni pregled le do N = {NAJVEC_IZCRPNO}")
    vsi = np.arange(2 ** n, dtype=np.uint32)
    enice = np.bitwise_count(vsi)
    vhodi_ = vsi[enice % 2 == 0]
    tarca = (np.bitwise_count(vhodi_) // 2) % 2
    najboljsa = 0.0
    for a in range(2 ** n):
        ujemanje = float(np.mean(np.bitwise_count(vhodi_ & np.uint32(a)) % 2 == tarca))
        najboljsa = max(najboljsa, ujemanje, 1.0 - ujemanje)
    return najboljsa


def vhodi(n, k, rng):
    """k veljavnih vhodov. Če jih je vseh največ k, vsi po vrsti (ponovljeni do k), sicer k različnih naključnih."""
    if 2 ** (n - 1) <= k:
        vsi = [[(j >> i) & 1 for i in range(n)] for j in range(2 ** n) if bin(j).count("1") % 2 == 0]
        return [vsi[i % len(vsi)] for i in range(k)]
    izbrani = {}
    while len(izbrani) < k:
        x = [int(b) for b in rng.integers(0, 2, n - 1)]
        x.append(sum(x) % 2)  # zadnji bit poskrbi za sodo vsoto: enakomerno po veljavnih vhodih
        izbrani.setdefault(tuple(x), x)
    return list(izbrani.values())


# -------------------------------------------------------------------------------------------- vezja


def vezje_igre(priprava, x, oznaka=""):
    """Priprava GHZ, nato poteza vsakega igralca: S pri x_i = 1, H, meritev."""
    if len(x) != priprava.num_qubits:
        raise ValueError("vhod mora imeti en bit na igralca")
    cilj(x)
    qc = priprava.copy(name=f"{priprava.name}_igra{oznaka}")
    qc.barrier()
    for i, b in enumerate(x):
        if b:
            qc.s(i)
    qc.h(range(len(x)))
    qc.measure_all()
    return qc


def delez_zmag(stetje, x):
    """Delež ponovitev, v katerih ima niz odgovorov zmagovalno pariteto."""
    c = cilj(x)
    vseh = sum(stetje.values())
    return sum(v for niz, v in stetje.items() if niz.replace(" ", "").count("1") % 2 == c) / vseh


def statistika(zmage, shots):
    """Povprečje deleža zmag po vhodih in njegova standardna napaka (sigma).

    Sigma je večja od dveh ocen: razpršenosti deležev med vhodi (standardni odklon / koren iz števila vhodov) in
    binomske napake zaradi končnega števila ponovitev (koren iz p(1-p) / število vseh ponovitev)."""
    m = float(np.mean(zmage))
    med_vhodi = float(np.std(zmage, ddof=1) / math.sqrt(len(zmage))) if len(zmage) > 1 else 0.0
    binomska = math.sqrt(max(m * (1 - m), 0.0) / (len(zmage) * shots))
    return {"zmage": [float(z) for z in zmage], "povprecje": m, "sigma": max(med_vhodi, binomska)}


def najglobje(vezja):
    """Indeks najglobljega izmed vezij; njegove lastnosti navedemo v tabeli."""
    return max(range(len(vezja)), key=lambda i: ghz.lastnosti(vezja[i])["globina"])


# ---------------------------------------------------------------------------------------- simulacija


def simuliraj(ns, k, shots, ime, seme=5):
    """Idealni in šumni simulator (FakeFez) za vsak N; N, ki so v datoteki že izračunani, preskoči.

    Seme klica simulatorja je odvisno od tega, koliko N je isti zagon že izračunal, zato se nadaljevani zagon
    ujema s celotnim le, če gre po N v enakem vrstnem redu od začetka."""
    from qiskit_ibm_runtime.fake_provider import FakeFez

    backend = FakeFez()
    pot = SIM / f"{ime}.json"
    podatki = json.loads(pot.read_text()) if pot.exists() else {
        "ime": ime, "backend": "FakeFez", "shots": shots, "k": k, "seme": seme,
        "seme_simulatorja": SEME_SIMULATORJA, "rezultati": []}
    narejeni = {r["n"] for r in podatki["rezultati"]}
    idealni, sumni = ghz.simulator(seed=SEME_SIMULATORJA), ghz.simulator(backend, seed=SEME_SIMULATORJA)
    for n in ns:
        if n in narejeni:
            continue
        z = ghz.zasnova_za(backend, n)
        xs = vhodi(n, k, np.random.default_rng(seme + n))
        logicna = [vezje_igre(z.priprava, x, f"_{j}") for j, x in enumerate(xs)]
        prevedena = ghz.prevedi(logicna, backend, z.postavitev)
        j = najglobje(logicna)
        vrstica = {"n": n, "postavitev": z.postavitev, "vhodi": xs, "klasicna_meja": klasicna_meja(n),
                   "lastnosti_pred": ghz.lastnosti(logicna[j]),
                   "lastnosti_po": ghz.lastnosti(prevedena[j], backend.target),
                   "idealno": statistika([delez_zmag(s, x) for s, x in zip(idealni.pozeni(logicna, shots), xs)], shots),
                   "sumno": statistika([delez_zmag(s, x) for s, x in zip(sumni.pozeni(prevedena, shots), xs)], shots)}
        podatki["rezultati"].append(vrstica)
        podatki["rezultati"].sort(key=lambda r: r["n"])
        SIM.mkdir(parents=True, exist_ok=True)
        pot.write_text(json.dumps(podatki, indent=1, ensure_ascii=False))
        print(f"N={n:>3}  meja {vrstica['klasicna_meja']:.4f}  idealno {vrstica['idealno']['povprecje']:.4f}  "
              f"šumno {vrstica['sumno']['povprecje']:.4f} +- {vrstica['sumno']['sigma']:.4f}", flush=True)
    return podatki


# ------------------------------------------------------------------------------- zagon na procesorju


def analiza_qpu(podatki):
    """Delež zmag po N iz shranjenih izidov posla (rezultati/qpu/<ime>.json, zgrajenega s `qpu.py`).

    Preveri, da ima vsak N toliko slovarjev izidov, kot je vhodov, in da ima vsak slovar vse posnetke."""
    vrstice = []
    for posel in podatki["posli"]:
        if not posel.get("stetja"):
            continue
        for p in posel["poskusi"]:
            kopij = p.get("kopij", 1)  # zapis prve naloge nima kopij: eno vezje na vhod
            stetja = posel["stetja"][p["zacetek"]:p["zacetek"] + p["st_vezij"]]
            if not len(stetja) == len(p["vhodi"]) * kopij == p["st_vezij"]:
                raise ValueError(f"N={p['n']}: {len(stetja)} slovarjev izidov, {len(p['vhodi'])} vhodov, "
                                 f"kopij {kopij}, st_vezij {p['st_vezij']}")
            if any(sum(st.values()) != podatki["shots"] for st in stetja):
                raise ValueError(f"N={p['n']}: slovar izidov nima {podatki['shots']} posnetkov")
            if kopij > 1:  # kopije istega vhoda so zapored: njihove izide seštejemo
                imena = (posel.get("imena_vezij") or [])[p["zacetek"]:p["zacetek"] + p["st_vezij"]]
                if imena:  # ime kopije je <ime vezja>_p<c>: vsaka skupina mora imeti eno ime, skupine različna
                    skupine = [{i.rsplit("_p", 1)[0] for i in imena[j * kopij:(j + 1) * kopij]}
                               for j in range(len(p["vhodi"]))]
                    if any(len(s) != 1 for s in skupine) or len(set().union(*skupine)) != len(skupine):
                        raise ValueError(f"N={p['n']}: kopije vezij niso zapored po vhodih")
                stetja = [sum((Counter(st) for st in stetja[j * kopij:(j + 1) * kopij]), Counter())
                          for j in range(len(p["vhodi"]))]
            shots = podatki["shots"] * kopij  # posnetkov na vhod
            s = statistika([delez_zmag(st, x) for st, x in zip(stetja, p["vhodi"])], shots)
            vrstice.append({"n": p["n"], "job_id": posel["job_id"], "postavitev": p["postavitev"], "vhodi": p["vhodi"],
                            "shots": shots, "kopij": kopij, "klasicna_meja": klasicna_meja(p["n"]),
                            "lastnosti_pred": p["lastnosti_pred"], "lastnosti_po": p["lastnosti_po"], **s})
    print(f"{'N':>4} {'meja':>7} {'zmage':>7} {'sigma':>7}  nad mejo (zmage - 2 sigma > meja)")
    for v in vrstice:
        nad = v["povprecje"] - 2 * v["sigma"] > v["klasicna_meja"]
        print(f"{v['n']:>4} {v['klasicna_meja']:>7.4f} {v['povprecje']:>7.4f} {v['sigma']:>7.4f}  {'da' if nad else 'ne'}")
    return vrstice


def analiza_branja(podatki):
    """Napaka branja vsakega kubita iz dveh kalibracijskih vezij posla (če ju ima): p10 je delež posnetkov, v katerih
    smo kubit v stanju 0 prebrali kot 1, p01 pa delež, v katerih smo kubit v stanju 1 prebrali kot 0.

    Vrne None, če posel nima kalibracijskih vezij ali izidov."""
    for posel in podatki["posli"]:
        kal = posel.get("kalibracija")
        if not kal or not posel.get("stetja"):
            continue
        kubiti = kal["kubiti"]
        imena = (posel.get("imena_vezij") or [])[kal["zacetek"]:kal["zacetek"] + 2]
        if kal["st_vezij"] != 2 or imena not in ([], ["kal_0", "kal_1"]):
            raise ValueError(f"kalibracijski vezji nista na pričakovanem mestu (st_vezij {kal['st_vezij']}, {imena})")
        nic, ena = posel["stetja"][kal["zacetek"]:kal["zacetek"] + 2]
        shots = podatki["shots"]
        if sum(nic.values()) != shots or sum(ena.values()) != shots:
            raise ValueError(f"kalibracijski vezji nimata {shots} posnetkov")
        if any(len(niz.replace(" ", "")) != len(kubiti) for st in (nic, ena) for niz in st):
            raise ValueError(f"izid kalibracijskega vezja nima {len(kubiti)} bitov")

        def delez_enic(stetje, i):  # klasični bit i je i-ti znak z desne
            return sum(k for niz, k in stetje.items() if niz.replace(" ", "")[len(kubiti) - 1 - i] == "1") / shots

        p10 = [delez_enic(nic, i) for i in range(len(kubiti))]
        p01 = [1 - delez_enic(ena, i) for i in range(len(kubiti))]
        napaka = [(a + b) / 2 for a, b in zip(p10, p01)]
        najslabsi = max(range(len(kubiti)), key=napaka.__getitem__)
        print(f"branje: {len(kubiti)} kubitov, povprečna napaka {float(np.mean(napaka)):.4f}, "
              f"največja {napaka[najslabsi]:.4f} (kubit {kubiti[najslabsi]})")
        return {"kubiti": kubiti, "shots": shots, "p10": p10, "p01": p01, "povprecje": float(np.mean(napaka)),
                "najslabsi_kubit": kubiti[najslabsi], "najvecja": napaka[najslabsi],
                "objavljena_napaka": kal.get("objavljena_napaka")}
    return None


# ------------------------------------------------------------------------------------ tabela in graf


def _zaokrozi(x, mest):
    """Desetiški zapis števila, zaokrožen na `mest` decimalk, polovica navzgor (0,53125 -> 0,5313; 0,8115 -> 0,812).

    Oblika f"{x:.4f}" zaokroži dvojiško vrednost float, ki je lahko malo pod polovico, in zato včasih navzdol."""
    return Decimal(str(float(x))).quantize(Decimal(1).scaleb(-mest), rounding=ROUND_HALF_UP)


def _st(x, mest=3):
    return str(_zaokrozi(x, mest)).replace(".", "{,}")


def _z_napako(m, s, mest=3):
    return f"{_st(m, mest)}({int(_zaokrozi(s, mest).scaleb(mest))})"


def _beri(mapa, ime):
    pot = mapa / f"{ime}.json"
    return {r["n"]: r for r in json.loads(pot.read_text())["rezultati"]} if pot.exists() else {}


def _datum(ime_qpu):
    """Dan oddaje naloge kot 'd. m.' (iz rezultati/qpu/<ime>.json; čas oddaje je zapisan v UTC)."""
    pot = QPU / f"{ime_qpu}.json"
    if not pot.exists():
        return ""
    cas = json.loads(pot.read_text())["posli"][0]["cas_oddaje"]
    return f"{int(cas[8:10])}. {int(cas[5:7])}."


def samo_napake_branja(postavitev, branje):
    """Ocena deleža zmag, če bi bile napake branja edina napaka. Vsak prebrani bit se neodvisno obrne z izmerjeno
    verjetnostjo e (povprečje p10 in p01 tega kubita); pariteta odgovorov ostane prava, če je obratov sodo, torej z
    verjetnostjo 1/2 + 1/2 * produkt(1 - 2 e)."""
    napaka = {q: (a + b) / 2 for q, a, b in zip(branje["kubiti"], branje["p10"], branje["p01"])}
    return 0.5 + 0.5 * math.prod(1 - 2 * napaka[q] for q in postavitev)


def tabela_ponovitve(ime_qpu, ime_qpu2):
    """Vrstice tabele ponovljene meritve: prva in druga naloga na procesorju pri istem N; zadnji stolpec je ocena
    deleža zmag samo z izmerjenimi napakami branja (če ima druga naloga vezji za preverbo branja)."""
    prva, druga = _beri(QPU, f"{ime_qpu}_analiza"), _beri(QPU, f"{ime_qpu2}_analiza")
    branje = json.loads((QPU / f"{ime_qpu2}_analiza.json").read_text()).get("branje")

    def kriterij(r, meja):
        return "da" if r["povprecje"] - 2 * r["sigma"] > meja else "ne"

    print(f"\n% N & ibm_fez {_datum(ime_qpu)} & kriterij & ibm_fez {_datum(ime_qpu2)} & kriterij & ponovitev na vhod "
          "& samo napake branja")
    for n in sorted(druga):
        a, b = prva.get(n), druga[n]
        meja = klasicna_meja(n)
        print(" & ".join([str(n), _z_napako(a["povprecje"], a["sigma"]) if a else "--", kriterij(a, meja) if a else "--",
                          _z_napako(b["povprecje"], b["sigma"]), kriterij(b, meja), str(b.get("shots", "")),
                          _st(samo_napake_branja(b["postavitev"], branje)) if branje else "--"]) + r" \\")


def tabela(ime_sim, ime_qpu):
    sim, qpu = _beri(SIM, ime_sim), _beri(QPU, f"{ime_qpu}_analiza")
    print("% N & klasična meja & idealni sim. & šumni sim. & ibm_fez & nad mejo")
    for n in sorted(set(sim) | set(qpu)):
        s, q = sim.get(n), qpu.get(n)
        meja = klasicna_meja(n)
        celice = [str(n), _st(meja, 4),
                  _st(s["idealno"]["povprecje"]) if s else "--",
                  _z_napako(s["sumno"]["povprecje"], s["sumno"]["sigma"]) if s else "--",
                  _z_napako(q["povprecje"], q["sigma"]) if q else "--",
                  ("da" if q["povprecje"] - 2 * q["sigma"] > meja else "ne") if q else "--"]
        print(" & ".join(celice) + r" \\")
    print("\n% N & globina pred & globina po & 2q pred & 2q po & trajanje [us]")
    for n in sorted(qpu):
        pred, po = qpu[n]["lastnosti_pred"], qpu[n]["lastnosti_po"]
        print(" & ".join([str(n), str(pred["globina"]), str(po["globina"]), str(pred["dvokubitna"]),
                          str(po["dvokubitna"]), _st(po["trajanje_us"], 2)]) + r" \\")


def graf(ime_sim, ime_qpu, ime_qpu2=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sim, qpu = _beri(SIM, ime_sim), _beri(QPU, f"{ime_qpu}_analiza")
    qpu2 = _beri(QPU, f"{ime_qpu2}_analiza") if ime_qpu2 else {}
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    # klasična meja pri vsakem celem N, ne le pri izmerjenih (stopnica se spremeni pri vsakem lihem N)
    vsi_n = list(range(3, max(set(sim) | set(qpu) | set(qpu2), default=3) + 1))
    ax.step(vsi_n, [klasicna_meja(n) for n in vsi_n], "k--", where="mid", linewidth=1, label="klasična meja")
    if sim:
        x = sorted(sim)
        ax.plot(x, [sim[n]["idealno"]["povprecje"] for n in x], ":", color="0.4", marker="s", markersize=3,
                linewidth=1, label="idealni simulator")
        ax.errorbar(x, [sim[n]["sumno"]["povprecje"] for n in x], yerr=[2 * sim[n]["sumno"]["sigma"] for n in x],
                    fmt="^-", color="#0072B2", markersize=4, linewidth=1, capsize=2, label="šumni simulator (FakeFez)")
    if qpu:
        x = sorted(qpu)
        ax.errorbar(x, [qpu[n]["povprecje"] for n in x], yerr=[2 * qpu[n]["sigma"] for n in x],
                    fmt="o-", color="#D55E00", markersize=4, linewidth=1, capsize=2,
                    label=f"ibm_fez, {_datum(ime_qpu)}" if qpu2 else "ibm_fez")
    if qpu2:
        x = sorted(qpu2)
        ax.errorbar(x, [qpu2[n]["povprecje"] for n in x], yerr=[2 * qpu2[n]["sigma"] for n in x],
                    fmt="D--", color="#009E73", markersize=3.5, linewidth=1, capsize=2,
                    label=f"ibm_fez, {_datum(ime_qpu2)}")
    ax.set_xlabel("število igralcev $N$")
    ax.set_ylabel("delež zmag")
    ax.set_ylim(0.45, 1.02)
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.1f}".replace(".", ","))  # decimalna vejica kot v poročilu
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(SLIKA)
    print(f"shranjeno: {SLIKA.relative_to(KOREN)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ukaz", choices=["meja", "sim", "tabela", "graf"])
    ap.add_argument("--n", type=int, nargs="+", default=[3, 5, 10])
    ap.add_argument("--k", type=int, default=8, help="število vhodov na N")
    ap.add_argument("--shots", type=int, default=1000)
    ap.add_argument("--ime", default="igra_fakefez", help="ime simulacije (rezultati/sim/<ime>.json)")
    ap.add_argument("--qpu", default="igra", help="ime zagona na procesorju (rezultati/qpu/<ime>_analiza.json)")
    ap.add_argument("--qpu2", default=None, help="ime ponovljenega zagona na procesorju (druga serija v grafu in tabeli)")
    ap.add_argument("--seme", type=int, default=5, help="seme vhodov: za N se uporabi seme + N")
    a = ap.parse_args()
    if a.ukaz == "meja":
        for n in a.n:
            izcrpno = f"{klasicna_meja_izcrpno(n):.6f}" if n <= NAJVEC_IZCRPNO else f"le do N = {NAJVEC_IZCRPNO}"
            print(f"N={n:>2}  formula {klasicna_meja(n):.6f}  izčrpno {izcrpno}")
    elif a.ukaz == "sim":
        simuliraj(a.n, a.k, a.shots, a.ime, a.seme)
    elif a.ukaz == "tabela":
        tabela(a.ime, a.qpu)
        if a.qpu2:
            tabela_ponovitve(a.qpu, a.qpu2)
    else:
        graf(a.ime, a.qpu, a.qpu2)


if __name__ == "__main__":
    main()
