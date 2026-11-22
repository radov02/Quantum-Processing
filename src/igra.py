"""Igra GHZ (Merminova paritetna igra): vezja, delež zmag, klasična meja, simulacija, tabeli in grafa.

Pravila (Brassard, Broadbent in Tapp, 2005): N >= 3 igralcev, igralec i dobi bit x_i, vsota vseh x_i je soda. Igralci
med igro ne komunicirajo; vsak odgovori z bitom y_i. Zmagajo, če velja
    y_1 xor ... xor y_N = (x_1 + ... + x_N) / 2 mod 2.
Kvantna strategija: stanje GHZ, igralec z x_i = 1 izvede vrata S, nato vsi vrata H in meritev.
Klasična meja: 1/2 + 2^(-ceil(N/2)) pri enakomerno naključnih veljavnih vhodih.

Ukazi:
  meja     formula klasične meje in izčrpni pregled strategij
  sim      idealni in šumni simulator (FakeFez) -> rezultati/sim/<ime>.json
  tabela   vrstice tabel za poročilo iz simulacije in zagona na procesorju
  graf     porocilo-seminarska-naloga/img/ghz_igra.pdf
  cip      porocilo-seminarska-naloga/img/ibm_fez_cip.pdf: povezave čipa in kubiti, izbrani za en N (npr. --n 20)
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
import ghz

KOREN = Path(__file__).resolve().parent.parent
SIM = KOREN / "rezultati" / "sim"
QPU = KOREN / "rezultati" / "qpu"
SLIKA = KOREN / "porocilo-seminarska-naloga" / "img" / "ghz_igra.pdf"
SLIKA_CIPA = KOREN / "porocilo-seminarska-naloga" / "img" / "ibm_fez_cip.pdf"
# i-ti klic simulatorja v istem zagonu dobi seme SEME_SIMULATORJA + 1000 i
SEME_SIMULATORJA = 11
# izčrpni pregled ima 2^N korakov nad 2^(N-1) vhodi
NAJVEC_IZCRPNO = 16


# ------------------------------------------------------------------------------------------ pravila


def veljaven(x):
    """Ali je x niz bitov s sodo vsoto."""
    return all(b in (0, 1) for b in x) and sum(x) % 2 == 0


def cilj(x):
    """Pariteta odgovorov, s katero igralci zmagajo pri vhodu x."""
    if not veljaven(x):
        raise ValueError("vhod igre mora biti niz bitov s sodo vsoto")
    return (sum(x) // 2) % 2


def klasicna_meja(n):
    """Klasična meja 1/2 + 2^(-ceil(n/2))."""
    if n < 3:
        raise ValueError("igra je določena za vsaj 3 igralce")
    return 0.5 + 2.0 ** (-math.ceil(n / 2))


def klasicna_meja_izcrpno(n):
    """Najboljša deterministična strategija, najdena s pregledom vseh.

    Strategija igralca je y_i = a_i x_i xor b_i, zato je pariteta odgovorov (a . x) xor b, kjer je b = xor vseh b_i.
    Pregled gre po vseh 2^n vektorjih a in obeh b."""
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
        x.append(sum(x) % 2)  # zadnji bit dopolni vsoto do sode
        izbrani.setdefault(tuple(x), x)
    return list(izbrani.values())


# -------------------------------------------------------------------------------------------- vezja


def vezje_igre(priprava, x, oznaka=""):
    """Priprava GHZ, nato S pri x_i = 1, H in meritev."""
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

    Sigma je večja od dveh ocen: standardni odklon med vhodi / koren(število vhodov) in binomska napaka
    koren(p (1 - p) / število vseh ponovitev)."""
    m = float(np.mean(zmage))
    med_vhodi = float(np.std(zmage, ddof=1) / math.sqrt(len(zmage))) if len(zmage) > 1 else 0.0
    binomska = math.sqrt(max(m * (1 - m), 0.0) / (len(zmage) * shots))
    return {"zmage": [float(z) for z in zmage], "povprecje": m, "sigma": max(med_vhodi, binomska)}


def najglobje(vezja):
    """Indeks najglobljega vezja; njegove lastnosti so v tabeli."""
    return max(range(len(vezja)), key=lambda i: ghz.lastnosti(vezja[i])["globina"])


# ---------------------------------------------------------------------------------------- simulacija


def simuliraj(ns, k, shots, ime, seme=5):
    """Idealni in šumni simulator (FakeFez) za vsak N; N, ki so že v datoteki, preskoči.

    Seme klica je odvisno od zaporedne številke N v zagonu, zato se nadaljevani zagon ujema s celotnim le pri enakem
    vrstnem redu N od začetka."""
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
    """Delež zmag po N iz izidov posla (rezultati/qpu/<ime>.json)."""
    vrstice = []
    for posel in podatki["posli"]:
        if not posel.get("stetja"):
            continue
        for p in posel["poskusi"]:
            kopij = p.get("kopij", 1)  # zapis prve naloge nima polja kopij
            stetja = posel["stetja"][p["zacetek"]:p["zacetek"] + p["st_vezij"]]
            if not len(stetja) == len(p["vhodi"]) * kopij == p["st_vezij"]:
                raise ValueError(f"N={p['n']}: {len(stetja)} slovarjev izidov, {len(p['vhodi'])} vhodov, "
                                 f"kopij {kopij}, st_vezij {p['st_vezij']}")
            if any(sum(st.values()) != podatki["shots"] for st in stetja):
                raise ValueError(f"N={p['n']}: slovar izidov nima {podatki['shots']} posnetkov")
            if kopij > 1:  # kopije istega vhoda so zapored, njihovi izidi se seštejejo
                imena = (posel.get("imena_vezij") or [])[p["zacetek"]:p["zacetek"] + p["st_vezij"]]
                if imena:  # ime kopije je <ime vezja>_p<c>
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


# ------------------------------------------------------------------------------------ tabela in graf


def _zaokrozi(x, mest):
    """Zaokroži na `mest` decimalk, polovica navzgor (0,53125 -> 0,5313; 0,8115 -> 0,812)."""
    return Decimal(str(float(x))).quantize(Decimal(1).scaleb(-mest), rounding=ROUND_HALF_UP)


def _st(x, mest=3):
    return str(_zaokrozi(x, mest)).replace(".", "{,}")


def _z_napako(m, s, mest=3):
    """Celica tabele z deležem zmag in standardno napako: 0,949 +- 0,003."""
    return f"{_st(m, mest)} $\\pm$ {_st(s, mest)}"


def _beri(mapa, ime):
    pot = mapa / f"{ime}.json"
    return {r["n"]: r for r in json.loads(pot.read_text())["rezultati"]} if pot.exists() else {}


def _datum(ime_qpu):
    """Dan oddaje naloge kot 'd. m.' (čas oddaje je zapisan v UTC)."""
    pot = QPU / f"{ime_qpu}.json"
    if not pot.exists():
        return ""
    cas = json.loads(pot.read_text())["posli"][0]["cas_oddaje"]
    return f"{int(cas[8:10])}. {int(cas[5:7])}."


def tabela_ponovitve(ime_qpu, ime_qpu2):
    """Vrstice tabele z obema meritvama na procesorju."""
    prva, druga = _beri(QPU, f"{ime_qpu}_analiza"), _beri(QPU, f"{ime_qpu2}_analiza")

    def kriterij(r, meja):
        return "da" if r["povprecje"] - 2 * r["sigma"] > meja else "ne"

    print(f"\n% N & ibm_fez {_datum(ime_qpu)} & kriterij & ibm_fez {_datum(ime_qpu2)} & kriterij & ponovitev na vhod")
    for n in sorted(druga):
        a, b = prva.get(n), druga[n]
        meja = klasicna_meja(n)
        print(" & ".join([str(n), _z_napako(a["povprecje"], a["sigma"]) if a else "--", kriterij(a, meja) if a else "--",
                          _z_napako(b["povprecje"], b["sigma"]), kriterij(b, meja), str(b.get("shots", ""))]) + r" \\")


def tabela(ime_sim, ime_qpu):
    sim, qpu = _beri(SIM, ime_sim), _beri(QPU, f"{ime_qpu}_analiza")
    print("% N & p_klas & p_ideal & p_šum & p_ibm & p_ibm - 2 sigma > p_klas")
    for n in sorted(set(sim) | set(qpu)):
        s, q = sim.get(n), qpu.get(n)
        meja = klasicna_meja(n)
        celice = [str(n), _st(meja, 4),
                  _st(s["idealno"]["povprecje"]) if s else "--",
                  _z_napako(s["sumno"]["povprecje"], s["sumno"]["sigma"]) if s else "--",
                  _z_napako(q["povprecje"], q["sigma"]) if q else "--",
                  ("da" if q["povprecje"] - 2 * q["sigma"] > meja else "ne") if q else "--"]
        print(" & ".join(celice) + r" \\")
    print("\n% N & globina pred & globina po & trajanje [us]")
    for n in sorted(qpu):
        pred, po = qpu[n]["lastnosti_pred"], qpu[n]["lastnosti_po"]
        print(" & ".join([str(n), str(pred["globina"]), str(po["globina"]), _st(po["trajanje_us"], 2)]) + r" \\")
    # dvokubitnih vrat mora biti N - 1 pred prevajanjem in po njem
    drugace = [n for n in sorted(qpu) if {qpu[n]["lastnosti_pred"]["dvokubitna"], qpu[n]["lastnosti_po"]["dvokubitna"]} != {n - 1}]
    print("% dvokubitnih vrat je pri vseh N pred prevajanjem in po njem N - 1" if not drugace
          else f"% POZOR: število dvokubitnih vrat ni N - 1 pri N = {drugace}")


def graf(ime_sim, ime_qpu, ime_qpu2=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sim, qpu = _beri(SIM, ime_sim), _beri(QPU, f"{ime_qpu}_analiza")
    qpu2 = _beri(QPU, f"{ime_qpu2}_analiza") if ime_qpu2 else {}
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    # klasična meja pri vsakem celem N, ne le pri izmerjenih
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
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.1f}".replace(".", ","))  # decimalna vejica
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(SLIKA)
    print(f"shranjeno: {SLIKA.relative_to(KOREN)}")


# ---------------------------------------------------------------------------------------- shema čipa


def koordinate_cipa(povezave, st_kubitov=156):
    """Položaj (x, y) vsakega kubita na shemi čipa Heron r2 s 156 kubiti (ibm_fez).

    Čip ima 8 vrstic po 16 kubitov: kubit 20 v + s je v vrstici v in stolpcu s (s < 16). Med zaporednima vrsticama so
    po 4 vmesni kubiti (20 v + 16 do 20 v + 19), ki povezujejo kubita istega stolpca. Povezava, ki na shemi ni med
    sosedoma, sproži ValueError."""
    if st_kubitov != 156:
        raise ValueError("shema je narejena za čip s 156 kubiti")
    sosedi = {q: set() for q in range(st_kubitov)}
    for a, b in povezave:
        sosedi[a].add(b)
        sosedi[b].add(a)
    xy = {}
    for q in range(st_kubitov):
        v, s = divmod(q, 20)
        if s < 16:
            xy[q] = (s, 2 * v)
        else:
            stolpci = {t % 20 for t in sosedi[q]}
            if len(stolpci) != 1:
                raise ValueError(f"vmesni kubit {q} ne povezuje dveh kubitov istega stolpca")
            xy[q] = (stolpci.pop(), 2 * v + 1)
    for a, b in povezave:
        (xa, ya), (xb, yb) = xy[a], xy[b]
        if abs(xa - xb) + abs(ya - yb) != 1:
            raise ValueError(f"povezava {a}-{b} na shemi ni med sosedoma")
    return xy


def drevo_iz_postavitve(postavitev, povezave):
    """Starši kubitov v drevesu GHZ iz shranjene postavitve (indeksi v postavitvi, začetni kubit ima None).

    Starš kubita je njegov sosed na čipu, ki je v postavitvi najbolj spredaj (vrstni red BFS iz `ghz.najdi_drevo`).
    Povezave z napako vrat 1 morajo biti izpuščene tudi tukaj."""
    sosedi = {}
    for a, b in povezave:
        sosedi.setdefault(a, set()).add(b)
        sosedi.setdefault(b, set()).add(a)
    starsi = [None]
    for i in range(1, len(postavitev)):
        prej = [j for j in range(i) if postavitev[j] in sosedi.get(postavitev[i], ())]
        if not prej:
            raise ValueError(f"kubit {postavitev[i]} ni povezan z nobenim prejšnjim kubitom postavitve")
        starsi.append(prej[0])
    return starsi


def graf_cipa(ime_qpu, n):
    """Shema čipa ibm_fez s kubiti in vrati CNOT zagona `ime_qpu` pri N = n."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from qiskit_ibm_runtime.fake_provider import FakeFez

    backend = FakeFez()  # iste povezave kot ibm_fez
    povezave = sorted({tuple(sorted(p)) for p in backend.coupling_map.get_edges()})
    xy = koordinate_cipa(povezave, backend.num_qubits)
    poskus = next(p for p in json.loads((QPU / f"{ime_qpu}.json").read_text())["posli"][0]["poskusi"] if p["n"] == n)
    postavitev = poskus["postavitev"]
    starsi = drevo_iz_postavitve(postavitev, povezave)
    # obnovljeno drevo mora dati enako globino vezja, kot je zapisana ob zagonu
    priprava = ghz.ghz_drevo(postavitev, starsi).priprava
    vezja = [vezje_igre(priprava, x) for x in poskus["vhodi"]]
    if ghz.lastnosti(vezja[najglobje(vezja)])["globina"] != poskus["lastnosti_pred"]["globina"]:
        raise ValueError("obnovljeno drevo se ne ujema z zapisano globino vezja")
    drevo = {tuple(sorted((postavitev[i], postavitev[starsi[i]]))) for i in range(1, n)}

    barva, siva, crnilo = "#D55E00", "0.62", "0.15"  # ista barva kot v grafu deleža zmag
    visina = 0.74  # razmik med vrsticami glede na razmik med stolpci
    tocka = {q: (x, -visina * y) for q, (x, y) in xy.items()}
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for a, b in povezave:
        v_drevesu = (a, b) in drevo
        ax.plot(*zip(tocka[a], tocka[b]), color=barva if v_drevesu else siva, linewidth=2.6 if v_drevesu else 0.7,
                solid_capstyle="round", zorder=2 if v_drevesu else 1)
    for q, (x, y) in tocka.items():
        izbran, zacetni = q in postavitev, q == postavitev[0]
        ax.add_patch(plt.Circle((x, y), 0.31, facecolor=barva if izbran else "white",
                                edgecolor="black" if zacetni else (barva if izbran else siva),
                                linewidth=1.6 if zacetni else 0.7, zorder=3))
        ax.text(x, y, str(q), ha="center", va="center_baseline", fontsize=5.2, zorder=4,
                color="black" if izbran else "0.35")
    ax.set_xlim(-0.6, 15.6)
    ax.set_ylim(-visina * 14 - 0.6, 0.6)
    ax.set_aspect("equal")
    ax.axis("off")
    krog = dict(marker="o", linestyle="none", markersize=7)
    ax.legend(handles=[
        Line2D([], [], markerfacecolor="white", markeredgecolor=siva, markeredgewidth=0.7, **krog),
        Line2D([], [], color=siva, linewidth=0.7),
        Line2D([], [], markerfacecolor=barva, markeredgecolor=barva, markeredgewidth=0.7, **krog),
        Line2D([], [], markerfacecolor=barva, markeredgecolor="black", markeredgewidth=1.6, **krog),
        Line2D([], [], color=barva, linewidth=2.6)],
        labels=["kubit", "povezava", f"izbran za $N={n}$", "začetni kubit (H)", "vrata CNOT"],
        loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=5, frameon=False, fontsize=7, labelcolor=crnilo,
        handlelength=1.4, handletextpad=0.5, columnspacing=1.2)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0.07)
    fig.savefig(SLIKA_CIPA)
    print(f"shranjeno: {SLIKA_CIPA.relative_to(KOREN)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ukaz", choices=["meja", "sim", "tabela", "graf", "cip"])
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
    elif a.ukaz == "cip":
        if len(a.n) != 1:
            ap.error("cip potrebuje natanko en N, npr. --n 20")
        graf_cipa(a.qpu, a.n[0])
    else:
        graf(a.ime, a.qpu, a.qpu2)


if __name__ == "__main__":
    main()
