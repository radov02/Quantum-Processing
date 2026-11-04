"""Stanje GHZ na procesorju IBM: izbira kubitov, vezje priprave, prevajanje in simulatorja.

Izbira kubitov: iz objavljene kalibracije procesorja (napake branja in dvokubitnih vrat) poiščemo drevo N povezanih
kubitov z najmanjšo oceno napake (`najdi_drevo`). Stanje GHZ pripravimo z vrati H na korenu in vrati CNOT od vsakega
starša k otroku, zato ima vezje točno N - 1 dvokubitnih vrat in prevajalnik ne doda nobenih vrat SWAP.

Simulatorja: idealni (Aer brez šuma) in šumni (Aer s šumnim modelom procesorja, npr. FakeFez). Pri šumnem napake
dvokubitnih vrat nadomestimo s Paulijevim približkom, napake branja pa dodamo po simulaciji za vsak merjeni fizični kubit.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager

# število niti simulatorja (0 = vse jedra); GHZ_NITI omeji porabo procesorja in pomnilnika
NITI = int(os.environ.get("GHZ_NITI") or 0)

# ---------------------------------------------------------------------------------------------- čip


def dvokubitna_vrata(backend):
    """Ime dvokubitnih vrat, ki jih procesor izvaja neposredno (pri ibm_fez so to vrata CZ)."""
    for ime in ("cz", "ecr", "cx"):
        if ime in backend.target.operation_names:
            return ime
    raise ValueError("procesor nima znanih dvokubitnih vrat")


# Kubiti, ki jih izbira ne sme uporabiti (qpu.py --brez-kubitov); privzeto nobeden.
IZKLJUCENI_KUBITI = set()


def napake(backend):
    """Napake branja po kubitih in napake dvokubitnih vrat po povezavah (iz kalibracije lažnega/pravega procesorja)."""
    t = backend.target
    branje = {}
    for q, lastnosti in t["measure"].items():
        branje[q[0]] = (lastnosti.error if lastnosti and lastnosti.error is not None else 0.05)
    vrata = {}
    for q, lastnosti in t[dvokubitna_vrata(backend)].items():
        e = lastnosti.error if lastnosti and lastnosti.error is not None else 0.05
        kljuc = tuple(sorted(q))
        vrata[kljuc] = min(e, vrata.get(kljuc, 1.0))
    sosedi = {q: set() for q in range(backend.num_qubits)}
    for a, b in vrata:
        sosedi[a].add(b)
        sosedi[b].add(a)
    # kubiti z napako 1 (pokvarjeni) ali napako branja >= 0,5 ne pridejo v poštev
    slabi = {q for q, e in branje.items() if e >= 0.5} | IZKLJUCENI_KUBITI
    for (a, b), e in vrata.items():
        if e >= 1.0:
            sosedi[a].discard(b)
            sosedi[b].discard(a)
    return branje, vrata, sosedi, slabi


def cena(kubiti, povezave, branje, vrata):
    """Skupna ocena napake izbire: vsota napak branja kubitov in napak dvokubitnih vrat na povezavah."""
    return sum(branje[q] for q in kubiti) + sum(vrata[tuple(sorted(p))] for p in povezave)


CENA_NIVOJA = 0.0015


def najdi_drevo(backend, n):
    """Izbere n povezanih kubitov čipa in drevo povezav med njimi.

    Za vsak uporaben kubit čipa kot koren z iskanjem v širino (BFS) zgradimo drevo: iz kubita gremo najprej na
    sosede z najmanjšo napako (napaka dvokubitnih vrat na povezavi + napaka branja soseda), dokler nimamo n kubitov.
    Vsako drevo ocenimo: seštejemo napake branja vseh kubitov in napake vrat na vseh povezavah drevesa (`cena`) ter
    dodamo majhno kazen za globino drevesa (CENA_NIVOJA na kubit in nivo; groba ocena izgube zaradi čakanja, glej
    komentar spodaj). Obdržimo drevo z najmanjšo oceno.

    Vrne (kubiti v vrstnem redu BFS, starsi), kjer je starsi[i] indeks starša kubita i (koren ima None)."""
    branje, vrata, sosedi, slabi = napake(backend)
    najboljse, najboljsi_kljuc = None, None
    for koren in sosedi:
        if koren in slabi:
            continue
        vrstni_red, starsi, nivo = [koren], {koren: None}, {koren: 0}
        vrsta = deque([koren])
        while vrsta and len(vrstni_red) < n:
            q = vrsta.popleft()
            for s in sorted(sosedi[q], key=lambda s: vrata[tuple(sorted((q, s)))] + branje[s]):
                if s in starsi or s in slabi or len(vrstni_red) >= n:
                    continue
                starsi[s], nivo[s] = q, nivo[q] + 1
                vrstni_red.append(s)
                vrsta.append(s)
        if len(vrstni_red) < n:
            continue
        povezave = [(q, starsi[q]) for q in vrstni_red[1:]]
        # napake kubitov in vrat + cena globine: vsak nivo drevesa pomeni ~2 plasti vrat (~0,15 us) čakanja za
        # ~n kubitov, pri T2 ~ 100 us je to ~0,0015 izgube koherence na kubit in nivo
        globina_drevesa = max(nivo[q] for q in vrstni_red)
        kljuc = cena(vrstni_red, povezave, branje, vrata) + CENA_NIVOJA * n * globina_drevesa
        if najboljsi_kljuc is None or kljuc < najboljsi_kljuc:
            indeks = {q: i for i, q in enumerate(vrstni_red)}
            najboljse = (vrstni_red, [None] + [indeks[starsi[q]] for q in vrstni_red[1:]])
            najboljsi_kljuc = kljuc
    if najboljse is None:
        raise ValueError(f"na čipu ni drevesa z {n} kubiti")
    return najboljse


# ------------------------------------------------------------------------------------------- vezja


@dataclass
class Zasnova:
    """Logično vezje priprave GHZ in kubiti na čipu, na katere ga postavimo (kubit i -> postavitev[i])."""
    ime: str
    priprava: QuantumCircuit
    postavitev: list


def ghz_drevo(postavitev, starsi):
    """Priprava GHZ po drevesu: H na korenu, nato CNOT od starša k otroku (starš je v BFS vedno pred otrokom)."""
    n = len(postavitev)
    qc = QuantumCircuit(n, name=f"drevo_{n}")
    qc.h(0)
    for i in range(1, n):
        qc.cx(starsi[i], i)
    return Zasnova("drevo", qc, list(postavitev))


def zasnova_za(backend, n):
    """Drevo GHZ z n kubiti na najboljših kubitih procesorja."""
    return ghz_drevo(*najdi_drevo(backend, n))


# ------------------------------------------------------------------------------ prevajanje in zagon


def prevedi(vezja, backend, postavitev, seed=7):
    """Prevajanje s fiksno postavitvijo kubitov in časovnim razporedom ALAP (da so čakanja vidna kot `delay`)."""
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend, initial_layout=postavitev,
                                      seed_transpiler=seed, scheduling_method="alap")
    return [_brez_praznih_kubitov(qc) for qc in pm.run(vezja)]


def _brez_praznih_kubitov(qc):
    """Razpored doda `delay` tudi na neuporabljene kubite čipa; odstranimo jih, da jih simulator lahko izpusti
    (na rezultat ne vplivajo, ker ti kubiti niso prepleteni z našimi)."""
    uporabljeni = {q for inst in qc.data if inst.operation.name not in ("barrier", "delay") for q in inst.qubits}
    novo = qc.copy_empty_like()
    for inst in qc.data:
        if inst.operation.name == "delay" and not set(inst.qubits) & uporabljeni:
            continue
        if inst.operation.name == "barrier":
            qubiti = [q for q in inst.qubits if q in uporabljeni]
            if qubiti:
                novo.barrier(qubiti)
            continue
        novo.append(inst)
    return novo


def lastnosti(qc, target=None):
    """Število uporabljenih kubitov, globina, dvokubitna vrata in ocenjeno trajanje [us] (če je podan target)."""
    uporabljeni = {qc.find_bit(q).index for inst in qc.data for q in inst.qubits
                   if inst.operation.name not in ("barrier", "delay")}
    dvo = sum(1 for inst in qc.data if inst.operation.num_qubits == 2 and inst.operation.name != "barrier")
    globina = qc.depth(lambda inst: inst.operation.name not in ("barrier", "delay"))
    trajanje = None
    if target is not None:
        try:
            trajanje = float(qc.estimate_duration(target, unit="u"))
        except Exception:  # noqa: BLE001  (logično vezje brez časov vrat)
            trajanje = None
    return {"kubiti": len(uporabljeni), "globina": globina, "dvokubitna": dvo, "trajanje_us": trajanje}


class SumniSimulator:
    """Šumni simulator: Aer s šumnim modelom procesorja, napako branja pa dodamo sami po simulaciji.

    Aer simulira vrata in čakanje kubitov po modelu procesorja (napake dvokubitnih vrat so poenostavljene, glej
    `_pauli_priblizek_dvokubitnih`), a brez napake branja. Razlog: Aer 0.18 z metodo MPS obvisi, ko ima vezje hkrati
    termično relaksacijo (izgubo stanja med čakanjem) in napako branja (opaženo na FakeMontrealV2). Napaka branja je
    klasična (prebrani bit se obrne), zato jo dodamo po simulaciji na vsak izmerjeni bit: stanje 0 se prebere kot 1 z
    verjetnostjo p(1|0), stanje 1 kot 0 z verjetnostjo p(0|1), in to za fizični kubit, ki ga vezje meri."""

    def __init__(self, backend, metoda="matrix_product_state", seed=11):
        """Prebere verjetnosti napake branja po kubitih in pripravi Aer s šumnim modelom brez napake branja."""
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel

        self.branje = {}
        for napaka in NoiseModel.from_backend(backend).to_dict()["errors"]:
            if napaka["type"] == "roerror":
                for qubiti in napaka["gate_qubits"]:
                    self.branje[qubiti[0]] = np.array(napaka["probabilities"], dtype=float)
        # Aer iz `target` naredi simetrično napako (povprečje obeh smeri); kalibracija procesorja pa ima obe
        # smeri ločeno (|1> se bere slabše, ker med branjem razpade). Če jih imamo, uporabimo te.
        lastnosti_ = backend.properties() if hasattr(backend, "properties") else None
        if lastnosti_ is not None:
            for q in range(backend.num_qubits):
                try:
                    p10 = lastnosti_.qubit_property(q, "prob_meas1_prep0")[0]
                    p01 = lastnosti_.qubit_property(q, "prob_meas0_prep1")[0]
                except Exception:  # noqa: BLE001  (kubit brez podatka: ostane simetrična napaka)
                    continue
                self.branje[q] = np.array([[1 - p10, p10], [p01, 1 - p01]], dtype=float)
        self.branje_vir = "properties" if lastnosti_ is not None else "šumni model (simetrično)"
        self.seed = seed
        self.klic = 0
        brez_branja = NoiseModel.from_backend(backend, readout_error=False)
        _pauli_priblizek_dvokubitnih(brez_branja, dvokubitna_vrata(backend))
        self.sim = AerSimulator(method=metoda, noise_model=brez_branja, seed_simulator=seed,
                                  max_parallel_threads=NITI)
        self.rng = np.random.default_rng(seed)

    def pozeni(self, prevedena, shots):
        """Izvede prevedena vezja in vrne slovar štetij za vsako vezje (niz bitov -> število ponovitev).

        Aer vrne izid vsake ponovitve posebej (`memory=True`); vsak prebrani bit nato z verjetnostjo napake branja
        njegovega kubita obrnemo in ponovitve preštejemo. Niz ima klasični bit 0 na desni, kot pri Qiskitu."""
        self.klic += 1
        rezultat = self.sim.run(prevedena, shots=shots, memory=True,
                                seed_simulator=self.seed + 1000 * self.klic).result()
        stetja = []
        for i, qc in enumerate(prevedena):
            klasicni_v_kubit = {qc.find_bit(inst.clbits[0]).index: qc.find_bit(inst.qubits[0]).index
                                for inst in qc.data if inst.operation.name == "measure"}
            nizi = rezultat.get_memory(i)
            st_bitov = len(nizi[0])
            biti = np.array([[c == "1" for c in reversed(s)] for s in nizi], dtype=bool)
            for k in range(st_bitov):
                p = self.branje.get(klasicni_v_kubit.get(k))
                if p is None:
                    continue
                r = self.rng.random(len(nizi))
                # p[t][m] = verjetnost, da stanje t preberemo kot m
                biti[:, k] = np.where(biti[:, k], r >= p[1][0], r < p[0][1])
            stetje = {}
            for vrstica in biti:
                niz = "".join("1" if b else "0" for b in reversed(vrstica))
                stetje[niz] = stetje.get(niz, 0) + 1
            stetja.append(stetje)
        return stetja


def _pauli_priblizek_dvokubitnih(sumni_model, ime_vrat):
    """Poenostavi šum dvokubitnih vrat v šumnem modelu (spremeni podani model).

    Šumni model zapiše napako dvokubitnih vrat z zapletenim matematičnim opisom (Krausovi operatorji:
    depolarizacija in izguba stanja med vrati). Aer 0.18 z metodo MPS pri takem opisu obvisi (opaženo na
    FakeMontrealV2). Zato napako nadomestimo z navadnim naključnim šumom: po vratih se na obeh kubitih z določeno
    verjetnostjo zgodi ena od 16 kombinacij napak I (nič), X (zamenja 0 in 1), Z (obrne predznak pri |1>) in Y
    (oboje). Verjetnost kombinacije P izračunamo iz Krausovih operatorjev K_k: p_P = vsota_k |sled(P K_k)|^2 / 16
    (v literaturi: Paulijev približek, angl. Pauli twirling). Tako ostane povprečna napaka vrat enaka kot prej.

    Izgubimo le nesimetrični del izgube stanja med samimi vrati, ki je majhen (vrata CZ na FakeFez trajajo
    ~0,1 us, T1 je ~100 us). Izguba stanja med čakanji in na enokubitnih vratih ostane nespremenjena.
    Uporablja notranji slovar `_local_quantum_errors` (Aer 0.18.2, različica je zaklenjena v requirements.txt)."""
    from qiskit.quantum_info import Kraus, Pauli
    from qiskit_aer.noise import pauli_error

    oznake = [a + b for a in "IXYZ" for b in "IXYZ"]
    matrike = [Pauli(o).to_matrix() for o in oznake]
    napake_vrat = sumni_model._local_quantum_errors.get(ime_vrat, {})
    if not napake_vrat:
        raise RuntimeError(f"šumni model nima napak za vrata {ime_vrat}: Paulijevega približka ni mogoče narediti")
    for qubiti, napaka in list(napake_vrat.items()):
        krausi = Kraus(napaka).data
        p = np.array([sum(abs(np.trace(m.conj().T @ k)) ** 2 for k in krausi) / 16 for m in matrike])
        p = np.clip(p.real, 0, None)
        p /= p.sum()
        sumni_model._local_quantum_errors[ime_vrat][qubiti] = pauli_error(
            [(o, float(v)) for o, v in zip(oznake, p) if v > 1e-12])


class IdealniSimulator:
    """Aer brez šuma: izvede vezja brez napak in vrne slovar štetij za vsako vezje."""

    def __init__(self, metoda="matrix_product_state", seed=11):
        from qiskit_aer import AerSimulator

        self.sim = AerSimulator(method=metoda, seed_simulator=seed, max_parallel_threads=NITI)
        self.seed, self.klic = seed, 0

    def pozeni(self, prevedena, shots):
        self.klic += 1
        rezultat = self.sim.run(prevedena, shots=shots, seed_simulator=self.seed + 1000 * self.klic).result()
        return [rezultat.get_counts(i) for i in range(len(prevedena))]


def simulator(backend=None, metoda="matrix_product_state", seed=11):
    """Idealni simulator (brez backend) ali šumni simulator z modelom procesorja `backend`."""
    return IdealniSimulator(metoda, seed) if backend is None else SumniSimulator(backend, metoda, seed)
