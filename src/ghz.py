"""Stanje GHZ na procesorju IBM: izbira kubitov, vezje priprave, prevajanje, idealni in šumni simulator.

Kubiti so drevo N povezanih kubitov z najmanjšo vsoto objavljenih napak branja in dvokubitnih vrat. Vezje ima vrata H
na korenu in CNOT od starša k otroku, torej N - 1 dvokubitnih vrat in nobenih vrat SWAP.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager

# število niti simulatorja Aer (0 = vsa jedra), iz spremenljivke okolja GHZ_NITI
NITI = int(os.environ.get("GHZ_NITI") or 0)

# ---------------------------------------------------------------------------------------------- čip


def dvokubitna_vrata(backend):
    """Ime dvokubitnih vrat procesorja (pri ibm_fez so to vrata CZ)."""
    for ime in ("cz", "ecr", "cx"):
        if ime in backend.target.operation_names:
            return ime
    raise ValueError("procesor nima znanih dvokubitnih vrat")


# kubiti, ki jih izbira preskoči (qpu.py --brez-kubitov)
IZKLJUCENI_KUBITI = set()


def napake(backend):
    """Napake branja po kubitih, napake dvokubitnih vrat po povezavah, sosedi na čipu in neuporabni kubiti.

    Kjer napaka ni objavljena, velja 0,05."""
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
    # neuporabni: napaka branja >= 0,5 ali izključen kubit; povezave z napako vrat 1 odpadejo
    slabi = {q for q, e in branje.items() if e >= 0.5} | IZKLJUCENI_KUBITI
    for (a, b), e in vrata.items():
        if e >= 1.0:
            sosedi[a].discard(b)
            sosedi[b].discard(a)
    return branje, vrata, sosedi, slabi


def cena(kubiti, povezave, branje, vrata):
    """Vsota napak branja kubitov in napak dvokubitnih vrat na povezavah."""
    return sum(branje[q] for q in kubiti) + sum(vrata[tuple(sorted(p))] for p in povezave)


# kazen za globino na kubit in nivo drevesa: nivo pomeni ~2 plasti vrat (~0,15 us) čakanja,
# kar je pri T2 ~ 100 us ~0,0015
CENA_NIVOJA = 0.0015


def najdi_drevo(backend, n):
    """Izbere n povezanih kubitov čipa in drevo povezav med njimi.

    Iz vsakega uporabnega kubita zgradi drevo z iskanjem v širino (sosedi po naraščajoči napaki vrat in branja) in
    obdrži tisto z najmanjšo vrednostjo cena + CENA_NIVOJA * n * globina drevesa.

    Vrne (kubiti v vrstnem redu BFS, starsi); starsi[i] je indeks starša kubita i, koren ima None."""
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
    """Vezje priprave GHZ in postavitev: logični kubit i je na kubitu čipa postavitev[i]."""
    ime: str
    priprava: QuantumCircuit
    postavitev: list


def ghz_drevo(postavitev, starsi):
    """Priprava GHZ po drevesu: H na korenu, nato CNOT od starša k otroku."""
    n = len(postavitev)
    qc = QuantumCircuit(n, name=f"drevo_{n}")
    qc.h(0)
    for i in range(1, n):
        qc.cx(starsi[i], i)
    return Zasnova("drevo", qc, list(postavitev))


def zasnova_za(backend, n):
    """Priprava GHZ z n kubiti na izbranem drevesu procesorja."""
    return ghz_drevo(*najdi_drevo(backend, n))


# ------------------------------------------------------------------------------ prevajanje in zagon


def prevedi(vezja, backend, postavitev, seed=7):
    """Prevajanje s podano postavitvijo in razporedom ALAP; čakanja ostanejo v vezju kot `delay`."""
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend, initial_layout=postavitev,
                                      seed_transpiler=seed, scheduling_method="alap")
    return [_brez_praznih_kubitov(qc) for qc in pm.run(vezja)]


def _brez_praznih_kubitov(qc):
    """Odstrani `delay` in `barrier` na kubitih čipa, ki jih vezje ne uporablja."""
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
    """Število uporabljenih kubitov, globina, število dvokubitnih vrat in trajanje [us] (le s podanim target)."""
    uporabljeni = {qc.find_bit(q).index for inst in qc.data for q in inst.qubits
                   if inst.operation.name not in ("barrier", "delay")}
    dvo = sum(1 for inst in qc.data if inst.operation.num_qubits == 2 and inst.operation.name != "barrier")
    globina = qc.depth(lambda inst: inst.operation.name not in ("barrier", "delay"))
    trajanje = None
    if target is not None:
        try:
            trajanje = float(qc.estimate_duration(target, unit="u"))
        except Exception:  # vezje brez časov vrat
            trajanje = None
    return {"kubiti": len(uporabljeni), "globina": globina, "dvokubitna": dvo, "trajanje_us": trajanje}


class SumniSimulator:
    """Aer s šumnim modelom procesorja; napaka branja se doda po simulaciji.

    Aer 0.18 z metodo MPS obvisi, ko ima vezje hkrati termično relaksacijo in napako branja (opaženo na
    FakeMontrealV2), zato model nima napake branja. Vsak prebrani bit se obrne z verjetnostjo p(1|0) oziroma p(0|1)
    kubita, ki ga vezje meri."""

    def __init__(self, backend, metoda="matrix_product_state", seed=11):
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel

        self.branje = {}
        for napaka in NoiseModel.from_backend(backend).to_dict()["errors"]:
            if napaka["type"] == "roerror":
                for qubiti in napaka["gate_qubits"]:
                    self.branje[qubiti[0]] = np.array(napaka["probabilities"], dtype=float)
        # šumni model ima le povprečje obeh smeri; properties() ima p(1|0) in p(0|1) ločeno
        lastnosti_ = backend.properties() if hasattr(backend, "properties") else None
        if lastnosti_ is not None:
            for q in range(backend.num_qubits):
                try:
                    p10 = lastnosti_.qubit_property(q, "prob_meas1_prep0")[0]
                    p01 = lastnosti_.qubit_property(q, "prob_meas0_prep1")[0]
                except Exception:  # kubit brez podatka: ostane simetrična napaka
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
        """Slovar štetij (niz bitov -> število ponovitev) za vsako vezje; klasični bit 0 je v nizu na desni."""
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
                # p[t][m]: verjetnost, da se stanje t prebere kot m
                biti[:, k] = np.where(biti[:, k], r >= p[1][0], r < p[0][1])
            stetje = {}
            for vrstica in biti:
                niz = "".join("1" if b else "0" for b in reversed(vrstica))
                stetje[niz] = stetje.get(niz, 0) + 1
            stetja.append(stetje)
        return stetja


def _pauli_priblizek_dvokubitnih(sumni_model, ime_vrat):
    """Napake dvokubitnih vrat v šumnem modelu zamenja s Paulijevimi (spremeni podani model).

    Aer 0.18 z metodo MPS obvisi pri napaki, zapisani s Krausovimi operatorji K_k (opaženo na FakeMontrealV2).
    Verjetnost Paulijeve kombinacije P je p_P = vsota_k |sled(P K_k)|^2 / 16 (Paulijev približek, angl. Pauli
    twirling). Povprečna napaka vrat ostane enaka, izgubi se nesimetrični del relaksacije med vrati.
    Uporablja notranji slovar `_local_quantum_errors` (Aer 0.18.2)."""
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
    """Aer brez šuma; vrne slovar štetij za vsako vezje."""

    def __init__(self, metoda="matrix_product_state", seed=11):
        from qiskit_aer import AerSimulator

        self.sim = AerSimulator(method=metoda, seed_simulator=seed, max_parallel_threads=NITI)
        self.seed, self.klic = seed, 0

    def pozeni(self, prevedena, shots):
        self.klic += 1
        rezultat = self.sim.run(prevedena, shots=shots, seed_simulator=self.seed + 1000 * self.klic).result()
        return [rezultat.get_counts(i) for i in range(len(prevedena))]


def simulator(backend=None, metoda="matrix_product_state", seed=11):
    """Idealni simulator (brez backend) ali šumni simulator za podani backend."""
    return IdealniSimulator(metoda, seed) if backend is None else SumniSimulator(backend, metoda, seed)
