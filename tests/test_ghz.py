"""Preverbe izbire kubitov, vezja GHZ in šumnega simulatorja, ki jih uporablja igra GHZ.

Zagon: .venv/bin/python -m pytest tests -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit_ibm_runtime.fake_provider import FakeFez

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import ghz  # noqa: E402

BACKEND = FakeFez()


def test_drevo_starsi_pred_otroki_in_brez_swap():
    """Vsak starš je v vrstnem redu pred otrokom, vsaka povezava drevesa je povezava čipa, po prevajanju pa ostane
    N - 1 dvokubitnih vrat (prevajalnik ni dodal nobenih vrat SWAP)."""
    n = 12
    kubiti, starsi = ghz.najdi_drevo(BACKEND, n)
    assert len(set(kubiti)) == n and starsi[0] is None
    assert all(starsi[i] < i for i in range(1, n))
    povezave_cipa = {tuple(sorted(p)) for p in BACKEND.coupling_map.get_edges()}
    assert all(tuple(sorted((kubiti[i], kubiti[starsi[i]]))) in povezave_cipa for i in range(1, n))
    z = ghz.zasnova_za(BACKEND, n)
    vezje = z.priprava.copy()
    vezje.measure_all()
    prevedeno = ghz.prevedi([vezje], BACKEND, z.postavitev)[0]
    assert ghz.lastnosti(prevedeno)["dvokubitna"] == n - 1
    assert "swap" not in prevedeno.count_ops()


def _izven_diagonale(kanal):
    """Največji element matrike prenosa Paulijev (PTM) izven diagonale; Paulijev kanal ima tam same ničle."""
    from qiskit.quantum_info import PTM

    m = PTM(kanal).data
    return np.abs(m - np.diag(np.diag(m))).max()


def test_pauli_priblizek_ohrani_povprecno_zvestobo():
    """Napaka vrat CZ kot v šumnem modelu (depolarizacija + termična relaksacija obeh kubitov s časi T1, T2 in
    trajanjem vrat iz FakeFez): po Paulijevem približku ima kanal enako povprečno zvestobo in je Paulijev, prvotni
    kanal pa ni bil (relaksacija ni Paulijev kanal)."""
    from qiskit.quantum_info import Kraus, average_gate_fidelity
    from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error

    (a, b), vrata = next((q, v) for q, v in BACKEND.target["cz"].items() if v.error is not None and v.error < 0.05)
    relaksacija = []
    for q in (a, b):
        t1, t2 = BACKEND.target.qubit_properties[q].t1, BACKEND.target.qubit_properties[q].t2
        relaksacija.append(thermal_relaxation_error(t1, min(t2, 2 * t1), vrata.duration))
    napaka = depolarizing_error(vrata.error, 2).compose(relaksacija[0].tensor(relaksacija[1]))
    model = NoiseModel()
    model.add_quantum_error(napaka, "cz", [a, b])
    ghz._pauli_priblizek_dvokubitnih(model, "cz")
    nova = Kraus(model._local_quantum_errors["cz"][(a, b)])
    assert average_gate_fidelity(nova) == pytest.approx(average_gate_fidelity(Kraus(napaka)), abs=1e-9)
    assert average_gate_fidelity(nova) < 1 - 1e-6  # kanal ni identiteta, test ni prazen
    assert _izven_diagonale(nova) < 1e-9
    assert _izven_diagonale(Kraus(napaka)) > 1e-6


class _IdealnoZNesimetricnimBranjem(ghz.SumniSimulator):
    """Idealen simulator, napako branja (nesimetrično) pa dodamo enako kot pri šumnem modelu."""

    def __init__(self, branje):
        from qiskit_aer import AerSimulator

        self.sim = AerSimulator(method="matrix_product_state", seed_simulator=3)
        self.branje = branje
        self.rng = np.random.default_rng(3)
        self.seed, self.klic = 3, 0


def _delez_enk(stetje, n):
    """Delež enic na klasičnem bitu i (v nizu Qiskit je bit 0 skrajno desno)."""
    vseh = sum(stetje.values())
    return np.array([sum(v for niz, v in stetje.items() if niz[n - 1 - i] == "1") / vseh for i in range(n)])


def test_sumni_simulator_preklopi_bit_z_napako_merjenega_kubita():
    """Vsak klasični bit se preklopi z napako fizičnega kubita, ki ga meri, v pravi smeri: iz stanja 0 z p(1|0),
    iz stanja 1 z p(0|1). Napake so po kubitih različne in obe smeri ločeni, zato napačna preslikava ali smer pade."""
    n = 6
    z = ghz.zasnova_za(BACKEND, n)
    branje = {q: np.array([[1 - p10, p10], [p01, 1 - p01]])
              for q, p10, p01 in zip(z.postavitev, [0.01, 0.03, 0.05, 0.07, 0.09, 0.11],
                                     [0.15, 0.19, 0.23, 0.27, 0.31, 0.35])}
    nic = QuantumCircuit(n)
    nic.measure_all()
    ena = QuantumCircuit(n)
    ena.x(range(n))
    ena.measure_all()
    prevedena = ghz.prevedi([nic, ena], BACKEND, z.postavitev)
    stetje_0, stetje_1 = _IdealnoZNesimetricnimBranjem(branje).pozeni(prevedena, 20000)
    assert _delez_enk(stetje_0, n) == pytest.approx([branje[q][0][1] for q in z.postavitev], abs=0.008)
    assert 1 - _delez_enk(stetje_1, n) == pytest.approx([branje[q][1][0] for q in z.postavitev], abs=0.012)


def test_sumni_simulator_bere_nesimetricno_branje_iz_kalibracije():
    """Šumni simulator FakeFez: branje iz kalibracije (obe smeri ločeno), napake vrat CZ pa so Paulijeve (približek je
    res uporabljen v modelu, ne le v pomožni funkciji)."""
    from qiskit.quantum_info import Kraus

    sim = ghz.SumniSimulator(BACKEND)
    napake_cz = sim.sim.options.noise_model._local_quantum_errors["cz"]
    assert len(napake_cz) > 100
    assert max(_izven_diagonale(Kraus(e)) for e in list(napake_cz.values())[:20]) < 1e-9
    lastnosti = BACKEND.properties()
    nesimetricni = 0
    for q in range(5):
        p10 = lastnosti.qubit_property(q, "prob_meas1_prep0")[0]
        p01 = lastnosti.qubit_property(q, "prob_meas0_prep1")[0]
        assert sim.branje[q][0][1] == pytest.approx(p10)
        assert sim.branje[q][1][0] == pytest.approx(p01)
        nesimetricni += abs(p10 - p01) > 1e-4
    assert nesimetricni > 0
    assert sim.branje_vir == "properties"


def test_izkljuceni_kubiti_niso_izbrani(monkeypatch):
    """Kubit iz privzete postavitve po izključitvi ni več v njej, število kubitov pa ostane enako."""
    n = 25
    osnovna = ghz.zasnova_za(BACKEND, n).postavitev
    izkljuceni = {osnovna[1], osnovna[-1]}
    monkeypatch.setattr(ghz, "IZKLJUCENI_KUBITI", izkljuceni)
    nova = ghz.zasnova_za(BACKEND, n).postavitev
    assert len(set(nova)) == n
    assert not izkljuceni & set(nova)
