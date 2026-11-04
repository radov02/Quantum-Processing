"""Skupno za qpu.py: povezava z IBM Quantum, štetje dvokubitnih vrat in dnevnik poslov (rezultati/naloge.jsonl).

Dnevnik ima en zapis JSON na vrstico; za vsak job ID velja zadnji zapis. Izidov ni v dnevniku: ti so v
rezultati/qpu/<ime>.json. Dnevnik služi kot zapis porabe (obračunani čas vsakega posla).
"""

import json
import os
import sys
from pathlib import Path

KOREN = Path(__file__).resolve().parent.parent
REZULTATI = KOREN / "rezultati"
DNEVNIK = REZULTATI / "naloge.jsonl"


def povezi():
    """Storitev IBM Quantum z žetonom in instanco iz okolja (QISKIT_IBM_TOKEN, QISKIT_IBM_INSTANCE) ali s shranjenim
    računom; datoteke .env ne bere."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    token = os.environ.get("QISKIT_IBM_TOKEN")
    instance = os.environ.get("QISKIT_IBM_INSTANCE") or None
    if token:
        return QiskitRuntimeService(channel="ibm_quantum_platform", token=token, instance=instance)
    try:
        return QiskitRuntimeService()
    except Exception as e:  # noqa: BLE001
        sys.exit(f"Ni žetona: nastavite QISKIT_IBM_TOKEN v okolju.\nPodrobnosti: {e}")


def st_dvokubitnih(qc):
    return sum(1 for inst in qc.data if inst.operation.num_qubits == 2)


def beri_dnevnik():
    """Zapisi iz naloge.jsonl; za vsak job ID velja zadnji zapis."""
    if not DNEVNIK.exists():
        return []
    zadnji = {}
    for vrstica in DNEVNIK.read_text(encoding="utf-8").splitlines():
        if vrstica.strip():
            z = json.loads(vrstica)
            zadnji[z["job_id"]] = z
    return list(zadnji.values())


def dodaj_v_dnevnik(zapis):
    REZULTATI.mkdir(exist_ok=True)
    with open(DNEVNIK, "a", encoding="utf-8") as f:
        f.write(json.dumps(zapis, ensure_ascii=False) + "\n")


def stetja_iz_rezultata(rezultat):
    """Izidi (slovar bitni niz -> število) za vsako vezje v rezultatu SamplerV2."""
    return [pub.join_data().get_counts() for pub in rezultat]


def dopolni_zapis(job_id, poraba_s):
    """Ko je poraba posla obračunana, v dnevnik doda zapis z obračunanim časom (če se je spremenil).

    Vrne zapis ali None, če posla ni v dnevniku."""
    zapis = next((z for z in beri_dnevnik() if z["job_id"] == job_id), None)
    if zapis is None or zapis.get("porabljen_cas_s") == poraba_s:
        return zapis
    zapis["porabljen_cas_s"] = poraba_s
    dodaj_v_dnevnik(zapis)
    return zapis
