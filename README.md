# Igra GHZ na procesorju IBM: kvantna in klasična strategija ob šumu

Seminarska naloga pri predmetu NPMP 2026/27 (kvantno procesiranje), Skupina 20: Erik Radovičevič in Dominik Krašovec.

Repozitorij: [github.com/radov02/Quantum-Processing](https://github.com/radov02/Quantum-Processing).

**Raziskovalno vprašanje.** Do koliko igralcev N kvantna strategija na šumnem procesorju IBM Heron (ibm_fez, 156
kubitov) v igri GHZ še premaga najboljšo klasično strategijo in koliko šum pri tem zmanjša delež zmag?

**Glavni rezultat.** V dveh meritvah na ibm_fez (11. in 17. 11. 2026) je bil delež zmag kvantne strategije do 40
igralcev v obeh za več kot dve standardni napaki nad klasično mejo, pri 50 in 60 igralcih v eni od dveh, pri 75 v
nobeni.

## Igra GHZ

N igralcev (N >= 3) dobi vsak en bit x_i, vsota vseh bitov je soda. Igralci se med igro ne pogovarjajo, vsak odgovori z
enim bitom y_i. Zmagajo, če je pariteta odgovorov enaka (x_1 + ... + x_N) / 2 po modulu 2.

- **Klasična strategija:** pri enakomerno naključnih veljavnih vhodih nobena ne zmaga z verjetnostjo, večjo od
  1/2 + 2^(-ceil(N/2)) (klasična meja; za N = 3 je 0,75, za velike N gre proti 0,5).
- **Kvantna strategija:** igralci si vnaprej razdelijo stanje GHZ. Igralec z x_i = 1 na svojem kubitu izvede vrata S,
  nato vsi izvedejo vrata H in kubit izmerijo; izid je odgovor. Brez šuma zmagajo vedno.
- **Merilo:** delež zmag je nad klasično mejo, če je povprečje po vhodih, zmanjšano za dve standardni napaki, še
  vedno večje od meje.

Stanje GHZ pripravimo na drevesu povezanih kubitov: iz vsakega uporabnega kubita čipa z iskanjem v širino zgradimo
drevo z N kubiti in obdržimo tisto z najmanjšo skupno oceno napake (objavljene napake branja in dvokubitnih vrat ter
majhna kazen za globino drevesa). Vezje ima zato N - 1 dvokubitnih vrat in prevajalnik ne doda nobenih vrat SWAP.
Podrobnosti so v poročilu (`porocilo-seminarska-naloga/main.tex`).

V besedilu izvedbo na procesorju imenujemo naloga (angl. job); v kodi in podatkih je to `posel`.

## Vsebina

| Pot | Vsebina |
|---|---|
| `src/ghz.py` | izbira kubitov iz objavljenih napak procesorja (drevo), vezje priprave GHZ, prevajanje, idealni in šumni simulator |
| `src/igra.py` | igra GHZ: pravila, vezje igre, delež zmag, klasična meja (formula in izčrpni pregled strategij), simulacija, tabela, graf in shema čipa |
| `src/qpu.py` | zagon igre na procesorju IBM z nadzorom porabe: `nacrt`, `poslji`, `prevzemi`, `analiza` |
| `src/ibm_skupno.py` | povezava z IBM Quantum in dnevnik nalog |
| `tests/` | samodejni testi: pravila in klasična meja, kvantna strategija na idealnem simulatorju, izbira kubitov, šumni simulator, naloga igre in varovala porabe |
| `rezultati/sim/igra_fakefez.json` | simulacija igre: idealni in šumni simulator (FakeFez), 8 vhodov x 250 ponovitev za N = 3 ... 50 (pri N = 3 so veljavni vhodi le štirje, zato vsak nastopi dvakrat). Zapis je napisala prejšnja različica `igra.py`, zato ima polje `zasnova` in nima polja `seme_simulatorja` (seme simulatorja je bilo 11, kot zdaj) |
| `rezultati/qpu/igra.json` | zagon na ibm_fez 11. 11. 2026: vezja, vhodi, postavitve kubitov, objavljena kakovost izbranih kubitov ob oddaji (`kakovost`: povprečna napaka branja in dvokubitnih vrat ter povprečni T2) in surova števila izidov (72 vezij x 1000 ponovitev). Zapis je napisala prejšnja različica `qpu.py`, zato je število vhodov na N v polju `igra` (zdaj `k`) |
| `rezultati/qpu/igra_analiza.json` | delež zmag in standardna napaka po N, izračunana iz `igra.json` (korak 7) |
| `rezultati/qpu/igra2.json` | druga meritev na ibm_fez 17. 11. 2026: 122 vezij x 1000 ponovitev (isti vhodi; pri N = 60 in 75 je vsako vezje v nalogi štirikrat), kubiti izbrani znova po objavljenih napakah tistega dne in brez kubita 148; surova števila izidov. Nalogo je oddala prejšnja različica `qpu.py`, ki ji je na konec dodala dve pomožni vezji (`kal_0`, `kal_1`, polje `kalibracija`); analiza ju ne uporablja, zato ima naloga 120 vezij igre in 122 vezij skupaj |
| `rezultati/qpu/igra2_analiza.json` | delež zmag po N za drugo meritev (izidi kopij istega vhoda sešteti) |
| `rezultati/kalibracija/` | objavljene lastnosti procesorja ibm_fez (napake branja in vrat ter T1 in T2 po kubitih) z zadnjo posodobitvijo pred oddajo vsake od obeh nalog (`ibm_fez_igra.json`, `ibm_fez_igra2.json`; preneseno 22. 11. 2026). Po teh podatkih so bili izbrani kubiti; iz prve datoteke je objavljena napaka branja kubita 148, ki jo navaja poročilo. Polje `lastnosti` ustreza klicu `service.backend("ibm_fez").properties(datetime=<čas oddaje naloge>)` knjižnice qiskit-ibm-runtime; skripta za prenos ni del repozitorija, za ponoven prenos je potreben račun IBM Quantum |
| `rezultati/naloge.jsonl` | dnevnik obeh nalog na procesorju (ena vrstica na nalogo: job ID, čas, meja, ocena in obračunani čas): igra 21 s, igra2 34 s |
| `porocilo-seminarska-naloga/` | poglavje za zbornik (`main.tex`, `naslovnica.tex`, `references.bib`, sliki v `img/`), prevedeno poročilo `main.pdf`, prosojnice (`prosojnice/prosojnice.tex` in `.pdf`) ter prijava teme (`tasks/`) |
| `requirements.txt` | različice paketov |
| `.env.example` | imeni spremenljivk okolja za dostop do IBM Quantum (brez vrednosti) |

## Namestitev

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Uporabljene različice: Qiskit 2.6.1, Qiskit Aer 0.18.2, qiskit-ibm-runtime 0.51.0 (vse v `requirements.txt`).
Potreben je Python 3.12 ali novejši (zahtevata ga različici paketov NumPy in SciPy v `requirements.txt`); preizkušeno
s Pythonom 3.13 v Linuxu. Ukazi so zapisani za lupino bash.

## Testi

```bash
.venv/bin/python -m pytest tests -q
```

Testi ne potrebujejo računa IBM in trajajo nekaj minut (največ časa porabi gradnja šumnega modela FakeFez).

## Ponovitev rezultatov

Vsi ukazi se poženejo iz korena repozitorija. Spremenljivka `GHZ_NITI` določi število niti simulatorja Aer
(npr. `GHZ_NITI=1`); privzeto jih Aer izbere sam. Koraki 1 do 5 in 7 ne potrebujejo računa IBM: tabeli, graf, shema
čipa in analiza nastanejo iz shranjenih izidov v `rezultati/`. Račun IBM in procesorski čas potrebuje le korak 6.

1. **Klasična meja.** Formula proti izčrpnemu pregledu vseh determinističnih strategij (izčrpni pregled le do
   N = 16, ker ima 2^N korakov):
   ```bash
   .venv/bin/python src/igra.py meja --n 3 4 5 6 7 8 9 10 11 12
   ```
2. **Simulacija** (idealni simulator in šumni simulator z modelom šuma FakeFez, 8 vhodov na N, 250 ponovitev; pri
   N = 3 so veljavni vhodi le štirje, zato vsak nastopi dvakrat). Izidi, ki jih uporablja poročilo, so shranjeni v
   `rezultati/sim/igra_fakefez.json`. Ukaz preskoči N, ki so v datoteki že izračunani, zato bi se z
   `--ime igra_fakefez` končal brez izpisa; ponovitev od začetka ima zato svoje ime:
   ```bash
   GHZ_NITI=1 .venv/bin/python src/igra.py sim --n 3 5 10 20 30 40 50 --k 8 --shots 250 --ime ponovitev
   ```
   Ukaz po vsakem N izpiše vrstico in dopolni `rezultati/sim/ponovitev.json`. Prva vrstica se izpiše po pol minute
   do minute (gradnja šumnega modela), N do 20 so izračunani v približno dveh minutah. Pri večjih N (do 50) simulacija
   na prenosniku traja dolgo; za N = 60 in 75 simulacije nimamo (pri N = 60 se v več kot eni uri ni končala). Vhodi
   za N imajo seme 5 + N; seme simulatorja (11) se pri vsakem naslednjem N v istem zagonu poveča, zato se števila
   ujemajo s shranjenimi, če se N izračunajo v istem vrstnem redu v enem zagonu, kot v zgornjem ukazu. Tabelo in graf
   s ponovljeno simulacijo dobite z `--ime ponovitev` pri ukazih `tabela` in `graf`.
3. **Tabeli** poročila (delež zmag, klasična meja, lastnosti vezij pred prevajanjem in po njem; z `--qpu2 igra2` še
   primerjava obeh meritev):
   ```bash
   .venv/bin/python src/igra.py tabela --qpu2 igra2
   ```
4. **Graf** `porocilo-seminarska-naloga/img/ghz_igra.pdf` (obe meritvi na procesorju):
   ```bash
   .venv/bin/python src/igra.py graf --qpu2 igra2
   ```
5. **Shema čipa** `porocilo-seminarska-naloga/img/ibm_fez_cip.pdf`: povezave med 156 kubiti procesorja ibm_fez (iz
   modela FakeFez) ter kubiti in vrata CNOT, ki smo jih uporabili pri N = 20 (iz `rezultati/qpu/igra.json`; drevo vrat
   CNOT se obnovi iz shranjene postavitve):
   ```bash
   .venv/bin/python src/igra.py cip --qpu igra --n 20
   ```
6. **Zagon na procesorju ibm_fez.** Potreben je račun IBM Quantum, zagon pa **porabi procesorski čas** (Open Plan:
   600 s v 28 dneh), zato najprej `nacrt`, ki ne odda ničesar. Pred zagonom v lupini izvozite spremenljivki iz
   `.env.example` (`QISKIT_IBM_TOKEN` in `QISKIT_IBM_INSTANCE`), ker skripte datoteke `.env` ne berejo same.
   Shranjeni meritvi imata imeni `igra` in `igra2`. `poslji` zapisa plačanega zagona nikoli ne prepiše in oddajo z
   že uporabljenim imenom zavrne, zato imata ponovitvi svoji imeni (`igra_ponovitev`, `igra2_ponovitev`).
   Ukaza z nastavitvami prve meritve z 11. 11. 2026 (8 vhodov na N, 1000 ponovitev, seme vhodov 5, največ 40 s):
   ```bash
   export QISKIT_IBM_TOKEN=<vaš žeton>
   export QISKIT_IBM_INSTANCE=<vaša instanca>
   .venv/bin/python src/qpu.py nacrt --ime igra_ponovitev --k 8 --n 3 5 10 20 30 40 50 60 75 --shots 1000 --seme 5 \
       --model-posnetek-ms 0.284
   .venv/bin/python src/qpu.py poslji --ime igra_ponovitev --k 8 --n 3 5 10 20 30 40 50 60 75 --shots 1000 --seme 5 \
       --max-sekund 40 --max-poslov 1 --faktor-omejitve 1.5 --model-posnetek-ms 0.284
   ```
   Nastavitve druge meritve (17. 11. 2026, največ 60 s): vezja z N >= 60 so v nalogi štirikrat (`--kopije-od 60 4`),
   kubit 148 je izločen. Ukaz zgradi 120 vezij igre; shranjena naloga ima še dve pomožni vezji prejšnje različice
   `qpu.py`:
   ```bash
   .venv/bin/python src/qpu.py poslji --ime igra2_ponovitev --k 8 --n 3 5 10 20 30 40 50 60 75 --shots 1000 --seme 5 \
       --kopije-od 60 4 --brez-kubitov 148 \
       --max-sekund 60 --max-poslov 1 --faktor-omejitve 1.5 --model-posnetek-ms 0.284
   ```
   `poslji` počaka na izide ter zapiše `rezultati/qpu/<ime>.json` in `rezultati/qpu/<ime>_analiza.json`. Tabelo
   ponovljenih meritev izpiše `.venv/bin/python src/igra.py tabela --qpu igra_ponovitev --qpu2 igra2_ponovitev`.
   Kubiti se izberejo po objavljenih napakah procesorja ob zagonu, zato se postavitve in deleži zmag ponovitve
   razlikujejo od shranjenih.

   `poslji` brez `--max-sekund` ne odda ničesar. Naloga dobi zgornjo mejo trajanja (faktor x ocena + 3 s; pri
   shranjenih zagonih 37 s in 58 s), ki se pred oddajo rezervira v meji `--max-sekund`. Nov zagon se zavrne, dokler
   katera od prejšnjih nalog v dnevniku ni obračunana. Če se čakanje prekine, izide in obračun prevzame
   `qpu.py prevzemi --ime <ime>`.

   Celotno pot lahko brez procesorja in brez računa preizkusite z `--fake` (FakeFez in šumni simulator; lokalni zapis
   z `--fake` se ob ponovitvi prepiše):
   ```bash
   .venv/bin/python src/qpu.py poslji --fake --ime preizkus --k 2 --n 3 5 --shots 50 --max-sekund 10
   ```
   Ukaz traja okoli minute in zapiše `rezultati/qpu/preizkus.json` ter `preizkus_analiza.json` (v `.gitignore`).
7. **Analiza** shranjenega zagona (iz surovih izidov v `rezultati/qpu/<ime>.json` ponovno zapiše
   `rezultati/qpu/<ime>_analiza.json`; procesor ni potreben):
   ```bash
   .venv/bin/python src/qpu.py analiza --ime igra
   .venv/bin/python src/qpu.py analiza --ime igra2
   ```

## Poročilo

Poročilo je poglavje zbornika: `porocilo-seminarska-naloga/main.tex` se prevede znotraj predloge zbornika kot mapa
`Skupina20` (v njej `main.tex` ter `img/ghz_igra.pdf` in `img/ibm_fez_cip.pdf`, ker `main.tex` išče slike v `Skupina20/img/`); vnosi iz
`references.bib` gredo v skupni `references.bib` zbornika. Predloga zbornika ni del repozitorija, zato je prevedeni
`main.pdf` priložen. Samostojni `main.pdf` namesto uvodnih strani zbornika vključi `naslovnica.tex` (naslovnica,
kazalo, številčenje).

Prosojnice se prevedejo samostojno, iz mape `porocilo-seminarska-naloga/prosojnice/` (dvakrat, zaradi številčenja
strani); sliki sta v `../img/`:

```bash
pdflatex prosojnice.tex
```
