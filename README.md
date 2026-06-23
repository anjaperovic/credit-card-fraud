# Primena neuronskih mreža za predviđanje prevara u finansijama

> **Credit Card Fraud Detection** — implementacija i komparativna analiza pet modela mašinskog učenja za binarnu klasifikaciju finansijskih prevara na visoko nebalansiranom skupu podataka.

---

## Sadržaj

1. [Opis problema](#1-opis-problema)
2. [Podaci](#2-podaci)
3. [Arhitektura modela](#3-arhitektura-modela)
4. [Trening](#4-trening)
5. [Analiza osetljivosti i hiperparametarska optimizacija](#5-analiza-osetljivosti-i-hiperparametarska-optimizacija)
6. [Rezultati evaluacije](#6-rezultati-evaluacije)
7. [Diskusija](#7-diskusija)
8. [Zaključak](#8-zaključak)

---

## 1. Opis problema

Detekcija prevara na kreditnim karticama predstavlja jedan od klasičnih problema binarne klasifikacije u finansijskom sektoru, ali sa jednom ključnom specifičnošću: **ekstreman disbalans klasa**. U realnim bankarskim sistemima, prevare čine tek mali deo ukupnog saobraćaja transakcija — u ovom projektu manje od **0.17%** svih transakcija je označeno kao prevara.

Ovakav disbalans stvara nekoliko izazova:

- **Varljiva metrika tačnosti (Accuracy):** Model koji svaku transakciju proglasi legitimnom postiže tačnost od >99%, ali nula detektovanih prevara — što je potpuno neupotrebljivo.
- **Poslovni kompromis:** Banka mora da balansira između dva dijametralno suprotna cilja — uhvatiti svaku prevaru (visok Recall) i ne blokirati legitimne korisnike (visoka Precision).
- **Relevantne metrike:** Fokus je isključivo na **Precision**, **Recall** i **F1-Score** za klasu prevara (klasa 1), a ne na ukupnoj tačnosti.

Neuronske mreže su odabrane kao primarni alat jer, zahvaljujući nelinearnim aktivacionim funkcijama, imaju kapacitet da detektuju suptilne anomalijske obrasce koji su nevidljivi linearnim klasifikatorima.

---

## 2. Podaci

### Izvor

Dataset je preuzet sa [Kaggle platforme](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — **Credit Card Fraud Detection** skup podataka koji je objavila ULB (Université Libre de Bruxelles) Machine Learning Group.

### Struktura

| Atribut | Tip | Opis |
|---------|-----|------|
| `V1` – `V28` | Float (PCA) | Anonimizovane karakteristike transakcije dobijene PCA transformacijom originalnih podataka |
| `Time` | Float | Broj sekundi proteklih od prve transakcije u skupu |
| `Amount` | Float | Iznos transakcije u eurima |
| `Class` | 0 / 1 | Ciljna promenljiva: 0 = legitimna transakcija, 1 = prevara |

### Statistike

| Parametar | Vrednost |
|-----------|----------|
| Ukupan broj transakcija | 284.807 |
| Broj prevara (Class=1) | 492 (0.173%) |
| Broj legitimnih (Class=0) | 284.315 (99.827%) |
| Podela train/test | 80% / 20% (stratifikovana) |
| Broj atributa | 30 |

### Analiza i preprocesiranje

**Distribucija klasa** potvrđuje ekstremni disbalans — prevare čine manje od 0.5% celokupnog skupa.

**Skaliranje:** Primenjen je `StandardScaler` isključivo na kolone `Time` i `Amount`. Atributi `V1`–`V28` su već skalirani jer su produkt PCA transformacije. Scaler je fitovan **isključivo na trening skupu** kako bi se izbeglo curenje informacija (data leakage) u test skup.

**Stratifikovana podela:** `train_test_split` sa parametrom `stratify=y` osigurava da obe klase budu proporcionalno zastupljene u trening i test skupu.

**SMOTE balansiranje:** Za modele 3 i 4, primenjena je **SMOTE (Synthetic Minority Over-sampling Technique)** tehnika na trening skupu. SMOTE matematičkom interpolacijom na osnovu KNN metode kreira sintetičke uzorke manjinske klase — za razliku od prostog dupliranja, ovi primeri su jedinstveni i realisti. Trening skup je balansiran na odnos 50:50 (sa 227.451 uzoraka po klasi). Test skup je ostao nepromenjen.

---

## 3. Arhitektura modela

Projekat obuhvata **5 eksperimentalnih konfiguracija** sa progresivnim uvođenjem tehnika balansiranja i različitim pristupima detekciji anomalija.

### Model 1 — Sklearn Baseline MLP

Polazna tačka istraživanja. `MLPClassifier` iz scikit-learn biblioteke sa arhitekturom **30 → 64 → 16 → 1** (arhitektura levka). Automatski early stopping.

### Model 2 — Osnovni PyTorch MLP

Ista logika kao M1, ali reimplementirana u PyTorch-u sa punom kontrolom nad procesom treninga:

```
Ulaz: 30 obeležja
  ↓ Linear(30, 64) + ReLU
  ↓ Linear(64, 16) + ReLU
  ↓ Linear(16, 1)
Izlaz: logit → Sigmoid → P(prevara)
```

| Hiperparametar | Vrednost |
|----------------|----------|
| Loss funkcija | BCEWithLogitsLoss |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 2048 |
| Epohe | 20 |
| Seed | 42 |

### Model 3 — XGBoost Klasifikator + SMOTE

Napuštanje arhitekture dubokih mreža u korist ansambla stabala odlučivanja treniranog nad SMOTE-balansiranim podacima. XGBoost gradi stabla sekvencijalno, pri čemu svako naredno ispravlja greške prethodnih.

| Hiperparametar | Vrednost |
|----------------|----------|
| n_estimators | 100 |
| learning_rate | 0.1 |
| max_depth | 6 |
| tree_method | hist |

### Model 4 — PyTorch MLP + SMOTE

Ista MLP arhitektura kao Model 2, ali trenirana nad balansiranim SMOTE skupom. BCEWithLogitsLoss bez `pos_weight` parametra jer su klase matematički ujednačene.

### Model 5 — Autoenkoder (nenadgledani pristup)

Fundamentalno drugačiji pristup: **nenadgledano učenje** zasnovano na anomalijama rekonstrukcije.

```
Enkoder:  30 → 16 → 6 → 3 (bottleneck)
Dekoder:  3 → 6 → 16 → 30
```

Model se trenira **isključivo na legitimnim transakcijama**. Prevare se detektuju na osnovu visoke greške rekonstrukcije — mreža ne zna kako da rekonstruiše anomalije koje nikada nije videla tokom treninga.

**Prag detekcije:** Dinamički postavljen na 99. percentil grešaka rekonstrukcije (top 1% sa najvećom greškom klasifikuje se kao prevara).

---

## 4. Trening

### Reproduktivnost

Fiksiran globalni seed (`RAND_STATE = 42`) za sve biblioteke: Python `random`, NumPy, PyTorch (CPU i CUDA), kao i determinizam CUDNN kernela.

### Trening MLP modela (M2, M4)

Standardna PyTorch trening petlja kroz 20 epoha:
1. `optimizer.zero_grad()` — resetovanje gradijenata
2. Forward pass kroz mrežu
3. `loss.backward()` — backpropagation
4. `optimizer.step()` — ažuriranje težina

**Kriva gubitka M2:** Strmoglavi pad već u prvim epohama (karakteristično za visoko nebalansirane podatke), zatim stabilna konvergencija.

**Kriva gubitka M4:** Prirodniji, postepeni silazni trend zahvaljujući balansiranom trening skupu.

### Trening Autoenkodera (M5)

- **50 epoha** trening isključivo na legitimnim transakcijama
- Loss: `MSELoss` (greška rekonstrukcije)
- Optimizer: Adam sa L2 regularizacijom (`weight_decay=1e-4`)
- `ReduceLROnPlateau` scheduler: prepolavlja learning rate ako se gubitak ne poboljša 3 uzastopne epohe
- Čuvanje najboljeg stanja modela (`best_ae_state`)

**Ključni rezultat:** Na kraju treninga, greška rekonstrukcije za prevare (**18.79**) bila je **41× veća** od greške za legitimne transakcije (**0.45**).

---

## 5. Analiza osetljivosti i hiperparametarska optimizacija

### Analiza osetljivosti — Model 2

Metoda zasnovana na gradijentima: merimo za svaki atribut **koliko bi se predikcija promenila minimalnom izmenom njegove vrednosti**. Primenjeno na prvih 500 uzoraka test skupa sa `requires_grad_(True)`.

**Top 5 obeležja po uticaju:**

| Rang | Atribut | Apsolutni gradijent |
|------|---------|---------------------|
| 1 | V14 | najveći uticaj |
| 2 | Amount | visok uticaj |
| 3 | V4 | srednji uticaj |
| 4 | V12 | srednji uticaj |
| 5 | V10 | manji uticaj |

**Zaključak:** V14 je ubedljivo najvažniji atribut (PCA komponenta sa jakim korelacijama sa originalnim obeležjima transakcije). Amount na visokom mestu potvrđuje intuiciju — prevaranti često testiraju karticama sitnim iznosima ili pokušavaju ekstremno velike transakcije.

### Hiperparametarska optimizacija — Model 5 (bottleneck)

Testirane su 4 vrednosti za dimenziju latentnog prostora (bottleneck) u kraćem treningu od 5 epoha:

| Bottleneck | MSE greška (legitimne) |
|------------|------------------------|
| 2 | 0.606773 |
| 3 | 0.580401 |
| 5 | 0.496129 |
| 8 | 0.460548 |

**Odabrana vrednost: 3** — uprkos tome što bottleneck=8 minimizuje grešku rekonstrukcije normalnih transakcija, veći latentni prostor bi omogućio mreži da uspešno rekonstruiše i prevare, čime bi se uništila osnova detekcije anomalija. Bottleneck=3 svesno forsira kompresiju koja garantuje visoku grešku za anomalije.

---

## 6. Rezultati evaluacije

Sve metrike se odnose na klasu prevara (Class=1) na originalnom, nebalansiranom test skupu.

| Model | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| M1 — Sklearn Baseline | **0.85** | 0.81 | **0.83** |
| M2 — PyTorch Base | 0.81 | 0.71 | 0.76 |
| M3 — XGBoost + SMOTE | 0.35 | 0.87 | 0.49 |
| M4 — PyTorch + SMOTE | 0.68 | **0.83** | 0.75 |
| M5 — Autoenkoder | 0.15 | 0.86 | 0.25 |

### Matrice konfuzije (klasa prevara)

| Model | True Positive | False Negative | False Positive |
|-------|--------------|----------------|----------------|
| M1 | 79 | 19 | 14 |
| M2 | 70 | 28 | 16 |
| M3 | 85 | 13 | 161 |
| M4 | 81 | 17 | 38 |
| M5 | 84 | 14 | 486 |

---

## 7. Diskusija

### Model 1 — Apsolutni pobednik po balansu metrika

Sa F1-score-om od **0.83**, Sklearn Baseline MLP je postigao optimalan balans u projektu. Preciznost od 85% minimizuje lažne uzbune (svega 14 na 56.864 legitimnih transakcija), dok odziv od 81% hvata četiri od pet pravih prevara. Ovo je idealan profil za produkcijsku upotrebu gde je operativni trošak lažnih uzbuna visok.

### Model 4 — Najboljji PyTorch model

PyTorch MLP sa SMOTE balansiranjem postiže Recall od 83% uz prihvatljiv pad preciznosti (68%). Sa svega 38 lažnih uzbuna (nasuprot 161 kod XGBoosta), ovo je kompromis koji je operativno opravdan — SMOTE je pomogao mreži da razvije osetljivost bez dramatičnog povećanja alarma.

### Zamka preagresivnog balansiranja

Model 3 (XGBoost + SMOTE) postiže najviši Recall (87%), ali cena od 161 lažnih uzbuna ga čini upitnim za realni bankovni sistem. Sličan problem bi se pojavio sa `pos_weight` tehnikama koje su testirane u prethodnim verzijama projekta.

### Vrednost nenadgledanog pristupa (Model 5)

Autoenkoder je konceptualno jedinstven: **nikada nije video prevaru tokom treninga**, a uspeo je da detektuje 86% prevara isključivo na osnovu toga koliko su mu one "nepoznate". Niska preciznost (15%) je direktna posledica fiksnog praga (1% saobraćaja), ali u relanom sistemu ova grupa ne ide na trajnu blokadu — prosleđuje se na drugi stepen autentifikacije. Autoenkoder je nezamenljiv za detekciju novih, evoluiranih oblika prevara koji nisu bili vidljivi tokom treninga.

### Ograničenja interpretabilnosti

V1–V28 su anonimizovane PCA komponente — direktna interpretacija nije moguća. Jedini puni interpretabilan signal je `Amount`, koji se potvrđuje na visokom mestu u analizi osetljivosti.

---

## 8. Zaključak

Projekat demonstrira da za problem detekcije finansijskih prevara ne postoji jedinstveno "best" rešenje — odabir modela zavisi od poslovnih prioriteta:

| Prioritet | Preporučeni model |
|-----------|-------------------|
| Maksimalan balans (F1) | **Model 1 — Sklearn MLP** |
| Visok Recall uz kontrolisane alarme | **Model 4 — PyTorch + SMOTE** |
| Apsolutna bezbednost (max Recall) | Model 3 — XGBoost + SMOTE |
| Detekcija nepoznatih prevara | Model 5 — Autoenkoder |

**Ključni uvidi:**

1. **Accuracy je varljiva metrika** — svi modeli postižu ~100% tačnosti, a jedine relevantne su Precision, Recall i F1 za klasu prevara.
2. **Sofisticiranije ≠ bolje** — Sklearn Baseline postiže najviši F1 na celom projektu uprkos svojoj jednostavnosti.
3. **SMOTE podiže Recall, ali snižava Precision** — kompromis koji se mora svesno doneti u skladu sa poslovnim zahtevima.
4. **Nenadgledano učenje otvara nove mogućnosti** — Autoenkoder detektuje anomalije bez labeled podataka, što ga čini korisnim za detekciju evoluiranih napada.

---

## Pokretanje projekta

### Zavisnosti

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn torch xgboost
```

### Podaci

Preuzmite `creditcard.csv` sa [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) i uploadujte ga u Google Colab okruženje pri pokretanju sveske.

### Pokretanje

Otvorite `Projekat.ipynb` u Google Colab-u i pokrenite ćelije sekvencijalno.

---

## Licenca

Ovaj projekat je objavljen pod **MIT licencom**.

```
MIT License — slobodno korišćenje, modifikacija i distribucija uz navođenje izvora.
```

---

## Autor

Projekat rađen u okviru predmeta **Duboko učenje i neuronske mreže**  
Fakultet organizacionih nauka, Univerzitet u Beogradu
