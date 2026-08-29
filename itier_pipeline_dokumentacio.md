# Itier et al. (2021) — Képhamisítás-felderítő Pipeline Részletes Dokumentációja

## `trainmodel_five.py` és `evaluatemodel_five.py`

---

## Tartalom

1. [Áttekintés és kontextus](#1-áttekintés-és-kontextus)
2. [Adatelőkészítés és feature pipeline](#2-adatelőkészítés-és-feature-pipeline)
3. [Model 1 — Random Forest Classifier (RFC)](#3-model-1--random-forest-classifier-rfc)
4. [Model 2 — Support Vector Machine (SVM)](#4-model-2--support-vector-machine-svm)
5. [Model 3 — K-Means](#5-model-3--k-means)
6. [Model 4 — Multi-Layer Perceptron (MLP)](#6-model-4--multi-layer-perceptron-mlp)
7. [Model 5 — Gradient Boosting / XGBoost](#7-model-5--gradient-boosting--xgboost)
8. [Valószínűség-kalibráció](#8-valószínűség-kalibráció)
9. [evaluatemodel_five.py — Kiértékelési pipeline](#9-evaluatemodel_fivepy--kiértékelési-pipeline)
10. [Küszöbválasztás és ROC analízis](#10-küszöbválasztás-és-roc-analízis)
11. [Teljesítmény-optimalizálás](#11-teljesítmény-optimalizálás)

---

## 1. Áttekintés és kontextus

A pipeline célja képhamisítás-felderítés (copy-move / splicing detekció), az Itier et al. (2021) cikk módszertana alapján. A feladat binomiális osztályozás blokkszinten: minden 32×32 pixeles képblokk `Border` (hamisítási határvonal, osztály=1) vagy `Uniform` (homogén, nem manipulált, osztály=0).

A pipeline lépései:
```
Kép → SRM szűrők (5×) → blokkszintű feature kinyerés → StandardScaler
    → 5 osztályozó → hőtérkép → küszöbölés → bináris maszk → kiértékelés
```

Az öt összehasonlított modell:
- **RFC** — a cikk eredeti modellje
- **SVM** — a cikk által említett alternatíva
- **K-Means** — unsupervised alternatíva (szintén a cikkből)
- **MLP** — saját kiegészítés, neurális háló
- **XGBoost / GradientBoosting** — saját kiegészítés, boosting ensemble

---

## 2. Adatelőkészítés és feature pipeline

### 2.1 `balance_classes(X, y)` — Osztályegyensúlyozás

```python
def balance_classes(X, y):
    bi, ui = np.where(y_arr == 1)[0], np.where(y_arr == 0)[0]
    mn = min(len(bi), len(ui))
    idx = np.concatenate([np.random.choice(bi, mn, replace=False),
                          np.random.choice(ui, mn, replace=False)])
```

**Miért szükséges?** A blokkszintű feature-kinyerés során a `Uniform` (háttér) blokkok száma természetesen jóval meghaladja a `Border` blokkok számát — egy tipikus képen a manipulált határ vékony sáv, az egész képterület töredéke. Kiegyensúlyozás nélkül az osztályozók a többségi osztályra (Uniform) specializálódnának, és bár magas overall accuracy-t érnének el, a ritka, fontos `Border` osztályt szinte soha nem ismernék fel (recall ≈ 0).

**Módszer: Undersampling.** A `Border` blokkok száma korlátozza: a `Uniform` blokkokból ugyanannyit tartunk meg véletlenszerűen, amennyi `Border` blokk van. Az `np.random.seed(42)` biztosítja a reprodukálhatóságot.

**Alternatívák és miért nem:** az Oversampling (SMOTE) szintetikus mintákat generál, ami zajt vihet a feature-térbe; a Class Weight beállítás pedig modell-függő, és nem minden sklearn estimator kezeli egyformán.

### 2.2 StandardScaler — Globális normalizáció

```python
scaler = StandardScaler()
X_sc   = scaler.fit_transform(X_bal)
joblib.dump(scaler, "itier_scaler.pkl")
```

**Miért kötelező?** A 60 feature (5 SRM szűrő × 12 statisztika) különböző nagyságrendű értékeket vesz fel: a variancia-jellemzők tág tartományban mozognak, míg a szög-jellemzők (θ, arccos értékek) [0, π] között vannak. A StandardScaler minden feature-t nulla átlaggal és egységnyi szórással normalizál:

```
z = (x - μ) / σ
```

Enélkül:
- Az SVM RBF kernele a nagy varianciájú feature-öket preferálná
- Az MLP gradiensáramlása instabil lenne (vanishing/exploding gradients)
- A távolságalapú módszereknél (K-Means) a nagy értékű feature-ök dominálnának

**Kritikus:** a scaler **csak a tanítóhalmazon kerül illesztésre** (`fit_transform`), a teszteléskor csak `transform` fut. Ez megakadályozza a data leakage-et.

### 2.3 MAX_SVM_SAMPLES és MAX_MLP_SAMPLES — Memóriakezelés

```python
MAX_SVM_SAMPLES = 30_000
MAX_MLP_SAMPLES = 50_000
```

Az SVM számítási komplexitása O(n²)–O(n³) a minták számában — 100 000+ mintán akár órákat venne igénybe. Az MLP esetén a 50 000 korlát inkább a tanítási időt szabályozza. Ha a teljes halmaz meghaladja ezt, véletlenszerű részhalmazt választunk (a seed rögzített a reprodukálhatósághoz).

---

## 3. Model 1 — Random Forest Classifier (RFC)

### 3.1 Hogyan működik az RFC?

A Random Forest egy **ensemble** módszer: sok, egymástól független döntési fát épít, majd a végső predikciót szavazással (osztályozásnál: majoritás) határozza meg.

Két kulcselemből áll:
1. **Bootstrap aggregation (Bagging):** minden fa a tanítóhalmaz véletlenszerű mintájából (visszatevéssel) épül — így mindegyik fa kis mértékben eltér a többitől.
2. **Feature randomizáció:** minden csomópontnál csak a feature-ök egy véletlenszerű részhalmazát vizsgálja a legjobb split keresésénél (`max_features`). Ez diverzifikálja a fákat, csökkenti a korrelációjukat.

A végeredmény: nagy bias-varianciája kompromisszumú, általánosító ensemble, amely robusztus az outlierekre és a feature-skálától független (ezért StandardScaler nem **kötelező** RFC-nél — de a többi modellel való konzisztencia miatt megtartjuk).

### 3.2 Paraméterek és indoklásuk

```python
gs_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid={
        "n_estimators":     [100, 200, 500],
        "max_depth":        [10, 20, None],
        "criterion":        ["gini", "entropy"],
        "min_samples_leaf": [1, 5, 10],
        "max_features":     ["sqrt", "log2"],
    },
    cv=3, n_jobs=-1, scoring="f1"
)
```

**`n_estimators` — [100, 200, 500]:**
A fák száma. Több fa → kisebb varianciájú ensemble, de nem lineárisan: 500 fa nem kétszerese 250-nek. A [100, 200, 500] lefedi a hasznos tartományt: 100 alatti értéknél a szórás még nagy, 500 fölött a javulás marginális, de a számítási idő nő.

**`max_depth` — [10, 20, None]:**
A fa maximális mélysége. `None` = korlátlan (a fa addig nő, amíg a levél tiszta, vagy `min_samples_leaf` meg nem állítja). Mély fák → overfit; sekély fák → underfit. A [10, 20, None] skálán a GridSearch megtalálja a feladathoz optimális kompromisszumot.

**`criterion` — ["gini", "entropy"]:**
A split minőségének mérőszáma. A **Gini-impurity** számítása gyorsabb:
```
Gini(t) = 1 - Σ p(c)²
```
Az **Entropy** Shannon-féle információtartalmon alapul:
```
Entropy(t) = -Σ p(c) · log₂(p(c))
```
Általában minimális különbség van teljesítményben — a GridSearch eldönti, melyik a jobb erre az adathalmazra.

**`min_samples_leaf` — [1, 5, 10] (FIX #3 bővítés):**
Az eredeti kód csak `n_estimators` és `max_depth` paramétert vizsgált. A `min_samples_leaf` meghatározza, hogy egy levélcsomópontban **minimum hány minta kell** — ez közvetlenül szabályozza az overfit-et:
- `1` = korlátlan (lehet 1 mintás levél → overfit)
- `10` = minden levél legalább 10 mintát tartalmaz → általánosabb fa

**`max_features` — ["sqrt", "log2"] (FIX #3 bővítés):**
60 feature esetén:
- `"sqrt"` ≈ 8 feature vizsgálata minden csomópontnál
- `"log2"` ≈ 6 feature vizsgálata

Kisebb érték → diverzebb, korrelációmentesebb fák → jobb ensemble teljesítmény. Ez az RFC egyik legérzékenyebb paramétere.

**`GridSearchCV` vs `RandomizedSearchCV`:**
Az RFC-nél GridSearch-t használunk, mert a keresési tér kezelhető: 3×3×2×3×2 = 108 kombináció, 3-fold CV-vel = 324 illesztés. Ez még számítható időn belül megvalósítható.

---

## 4. Model 2 — Support Vector Machine (SVM)

### 4.1 Hogyan működik az SVM?

Az SVM a legjobb **szeparáló hipersíkot** keresi az osztályok között, maximalizálva a margót (az osztályok legközelebbi pontjától mért távolságot). Nem lineárisan szeparálható adatoknál a **kernel trick** emeli magasabb dimenziós térbe az adatokat, ahol lineáris szeparáció lehetséges.

Az RBF (Radial Basis Function) kernel:
```
K(x, x') = exp(-γ · ||x - x'||²)
```
Ez egy Gauss-szerű hasonlóságmértéket számít két pont között — a γ szabja, milyen gyorsan csökken a hasonlóság.

### 4.2 Paraméterek

```python
gs_svm = GridSearchCV(
    SVC(probability=False, random_state=42),
    param_grid={
        "C":      [0.1, 1.0, 10.0],
        "kernel": ["rbf", "linear"],
        "gamma":  ["scale", "auto"],
    },
    cv=3, n_jobs=-1, scoring="f1"
)
```

**`C` — [0.1, 1.0, 10.0] — Regularizációs paraméter:**
A `C` az ún. "soft margin" paraméter — meghatározza, mennyire engedünk meg hibás osztályozásokat a margó szélesítéséért cserébe:
- Kis `C` (0.1): széles margó, több hibás osztályozás megengedett → jobb általánosítás
- Nagy `C` (10.0): szűk margó, kevés hibás osztályozás → overfit veszélye

**`kernel` — ["rbf", "linear"]:**
- `"rbf"`: nem lineáris határhoz alkalmas, általában jobb valós adatokon
- `"linear"`: lineárisan szeparálható adatoknál elegendő, és lényegesen gyorsabb

**`gamma` — ["scale", "auto"]:**
- `"scale"` = 1 / (n_features × X.var()) — az adatok varianciájával skálázva, általában jobb
- `"auto"` = 1 / n_features — egyszerűbb

**`probability=False` a GridSearch-ben:**
Az SVM nyers decision function-je nem igényli a valószínűség-számítást (`probability=True` drágítja a tanítást). A GridSearch F1 scoring-hoz csak a `predict()` kell. A kalibrált valószínűségek csak utána kerülnek be.

### 4.3 CalibratedClassifierCV (FIX #1)

```python
raw_svm        = SVC(probability=False, random_state=42, **gs_svm.best_params_)
calibrated_svm = CalibratedClassifierCV(raw_svm, method='isotonic', cv=3)
calibrated_svm.fit(X_svm, y_svm)
```

**Probléma:** az eredeti `SVC(probability=True)` Platt-skálázást végez, de a 30 000 mintás subsample-en illesztett modell authentic tesztképeken is `max_p ≈ 1.0`-t adott — a kalibráció a tanítóadatokon volt illesztve → overfit.

**Megoldás:** a `CalibratedClassifierCV` 3-fold keresztvalidációval illeszt: minden foldban a SVM az adatok 2/3-án tanul, a kalibrálás az 1/3-on történik → a kalibrált valószínűségek nem overfit-eltek a tanítóadatokra.

---

## 5. Model 3 — K-Means

### 5.1 Hogyan működik?

A K-Means egy felügyelet nélküli klaszterezési algoritmus: a feature-teret `k` klaszterre osztja, minimalizálva a klaszterekhez tartozó minták és a klasztercentroid közötti négyzetes eltérés összegét. `k=2` esetén két klaszter keletkezik, amelyeket a cikk szerint a `Border` és `Uniform` osztályokhoz rendelünk.

```python
km_clf = KMeansClassifier(n_init=10, max_iter=300, random_state=42)
km_clf.fit(X_sc, y_bal)
```

**`n_init=10`:** 10 különböző véletlenszerű inicializálásból a legjobbat tartja meg (az inicializálás befolyásolja, melyik lokális minimumba konvergál).

**`max_iter=300`:** maximális iterációszám konvergencia előtt.

**`cross_val_score`:** a K-Means esetén nincs built-in CV, ezért `cross_val_score` hívással mérjük a stabilitást.

A K-Means **gyenge baseline**: nem garantált, hogy a két klaszter pontosan leképezi a `Border`/`Uniform` határt — különösen magas dimenziós feature-térben. Az összehasonlításban mégis fontos, mert a cikk is említi mint alternatívát.

---

## 6. Model 4 — Multi-Layer Perceptron (MLP)

### 6.1 Hogyan működik az MLP?

Az MLP egy előrecsatolt neurális háló: bemeneti réteg → egy vagy több rejtett réteg → kimeneti réteg. Minden réteg lineáris transzformációt végez (súlymátrix + bias), amit egy nemlineáris aktivációs függvény követ. A tanulás **backpropagation** + **Adam optimalizáló** segítségével keresztentrópia loss minimalizálásával történik.

Az architektúra esetünkben: `60 → 128 → 64 → 2` (az alapeset), ahol 60 a feature-ök száma, 2 a kimeneti osztályok (Uniform / Border).

### 6.2 Paraméterek — eredeti és javítások

```python
rs_mlp = RandomizedSearchCV(
    MLPClassifier(
        solver="adam",
        max_iter=500, early_stopping=True,
        n_iter_no_change=10, random_state=42
    ),
    param_distributions={
        "hidden_layer_sizes": [(128, 64), (256, 128), (64, 32), (128, 64, 32), (100,)],
        "alpha":              [1e-4, 1e-3, 1e-2, 1e-1],
        "learning_rate_init": [1e-4, 1e-3, 5e-3],
        "activation":         ["relu", "tanh"],           # ÚJ
        "learning_rate":      ["constant", "adaptive"],   # ÚJ
        "batch_size":         [64, 128, 200],             # ÚJ
    },
    n_iter=30, cv=3, n_jobs=-1, scoring="f1", random_state=42
)
```

**`solver="adam"` — Adam optimalizáló:**
Az Adam (Adaptive Moment Estimation) az SGD egy adaptív változata, amely minden paraméterhez külön tanulási rátát tart fenn, az első (momentum) és második (RMS) gradiensmomentek becslése alapján. Nagy adathalmazon (>1000 minta) az `lbfgs` és `sgd` solverekhez képest gyorsabb konvergenciát biztosít.

**`hidden_layer_sizes` — architektúra keresés:**
Az 5 lehetőség különböző komplexitású hálókat fed le:
- `(128, 64)` — 2 réteg, fokozatos szűkülés (alapeset)
- `(256, 128)` — nagyobb kapacitás, több paraméter
- `(64, 32)` — kisebb, gyorsabb, kevésbé overfit-hajlamos
- `(128, 64, 32)` — 3 réteg, mélyebb hierarchikus reprezentáció
- `(100,)` — egyetlen rejtett réteg, a sklearn alapértelmezette

**`alpha` — L2 regularizáció [1e-4, 1e-3, 1e-2, 1e-1]:**
Az `alpha` az L2 penalizációs tag erőssége: a loss függvényhez `α·||w||²` kerül hozzáadásra. Ez bünteti a nagy súlyokat → simább döntési határok → kevésbé overconfident softmax. Nagyobb `alpha` értékek különösen fontosak, mert a softmax kimeneti réteg hajlamos extrém valószínűségeket (≈1.0) produkálni kisebb regularizáció esetén.

**`early_stopping=True, n_iter_no_change=10`:**
Ha a validációs loss 10 egymást követő epokon keresztül nem javul, a tanítás leáll. Ez két előnnyel jár:
1. Megakadályozza az overfittingot (a modell nem "memorizálja" a tanítóadatokat)
2. Felesleges számítást takarít meg

**`activation` — ["relu", "tanh"] (ÚJ javítás):**
- **ReLU** (Rectified Linear Unit): `f(x) = max(0, x)`. Gyors, nem szenved vanishing gradient problémától. Negatív inputra nullát ad → ritka aktiváció.
- **Tanh** (hiperbolikus tangens): `f(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)`, értékkészlete (-1, 1). StandardScaler után a bemenetek nulla körül szimmetrikusak — a tanh szimmetriája ezt kihasználja, és néha jobb osztályozási határt ad zéró-közeli, normalizált adatokon.

**`learning_rate` — ["constant", "adaptive"] (ÚJ javítás):**
- **`"constant"`**: a `learning_rate_init` végig változatlan marad
- **`"adaptive"`**: ha az edzési loss nem csökken az adott tolerancián belül, a tanulási ráta felére csökken. Hatása: a konvergencia közelében finom lépéseket tesz → pontosabb végső optimum. Az `early_stopping=True`-val természetes párba illik: ha a loss nem javul, először lassít, majd megáll.

**`batch_size` — [64, 128, 200] (ÚJ javítás):**
A mini-batch mérete meghatározza, mennyi mintát lát egyszerre a hálózat egy gradienslépés előtt:
- **Kis batch (64):** több gradienslépés epochonként, zajos (de regularizáló hatású) gradiens-becslések → jobb általánosítás, de lassabb
- **Nagy batch (200):** pontosabb gradiens-becslés, stabilabb konvergencia, de esetlegesen élesebb, kevésbé általánosító minimum ("sharp minima")
- Az implicit `'auto'` = `min(200, n_samples)` — nagy mintaszámnál ez 200, ami a legnagyobb batch.

**`n_iter=30` (20-ról növelve):**
A megnövelt keresési tér (5×4×3×2×2×3 = 720 kombináció) esetén a 30 véletlenszerű próba 4,2%-ot fed le — ez statisztikailag elégséges RandomizedSearch esetén (a kutatások szerint 60 véletlenszerű próba a legtöbb esetben közelíti a GridSearch eredményét a 20-30-szoros kombinációs tér esetén is).

**`RandomizedSearchCV` vs `GridSearchCV`:**
Az MLP esetén az RFC-vel ellentétben a keresési tér nagy (720 kombináció), ezért GridSearch helyett RandomizedSearchCV-t használunk. Az `n_iter=30` biztosítja, hogy a véletlenszerű keresés végigjárja a legfontosabb tartományokat.

---

## 7. Model 5 — Gradient Boosting / XGBoost

### 7.1 Hogyan működik a Gradient Boosting?

A Gradient Boosting szintén ensemble módszer, de az RFC-vel ellentétben **szekvenciálisan** épít fákat: minden új fa az előző modell hibáját (a loss negatív gradiensét) próbálja korrigálni.

Az algoritmus:
1. Indul egy konstans (átlag) predikcióval
2. Kiszámolja az aktuális rezidumokat (pszeudo-reziduumok = loss negatív gradiense)
3. Új fát illeszt a rezidumokra
4. A fa predikciójához hozzáad `learning_rate × fa_predikció`-t
5. Megismétli 2-4-et `n_estimators` szor

Az XGBoost az alap Gradient Boosting kiterjesztése: másodrendű Taylor-közelítést alkalmaz (Hesse-mátrix is), beépített L1+L2 regularizációt tartalmaz, és párhuzamos fa-építést végez — ezért általában gyorsabb és jobb teljesítményű.

### 7.2 Közös paraméterek (XGBoost és GB)

**`n_estimators` — [100, 200, 300]:**
A fák (boosting körök) száma. Több fa → kisebb bias, de túl sok kör → overfit. A `learning_rate`-tel együtt értelmezendő: kis tanulási ráta (0.01) sok fát igényel, nagy (0.2) kevesebbet.

**`learning_rate` — [0.01, 0.05, 0.1, 0.2]:**
Minden új fa hozzájárulásának "súlya". Kis érték → több fa szükséges, de általánosabb modell. A boosting-ban ez a "shrinkage" — az overfittel szembeni legfontosabb védelmi vonal.

**`max_depth` — [3, 4, 5, 6]:**
A fák maximális mélysége. A boosting általában sekély fákat preferál (3-5), mivel sok ilyen fa együtt már komplex mintákat tanul meg anélkül, hogy egyenként overfit-elnének. Az RFC-nél mélység 10-20 volt — boosting-nál ennél kisebb értékek az optimálisak.

**`subsample` — [0.7, 0.8, 1.0]:**
Stochastic Gradient Boosting: minden fához csak a minták megadott arányát használja. `0.8` = minden faépítéshez véletlenszerűen választott 80% — ez csökkenti a varianciát és növeli a robusztusságot (az RFC bagging-jéhez hasonlóan).

### 7.3 XGBoost-specifikus paraméterek

**`colsample_bytree` — [0.7, 0.8, 1.0]:**
Feature subsampling: minden fa építésekor a feature-ök csak egy arányát veszi figyelembe. Hasonló a `subsample`-höz, de az oszlopokra (feature-ökre) vonatkozik. Az RFC `max_features`-ének megfelelője boosting-ban.

**`gamma` (min_split_loss) — [0, 0.1, 0.5, 1.0] (ÚJ javítás):**
A minimális loss reduction, amely ahhoz szükséges, hogy egy csomópont egyáltalán felosztódjon. `0` = korlátlan felosztás, `1.0` = csak akkor oszt fel, ha legalább 1.0 loss reduction érhető el. Ez az XGBoost egyik legfontosabb regularizációs paramétere — a keresési tér eredetileg nem tartalmazta.

**`reg_alpha` (L1) — [0, 0.01, 0.1] (ÚJ javítás):**
L1 regularizáció a fa levél-súlyaira. Az L1 "ritka" megoldásokat ösztönöz: sok súly nullára csúszik → implicit feature selection hatás.

**`reg_lambda` (L2) — [0.5, 1.0, 2.0] (ÚJ javítás):**
L2 regularizáció a levél-súlyokra. Az XGBoost alapértelmezése 1.0. Kisebb érték → kevésbé bünteti a nagy súlyokat; nagyobb → simább döntési felületek.

### 7.4 `n_jobs` thread contention javítás

**Probléma (eredeti kód):**
```python
# EREDETI — HIBÁS
XGBClassifier(..., n_jobs=-1)           # minden CPU-t kér
RandomizedSearchCV(..., n_jobs=-1)      # szintén minden CPU-t kér
```
Amikor mindkettő egyszerre `n_jobs=-1`, az operációs rendszer thread-eket oszt el versenyszituációban — a context switching overhead miatt mindkettő **lassabb**, mintha az egyik `n_jobs=1` lett volna.

**Megoldás:**
```python
# JAVÍTOTT
XGBClassifier(..., n_jobs=1)           # az XGBoost nem párhuzamosít
RandomizedSearchCV(..., n_jobs=-1)     # a CV párhuzamosít (foldok párhuzamosan)
```
A RandomizedSearchCV `n_jobs=-1` beállítása a CV foldokat párhuzamosítja — ez a hatékonyabb megközelítés, mivel a foldok teljesen független számítások.

### 7.5 GradientBoosting early stopping (ÚJ javítás)

```python
base_boost = GradientBoostingClassifier(
    random_state=42,
    n_iter_no_change=15,
    validation_fraction=0.1,
)
```
Az sklearn GradientBoosting fallback esetén az early stopping biztosítja, hogy 300 fa helyett a tanítás hamarabb leálljon, ha a loss nem javul. `validation_fraction=0.1` = a tanítóhalmaz 10%-a validációra van fenntartva az early stopping ellenőrzéséhez.

### 7.6 HistGradientBoosting megjegyzés

A kód dokumentálja, miért nem váltottunk `HistGradientBoostingClassifier`-re, noha az 10 000+ mintán 5-10× gyorsabb lenne:
- A `HistGradientBoosting` paraméterei eltérnek (pl. `max_iter` az `n_estimators` helyett)
- Nincs `subsample` paramétere
- A meglévő keresési tér átírása szükséges lenne

---

## 8. Valószínűség-kalibráció

### 8.1 Mi a probléma?

Az osztályozók kimenetének értelmezéséhez szükség van **jól kalibrált valószínűségekre**: ha egy modell egy mintára 0.8-as valószínűséget ad, akkor valóban az esetek ~80%-ában kellene, hogy az a minta a pozitív osztályba tartozzon.

Az MLP és SVM esetén ez nem teljesül automatikusan:
- **SVM:** a decision function skálája tetszőleges, az egyszerű szigmoid illesztés overfit-el
- **MLP:** a softmax keresztentrópia loss-t optimalizál, nem kalibrált valószínűségeket

### 8.2 CalibratedClassifierCV

```python
CalibratedClassifierCV(estimator, method='sigmoid'/'isotonic', cv=3/5)
```

A `CalibratedClassifierCV` a kalibrálást **keresztvalidációban** végzi:
- Az adatokat `cv` (3 vagy 5) részre osztja
- Minden foldban a base estimator a többi foldon tanul, a kalibrálás az aktuális foldon
- A végső modell `cv` db kalibrált modell átlaga

**Isotonic vs. Sigmoid:**

| | Isotonic | Sigmoid (Platt) |
|---|---|---|
| Paraméterei | N (lépésszám) | 2 (A, B) |
| Alkalmazható | >1000 kalibráló minta | Kisebb halmazra is |
| Overfit veszélye | Magas, ha a score-ok szűk tartományban vannak | Alacsony |
| SVM-re | ✅ (cv=3 elégséges) | — |
| MLP-re | ❌ (AUC<0.5 anomália!) | ✅ (cv=5) |

**MLP esetén miért sigmoid?**
Az isotonic regresszió invertálta a score-okat (AUC = 0.383 < 0.5), mert:
- Spliced képeken az MLP `max_p = 0.9976 ± 0.0001` (nagyon szoros klaszter)
- Authentic képeken az MLP `max_p = 0.9687 ± 0.1142` (nagy szórás, egyes authentic képek `max_p > 0.9976`)
- Az isotonic regresszió "megtanulta" ezt az anomáliát, és fordított leképezést épített
- A sigmoid 2 paraméteres egyszerűsége ezt az invertálást nem tudja elvégezni → megőrzi az irányt

---

## 9. `evaluatemodel_five.py` — Kiértékelési pipeline

### 9.1 `load_models()` — Modellek betöltése

```python
def load_models(scaler_path="itier_scaler.pkl"):
    scaler = joblib.load(scaler_path)
    model_files = {
        "RFC":     "itier_rf_model.pkl",
        "SVM":     "itier_svm_model.pkl",
        "K-Means": "itier_kmeans_model.pkl",
        "MLP":     "itier_mlp_model.pkl",
    }
    if os.path.exists("itier_boost_info.txt"):
        # dinamikusan tölti be az XGBoost vagy GradientBoosting modellt
```

Az `itier_boost_info.txt` tárolja, hogy XGBoost vagy sklearn GradientBoosting lett elmentve — így a kiértékelés nem feltételezi, hogy az XGBoost telepítve van.

### 9.2 `get_image_max_prob()` — Képszintű detekciós score

```python
def get_image_max_prob(img_path, model, scaler, block_size=32, step=8):
    feats, _, positions, dims = extract_features_from_image(img_path, ...)
    fs = scaler.transform(feats)
    probs = model.predict_proba(fs)[:, 1]
    # Hőtérkép: átlagolt blokk-valószínűségek
    heat_sum / heat_count → heat_map
    return float(np.nanmax(heat_map))  # max(P_I)
```

Ez implementálja a cikk 3.5 fejezetének `max(P_I)` értékét: az egész képen lévő összes 32×32-es blokk `P(Border)` valószínűségéből épített hőtérkép maximumát. Ha egy kép hamisított, várhatóan legalább egy régióban magas `P(Border)` értékek lesznek.

### 9.3 Teljesítmény-optimalizálás: feature-kinyerés egyszer

```python
# FIX: Feature-kinyerés EGYSZER képenként, nem modellnként
feats, _, positions, dims = extract_features_from_image(img_p, ...)
feats_scaled = scaler.transform(feats)

for model_name, model in models.items():
    probs = model.predict_proba(feats_scaled)[:, 1]
    # ... hőtérkép számítás minden modellhez
```

**Eredeti (hibás) megközelítés:** 45 kép × 5 modell = 225 `extract_features_from_image` hívás, ami az SRM konvolúciók és PCA számítások miatt fél óra+ futási időt jelent.

**Javított megközelítés:** 45 kép × 1 = 45 hívás. A feature-k ugyanazok minden modellnek — csak az osztályozás lépés különbözik modellenként.

### 9.4 Vizualizáció — 5 panel minden képnél

Minden tesztképhez 5 paneles vizualizáció készül:
1. **Eredeti kép** — referencia
2. **GT maszk** — ground truth (manuálisan annotált hamisított régió)
3. **Hőtérkép** (jet colormap, 0-1) — a `P(Border)` valószínűségek pixelenkénti átlaga
4. **Predikált maszk** — bináris maszk a t_hat küszöb után, post-processing utón
5. **Overlay** — Piros = False Positive, Narancs = True Positive, Zöld = False Negative

Az overlay-kódolás egyértelmű diagnózist ad: ha sok piros van, a modell hamis riasztásokat produkál; ha sok zöld van, nem találja meg a hamisított területet.

---

## 10. Küszöbválasztás és ROC analízis

### 10.1 `compute_roc_threshold()` — L2-minimális ROC küszöb

```python
distances = np.sqrt(fpr**2 + (tpr - 1)**2)
t_hat = float(thresholds[np.argmin(distances)])
```

A cikk 3.5 fejezetében leírt módszer: a ROC görbe azon pontján lévő küszöböt választjuk, amely euklidészi távolsága minimális a (0, 1) ideális ponttól — ahol FPR=0 és TPR=1 lenne. Ez a "legközelebb az ideálishoz" heurisztika.

### 10.2 `safe_threshold()` — Három speciális eset kezelése

**1. eset: AUC < 0.5 (invertált score):**
```python
if auc < 0.5:
    flipped_scores = 1.0 - y_scores_arr
    # újraszámol az invertált score-okkal
    return t_hat2, auc2, True  # flipped=True jelzés
```
Ha egy modell AUC < 0.5, a score-jai **fordítva** korrelálnak a valódi osztállyal — az "1-score" inverz pontosan az AUC=(1-eredeti_AUC) valódi teljesítményt adja vissza. Ez az MLP isotonic kalibráció összeomlásának utólagos kezelése.

**2. eset: GB degeneráció (spliced szórás ≈ 0):**
```python
if np.std(spl_scores) < 0.001:
    if aut_max < spl_min:
        t_hat = (spl_min + aut_max) / 2.0  # midpoint
    else:
        t_hat = np.percentile(aut_arr, 90)  # 90. percentilis
```
Ha minden spliced kép `max_p = 1.000 ± 0.000` (tökéletes klaszter), a ROC görbe degenerált (egyetlen ugrás). A midpoint küszöb tökéletes szeparáció esetén optimális; a 90. percentilis maximum 10% FP-t enged átlagosan.

**3. eset: t_hat ≥ 0.999 fallback:**
Lépcsős fallback hierarchia:
1. `global_t_hat` (RFC-re optimalizált, `optimal_threshold.txt`-ből)
2. Auth. képek 95. percentilise + 0.01 rés
3. 0.5 abszolút fallback

### 10.3 Score eloszlás debug (FIX #2)

```
Modell                  Spl.átlag  Spl.std  Auth.átlag  Auth.std
RFC                        0.8234   0.1021      0.4123    0.1534
SVM                        0.9123   0.0456      0.8987    0.1234  ← OVERCONFIDENT
MLP                        0.9976   0.0001      0.9687    0.1142  ← OVERCONFIDENT
```

Az `← OVERCONFIDENT` jelzés automatikusan megjelenik, ha egy modell authentic képeken is `aut_mu > 0.9` értéket ad — ez azonnal jelzi, hogy a modell nem diszkriminál megfelelően.

---

## 11. Összefoglalás — Módszerek motivációjának táblázata

| Összetevő | Módszer | Miért? |
|---|---|---|
| Osztályegyensúlyozás | Undersampling | Border blokkok ritkák → imbalanced data |
| Normalizáció | StandardScaler | SVM, MLP skálafüggő; konzisztencia az összes modellnél |
| RFC hyperparameter | GridSearchCV | Kisebb keresési tér, kimerítő keresés megengedhető |
| RFC regularizáció | min_samples_leaf, max_features | Overfit elleni, diverzebb fák |
| SVM kalibráció | CalibratedClassifierCV isotonic | Platt-skálázás overfit-el a subsample-en |
| MLP solver | Adam | Nagy adathalmazon legjobb konvergencia |
| MLP aktiváció | relu + tanh keresés | Tanh jobb lehet normalizált, szimmetrikus adatokon |
| MLP learning rate | constant + adaptive | Adaptive finomabb konvergencia az early stopping mellé |
| MLP batch size | 64/128/200 | Kisebb batch → jobb generalizálás |
| MLP kalibráció | CalibratedClassifierCV sigmoid | Isotonic invertálta a score-okat (AUC=0.383) |
| Boosting keresés | RandomizedSearchCV | Nagy keresési tér, kimerítő nem hatékony |
| XGBoost regularizáció | gamma, reg_alpha, reg_lambda | Az XGBoost legfontosabb regularizációs paraméterei |
| XGBoost párhuzamosítás | n_jobs=1 + CV n_jobs=-1 | Thread contention elkerülése |
| GB early stopping | n_iter_no_change=15 | Felesleges iterációk és overfit megelőzése |
| Küszöbválasztás | ROC L2-minimum + safe_threshold | 3 degenerált eset robusztus kezelése |
| Feature-kinyerés | Egyszer képenként | 5× gyorsabb kiértékelés (5 modell esetén) |

---

*Dokumentáció: trainmodel_five.py + evaluatemodel_five.py, Itier et al. (2021) pipeline*
