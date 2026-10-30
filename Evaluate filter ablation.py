"""
evaluate_filter_ablation.py

CHECKLIST #7 — Jellemzőkinyerés szétszedése szűrők szerint
=============================================================
A cikk (Itier et al., 2021) Table 1-je az 5 SRM magas-passzív szűrőt
(HSB, VSB, RF0, RF1, RF2) KÜLÖN-KÜLÖN is kiértékeli, majd a végén az
összes konkatenált változatával ("Concatenated") hasonlítja össze —
hogy megmutassa, melyik szűrő önmagában mennyire informatív, és hogy a
kombináció valóban jobb-e mindegyiknél külön-külön.

Korábban a pipeline (itier.py) mindig azonnal konkatenálta mind az 5
szűrőt (60 dim), nem volt kapcsoló/ág egyetlen szűrő önálló
kiértékelésére. Az itier.py mostantól elfogadja a `filter_names`
paramétert (lásd extract_features_from_image), ez a szkript pedig ezt
használva reprodukálja a cikk Table 1-es ablációját:

    Metrika          HSB    VSB    RF0    RF1    RF2   Concatenated
    Mean precision   ...    ...    ...    ...    ...   ...
    Mean recall      ...    ...    ...    ...    ...   ...
    Mean F1 score    ...    ...    ...    ...    ...   ...
    Variance F1      ...    ...    ...    ...    ...   ...

Módszertan (a cikk Section 4.1 alapján, KÉP-SZINTŰ osztályozás):
  - Az adatbázis (spliced + authentic képek) 5 kiegyensúlyozott
    (stratifikált) partícióra van osztva.
  - 5 fordulóban, mindig EGY partíción tanítunk, a MARADÉK NÉGYEN
    tesztelünk — szó szerint a cikk módszertana: "one is used for
    training and the other four are grouped to form the validation
    set" (Section 4.1). Ez fordítva van a megszokott k-fold CV-hez
    képest (ahol 4/5-öd a tanító), de a cikk ezt indokolja: "a fifth
    of the images for training is enough, because the training is on
    the images' blocks using sliding windows".
  - Osztályozó: Random Forest Classifier, FIX (nem grid-search-elt)
    hiperparaméterekkel. A cikk Table 1-je is egyetlen rögzített
    modell-konfigurációt hasonlít össze szűrőnként, nem hiperparaméter-
    keresést végez soronként — 6 szűrő-változat × 5 fold × GridSearchCV
    kombinatorikusan kezelhetetlen futásidőt adna. (A hiperparaméter-
    finomhangolás a dedikált trainmodel_*.py szkriptek dolga —
    checklist #9/#12 — ez a szkript kifejezetten a SZŰRŐ-összehasonlításra
    fókuszál, változatlan osztályozó mellett, mint a cikk is teszi.)
  - Kép-szintű pontszám: max(1 - P(Uniform)) minden blokkra — 3-OSZTÁLYOS
    SÉMÁNÁL (checklist #1) ez P(Border)+P(Spliced), 2 osztálynál pontosan
    a régi P(Border)-nek felel meg.
  - Küszöb: ROC L2-távolság-optimum a (0,1) ponttól (cikk Section 3.5),
    fold-onként újraszámolva a fold teszthalmazán.

FONTOS KORLÁTOZÁS / HATÁRVONAL: ez a szkript ÖNÁLLÓAN, csak saját magára
nézve implementál egy valódi rotáló 5-fold CV-t a KÉP-SZINTŰ
osztályozáshoz. Ez NEM ütközik a checklist #10 hiányával (a fő
trainmodel_*.py pipeline-ok jelenleg egyetlen, fix train/test split-et
használnak, prepare_database.py-ból) — az változatlan marad, ez egy
KÜLÖN, önmagában futó összehasonlító szkript, ami nem módosítja és nem
használja a dataset_splits.json-ban rögzített split-et (a teljes
elérhető képhalmazt saját maga particionálja újra).
"""

import os
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve
from itier import extract_features_from_image, CLASS_UNIFORM, ALL_FILTER_NAMES

# ============================================================
# KONFIGURÁCIÓ
# ============================================================
BLOCK_SIZE       = 32
STEP             = 8
N_FOLDS          = 5      # a cikk 5-fold rotáló particionálása (Section 4.1)
MAX_AUTH_IMAGES  = 180     # authentic képek felső korlátja (futásidő miatt;
                          # None = az összes elérhető, mint a cikkben)
RFC_PARAMS = dict(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)

# A cikk Table 1 oszlopai: mind az 5 szűrő külön-külön, majd konkatenálva.
FILTER_SETS = {
    "HSB":          ["HSB"],
    "VSB":          ["VSB"],
    "RF0":          ["RF0"],
    "RF1":          ["RF1"],
    "RF2":          ["RF2"],
    "Concatenated": list(ALL_FILTER_NAMES),   # mind az 5 együtt (60 dim)
}


# ---------------------------------------------------------------------------
# Osztályok kiegyensúlyozása (ugyanaz az elv, mint trainmodel_*.py-ban —
# N-osztályos, önállóan itt is definiálva, hogy a szkript ne függjön egyik
# trainmodel_X.py-tól sem)
# ---------------------------------------------------------------------------

def balance_classes(X, y):
    X_arr, y_arr = np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32)
    classes = np.unique(y_arr)
    class_indices = {c: np.where(y_arr == c)[0] for c in classes}
    mn = min(len(idx) for idx in class_indices.values())
    if mn == 0:
        return np.zeros((0, X_arr.shape[1]), dtype=np.float32), np.array([], dtype=np.int32)
    np.random.seed(42)
    idx = np.concatenate([
        np.random.choice(class_indices[c], mn, replace=False) for c in classes
    ])
    np.random.shuffle(idx)
    return X_arr[idx], y_arr[idx]


# ---------------------------------------------------------------------------
# ROC L2-optimális küszöb (cikk Section 3.5, ugyanaz mint tune_treshold.py)
# ---------------------------------------------------------------------------

def get_optimal_roc_threshold(y_true, y_scores):
    """
    HIBAJAVÍTÁS: az sklearn roc_curve() az ELSŐ pontot MESTERSÉGESEN fűzi be
    (FPR=0, TPR=0, threshold=+inf) — ez szó szerint "mindent elutasítok",
    triviális, a gyakorlatban hasznavehetetlen működési pont, csak a görbe
    (0,0)-ból induló teljességéhez kell a matematikai definícióhoz.

    Ha egy szűrő gyenge/nem-diszkriminatív jelet ad (pl. a legtöbb kép közel
    azonos pontszámot kap), a ROC görbe majdnem a (0,0)-(1,1) átlóhoz simul,
    és a (0,0) mesterséges pont L2-távolsága (=1.0) UGYANOLYAN VAGY JOBB
    lehet, mint bármelyik valódi, adatból számolt pont távolsága. Egyenlőség
    esetén az np.argmin az ELSŐ (tehát ezt a mesterséges) indexet választja
    → semmilyen képet nem sorol "hamisítottnak" → precision=recall=F1=
    PONTOSAN 0.0, minden foldban ugyanígy (variance_f1=0.0) — ez pontosan
    a megfigyelt "HSB-nél minden 0" tünetegyüttes.

    Javítás: a mesterséges első pontot KIZÁRJUK a keresésből (hacsak nem ez
    az egyetlen elérhető pont), így mindig egy VALÓS, adatból származó
    küszöböt választunk — ha van tényleges (akár gyenge) jel, azt megtaláljuk;
    ha a szűrő tényleg teljesen informálatlan, az eredmény őszintén alacsony
    F1 lesz, nem a triviális "elutasítok mindent" művi nullázás.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    if len(thresholds) > 1:
        fpr, tpr, thresholds = fpr[1:], tpr[1:], thresholds[1:]
    distances = np.sqrt(fpr ** 2 + (1 - tpr) ** 2)
    return float(thresholds[np.argmin(distances)])


# ---------------------------------------------------------------------------
# Authentic (hamisítatlan) képek összegyűjtése — ugyanaz a mintázat, mint
# evaluatemodel_seven.py find_authentic_images()-e
# ---------------------------------------------------------------------------

def find_authentic_images(max_count=None):
    files = []
    for d in ["4cam_auth", "Au"]:
        if os.path.exists(d):
            for fn in sorted(os.listdir(d)):
                if fn.lower().endswith((".jpg", ".png", ".tif", ".jpeg")):
                    files.append(os.path.join(d, fn))
    if not files:
        return []
    np.random.seed(42)
    np.random.shuffle(files)
    return files[:max_count] if max_count is not None else files


# ---------------------------------------------------------------------------
# Kép-szintű, címkézett elemek összegyűjtése: (kép, maszk_vagy_None, image_label)
# image_label: 1 = hamisított (spliced), 0 = hiteles (authentic)
# ---------------------------------------------------------------------------

def collect_labeled_items():
    with open("dataset_splits.json", "r") as f:
        data = json.load(f)

    # dataset_splits.json (prepare_database.py) KIZÁRÓLAG olyan kép-maszk
    # párokat tartalmaz, ahol talált maszkot — azaz mind hamisított (spliced)
    # kép (Columbia 4cam_splc/ vagy CASIA Tp_D_*), tehát image_label=1.
    spliced_pairs = (list(zip(data["train_images"], data["train_masks"])) +
                     list(zip(data["test_images"],  data["test_masks"])))
    items = [(img, msk, 1) for img, msk in spliced_pairs]

    auth_images = find_authentic_images(MAX_AUTH_IMAGES)
    items += [(img, None, 0) for img in auth_images]

    return items


# ---------------------------------------------------------------------------
# Adott szűrőkészlethez TARTOZÓ jellemzők kinyerése, KÉPENKÉNT KÉSZLETEZVE
# (cache) — így az 5-fold rotáció alatt egy képet csak EGYSZER kell
# feldolgozni, nem foldonként újra (5x - 6 szűrő-variáns * 5x helyett).
# ---------------------------------------------------------------------------

def extract_dataset_features(items, filter_names, verbose=True):
    cache = {}
    n = len(items)
    for i, (img_path, mask_path, image_label) in enumerate(items):
        if verbose and (i + 1) % 20 == 0:
            print(f"    jellemzőkinyerés: {i+1}/{n}")
        feats, labels, positions, dims = extract_features_from_image(
            img_path, mask_path, block_size=BLOCK_SIZE, step=STEP,
            filter_names=filter_names)
        if feats is None or dims is None or len(feats) == 0:
            continue
        cache[img_path] = dict(
            feats=feats, labels=labels, positions=positions,
            dims=dims, image_label=image_label)
    return cache


# ---------------------------------------------------------------------------
# Egy kép max(1 - P(Uniform)) pontszámának kiszámítása egy adott (illesztett)
# RFC-vel és scaler-rel — 3-OSZTÁLYOS SÉMA-biztos (checklist #1 folytonossága)
# ---------------------------------------------------------------------------

def image_max_score(entry, rfc, scaler):
    feats_sc = np.nan_to_num(
        scaler.transform(entry["feats"]), nan=0.0, posinf=0.0, neginf=0.0)
    proba = rfc.predict_proba(feats_sc)

    if CLASS_UNIFORM in rfc.classes_:
        uniform_col = int(np.where(rfc.classes_ == CLASS_UNIFORM)[0][0])
        probs = 1.0 - proba[:, uniform_col]
    else:
        # Extrém ritka eset: az adott fold tanító-partíciójában egyetlen
        # Uniform-blokk sem volt — ekkor minden előrejelzett osztály
        # "nem Uniform", tehát az összegük ~1.0 mindenhol.
        probs = np.sum(proba, axis=1)

    return float(np.max(probs)) if len(probs) > 0 else 0.0


# ---------------------------------------------------------------------------
# Egyetlen fold lefuttatása: tanítás a train-képek blokkjain, kiértékelés a
# test-képek kép-szintű max-score-ján (Precision/Recall/F1)
# ---------------------------------------------------------------------------

def run_fold(cache, train_paths, test_paths):
    # --- Blokk-szintű tanítóhalmaz összeállítása (csak a HAMISÍTOTT
    #     képekből, mert csak azoknak van blokk-címkéjük — lásd itier.py:
    #     mask_path=None -> labels=None) ---
    X_list, y_list = [], []
    for p in train_paths:
        entry = cache.get(p)
        if entry is None or entry["labels"] is None:
            continue
        X_list.append(entry["feats"])
        y_list.append(entry["labels"])

    if not X_list:
        return None   # nincs tanítható blokk-adat ebben a foldban

    X_train = np.vstack(X_list)
    y_train = np.concatenate(y_list)
    if len(np.unique(y_train)) < 2:
        return None   # csak 1 osztály — nem tanítható klasszifikátor

    X_bal, y_bal = balance_classes(X_train, y_train)
    if len(X_bal) == 0:
        return None

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_bal)

    rfc = RandomForestClassifier(**RFC_PARAMS)
    rfc.fit(X_sc, y_bal)

    # --- Kép-szintű teszt-kiértékelés ---
    y_true, y_score = [], []
    for p in test_paths:
        entry = cache.get(p)
        if entry is None:
            continue
        y_true.append(entry["image_label"])
        y_score.append(image_max_score(entry, rfc, scaler))

    y_true  = np.array(y_true)
    y_score = np.array(y_score)
    if len(np.unique(y_true)) < 2:
        return None   # degenerált teszt-partíció (csak 1 osztály) — kihagyjuk

    # DIAGNOSZTIKA: ha a spliced és authentic pontszám-eloszlások szinte
    # teljesen átfednek (a szűrő gyakorlatilag nem diszkriminál), ezt itt
    # LÁTHATÓVÁ tesszük, ne csak egy rejtélyes F1=0.0 jelenjen meg a végén.
    spl_scores  = y_score[y_true == 1]
    auth_scores = y_score[y_true == 0]
    if CLASS_UNIFORM not in rfc.classes_:
        print(f"      [FIGYELEM] ebben a foldban nem volt Uniform-blokk a "
              f"tanítóhalmazban (rfc.classes_={rfc.classes_}) — a pontszám "
              f"degenerált lehet.")
    if np.ptp(y_score) < 1e-6:
        print(f"      [FIGYELEM] a kép-szintű pontszám GYAKORLATILAG KONSTANS "
              f"minden képen (min={y_score.min():.4f}, max={y_score.max():.4f}) "
              f"— ez a szűrő ebben a foldban nem hordoz hasznosítható jelet.")
    else:
        print(f"      spliced score: min={spl_scores.min():.3f} "
              f"mean={spl_scores.mean():.3f} max={spl_scores.max():.3f}  |  "
              f"authentic score: min={auth_scores.min():.3f} "
              f"mean={auth_scores.mean():.3f} max={auth_scores.max():.3f}")

    t_hat = get_optimal_roc_threshold(y_true, y_score)
    # HIBAJAVÍTÁS: az sklearn roc_curve() a küszöböt ">=" konvencióval
    # definiálja (egy mintát pozitívnak számol, ha score >= threshold).
    # A korábbi szigorú "score > t_hat" emiatt PONTOSAN akkor omlott össze
    # (minden képet elutasítva, P=R=F1=0.0), amikor a legjobb küszöb éppen
    # a megfigyelt pontszámok MAXIMUMÁVAL esett egybe (t_hat=1.0000) — ekkor
    # semmi nem lehetett SZIGORÚAN nagyobb 1.0-nál, még akkor sem, ha a
    # spliced/authentic eloszlások valójában jól szétváltak (lásd a
    # valós futásban: minden olyan foldnál 0 lett az F1, ahol t_hat pontosan
    # 1.0 volt — HSB-nél mind az 5, de VSB/RF0/RF1/RF2/Concatenated egyes
    # foldjaiban is ugyanez történt). A ">=" a roc_curve saját, belső
    # definíciójával konzisztens, és ez a fold-összeomlás megszűnik.
    y_pred = (y_score >= t_hat).astype(int)

    return dict(
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        threshold=t_hat,
        n_train=len(train_paths), n_test=len(test_paths),
    )


# ---------------------------------------------------------------------------
# Egy szűrő-variáns teljes 5-fold rotációja + eredmények aggregálása
# (Table 1 formátum: mean + variance)
# ---------------------------------------------------------------------------

def run_ablation_for_filter(display_name, filter_names, items):
    print(f"\n{'='*60}")
    print(f"  Szűrő: {display_name}  (filter_names={filter_names})")
    print(f"{'='*60}")

    print("  Jellemzőkinyerés (egyszer, képenként gyorsítótárazva)...")
    cache = extract_dataset_features(items, filter_names)
    valid_paths  = [p for p, m, lbl in items if p in cache]
    valid_labels = [cache[p]["image_label"] for p in valid_paths]

    if len(valid_paths) < N_FOLDS:
        print(f"  FIGYELEM: túl kevés érvényes kép ({len(valid_paths)}) "
              f"{N_FOLDS}-fold CV-hez — kihagyva.")
        return None

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_results = []

    for fold_i, (big_idx, small_idx) in enumerate(
            skf.split(valid_paths, valid_labels)):
        # A CIKK MÓDSZERTANA SZERINT FORDÍTVA: a KIS partíción (1/5) tanítunk,
        # a NAGY maradékon (4/5) tesztelünk (lásd modul docstring).
        train_idx, test_idx = small_idx, big_idx
        train_paths = [valid_paths[i] for i in train_idx]
        test_paths  = [valid_paths[i] for i in test_idx]

        res = run_fold(cache, train_paths, test_paths)
        if res is None:
            print(f"  [{fold_i+1}/{N_FOLDS}] kihagyva (nem volt elég adat)")
            continue

        fold_results.append(res)
        print(f"  [{fold_i+1}/{N_FOLDS}] train={res['n_train']:3d} "
              f"test={res['n_test']:3d}  t_hat={res['threshold']:.4f}  "
              f"P={res['precision']:.4f}  R={res['recall']:.4f}  "
              f"F1={res['f1']:.4f}")

    if not fold_results:
        print("  Nem sikerült egyetlen érvényes foldot sem lefuttatni.")
        return None

    precisions = [r["precision"] for r in fold_results]
    recalls    = [r["recall"]    for r in fold_results]
    f1s        = [r["f1"]        for r in fold_results]

    summary = dict(
        mean_precision=float(np.mean(precisions)),
        mean_recall=float(np.mean(recalls)),
        mean_f1=float(np.mean(f1s)),
        variance_f1=float(np.var(f1s)),
        n_folds_completed=len(fold_results),
    )
    print(f"  --> Mean P={summary['mean_precision']:.4f}  "
          f"Mean R={summary['mean_recall']:.4f}  "
          f"Mean F1={summary['mean_f1']:.4f}  "
          f"Var F1(x1e-4)={summary['variance_f1']*10000:.2f}")
    return summary


# ---------------------------------------------------------------------------
# Fő futtatás
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Szűrő-abláció (checklist #7) — a cikk Table 1 reprodukciója")
    print("  HSB / VSB / RF0 / RF1 / RF2 külön-külön, majd Concatenated")
    print("=" * 60)

    items = collect_labeled_items()
    n_spliced = sum(1 for _, _, lbl in items if lbl == 1)
    n_auth    = sum(1 for _, _, lbl in items if lbl == 0)
    print(f"\nÖsszegyűjtött képek: {len(items)}  "
          f"(spliced={n_spliced}, authentic={n_auth})")
    if n_auth == 0:
        print("FIGYELEM: nem található authentic könyvtár "
              "(4cam_auth/ vagy Au/) — a kép-szintű P/R/F1 nem lesz "
              "értelmezhető authentic negatívok nélkül!")

    all_results = {}
    for display_name, filter_names in FILTER_SETS.items():
        summary = run_ablation_for_filter(display_name, filter_names, items)
        if summary is not None:
            all_results[display_name] = summary

    # --- Table 1 formátumú összefoglaló ---
    print("\n" + "=" * 78)
    print("  ÖSSZEFOGLALÓ (a cikk Table 1 formátuma)")
    print("=" * 78)
    header = f"  {'Metrika':<20s}" + "".join(
        f"{name:>10s}" for name in FILTER_SETS if name in all_results)
    print(header)
    for metric_key, metric_label in [
            ("mean_precision", "Mean precision"),
            ("mean_recall",    "Mean recall"),
            ("mean_f1",        "Mean F1 score"),
            ("variance_f1",    "Variance F1 (x1e-4)")]:
        row = f"  {metric_label:<20s}"
        for name in FILTER_SETS:
            if name not in all_results:
                continue
            val = all_results[name][metric_key]
            if metric_key == "variance_f1":
                val *= 10000
            row += f"{val:>10.4f}"
        print(row)
    print("=" * 78)

    with open("filter_ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=4)
    print("\nJSON export: filter_ablation_results.json")