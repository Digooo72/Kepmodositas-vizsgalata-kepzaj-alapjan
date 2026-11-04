"""
Evaluate filter ablation.py

CHECKLIST #7 — Jellemzőkinyerés szétszedése SRM szűrők szerint (Ablációs vizsgálat)
====================================================================================
Az Itier et al. (2021) cikk 1. táblázatának (Table 1) reprodukciója[cite: 2, 19]:
az 5 darab SRM magas-passzív szűrőt (HSB, VSB, RF0, RF1, RF2) KÜLÖN-KÜLÖN (12 dimenzió)
is kiértékeli, majd a végén az összes szűrő konkatenált változatával ("Concatenated",
60 dimenzió) hasonlítja össze[cite: 2, 5, 19]:

    Metrika               HSB    VSB    RF0    RF1    RF2   Concatenated
    Mean precision        ...    ...    ...    ...    ...   ...
    Mean recall           ...    ...    ...    ...    ...   ...
    Mean F1 score         ...    ...    ...    ...    ...   ...
    Variance F1 (x1e-4)   ...    ...    ...    ...    ...   ...

─── Módszertan (Cikk Section 4.1 alapján, KÉP-SZINTŰ osztályozás) ───────────────
  1. Fordított 5-szörös keresztvalidáció (1/5 tanító, 4/5 teszt):
     Az adatbázis (spliced + authentic képek) 5 kiegyensúlyozott (stratifikált)
     partícióra van osztva. 5 fordulóban mindig EGY partíción (20%) tanítunk, és a
     MARADÉK NÉGY partíción (80%) tesztelünk ("one is used for training and the
     other four are grouped to form the validation set", Section 4.1)[cite: 2, 19].
     Indoklás: egyetlen képen is több száz 32x32-es csúszóablak keletkezik, így a
     képek 1/5-e már elegendő blokkot biztosít a Random Forest betanításához[cite: 2, 19].

  2. Rögzített Random Forest osztályozó:
     n_estimators=100, max_depth=20, random_state=42. A cikk Table 1 kísérlete a
     szűrők informativitását hasonlítja össze változatlan osztályozó mellett.

  3. 3-osztályos képszintű pontszám (checklist #1):
     Minden blokkra P(Manipulált) = 1 - P(CLASS_UNIFORM) = P(Border) + P(Spliced),
     a képszintű pontszám pedig a kép összes blokkján vett maximum: max(P_I)[cite: 2, 16].

  4. Numerikusan stabilizált ROC L2-küszöbölés (Cikk Section 3.5):
     - A roc_curve() mesterséges első pontját (FPR=0, TPR=0, thr=+inf) kizárjuk
       a keresésből, hogy gyengébb szűrőnél se ragadjon be a "mindent elutasító"
       küszöbre[cite: 2].
     - A döntésnél szigorú '>' helyett '>=' relációt (y_score >= t_hat) alkalmazunk,
       így t_hat = 1.0000 esetén sem omlik össze 0-ra a detektálás[cite: 2].
────────────────────────────────────────────────────────────────────────────────────
"""

import os
import gc
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve
from itier import extract_features_from_image, CLASS_UNIFORM, ALL_FILTER_NAMES

# ===========================================================================
# KONFIGURÁCIÓ
# ===========================================================================
BLOCK_SIZE      = 32     # Csúszóablak mérete (32x32 pixel, Cikk Section 4.1)[cite: 2, 18]
STEP            = 8      # Csúszóablak lépésköze (8 pixel, Cikk Section 4.1)[cite: 2, 18]
N_FOLDS         = 5      # 5-fold rotáló partícionálás (1 fold train, 4 fold test)[cite: 2, 19]
MAX_AUTH_IMAGES = 180    # Hiteles (authentic) képek felső korlátja (Columbia = 180)[cite: 2, 17]
RFC_PARAMS      = dict(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)

# A cikk Table 1 oszlopai: az 5 SRM szűrő külön-külön (12D), majd konkatenálva (60D)[cite: 2, 5, 19]
FILTER_SETS = {
    "HSB":          ["HSB"],
    "VSB":          ["VSB"],
    "RF0":          ["RF0"],
    "RF1":          ["RF1"],
    "RF2":          ["RF2"],
    "Concatenated": list(ALL_FILTER_NAMES),   # Mind az 5 szűrő együtt (60 jellemző)[cite: 2, 5]
}
# ===========================================================================


# ---------------------------------------------------------------------------
# Osztályok kiegyensúlyozása (N-osztályos alulmintavételezés — Cikk Section 3.4)
# ---------------------------------------------------------------------------

def balance_classes(X, y):
    """
    Alulmintavételezéssel (undersampling) egyenlő elemszámra hozza a tanítóhalmazban
    szereplő blokk-osztályokat (Uniform=0, Border=1, Spliced=2)[cite: 2, 5].
    Önállóan itt is definiálva, hogy a szkript ne függjön a trainmodel_*.py fájloktól[cite: 2].
    """
    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.int32)
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
# ROC L2-optimális küszöb (Cikk Section 3.5, mesterséges kezdőpont kiszűrésével)
# ---------------------------------------------------------------------------

def get_optimal_roc_threshold(y_true, y_scores):
    """
    Meghatározza a ROC görbe (0,1) ideális pontjától mért euklideszi (L2)
    távolságot minimalizáló küszöbértéket[cite: 2, 17].

    KRITIKUS HIBAJAVÍTÁS:
      Az sklearn roc_curve() az első (0. indexű) pontot mesterségesen szúrja be:
      (FPR=0, TPR=0, threshold = max(y_scores) + 1)[cite: 2]. Ennek a pontnak az (0,1)-től
      vett L2-távolsága pontosan 1.0[cite: 2]. Ha egy önálló szűrő gyengébb jelet ad és a
      ROC-görbe közel simul az átlóhoz, a valódi küszöbök távolsága >= 1.0 lehet,
      így az np.argmin a 0. indexet választaná -> minden képet elutasítana
      (Precision = Recall = F1 = 0.0)[cite: 2].
      A `fpr[1:], tpr[1:], thresholds[1:]` csonkolással kizárjuk a mesterséges
      kezdőpontot, így mindig valódi, adatból származó küszöböt választunk[cite: 2].
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    if len(thresholds) > 1:
        fpr, tpr, thresholds = fpr[1:], tpr[1:], thresholds[1:]
    distances = np.sqrt(fpr ** 2 + (1.0 - tpr) ** 2)
    return float(thresholds[np.argmin(distances)])


# ---------------------------------------------------------------------------
# Hiteles (authentic) és hamisított (spliced) képek összegyűjtése
# ---------------------------------------------------------------------------

def find_authentic_images(max_count=None):
    """Összegyűjti a hiteles képeket a 4cam_auth/ és Au/ könyvtárakból[cite: 2]."""
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


def collect_labeled_items():
    """
    Összegyűjti a teljes elérhető képhalmazt képszintű címkékkel:
      (img_path, mask_path_vagy_None, image_label)
      ahol image_label = 1 (hamisított / spliced) vagy 0 (hiteles / authentic)[cite: 2].
    """
    with open("dataset_splits.json", "r") as f:
        data = json.load(f)

    # A dataset_splits.json train és test listáiban szereplő összes maszkos kép
    # hamisított (spliced -> image_label = 1)[cite: 2, 6].
    spliced_pairs = (list(zip(data["train_images"], data["train_masks"])) +
                     list(zip(data["test_images"],  data["test_masks"])))
    items = [(img, msk, 1) for img, msk in spliced_pairs]

    # Hiteles képek hozzáadása (maszk = None, image_label = 0)[cite: 2]
    auth_images = find_authentic_images(MAX_AUTH_IMAGES)
    items += [(img, None, 0) for img in auth_images]

    return items


# ---------------------------------------------------------------------------
# Képenkénti jellemzőkinyerés gyorsítótárazással (cache)
# ---------------------------------------------------------------------------

def extract_dataset_features(items, filter_names, verbose=True):
    """
    Kinyeri az adott `filter_names` szűrőkészlethez tartozó blokkjellemzőket
    az összes képre, és memóriabeli szótárban (`cache`) eltárolja őket[cite: 2].
    Így az 5-fold rotáció során minden képet csak EGYSZER kell feldolgozni[cite: 2].
    """
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
# Képszintű pontszám: max(1 - P(Uniform)) — 3-osztályos séma kompatibilis
# ---------------------------------------------------------------------------

def image_max_score(entry, rfc, scaler):
    """
    Kiszámítja egy kép blokkjain a manipuláltság valószínűségének maximumát:
      P(Manipulált) = 1 - P(CLASS_UNIFORM) = P(Border) + P(Spliced)[cite: 2, 5].
    """
    feats_sc = np.nan_to_num(
        scaler.transform(entry["feats"]), nan=0.0, posinf=0.0, neginf=0.0)
    proba = rfc.predict_proba(feats_sc)

    if CLASS_UNIFORM in rfc.classes_:
        uniform_col = int(np.where(rfc.classes_ == CLASS_UNIFORM)[0][0])
        probs = 1.0 - proba[:, uniform_col]
    else:
        probs = np.sum(proba, axis=1)

    return float(np.max(probs)) if len(probs) > 0 else 0.0


# ---------------------------------------------------------------------------
# Egyetlen fold lefuttatása (tanítás 1/5 partíción, tesztelés 4/5 partíción)
# ---------------------------------------------------------------------------

def run_fold(cache, train_paths, test_paths):
    """
    1. Összegyűjti a tanító-partícióba eső hamisított képek blokkjait.
    2. Kiegyensúlyozza az osztályokat (`balance_classes`) és illeszti a `StandardScaler`-t.
    3. Betanítja a rögzített paraméterű Random Forest osztályozót.
    4. Kiértékeli a teszt-partíció (4/5) összes képét (Precision, Recall, F1)[cite: 2].
    """
    # --- 1. Blokk-szintű tanítóhalmaz összeállítása (maszkos képekből) ---
    X_list, y_list = [], []
    for p in train_paths:
        entry = cache.get(p)
        if entry is None or entry["labels"] is None:
            continue
        X_list.append(entry["feats"])
        y_list.append(entry["labels"])

    if not X_list:
        return None   # Nincs tanítható blokk-adat ebben a foldban

    X_train = np.vstack(X_list)
    y_train = np.concatenate(y_list)
    if len(np.unique(y_train)) < 2:
        return None   # Csak 1 osztály van jelen -> nem tanítható osztályozó

    X_bal, y_bal = balance_classes(X_train, y_train)
    if len(X_bal) == 0:
        return None

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_bal)

    rfc = RandomForestClassifier(**RFC_PARAMS)
    rfc.fit(X_sc, y_bal)

    # --- 2. Képszintű teszt-kiértékelés a maradék 4/5 partíción ---
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
        return None   # Degenerált teszt-partíció

    # --- 3. Diagnosztikai eloszlás-kiírás ---
    spl_scores  = y_score[y_true == 1]
    auth_scores = y_score[y_true == 0]
    if CLASS_UNIFORM not in rfc.classes_:
        print(f"      [FIGYELEM] Ebben a foldban nem volt Uniform-blokk a "
              f"tanítóhalmazban (rfc.classes_={rfc.classes_}).")
    if np.ptp(y_score) < 1e-6:
        print(f"      [FIGYELEM] A képszintű pontszám konstans minden képen "
              f"(min={y_score.min():.4f}, max={y_score.max():.4f}).")
    else:
        print(f"      spliced score: min={spl_scores.min():.3f} "
              f"mean={spl_scores.mean():.3f} max={spl_scores.max():.3f}  |  "
              f"authentic score: min={auth_scores.min():.3f} "
              f"mean={auth_scores.mean():.3f} max={auth_scores.max():.3f}")

    # --- 4. Optimális ROC küszöb és predikció ('>=' relációval!) ---
    t_hat = get_optimal_roc_threshold(y_true, y_score)
    # HIBAJAVÍTÁS: Az sklearn roc_curve() a küszöböt '>=' konvencióval definiálja.
    # A korábbi szigorú '>' reláció t_hat = 1.0000 esetén minden képet elutasított
    # (P=R=F1=0.0). A '>=' relációval ez a határeset-összeomlás megszűnik[cite: 2].
    y_pred = (y_score >= t_hat).astype(int)

    return dict(
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        threshold=t_hat,
        n_train=len(train_paths),
        n_train_spliced=len(X_list),
        n_test=len(test_paths),
    )


# ---------------------------------------------------------------------------
# Egy szűrő-variáns teljes 5-fold rotációja és Table 1 aggregálása
# ---------------------------------------------------------------------------

def run_ablation_for_filter(display_name, filter_names, items):
    """Lefuttatja az 5-fold fordított keresztvalidációt egy adott szűrőkészletre[cite: 2]."""
    print(f"\n{'='*65}")
    print(f"  Szűrő: {display_name}  (filter_names={filter_names})")
    print(f"{'='*65}")

    print("  Jellemzőkinyerés (egyszer, képenként gyorsítótárazva)...")
    cache = extract_dataset_features(items, filter_names)
    valid_paths  = [p for p, m, lbl in items if p in cache]
    valid_labels = [cache[p]["image_label"] for p in valid_paths]

    if len(valid_paths) < N_FOLDS:
        print(f"  FIGYELEM: Túl kevés érvényes kép ({len(valid_paths)}) "
              f"{N_FOLDS}-fold CV-hez — kihagyva.")
        return None

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_results = []

    for fold_i, (big_idx, small_idx) in enumerate(
            skf.split(valid_paths, valid_labels)):
        # A CIKK SECTION 4.1 MÓDSZERTANA SZERINT FORDÍTVA:
        # A kis partíción (1/5, small_idx) tanítunk, és a nagy maradékon (4/5, big_idx)
        # tesztelünk[cite: 2, 19].
        train_idx, test_idx = small_idx, big_idx
        train_paths = [valid_paths[i] for i in train_idx]
        test_paths  = [valid_paths[i] for i in test_idx]

        res = run_fold(cache, train_paths, test_paths)
        if res is None:
            print(f"  [{fold_i+1}/{N_FOLDS}] kihagyva (nem volt elég adat)")
            continue

        fold_results.append(res)
        print(f"  [{fold_i+1}/{N_FOLDS}] train={res['n_train']:3d} "
              f"(spliced={res['n_train_spliced']:2d})  test={res['n_test']:3d}  "
              f"t_hat={res['threshold']:.4f}  P={res['precision']:.4f}  "
              f"R={res['recall']:.4f}  F1={res['f1']:.4f}")

    # Memória felszabadítása a következő szűrőkészlet előtt
    del cache
    gc.collect()

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


# ===========================================================================
# Fő futtatás: Table 1 generálása és JSON mentés
# ===========================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  Szűrő-abláció (checklist #7) — a cikk Table 1 reprodukciója")
    print("  HSB / VSB / RF0 / RF1 / RF2 külön-külön, majd Concatenated")
    print("=" * 65)

    items = collect_labeled_items()
    n_spliced = sum(1 for _, _, lbl in items if lbl == 1)
    n_auth    = sum(1 for _, _, lbl in items if lbl == 0)
    print(f"\nÖsszegyűjtött képek: {len(items)}  "
          f"(spliced={n_spliced}, authentic={n_auth})")
    if n_auth == 0:
        print("FIGYELEM: Nem található authentic könyvtár (4cam_auth/ vagy Au/) — "
              "a képszintű P/R/F1 nem lesz értelmezhető negatív minták nélkül!")

    all_results = {}
    for display_name, filter_names in FILTER_SETS.items():
        summary = run_ablation_for_filter(display_name, filter_names, items)
        if summary is not None:
            all_results[display_name] = summary

    # --- Table 1 formátumú összefoglaló táblázat kiírása ---
    print("\n" + "=" * 82)
    print("  ÖSSZEFOGLALÓ TÁBLÁZAT (az Itier et al., 2021 cikk Table 1 formátuma)")
    print("=" * 82)
    header = f"  {'Metrika':<22s}" + "".join(
        f"{name:>12s}" for name in FILTER_SETS if name in all_results)
    print(header)
    print("  " + "-" * (22 + 12 * len(all_results)))
    for metric_key, metric_label in [
            ("mean_precision", "Mean precision"),
            ("mean_recall",    "Mean recall"),
            ("mean_f1",        "Mean F1 score"),
            ("variance_f1",    "Variance F1 (x1e-4)")]:
        row = f"  {metric_label:<22s}"
        for name in FILTER_SETS:
            if name not in all_results:
                continue
            val = all_results[name][metric_key]
            if metric_key == "variance_f1":
                val *= 10000.0
            row += f"{val:>12.4f}"
        print(row)
    print("=" * 82)

    with open("filter_ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=4)
    print("\nJSON export: filter_ablation_results.json")