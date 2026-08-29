"""
trainmodel_three.py

Három osztályozó betanítása és összehasonlítása (Section 3.4):
  1. Random Forest Classifier (RFC) — a cikk eredeti modellje
  2. Support Vector Machine (SVM) — a cikk által említett alternatíva
  3. K-means — a cikk által említett alternatíva (2 klaszter: Uniform / Border)

Minden modell ugyanazon a kiegyensúlyozott, standardizált adaton tanít.
Minden modell külön .pkl fájlba kerül mentésre.

Javítások (trainmodel.py alapján):
  FIX #1: A GridSearchCV best_score_ attribútumát használjuk a felesleges
          cross_val_score újrafuttatás helyett (RFC, SVM esetén).
          K-means esetén (nincs GridSearch) a cross_val_score megmarad.

  FIX #2: SVM best_score_ a subsample-en futott CV-re vonatkozik — jelölve.

  FIX #3 (ÚJ): K-Means predict_proba NaN guard — ha exp(-dist) → 0 (nagyon
          nagy távolság), az összeg nulla lesz és 0/0 = nan. Megoldás: eps
          hozzáadása a nevezőhöz, hogy soha ne legyen osztás nullával.
          A NaN max_p = nan eredményeket okozta az evaluatemodel_three.py-ban.
"""

import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from itier import extract_features_from_image


# ---------------------------------------------------------------------------
# K-means wrapper: sklearn-kompatibilis osztályozóvá alakítja
# ---------------------------------------------------------------------------

class KMeansClassifier(BaseEstimator, ClassifierMixin):
    """
    K-means alapú 2-osztályos osztályozó wrapper.

    A K-means önmagában felügyelet nélküli. A cikk "k-means" alatt
    valószínűleg 2-klaszteres megközelítést ért:
      1. K=2 klaszter illesztése a tanítóadatokra
      2. A klaszterközpontok alapján meghatározzuk melyik klaszter = Border,
         melyik = Uniform (a tanítócímkék majority vote alapján)
      3. Teszteléskor: a közelebb lévő klaszterközponthoz rendeljük a blokkot

    predict_proba: a két klaszterközponttól való távolság alapján szoftmax-szerű
    valószínűséget számolunk.
    """

    def __init__(self, n_init=10, max_iter=300, random_state=42):
        self.n_init       = n_init
        self.max_iter     = max_iter
        self.random_state = random_state
        self.kmeans_      = None
        self.border_cluster_ = None

    def fit(self, X, y):
        self.kmeans_ = KMeans(
            n_clusters=2,
            n_init=self.n_init,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        self.kmeans_.fit(X)

        labels      = self.kmeans_.labels_
        border_in_0 = np.sum((labels == 0) & (y == 1))
        border_in_1 = np.sum((labels == 1) & (y == 1))
        self.border_cluster_ = 0 if border_in_0 > border_in_1 else 1
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        cluster_labels = self.kmeans_.predict(X)
        return (cluster_labels == self.border_cluster_).astype(int)

    def predict_proba(self, X):
        """
        Soft valószínűség a klaszterközponttól való euklideszi távolság alapján.

        FIX #3: eps guard a 0/0 NaN ellen.
        Ha a távolságok nagyon nagyok, exp(-dist) underflow → 0.
        eps = 1e-12 biztosítja, hogy a nevező soha ne legyen pontosan 0.
        """
        centers   = self.kmeans_.cluster_centers_
        dist_to_0 = np.linalg.norm(X - centers[0], axis=1)
        dist_to_1 = np.linalg.norm(X - centers[1], axis=1)

        d0    = np.exp(-dist_to_0)
        d1    = np.exp(-dist_to_1)
        # FIX #3: eps a 0/0 NaN ellen (nagyon nagy távolság → exp underflow)
        eps   = 1e-12
        total = d0 + d1 + eps

        if self.border_cluster_ == 1:
            prob_border  = d1 / total
            prob_uniform = d0 / total
        else:
            prob_border  = d0 / total
            prob_uniform = d1 / total

        return np.column_stack([prob_uniform, prob_border])


# ---------------------------------------------------------------------------
# Osztályok kiegyensúlyozása (trainmodel.py-ból átvéve, változtatás nélkül)
# ---------------------------------------------------------------------------

def balance_classes(X, y):
    """
    Undersampling alapú kiegyensúlyozás (Section 3.4).
    A cikk kiegyensúlyozott Border/Uniform arányt ír elő.
    """
    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.int32)

    border_idx  = np.where(y_arr == 1)[0]
    uniform_idx = np.where(y_arr == 0)[0]
    min_size    = min(len(border_idx), len(uniform_idx))

    if min_size == 0:
        return np.array([]), np.array([])

    np.random.seed(42)
    b_sampled = np.random.choice(border_idx,  min_size, replace=False)
    u_sampled = np.random.choice(uniform_idx, min_size, replace=False)

    idx = np.concatenate([b_sampled, u_sampled])
    np.random.shuffle(idx)
    return X_arr[idx], y_arr[idx]


# ---------------------------------------------------------------------------
# Fő betanítás
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("  Három osztályozó betanítása (RFC / SVM / K-means)")
    print("=" * 55)

    # ==========================================================
    # KONFIGURÁCIÓ
    MAX_IMAGES = None   # None = teljes adatbázis; szám = gyors teszt
    # ==========================================================

    with open("dataset_splits.json", "r") as f:
        data = json.load(f)

    train_images = data["train_images"]
    train_masks  = data["train_masks"]

    if MAX_IMAGES is not None:
        columbia_pairs = [(img, msk) for img, msk in zip(train_images, train_masks)
                          if "4cam" in img.lower()]
        casia_pairs    = [(img, msk) for img, msk in zip(train_images, train_masks)
                          if "tp" in img.lower()]
        half     = MAX_IMAGES // 2
        selected = columbia_pairs[:half] + casia_pairs[:half]
        if not selected:
            selected = list(zip(train_images, train_masks))[:MAX_IMAGES]
        train_images = [p[0] for p in selected]
        train_masks  = [p[1] for p in selected]
        print(f"Gyors teszt: {len(train_images)} képen tanítunk.")
    else:
        print(f"Teljes tanítóhalmaz: {len(train_images)} kép.")

    # --- 1. Nyers jellemzők összegyűjtése ---
    X_raw, y_raw = [], []
    for i, (img_p, mask_p) in enumerate(zip(train_images, train_masks)):
        print(f"  Feldolgozás ({i+1}/{len(train_images)}): {img_p}")
        feats, labels, _, _ = extract_features_from_image(img_p, mask_p)
        if feats is not None and labels is not None and len(feats) > 0:
            X_raw.append(feats)
            y_raw.append(labels)

    if not X_raw:
        print("Hiba: Nem sikerült jellemzőket kinyerni!")
        exit()

    X_all = np.vstack(X_raw)
    y_all = np.concatenate(y_raw)
    print(f"\nNyers eloszlás → Uniform(0): {int(np.sum(y_all==0))}, Border(1): {int(np.sum(y_all==1))}")

    # --- 2. Osztályok kiegyensúlyozása ---
    X_bal, y_bal = balance_classes(X_all, y_all)
    if len(X_bal) == 0:
        print("Hiba: Nincs elég adat!")
        exit()
    print(f"Kiegyensúlyozott → Uniform: {int(np.sum(y_bal==0))}, Border: {int(np.sum(y_bal==1))}")

    # --- 3. Globális StandardScaler (trainmodel.py FIX #1 alapján) ---
    print("\nGlobális StandardScaler illesztése...")
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_bal)
    joblib.dump(scaler, "itier_scaler.pkl")
    print("Scaler elmentve: itier_scaler.pkl")

    results = {}

    # =========================================================
    # MODEL 1: Random Forest Classifier (Section 3.4, cikk modellje)
    # =========================================================
    print("\n" + "=" * 55)
    print("  [1/3] Random Forest Classifier (GridSearchCV)")
    print("=" * 55)

    base_rf  = RandomForestClassifier(random_state=42)
    param_rf = {
        "n_estimators": [10,20,30,40,50,75, 100], #10 20 30 40 50 75 100
        "max_depth":    [10, 20, None],
        "criterion":    ["gini", "entropy"]
    }
    gs_rf = GridSearchCV(base_rf, param_rf, cv=3, n_jobs=-1, verbose=1, scoring="f1")
    gs_rf.fit(X_sc, y_bal)

    best_rf = gs_rf.best_estimator_
    print(f"Legjobb paraméterek: {gs_rf.best_params_}")

    # FIX #1: best_score_ a GridSearchCV belső CV eredménye — nem futtatunk
    # újabb cross_val_score-t, az csak felesleges re-fit lenne ugyanazon az adaton.
    cv_rf = gs_rf.best_score_
    print(f"CV F1 (3-fold, GridSearchCV): {cv_rf:.4f}")

    joblib.dump(best_rf, "itier_rf_model.pkl")
    print("Mentve: itier_rf_model.pkl")
    results["RFC"] = cv_rf

    # =========================================================
    # MODEL 2: Support Vector Machine
    # =========================================================
    print("\n" + "=" * 55)
    print("  [2/3] Support Vector Machine (GridSearchCV)")
    print("=" * 55)
    print("Megjegyzés: SVM lassan skálázódik nagy adathalmazon.")
    print("Ha túl lassú, csökkentsd MAX_IMAGES értékét.")

    MAX_SVM_SAMPLES = 30000
    if len(X_sc) > MAX_SVM_SAMPLES:
        np.random.seed(42)
        idx_svm = np.random.choice(len(X_sc), MAX_SVM_SAMPLES, replace=False)
        X_svm   = X_sc[idx_svm]
        y_svm   = y_bal[idx_svm]
        svm_note = f"(subsample: {MAX_SVM_SAMPLES}/{len(X_sc)} minta)"
        print(f"SVM subsample: {MAX_SVM_SAMPLES} mintára csökkentve (volt: {len(X_sc)})")
    else:
        X_svm, y_svm = X_sc, y_bal
        svm_note = "(teljes halmaz)"

    base_svm  = SVC(probability=True, random_state=42)
    param_svm = {
        "C":      [0.1, 1.0, 10.0],
        "kernel": ["rbf", "linear"],
        "gamma":  ["scale", "auto"]
    }
    gs_svm = GridSearchCV(base_svm, param_svm, cv=3, n_jobs=-1, verbose=1, scoring="f1")
    gs_svm.fit(X_svm, y_svm)

    best_svm = gs_svm.best_estimator_
    print(f"Legjobb paraméterek: {gs_svm.best_params_}")

    # FIX #1 + FIX #2: best_score_ a GridSearchCV belső CV eredménye.
    cv_svm = gs_svm.best_score_
    print(f"CV F1 (3-fold, GridSearchCV) {svm_note}: {cv_svm:.4f}")

    joblib.dump(best_svm, "itier_svm_model.pkl")
    print("Mentve: itier_svm_model.pkl")
    results["SVM"] = cv_svm

    # =========================================================
    # MODEL 3: K-means (GridSearch nem szükséges)
    # =========================================================
    print("\n" + "=" * 55)
    print("  [3/3] K-means Classifier (k=2)")
    print("=" * 55)

    km_clf = KMeansClassifier(n_init=10, max_iter=300, random_state=42)
    km_clf.fit(X_sc, y_bal)

    # K-means esetén nincs GridSearchCV, ezért cross_val_score indokolt.
    cv_km = cross_val_score(km_clf, X_sc, y_bal, cv=3, scoring="f1").mean()
    print(f"Border klaszter index: {km_clf.border_cluster_}")
    print(f"CV F1 (3-fold, cross_val_score): {cv_km:.4f}")

    joblib.dump(km_clf, "itier_kmeans_model.pkl")
    print("Mentve: itier_kmeans_model.pkl")
    results["K-means"] = cv_km

    # =========================================================
    # Összefoglaló
    # =========================================================
    print("\n" + "=" * 55)
    print("  ÖSSZEFOGLALÓ (tanítóhalmazon, 3-fold CV F1)")
    print("=" * 55)
    print(f"  {'Modell':<12} {'CV F1':>8}  {'Megjegyzés'}")
    print("  " + "-" * 50)
    for name, score in sorted(results.items(), key=lambda x: -x[1]):
        note = "(subsample CV)" if name == "SVM" and len(X_sc) > MAX_SVM_SAMPLES else ""
        bar  = "#" * int(score * 40)
        print(f"  {name:<12}: {score:.4f}  {bar}  {note}")

    best_name = max(results, key=results.get)
    print(f"\nLegjobb modell (CV F1 alapján): {best_name}")
    print("\nAz evaluatemodel_three.py a modell fájlokat automatikusan betölti:")
    print("  RFC:     itier_rf_model.pkl")
    print("  SVM:     itier_svm_model.pkl")
    print("  K-means: itier_kmeans_model.pkl")
    print("  Scaler:  itier_scaler.pkl  (minden modellhez közös)")