"""
trainmodel_seven.py

Hét különböző gépi tanulási osztályozó betanítása, hiperparaméter-optimalizálása,
valószínűség-kalibrálása és összehasonlítása a 3-osztályos SRM-PCA blokkjellemzőkön:

  1. Random Forest Classifier (RFC)         — Az Itier et al. (2021) cikk alapmodellje
  2. Support Vector Machine (SVM)           — Cikk által említett alternatíva (isotonic kalibrációval)
  3. K-Means Classifier (k=6, Softmax)      — Cikk által említett alternatíva (N-osztályos wrapper)
  4. Multi-Layer Perceptron (MLP)           — Sklearn neurális háló (sigmoid/Platt kalibrációval)
  5. Gradient Boosting / XGBoost            — Szekvenciális boosting ensemble
  6. Gaussian Mixture Model (GMM)           — Standard EM alapú generatív puha klaszterezés
  7. Bayesian Gaussian Mixture (BayesianEM) — Variational EM, Dirichlet-folyamat priorral

─── Fő módszertani és szoftvermérnöki pillérek ─────────────────────────────────
  - ADATSZIVÁRGÁS-MENTES GLOBÁLIS SKÁLÁZÁS (FIX #1):
    A StandardScaler kizárólag a kiegyensúlyozott tanítóblokkokon (X_bal)
    illeszkedik, és `itier_scaler.pkl` néven kerül mentésre.
  - 3-OSZTÁLYOS KIEGYENSÚLYOZÁS (checklist #1):
    A `balance_classes()` függvény alulmintavételezéssel (undersampling) egyenlő
    elemszámra hozza a CLASS_UNIFORM (0), CLASS_BORDER (1) és CLASS_SPLICED (2)
    osztályokat (lásd Cikk Section 3.4).
  - KILENCPONTOS MEMÓRIA-OPTIMALIZÁCIÓ (MEM #1 – MEM #9):
    1. del X_raw, y_raw          -> Nyers listák törlése np.vstack után
    2. MAX_TRAIN_BLOCKS korlát   -> Elő-mintavételezés az OOM (Out-Of-Memory) ellen
    3. np.asarray (float32)      -> Másolásmentes típuskonverzió
    4. del X_all, y_all          -> Teljes blokkmátrix törlése kiegyensúlyozás után
    5. StandardScaler -> float32 -> A float64 kimenet felezése (8 bájt -> 4 bájt)
    6. del X_bal                 -> Skálázatlan tömb törlése illesztés után
    7. N_JOBS_CV = 1             -> Nincs multiprocess worker-fork memóriaduplikáció
    8. Explicit gc.collect()     -> Memória-visszanyerés minden egyes modell után
    9. Részhalmaz-indexek törlése -> idx_svm, idx_mlp, idx_gmm, idx_em felszabadítása
────────────────────────────────────────────────────────────────────────────────
"""

import gc
import json
import os
import cv2
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Hatterbeli (GUI nelkuli) kepmentes
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from itier import extract_features_from_image, load_binary_mask
from clustering_models import GMMClassifier, BayesianEMClassifier

# Opcionális XGBoost betöltés (ha nincs telepítve, sklearn GradientBoosting a fallback)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Megjegyzés: xgboost nincs telepítve -> sklearn GradientBoosting fallback.")
    print("Telepítés: pip install xgboost\n")


# ===========================================================================
# Önálló K-Means osztályozó wrapper (3-osztályos séma, Softmax távolsággal)
# ===========================================================================

class KMeansClassifier(BaseEstimator, ClassifierMixin):
    """
    K-Means alapú, N-osztályos scikit-learn kompatibilis osztályozó wrapper.

    Működési elv:
      1. fit(X, y):
         - Felügyelet nélkül `n_clusters` (alapértelmezetten 6) Voronoï-cellára
           bontja a 60 dimenziós jellemzőteret.
         - A tanítócímkék (y) alapján minden klasztert a benne lévő minták
           TÖBBSÉGI osztályához (Uniform / Border / Spliced) rendel.
         - Ütközésmentes biztonsági mentéssel garantálja, hogy minden osztályhoz
           tartozzon legalább egy klaszter.
      2. predict_proba(X):
         - A klaszterközéppontoktól mért euklideszi távolságokat (d_k) az átlagos
           saját-klaszter távolsággal (_scale_) normálja, majd numerikusan stabil
           (subtract-max) Softmax függvénnyel valószínűségi súlyokká alakítja.
         - Az azonos osztályhoz rendelt klaszterek súlyait összeadja.
    """

    def __init__(self, n_clusters=6, n_init=10, max_iter=300, random_state=42):
        self.n_clusters   = n_clusters
        self.n_init       = n_init
        self.max_iter     = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32)
        self.classes_ = np.unique(y)
        n_classes     = len(self.classes_)

        effective_clusters = max(int(self.n_clusters), n_classes)
        self.effective_n_clusters_ = effective_clusters

        self.kmeans_ = KMeans(
            n_clusters   = effective_clusters,
            n_init       = self.n_init,
            max_iter     = self.max_iter,
            random_state = self.random_state
        )
        hard_labels = self.kmeans_.fit_predict(X)

        # Klaszterenkénti osztályarányok kiszámítása
        class_rates = np.zeros((effective_clusters, n_classes), dtype=np.float64)
        for k in range(effective_clusters):
            mask = (hard_labels == k)
            if np.any(mask):
                for ci, c in enumerate(self.classes_):
                    class_rates[k, ci] = np.mean(y[mask] == c)

        cluster_to_class_idx = np.argmax(class_rates, axis=1)

        # Ütközésmentes biztonsági mentés: minden osztálynak jusson legalább 1 klaszter
        reserved = set()
        for ci in range(n_classes):
            if not np.any(cluster_to_class_idx == ci):
                for cand_k in np.argsort(-class_rates[:, ci]):
                    cand_k = int(cand_k)
                    if cand_k in reserved:
                        continue
                    owner = cluster_to_class_idx[cand_k]
                    if np.sum(cluster_to_class_idx == owner) > 1:
                        cluster_to_class_idx[cand_k] = ci
                        reserved.add(cand_k)
                        break

        self.cluster_to_class_idx_ = cluster_to_class_idx
        self.class_rates_          = class_rates

        # Skálatényező a Softmax hőmérsékletéhez: átlagos távolság a saját középponttól
        dists_to_own = np.min(self.kmeans_.transform(X), axis=1)
        self._scale_ = max(float(np.mean(dists_to_own)), 1e-6)

        # Naplózási attribútumok a Border (1) osztályhoz
        if 1 in self.classes_:
            border_idx = int(np.where(self.classes_ == 1)[0][0])
            self.border_rates_   = class_rates[:, border_idx]
            self.border_cluster_ = int(np.argmax(self.border_rates_))
        else:
            self.border_rates_   = np.zeros(effective_clusters, dtype=np.float64)
            self.border_cluster_ = 0

        return self

    def predict_proba(self, X):
        """
        Numerikusan stabil Softmax távolság-transzformáció:
          logits_k(x) = -||x - mu_k||_2 / _scale_
          w_k(x)      = exp(logits_k - max_j logits_j) / Σ exp(...)
        """
        check_is_fitted(self, ['kmeans_', 'cluster_to_class_idx_', '_scale_'])
        X = check_array(X)
        X = np.asarray(X, dtype=np.float64)

        dists  = self.kmeans_.transform(X)
        logits = -dists / self._scale_
        logits -= logits.max(axis=1, keepdims=True)   # Túlcsordulás (overflow) elleni védelem
        weights = np.exp(logits)
        weights /= weights.sum(axis=1, keepdims=True)

        n_classes = len(self.classes_)
        proba = np.zeros((X.shape[0], n_classes), dtype=np.float64)
        for k in range(self.effective_n_clusters_):
            ci = self.cluster_to_class_idx_[k]
            proba[:, ci] += weights[:, k]

        row_sums = proba.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return proba / row_sums

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


# ===========================================================================
# MEMÓRIA- ÉS FUTÁSI KONFIGURÁCIÓ
# ===========================================================================
MAX_TRAIN_BLOCKS = 220_000   # MEM #2: Nyers tanítóblokkok felső korlátja (~34.3 MB float32)
N_JOBS_CV        = 2         # MEM #7: 1 = szekvenciális CV (nincs worker-fork memóriaduplikáció)

# Modellspecifikus mintakorlátok a nagy számítási igényű algoritmusokhoz:
MAX_SVM_SAMPLES  = 20_000    # Az SVM kvadratikus/köbös skálázódása miatt
MAX_MLP_SAMPLES  = 40_000    # MLP RandomizedSearchCV (30 iteráció x 3 fold) gyorsításához
MAX_GMM_SAMPLES  = 30_000    # 60x60-as teljes kovarianciamátrixok EM becsléséhez
MAX_BGMM_SAMPLES = 15_000    # Variational EM (BayesianGMM) számítási igénye miatt
# ===========================================================================


# ---------------------------------------------------------------------------
# Osztályok kiegyensúlyozása (N-osztályos undersampling — Cikk Section 3.4)
# ---------------------------------------------------------------------------

def balance_classes(X, y):
    """
    Alulmintavételezéssel (undersampling) kiegyensúlyozza a tanítóhalmazt:
    minden osztályból (Uniform=0, Border=1, Spliced=2) pontosan annyit tart meg,
    amennyi a legkisebb elemszámú osztályban (jellemzően a Border osztályban) van.

    MEM #3: np.asarray(..., dtype=np.float32) — nem készít felesleges másolatot,
    ha a bemenet már float32 típusú.
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
    return X_arr[idx].copy(), y_arr[idx].copy()


# ---------------------------------------------------------------------------
# Post-training képszintű küszöbbecslés (ROC / max_p és KS-statisztika)
# ---------------------------------------------------------------------------

def _build_heat_map(img_path, clf, scaler, block_size=32, step=8):
    """Felépíti egy kép (H, W) méretű valószínűségi hőtérképét."""
    feats, _, positions, dims = extract_features_from_image(
        img_path, block_size=block_size, step=step)
    if feats is None or len(feats) == 0 or dims is None:
        return None, None, None
    h, w = dims
    feats_sc = np.nan_to_num(
        scaler.transform(feats), nan=0.0, posinf=0.0, neginf=0.0)

    # 3-OSZTÁLYOS SÉMA: P(Manipulált) = 1 - P(Uniform) = P(Border) + P(Spliced)
    probs = 1.0 - clf.predict_proba(feats_sc)[:, 0]

    hs = np.zeros((h, w), np.float32)
    hc = np.zeros((h, w), np.float32)
    for p, (y, x) in zip(probs, positions):
        hs[y:y+block_size, x:x+block_size] += p
        hc[y:y+block_size, x:x+block_size] += 1.0
    hc[hc == 0] = 1.0
    return hs / hc, h, w


def _both_scores(img_path, clf, scaler, mask_path=None, block_size=32, step=8):
    """
    Kiszámítja mindkét képszintű pontszámot egyetlen hőtérkép-felépítéssel:
      1. mp : max(P_I) — a hőtérkép globális maximuma (Cikk Section 3.5, Eq. 4-5)
      2. ks : max|C(S) - C(H)| — kétmintás Kolmogorov-Smirnov statisztika (Cikk Eq. 8)
    """
    heat_map, h, w = _build_heat_map(img_path, clf, scaler, block_size, step)
    if heat_map is None:
        return 0.0, 0.0

    mp = float(np.nanmax(heat_map)) if np.any(np.isfinite(heat_map)) else 0.0

    mask_binary = None
    if mask_path and os.path.exists(mask_path):
        mb = load_binary_mask(mask_path)
        if mb is not None:
            if mb.shape != (h, w):
                mb = cv2.resize(mb, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_binary = mb

    if mask_binary is not None:
        S_mask = mask_binary > 127
    else:
        # Hiteles (authentic) képnél szintetikus GT maszk: középső negyed négyzet
        cy, cx     = h // 2, w // 2
        sq_h, sq_w = max(1, h // 4), max(1, w // 4)
        S_mask     = np.zeros((h, w), dtype=bool)
        S_mask[cy - sq_h:cy + sq_h, cx - sq_w:cx + sq_w] = True

    S_vals = heat_map[S_mask].flatten()
    H_vals = heat_map[~S_mask].flatten()
    ks = float(ks_2samp(S_vals, H_vals)[0]) if (len(S_vals) >= 2 and len(H_vals) >= 2) else 0.0
    return mp, ks


def _roc_optimal(y_true, y_scores):
    """
    Kiszámítja a ROC görbe (0,1) ideális pontjától vett L2-távolságot minimalizáló
    optimális döntési küszöböt (Cikk Section 3.5).

    HIBAJAVÍTÁS: Az sklearn roc_curve() által beszúrt mesterséges 0. indexű pontot
    (FPR=0, TPR=0, threshold=max+1) kizárjuk a minimumkeresésből, hogy a küszöb
    sose ragadjon be egy triviális "mindent elutasító" értékre.
    """
    yt = np.array(y_true)
    ys = np.array(y_scores, dtype=float)
    valid = np.isfinite(ys)
    if not np.any(valid) or len(np.unique(yt[valid])) < 2:
        return 0.5, 0.0, 0.0, 0.5, np.array([0, 1]), np.array([0, 1])

    auc = roc_auc_score(yt[valid], ys[valid])
    fpr_full, tpr_full, thrs_full = roc_curve(yt[valid], ys[valid])

    # A mesterséges első pont kizárása a küszöbkeresésből
    if len(thrs_full) > 1:
        fpr_search, tpr_search, thrs_search = fpr_full[1:], tpr_full[1:], thrs_full[1:]
    else:
        fpr_search, tpr_search, thrs_search = fpr_full, tpr_full, thrs_full

    best = int(np.argmin(np.sqrt(fpr_search**2 + (tpr_search - 1.0)**2)))
    return (
        float(thrs_search[best]),
        float(fpr_search[best]),
        float(tpr_search[best]),
        float(auc),
        fpr_full,
        tpr_full,
    )


def _save_threshold(t_hat, method, auc, fpr_opt, tpr_opt, n_spl, n_aut):
    """Elmenti a kalibrált küszöböt .txt és részletes .json formátumban."""
    txt_path  = f"optimal_threshold_{method}.txt"
    meta_path = f"optimal_threshold_meta_{method}.json"
    with open(txt_path, "w") as f:
        f.write(str(t_hat))
    with open(meta_path, "w") as f:
        json.dump({
            "threshold":       t_hat,
            "method":          method,
            "auc":             auc,
            "fpr_at_optimum":  fpr_opt,
            "tpr_at_optimum":  tpr_opt,
            "spliced_count":   n_spl,
            "authentic_count": n_aut
        }, f, indent=4)
    print(f"    {txt_path:<40s}  t_hat={t_hat:.4f}  AUC={auc:.4f}")
    print(f"    {meta_path}")


def estimate_both_thresholds(clf, scaler, data, block_size=32, step=8):
    """Post-training küszöbbecslés mindkét módszerrel (ROC és KS) az RFC modellen."""
    print(f"\n{'='*62}")
    print("  POST-TRAINING KÜSZÖBBECSLÉS — mindkét módszer (ROC + KS)")
    print(f"{'='*62}")

    test_images = data["test_images"]
    test_masks  = data["test_masks"]
    y_true, scores_roc, scores_ks = [], [], []
    spliced_count = 0

    print(f"\n  --- Hamisított (test) képek ---")
    for img_path, mask_path in zip(test_images, test_masks):
        lower = img_path.lower()
        base  = os.path.basename(img_path)
        if not ("splc" in lower or "\\tp\\" in lower or "/tp/" in lower
                or base.startswith("Tp_")):
            continue
        mp, ks = _both_scores(img_path, clf, scaler, mask_path=mask_path,
                               block_size=block_size, step=step)
        y_true.append(1)
        scores_roc.append(mp)
        scores_ks.append(ks)
        spliced_count += 1
        print(f"    Spliced:   {base:42s}  max_p={mp:.4f}  ks={ks:.4f}")

    if spliced_count == 0:
        print("  FIGYELEM: Nem találtam spliced képet -> küszöbbecslés kihagyva.")
        return

    print(f"\n  --- Hiteles (authentic) képek (max {spliced_count} db) ---")
    auth_files = []
    for adir in ["4cam_auth", "Au"]:
        if os.path.exists(adir):
            for fn in sorted(os.listdir(adir)):
                if fn.lower().endswith((".jpg", ".png", ".tif", ".jpeg")):
                    auth_files.append(os.path.join(adir, fn))

    if not auth_files:
        print("  FIGYELEM: Nem találtam authentic képeket -> küszöbbecslés kihagyva.")
        return

    np.random.seed(42)
    np.random.shuffle(auth_files)
    auth_files = auth_files[:spliced_count]
    for img_path in auth_files:
        base   = os.path.basename(img_path)
        mp, ks = _both_scores(img_path, clf, scaler, mask_path=None,
                               block_size=block_size, step=step)
        y_true.append(0)
        scores_roc.append(mp)
        scores_ks.append(ks)
        print(f"    Authentic: {base:42s}  max_p={mp:.4f}  ks={ks:.4f}")

    authentic_count = len(auth_files)
    y_true_arr = np.array(y_true)
    sc_roc     = np.array(scores_roc)
    sc_ks      = np.array(scores_ks)

    if len(np.unique(y_true_arr)) < 2:
        print("  HIBA: Nem elegendő osztály -> küszöbbecslés kihagyva.")
        return

    print(f"\n  --- ROC számítás ---")
    t_roc, fpr_roc_opt, tpr_roc_opt, auc_roc, fpr_roc, tpr_roc = _roc_optimal(y_true_arr, sc_roc)
    t_ks,  fpr_ks_opt,  tpr_ks_opt,  auc_ks,  fpr_ks,  tpr_ks  = _roc_optimal(y_true_arr, sc_ks)
    print(f"    ROC (max_p) :  t_hat={t_roc:.4f}  FPR={fpr_roc_opt:.4f}  TPR={tpr_roc_opt:.4f}  AUC={auc_roc:.4f}")
    print(f"    KS (ks_stat):  t_hat={t_ks:.4f}  FPR={fpr_ks_opt:.4f}  TPR={tpr_ks_opt:.4f}  AUC={auc_ks:.4f}")

    print(f"\n  --- Mentés ---")
    _save_threshold(t_roc, "roc", auc_roc, fpr_roc_opt, tpr_roc_opt, spliced_count, authentic_count)
    _save_threshold(t_ks,  "ks",  auc_ks,  fpr_ks_opt,  tpr_ks_opt,  spliced_count, authentic_count)

    # 2x2-es diagnosztikai ábra mentése
    spl_roc = sc_roc[y_true_arr == 1]; aut_roc = sc_roc[y_true_arr == 0]
    spl_ks  = sc_ks[y_true_arr == 1];  aut_ks  = sc_ks[y_true_arr == 0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Post-training küszöbbecslés — ROC módszer (bal) vs KS módszer (jobb)",
                 fontsize=13, fontweight="bold")
    configs = [
        (fpr_roc, tpr_roc, auc_roc, t_roc, fpr_roc_opt, tpr_roc_opt,
         f"ROC görbe — max(P_I)\nAUC={auc_roc:.3f}",
         spl_roc, aut_roc, "max_p értéke", f"max(P_I) eloszlás — t_hat={t_roc:.3f}"),
        (fpr_ks, tpr_ks, auc_ks, t_ks, fpr_ks_opt, tpr_ks_opt,
         f"ROC görbe — KS-statisztika\nAUC={auc_ks:.3f}",
         spl_ks, aut_ks, "KS-statisztika értéke", f"KS eloszlás — t_hat={t_ks:.3f}"),
    ]
    for col, (fpr_a, tpr_a, auc_a, t_a, fpr_opt_a, tpr_opt_a, roc_title,
              spl_s, aut_s, xlabel, dist_title) in enumerate(configs):
        ax0 = axes[0][col]
        ax0.plot(fpr_a, tpr_a, color="darkorange", lw=2, label=f"AUC={auc_a:.3f}")
        ax0.plot([0, 1], [0, 1], "navy", lw=1, linestyle="--")
        ax0.scatter([fpr_opt_a], [tpr_opt_a], color="red", s=100, zorder=5, label=f"t_hat={t_a:.3f}")
        ax0.set_xlabel("FPR"); ax0.set_ylabel("TPR")
        ax0.set_title(roc_title); ax0.legend(); ax0.grid(True, alpha=0.3)

        ax1 = axes[1][col]
        all_s = np.concatenate([spl_s, aut_s])
        bins  = np.linspace(0, max(float(all_s.max()), 0.01), 40)
        ax1.hist(spl_s, bins=bins, alpha=0.6, color="red",  label=f"Spliced (n={len(spl_s)})")
        ax1.hist(aut_s, bins=bins, alpha=0.6, color="blue", label=f"Authentic (n={len(aut_s)})")
        ax1.axvline(t_a, color="black", linestyle="--", lw=2, label=f"t_hat={t_a:.3f}")
        ax1.set_xlabel(xlabel); ax1.set_ylabel("Képek száma")
        ax1.set_title(dist_title); ax1.legend(); ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("roc_curve_both.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("\n  Ábra mentve: roc_curve_both.png")
    print(f"\n  {'='*58}\n  ÖSSZEFOGLALÓ\n  {'='*58}")
    print(f"  {'ROC / max_p':<14} {t_roc:>8.4f} {auc_roc:>8.4f} {fpr_roc_opt:>8.4f} {tpr_roc_opt:>8.4f}  optimal_threshold_roc.txt")
    print(f"  {'KS-stat':<14} {t_ks:>8.4f} {auc_ks:>8.4f} {fpr_ks_opt:>8.4f} {tpr_ks_opt:>8.4f}  optimal_threshold_ks.txt")


# ===========================================================================
# Fő betanítási folyamat (7 modell + globális StandardScaler)
# ===========================================================================

if __name__ == "__main__":
    print("=" * 62)
    print("  Hét osztályozó betanítása — Memória-optimalizált verzió")
    print("  RFC / SVM / K-Means / MLP / Boosting / GMM / BayesianEM")
    print("=" * 62)
    print(f"\n  Konfiguráció:")
    print(f"    MAX_TRAIN_BLOCKS = {MAX_TRAIN_BLOCKS:,}  (nyers minták felső határa)")
    print(f"    N_JOBS_CV        = {N_JOBS_CV}             (GridSearchCV parallelizmus)")
    print(f"    Becsült max. mátrixméret: {MAX_TRAIN_BLOCKS * 60 * 4 / 1024**2:.1f} MB (float32)\n")

    MAX_IMAGES = 60   # None = összes tanítókép (135 db); egész szám = gyors teszt

    with open("dataset_splits.json", "r") as f:
        data = json.load(f)

    train_images = data["train_images"]
    train_masks  = data["train_masks"]

    if MAX_IMAGES is not None:
        col  = [(i, m) for i, m in zip(train_images, train_masks) if "4cam" in i.lower()]
        cas  = [(i, m) for i, m in zip(train_images, train_masks) if "tp"   in i.lower()]
        half = MAX_IMAGES // 2
        sel  = col[:half] + cas[:half] or list(zip(train_images, train_masks))[:MAX_IMAGES]
        train_images, train_masks = zip(*sel)
        print(f"Gyors teszt: {len(train_images)} képen tanítunk.")
    else:
        print(f"Teljes tanítóhalmaz: {len(train_images)} kép.")

    # ── 1. Jellemzőkinyerés a tanítóképekből ──────────────────────────────
    X_raw, y_raw = [], []
    for i, (img_p, mask_p) in enumerate(zip(train_images, train_masks)):
        print(f"  Feldolgozás ({i+1}/{len(train_images)}): {img_p}")
        feats, labels, _, _ = extract_features_from_image(img_p, mask_p)
        if feats is not None and labels is not None and len(feats) > 0:
            X_raw.append(feats.astype(np.float32))   # Azonnali float32 tárolás
            y_raw.append(labels.astype(np.int32))

    if not X_raw:
        print("Hiba: Nem sikerült jellemzőket kinyerni!"); exit()

    X_all = np.vstack(X_raw)
    y_all = np.concatenate(y_raw)
    del X_raw, y_raw          # MEM #1: Nyers listák azonnali felszabadítása
    gc.collect()

    print(f"\nNyers -> Uniform(0): {int(np.sum(y_all==0))}, Border(1): {int(np.sum(y_all==1))}, "
          f"Spliced(2): {int(np.sum(y_all==2))}")
    print(f"Nyers blokkok: {len(X_all):,}  ->  {X_all.nbytes / 1024**2:.1f} MB (float32)")

    # MEM #2: Pre-balance subsample — X_all korlátozása MAX_TRAIN_BLOCKS-ra
    if len(X_all) > MAX_TRAIN_BLOCKS:
        np.random.seed(42)
        idx_pre = np.random.choice(len(X_all), MAX_TRAIN_BLOCKS, replace=False)
        X_all   = X_all[idx_pre]
        y_all   = y_all[idx_pre]
        del idx_pre
        gc.collect()
        print(f"Pre-balance subsample -> {len(X_all):,} minta (MAX_TRAIN_BLOCKS={MAX_TRAIN_BLOCKS:,})")

    # ── 2. Osztályok kiegyensúlyozása (Uniform / Border / Spliced) ────────
    X_bal, y_bal = balance_classes(X_all, y_all)
    del X_all, y_all          # MEM #4: X_all felszabadítása
    gc.collect()

    if len(X_bal) == 0:
        print("Hiba: Nincs elég adat az osztályok kiegyensúlyozása után!"); exit()
    print(f"Kiegyensúlyozott -> Uniform: {int(np.sum(y_bal==0))}, Border: {int(np.sum(y_bal==1))}, "
          f"Spliced: {int(np.sum(y_bal==2))}")
    print(f"Balanced méret: {X_bal.nbytes / 1024**2:.1f} MB (float32)")

    # ── 3. Globális StandardScaler illesztése és mentése (FIX #1) ─────────
    # MEM #5: A StandardScaler float64 kimenetét visszakonvertáljuk float32-re
    print("\nGlobális StandardScaler illesztése...")
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_bal).astype(np.float32)
    joblib.dump(scaler, "itier_scaler.pkl")
    print("Scaler elmentve: itier_scaler.pkl")
    print(f"X_sc méret: {X_sc.nbytes / 1024**2:.1f} MB (float32)")
    del X_bal                 # MEM #6: X_bal felszabadítása
    gc.collect()

    results = {}

    # ── MODEL 1: Random Forest Classifier (RFC) ───────────────────────────
    print("\n" + "=" * 62)
    print("  [1/7] Random Forest Classifier (GridSearchCV)")
    print(f"  n_jobs={N_JOBS_CV}  (MEM #7: nincs fork-másolat)")
    print("=" * 62)
    gs_rf = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid={
            "n_estimators":     [100, 300],
            "max_depth":        [15, None],
            "min_samples_leaf": [1, 5],
            "max_features":     ["sqrt", "log2"],
        },
        cv=3, n_jobs=N_JOBS_CV, verbose=1, scoring="f1_macro")
    gs_rf.fit(X_sc, y_bal)
    best_rf = gs_rf.best_estimator_
    print(f"Legjobb paraméterek: {gs_rf.best_params_}")
    print(f"CV F1 (3-fold): {gs_rf.best_score_:.4f}")
    joblib.dump(best_rf, "itier_rf_model.pkl")
    print("Mentve: itier_rf_model.pkl")
    results["RFC"] = float(gs_rf.best_score_)
    del gs_rf                 # MEM #8
    gc.collect()

    # ── MODEL 2: Support Vector Machine (SVM + Isotonic kalibráció) ───────
    print("\n" + "=" * 62)
    print("  [2/7] SVM + CalibratedClassifierCV (isotonic)")
    print("=" * 62)
    n_svm = min(MAX_SVM_SAMPLES, len(X_sc))
    if len(X_sc) > n_svm:
        np.random.seed(42)
        idx_svm      = np.random.choice(len(X_sc), n_svm, replace=False)
        X_svm, y_svm = X_sc[idx_svm], y_bal[idx_svm]
        del idx_svm
        svm_note = f"subsample {n_svm}/{len(X_sc)}"
        print(f"SVM subsample: {n_svm:,} minta.")
    else:
        X_svm, y_svm = X_sc, y_bal
        svm_note = "teljes halmaz"

    gs_svm = GridSearchCV(
        SVC(probability=False, random_state=42),
        param_grid={
            "C":      [0.1, 1.0, 10.0],
            "kernel": ["rbf", "linear"],
            "gamma":  ["scale", "auto"]
        },
        cv=3, n_jobs=N_JOBS_CV, verbose=1, scoring="f1_macro")
    gs_svm.fit(X_svm, y_svm)
    print(f"Legjobb SVM paraméterek: {gs_svm.best_params_}")
    print(f"CV F1 (3-fold, {svm_note}): {gs_svm.best_score_:.4f}")
    calibrated_svm = CalibratedClassifierCV(
        SVC(probability=False, random_state=42, **gs_svm.best_params_),
        method="isotonic", cv=3)
    calibrated_svm.fit(X_svm, y_svm)
    joblib.dump(calibrated_svm, "itier_svm_model.pkl")
    print("Mentve: itier_svm_model.pkl")
    results["SVM"] = float(gs_svm.best_score_)
    del gs_svm, X_svm, y_svm, calibrated_svm  # MEM #9
    gc.collect()

    # ── MODEL 3: K-Means Classifier (k=6, Softmax távolság-leképezés) ─────
    print("\n" + "=" * 62)
    print("  [3/7] K-Means Classifier (k=6, 3-osztályos Softmax wrapper)")
    print("=" * 62)
    km_clf = KMeansClassifier(n_clusters=6, n_init=10, max_iter=300, random_state=42)
    km_clf.fit(X_sc, y_bal)
    cv_km = float(cross_val_score(
        km_clf, X_sc, y_bal, cv=3, scoring="f1_macro", n_jobs=N_JOBS_CV
    ).mean())
    print(f"Border klaszter index: {km_clf.border_cluster_}")
    print(f"CV F1 (3-fold): {cv_km:.4f}")
    joblib.dump(km_clf, "itier_kmeans_model.pkl")
    print("Mentve: itier_kmeans_model.pkl")
    results["K-Means"] = cv_km
    del km_clf
    gc.collect()

    # ── MODEL 4: Multi-Layer Perceptron (MLP + Sigmoid/Platt kalibráció) ──
    print("\n" + "=" * 62)
    print("  [4/7] MLP + CalibratedClassifierCV (sigmoid)")
    print("=" * 62)
    n_mlp = min(MAX_MLP_SAMPLES, len(X_sc))
    if len(X_sc) > n_mlp:
        np.random.seed(42)
        idx_mlp      = np.random.choice(len(X_sc), n_mlp, replace=False)
        X_mlp, y_mlp = X_sc[idx_mlp], y_bal[idx_mlp]
        del idx_mlp
        mlp_note = f"subsample {n_mlp}/{len(X_sc)}"
        print(f"MLP subsample: {n_mlp:,} minta.")
    else:
        X_mlp, y_mlp = X_sc, y_bal
        mlp_note = "teljes halmaz"

    rs_mlp = RandomizedSearchCV(
        MLPClassifier(solver="adam", max_iter=500, early_stopping=True,
                      n_iter_no_change=10, random_state=42),
        param_distributions={
            "hidden_layer_sizes": [(128, 64), (256, 128), (64, 32), (128, 64, 32), (100,)],
            "alpha":              [1e-4, 1e-3, 1e-2, 1e-1],
            "learning_rate_init": [1e-4, 1e-3, 5e-3],
            "activation":         ["relu", "tanh"],
            "learning_rate":      ["constant", "adaptive"],
            "batch_size":         [64, 128, 200],
        },
        n_iter=20, cv=3, n_jobs=N_JOBS_CV, verbose=1, scoring="f1_macro", random_state=42)
    rs_mlp.fit(X_mlp, y_mlp)
    print(f"Legjobb MLP paraméterek: {rs_mlp.best_params_}")
    print(f"CV F1 (3-fold, {mlp_note}): {rs_mlp.best_score_:.4f}")
    calibrated_mlp = CalibratedClassifierCV(rs_mlp.best_estimator_,
                                            method="sigmoid", cv=5)
    calibrated_mlp.fit(X_mlp, y_mlp)
    joblib.dump(calibrated_mlp, "itier_mlp_model.pkl")
    print("Mentve: itier_mlp_model.pkl")
    results["MLP"] = float(rs_mlp.best_score_)
    del rs_mlp, X_mlp, y_mlp, calibrated_mlp  # MEM #9
    gc.collect()

    # ── MODEL 5: Gradient Boosting / XGBoost ──────────────────────────────
    print("\n" + "=" * 62)
    print(f"  [5/7] {'XGBoost' if XGBOOST_AVAILABLE else 'GradientBoosting'} "
          f"(RandomizedSearchCV)")
    print("=" * 62)

    if XGBOOST_AVAILABLE:
        base_boost = XGBClassifier(eval_metric="mlogloss",
                                   random_state=42, n_jobs=1)
        boost_params = {
            "n_estimators":     [100, 200, 300],
            "max_depth":        [3, 4, 5, 6],
            "learning_rate":    [0.01, 0.05, 0.1, 0.2],
            "subsample":        [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
            "gamma":            [0, 0.1, 0.5, 1.0],
            "reg_alpha":        [0, 0.01, 0.1],
            "reg_lambda":       [0.5, 1.0, 2.0],
        }
        model_filename, model_key = "itier_xgb_model.pkl", "XGBoost"
    else:
        base_boost = GradientBoostingClassifier(random_state=42,
                                                n_iter_no_change=15,
                                                validation_fraction=0.1)
        boost_params = {
            "n_estimators":     [100, 200, 300],
            "max_depth":        [3, 4, 5],
            "learning_rate":    [0.01, 0.05, 0.1, 0.2],
            "subsample":        [0.7, 0.8, 1.0],
            "min_samples_leaf": [1, 5, 10],
            "ccp_alpha":        [0.0, 0.001, 0.01],
        }
        model_filename, model_key = "itier_gb_model.pkl", "GradientBoosting"

    rs_boost = RandomizedSearchCV(base_boost, param_distributions=boost_params,
                                  n_iter=15, cv=3, n_jobs=N_JOBS_CV, verbose=1,
                                  scoring="f1_macro", random_state=42)
    rs_boost.fit(X_sc, y_bal)
    best_boost = rs_boost.best_estimator_
    print(f"Legjobb paraméterek: {rs_boost.best_params_}")
    print(f"CV F1 (3-fold): {rs_boost.best_score_:.4f}")
    joblib.dump(best_boost, model_filename)
    print(f"Mentve: {model_filename}")
    results[model_key] = float(rs_boost.best_score_)
    with open("itier_boost_info.txt", "w") as f:
        f.write(f"{model_key}\n{model_filename}")
    del rs_boost, base_boost, best_boost  # MEM #8
    gc.collect()

    # ── MODEL 6: Gaussian Mixture Model (GMM — Standard EM) ───────────────
    print("\n" + "=" * 62)
    print("  [6/7] Gaussian Mixture Model (GMM) — GridSearchCV")
    print("  EM algoritmus: puha klaszterezés, kovariancia-modellezés")
    print("=" * 62)

    n_gmm = min(MAX_GMM_SAMPLES, len(X_sc))
    if len(X_sc) > n_gmm:
        np.random.seed(42)
        idx_gmm      = np.random.choice(len(X_sc), n_gmm, replace=False)
        X_gmm, y_gmm = X_sc[idx_gmm], y_bal[idx_gmm]
        del idx_gmm
        gmm_note = f"subsample {n_gmm}/{len(X_sc)}"
        print(f"GMM subsample: {n_gmm:,} minta.")
    else:
        X_gmm, y_gmm = X_sc, y_bal
        gmm_note = "teljes halmaz"

    gs_gmm = GridSearchCV(
        GMMClassifier(random_state=42),
        param_grid={
            # 3-OSZTÁLYOS SÉMA: minimum 3 komponens szükséges, hogy
            # Uniform, Border és Spliced mindegyike kaphasson saját komponenst
            "n_components":    [3, 4, 6],
            "covariance_type": ["full", "tied", "diag"],
            "n_init":          [5, 10],
        },
        cv=3, n_jobs=N_JOBS_CV, verbose=1, scoring="f1_macro")
    gs_gmm.fit(X_gmm, y_gmm)
    best_gmm = gs_gmm.best_estimator_
    print(f"Legjobb GMM paraméterek: {gs_gmm.best_params_}")
    print(f"CV F1 (3-fold, {gmm_note}): {gs_gmm.best_score_:.4f}")
    print(f"Border komponens index:   {best_gmm.border_component_}")
    print(f"Border-arányok / komponens: "
          f"{[f'{r:.3f}' for r in best_gmm.border_rates_]}")
    joblib.dump(best_gmm, "itier_gmm_model.pkl")
    print("Mentve: itier_gmm_model.pkl")
    results["GMM"] = float(gs_gmm.best_score_)
    del gs_gmm, X_gmm, y_gmm, best_gmm  # MEM #9
    gc.collect()

    # ── MODEL 7: Bayesian Gaussian Mixture (Variational EM) ───────────────
    print("\n" + "=" * 62)
    print("  [7/7] Bayesian Gaussian Mixture (Variational EM)")
    print("  Dirichlet-prior -> automatikus komponensszám-meghatározás")
    print("=" * 62)

    n_em = min(MAX_BGMM_SAMPLES, len(X_sc))
    if len(X_sc) > n_em:
        np.random.seed(42)
        idx_em     = np.random.choice(len(X_sc), n_em, replace=False)
        X_em, y_em = X_sc[idx_em], y_bal[idx_em]
        del idx_em
        em_note = f"subsample {n_em}/{len(X_sc)}"
        print(f"BayesianEM subsample: {n_em:,} minta.")
    else:
        X_em, y_em = X_sc, y_bal
        em_note = "teljes halmaz"

    rs_em = RandomizedSearchCV(
        BayesianEMClassifier(random_state=42),
        param_distributions={
            "n_components":               [4, 8, 12],
            "covariance_type":            ["full", "diag"],
            "weight_concentration_prior": [None, 0.001, 0.01, 0.1, 1.0],
            "n_init":                     [1, 3],
        },
        n_iter=10, cv=3, n_jobs=1, verbose=1, scoring="f1_macro", random_state=42)
    rs_em.fit(X_em, y_em)
    best_em = rs_em.best_estimator_
    print(f"Legjobb BayesianEM paraméterek: {rs_em.best_params_}")
    print(f"CV F1 (3-fold, {em_note}): {rs_em.best_score_:.4f}")
    print(f"Aktív komponensek (weight > 1e-3): {best_em.n_active_components_}")
    print(f"Border komponens index:            {best_em.border_component_}")
    joblib.dump(best_em, "itier_bayesian_em_model.pkl")
    print("Mentve: itier_bayesian_em_model.pkl")
    results["BayesianEM"] = float(rs_em.best_score_)
    del rs_em, X_em, y_em, best_em  # MEM #9
    gc.collect()

    # ── Összefoglaló és JSON mentés ───────────────────────────────────────
    print("\n" + "=" * 62)
    print("  ÖSSZEFOGLALÓ — tanítóhalmazon (3-fold CV Macro-F1)")
    print("=" * 62)
    for name, score in sorted(results.items(), key=lambda x: -x[1]):
        tag = " [kalibrált]" if name in ("SVM", "MLP") else ""
        tag += " [EM/GMM]"    if name in ("GMM", "BayesianEM") else ""
        print(f"  {name:<20}: {score:.4f}  {'#'*int(score*40)}{tag}")
    print(f"\n  Legjobb blokk-szintű modell: {max(results, key=results.get)}")

    with open("training_summary_seven.json", "w") as f:
        json.dump({
            "cv_f1_scores":     results,
            "best_model":       max(results, key=results.get),
            "n_train":          len(train_images),
            "n_balanced":       int(len(y_bal)),
            "n_classes":        int(len(np.unique(y_bal))),
            "max_train_blocks": MAX_TRAIN_BLOCKS,
            "n_jobs_cv":        N_JOBS_CV,
        }, f, indent=4)
    print("JSON export: training_summary_seven.json")

    # ── Post-training küszöbbecslés indítása az RFC modellen ──────────────
    print("\nPost-training küszöbbecslés indítása (RFC modell, mindkét módszer)...")
    estimate_both_thresholds(best_rf, scaler, data)