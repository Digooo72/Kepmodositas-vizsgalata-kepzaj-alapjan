"""
trainmodel_seven.py

Hét osztályozó betanítása és összehasonlítása:
  1. Random Forest Classifier (RFC)            — cikk eredeti modellje
  2. Support Vector Machine (SVM)              — cikk által említett alternatíva
  3. K-Means (2 klaszter)                      — cikk által említett alternatíva
  4. Multi-Layer Perceptron (MLP)              — neurális háló, sklearn
  5. Gradient Boosting / XGBoost               — boosting ensemble
  6. Gaussian Mixture Model (GMM)              — EM alapú puha klaszterezés [ÚJ]
  7. Bayesian Gaussian Mixture (BayesianEM)    — Variational EM, auto komponensszám [ÚJ]

─── MEMÓRIA-OPTIMALIZÁCIÓK (javított verzió) ────────────────────────────────────
  MEM #1  del X_raw, y_raw  →  a listák felszabadítása np.vstack után
  MEM #2  Pre-balance subsample  →  MAX_TRAIN_BLOCKS cap X_all-on
  MEM #3  np.asarray (nem np.array)  →  nem másol ha már float32
  MEM #4  del X_all, y_all  →  vstack eredménye felszabadítva balance után
  MEM #5  X_sc float32-re castolása  →  StandardScaler float64 kimenete felezve
  MEM #6  del X_bal  →  balance_classes kimenete felszabadítva scaler után
  MEM #7  N_JOBS_CV = 1  →  nincs worker-fork memória-sokszorozás
  MEM #8  gc.collect()  →  minden modell után
  MEM #9  Subsample tömbök törlése  →  idx_svm, idx_mlp stb.
────────────────────────────────────────────────────────────────────────────────
"""

import gc
import json
import os
import cv2
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
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
from itier import extract_features_from_image, load_binary_mask
from clustering_models import GMMClassifier, BayesianEMClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Megjegyzés: xgboost nincs telepítve → sklearn GradientBoosting fallback.")
    print("Telepítés: pip install xgboost\n")


# ---------------------------------------------------------------------------
# K-Means wrapper — ÖNÁLLÓ, ITT DEFINIÁLVA (nem importálva trainmodel_three-ból,
# checklist #1 kérés: minden trainmodel_X.py önmagában tartalmazza a saját
# KMeansClassifier-ét). A GMM/BayesianEM import (clustering_models.py) marad,
# mert az egy önálló, kifejezetten erre a két klaszterezőre épített modul —
# nem egy "korábbi számozott" trainmodel fájl duplikátuma.
#
# 3-OSZTÁLYOS SÉMA: N klaszter → többségi szavazattal Uniform/Border/Spliced
# osztályhoz rendelve (ugyanaz az elv, mint trainmodel_three.py/five.py-ban
# és a clustering_models.py GMM/BayesianEM-jénél).
# ---------------------------------------------------------------------------

class KMeansClassifier(BaseEstimator, ClassifierMixin):
    """K-means alapú, N-osztályos osztályozó wrapper (lásd trainmodel_three.py)."""

    def __init__(self, n_clusters=6, n_init=10, max_iter=300, random_state=42):
        self.n_clusters      = n_clusters
        self.n_init          = n_init
        self.max_iter        = max_iter
        self.random_state    = random_state
        self.kmeans_         = None
        self.border_cluster_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32)
        self.classes_ = np.unique(y)
        n_classes     = len(self.classes_)

        self.kmeans_ = KMeans(n_clusters=self.n_clusters, n_init=self.n_init,
                              max_iter=self.max_iter, random_state=self.random_state)
        hard_labels = self.kmeans_.fit_predict(X)

        class_rates = np.zeros((self.n_clusters, n_classes), dtype=np.float64)
        for k in range(self.n_clusters):
            mask = hard_labels == k
            if mask.sum() > 0:
                for ci, c in enumerate(self.classes_):
                    class_rates[k, ci] = np.mean(y[mask] == c)

        cluster_to_class_idx = np.argmax(class_rates, axis=1)
        for ci in range(n_classes):
            if not np.any(cluster_to_class_idx == ci):
                best_k = int(np.argmax(class_rates[:, ci]))
                cluster_to_class_idx[best_k] = ci

        self.cluster_to_class_idx_ = cluster_to_class_idx
        self.class_rates_          = class_rates

        dists_to_own = np.min(self.kmeans_.transform(X), axis=1)
        self._scale_ = max(float(np.mean(dists_to_own)), 1e-6)

        if 1 in self.classes_:
            border_idx = int(np.where(self.classes_ == 1)[0][0])
            self.border_rates_   = class_rates[:, border_idx]
            self.border_cluster_ = int(np.argmax(self.border_rates_))

        return self

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X):
        """Softmax "subtract-max" trükk a 0/0 NaN ellen (numerikusan stabil)."""
        X = np.asarray(X, dtype=np.float64)
        dists  = self.kmeans_.transform(X)
        logits = -dists / self._scale_
        logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(logits)
        weights /= weights.sum(axis=1, keepdims=True)

        n_classes = len(self.classes_)
        proba = np.zeros((X.shape[0], n_classes), dtype=np.float64)
        for k in range(self.n_clusters):
            ci = self.cluster_to_class_idx_[k]
            proba[:, ci] += weights[:, k]
        row_sums = proba.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return proba / row_sums


# ============================================================
# MEMÓRIA-KONFIGURÁCIÓ — itt állítsd be a géped kapacitásának megfelelően
# ============================================================
MAX_TRAIN_BLOCKS = 150_000   # Balanced mintaszám felső határa (MEM #2)
                             #   150_000 → ~34 MB (float32, 60 feature)
                             #   Csökkentsd 80_000-re ha < 4 GB szabad RAM
                             #   Növeld 300_000-re ha > 16 GB szabad RAM

N_JOBS_CV = 1                # GridSearchCV worker-folyamatok száma (MEM #7)
                             #   1 = szekvenciális (legkisebb memória)
                             #   2 = enyhe párhuzamosság (>6 GB szabad RAM)
                             #  -1 = összes mag (csak >16 GB RAM esetén)

MAX_SVM_SAMPLES  = 30_000   # SVM subsample (SVM nem skálázódik jól nagy N-re)
MAX_MLP_SAMPLES  = 50_000   # MLP subsample
MAX_GMM_SAMPLES  = 40_000   # GMM subsample
MAX_BGMM_SAMPLES = 20_000   # BayesianEM subsample (lassabb modell)
# ============================================================


# ---------------------------------------------------------------------------
# Segédfüggvények — osztályegyensúly
# ---------------------------------------------------------------------------

def balance_classes(X, y):
    """
    Undersampling alapú kiegyensúlyozás (Section 3.4).

    3-OSZTÁLYOS SÉMA (checklist #1): tetszőleges osztályszámra
    általánosítva (korábban csak Border(1)/Uniform(0) bináris esetre volt
    hardcode-olva). Minden osztályt a legkisebb osztály méretére csökkent
    (undersampling), így Uniform/Border/Spliced egyenlő arányban szerepel
    a tanítóhalmazban — ez visszafelé kompatibilis a 2-osztályos esettel is.

    MEM #3: np.asarray helyett np.array — nem másol ha X már float32 tömb.
    """
    X_arr = np.asarray(X, dtype=np.float32)  # MEM #3: másolat nélkül ha float32
    y_arr = np.asarray(y, dtype=np.int32)

    classes = np.unique(y_arr)
    class_indices = {c: np.where(y_arr == c)[0] for c in classes}
    mn = min(len(idx) for idx in class_indices.values())
    if mn == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int32)

    np.random.seed(42)
    idx = np.concatenate([
        np.random.choice(class_indices[c], mn, replace=False) for c in classes
    ])
    np.random.shuffle(idx)
    return X_arr[idx].copy(), y_arr[idx].copy()  # .copy() → független tömb


# ---------------------------------------------------------------------------
# Post-training küszöbbecslés — mindkét módszer
# ---------------------------------------------------------------------------

def _build_heat_map(img_path, clf, scaler, block_size=32, step=8):
    """Hőtérkép-építő belső segédfüggvény."""
    feats, _, positions, dims = extract_features_from_image(
        img_path, block_size=block_size, step=step)
    if feats is None or len(feats) == 0 or dims is None:
        return None, None, None
    h, w = dims
    feats_sc = np.nan_to_num(
        scaler.transform(feats), nan=0.0, posinf=0.0, neginf=0.0)
    # 3-OSZTÁLYOS SÉMA: P(nem Uniform) = P(Border)+P(Spliced) = 1-P(Uniform),
    # 2 osztálynál visszafelé kompatibilis a régi [:, 1] oszloppal.
    probs = 1.0 - clf.predict_proba(feats_sc)[:, 0]
    hs = np.zeros((h, w), np.float32)
    hc = np.zeros((h, w), np.float32)
    for p, (y, x) in zip(probs, positions):
        hs[y:y+block_size, x:x+block_size] += p
        hc[y:y+block_size, x:x+block_size] += 1
    hc[hc == 0] = 1
    return hs / hc, h, w


def _both_scores(img_path, clf, scaler, mask_path=None, block_size=32, step=8):
    """
    Kiszámítja mindkét score-t (max_p és ks_stat) egyetlen hőtérképből.
    mask_path: spliced képeknél GT maszk; None → authentic, szintetikus GT.
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
                mb = cv2.resize(mb, (w, h), cv2.INTER_NEAREST)
            mask_binary = mb

    if mask_binary is not None:
        S_mask = mask_binary > 127
    else:
        cy, cx     = h // 2, w // 2
        sq_h, sq_w = max(1, h // 4), max(1, w // 4)
        S_mask     = np.zeros((h, w), dtype=bool)
        S_mask[cy - sq_h:cy + sq_h, cx - sq_w:cx + sq_w] = True

    S_vals = heat_map[S_mask].flatten()
    H_vals = heat_map[~S_mask].flatten()
    ks = float(ks_2samp(S_vals, H_vals)[0]) if (len(S_vals) >= 2 and len(H_vals) >= 2) else 0.0
    return mp, ks


def _roc_optimal(y_true, y_scores):
    """ROC L2-optimális küszöb."""
    yt = np.array(y_true); ys = np.array(y_scores, dtype=float)
    valid = np.isfinite(ys)
    if not np.any(valid) or len(np.unique(yt[valid])) < 2:
        return 0.5, 0.0, 0.0, 0.5, np.array([0, 1]), np.array([0, 1])
    auc = roc_auc_score(yt[valid], ys[valid])
    fpr, tpr, thrs = roc_curve(yt[valid], ys[valid])
    best = np.argmin(np.sqrt(fpr**2 + (tpr - 1)**2))
    return float(thrs[best]), float(fpr[best]), float(tpr[best]), float(auc), fpr, tpr


def _save_threshold(t_hat, method, auc, fpr_opt, tpr_opt, n_spl, n_aut):
    txt_path  = f"optimal_threshold_{method}.txt"
    meta_path = f"optimal_threshold_meta_{method}.json"
    with open(txt_path, "w") as f:
        f.write(str(t_hat))
    with open(meta_path, "w") as f:
        json.dump({"threshold": t_hat, "method": method, "auc": auc,
                   "fpr_at_optimum": fpr_opt, "tpr_at_optimum": tpr_opt,
                   "spliced_count": n_spl, "authentic_count": n_aut}, f, indent=4)
    print(f"    {txt_path:<40s}  t_hat={t_hat:.4f}  AUC={auc:.4f}")
    print(f"    {meta_path}")


def estimate_both_thresholds(clf, scaler, data, block_size=32, step=8):
    """Post-training küszöbbecslés mindkét módszerrel a test split képein."""
    print(f"\n{'='*62}")
    print("  POST-TRAINING KÜSZÖBBECSLÉS — mindkét módszer")
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
        y_true.append(1); scores_roc.append(mp); scores_ks.append(ks)
        spliced_count += 1
        print(f"    Spliced:   {base:42s}  max_p={mp:.4f}  ks={ks:.4f}")

    if spliced_count == 0:
        print("  FIGYELEM: Nem találtam spliced képet → küszöbbecslés kihagyva.")
        return

    print(f"\n  --- Eredeti képek (max {spliced_count} db) ---")
    auth_files = []
    for adir in ["4cam_auth", "Au"]:
        if os.path.exists(adir):
            for fn in sorted(os.listdir(adir)):
                if fn.lower().endswith((".jpg", ".png", ".tif")):
                    auth_files.append(os.path.join(adir, fn))

    if not auth_files:
        print("  FIGYELEM: Nem találtam authentic képeket → küszöbbecslés kihagyva.")
        return

    np.random.seed(42); np.random.shuffle(auth_files)
    auth_files = auth_files[:spliced_count]
    for img_path in auth_files:
        base   = os.path.basename(img_path)
        mp, ks = _both_scores(img_path, clf, scaler, mask_path=None,
                               block_size=block_size, step=step)
        y_true.append(0); scores_roc.append(mp); scores_ks.append(ks)
        print(f"    Authentic: {base:42s}  max_p={mp:.4f}  ks={ks:.4f}")

    authentic_count = len(auth_files)
    y_true_arr = np.array(y_true)
    sc_roc     = np.array(scores_roc)
    sc_ks      = np.array(scores_ks)

    if len(np.unique(y_true_arr)) < 2:
        print("  HIBA: Nem elegendő osztály → küszöbbecslés kihagyva."); return

    print(f"\n  --- ROC számítás ---")
    t_roc, fpr_roc_opt, tpr_roc_opt, auc_roc, fpr_roc, tpr_roc = _roc_optimal(y_true_arr, sc_roc)
    t_ks,  fpr_ks_opt,  tpr_ks_opt,  auc_ks,  fpr_ks,  tpr_ks  = _roc_optimal(y_true_arr, sc_ks)
    print(f"    ROC (max_p) :  t_hat={t_roc:.4f}  FPR={fpr_roc_opt:.4f}  TPR={tpr_roc_opt:.4f}  AUC={auc_roc:.4f}")
    print(f"    KS (ks_stat):  t_hat={t_ks:.4f}  FPR={fpr_ks_opt:.4f}  TPR={tpr_ks_opt:.4f}  AUC={auc_ks:.4f}")

    print(f"\n  --- Mentés ---")
    _save_threshold(t_roc, "roc", auc_roc, fpr_roc_opt, tpr_roc_opt, spliced_count, authentic_count)
    _save_threshold(t_ks,  "ks",  auc_ks,  fpr_ks_opt,  tpr_ks_opt,  spliced_count, authentic_count)

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
    plt.close()
    print("\n  Ábra mentve: roc_curve_both.png")
    print(f"\n  {'='*58}\n  ÖSSZEFOGLALÓ\n  {'='*58}")
    print(f"  {'ROC / max_p':<14} {t_roc:>8.4f} {auc_roc:>8.4f} {fpr_roc_opt:>8.4f} {tpr_roc_opt:>8.4f}  optimal_threshold_roc.txt")
    print(f"  {'KS-stat':<14} {t_ks:>8.4f} {auc_ks:>8.4f} {fpr_ks_opt:>8.4f} {tpr_ks_opt:>8.4f}  optimal_threshold_ks.txt")


# ===========================================================================
# Fő betanítási folyamat
# ===========================================================================

if __name__ == "__main__":
    print("=" * 62)
    print("  Hét osztályozó betanítása — Memória-optimalizált verzió")
    print("  RFC / SVM / K-Means / MLP / Boosting / GMM / BayesianEM")
    print("=" * 62)
    print(f"\n  Konfiguráció:")
    print(f"    MAX_TRAIN_BLOCKS = {MAX_TRAIN_BLOCKS:,}  (balanced minták felső határa)")
    print(f"    N_JOBS_CV        = {N_JOBS_CV}             (GridSearchCV parallelizmus)")
    print(f"    Becsült X_sc méret: {MAX_TRAIN_BLOCKS * 60 * 4 / 1024**2:.1f} MB (float32)\n")

    MAX_IMAGES = None

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

    # ── Jellemzőkinyerés ──────────────────────────────────────────────────
    X_raw, y_raw = [], []
    for i, (img_p, mask_p) in enumerate(zip(train_images, train_masks)):
        print(f"  Feldolgozás ({i+1}/{len(train_images)}): {img_p}")
        feats, labels, _, _ = extract_features_from_image(img_p, mask_p)
        if feats is not None and labels is not None and len(feats) > 0:
            X_raw.append(feats.astype(np.float32))   # float32 rögtön
            y_raw.append(labels.astype(np.int32))

    if not X_raw:
        print("Hiba: Nem sikerült jellemzőket kinyerni!"); exit()

    X_all = np.vstack(X_raw)
    y_all = np.concatenate(y_raw)
    del X_raw, y_raw          # MEM #1: listák felszabadítása
    gc.collect()

    print(f"\nNyers → Uniform(0): {int(np.sum(y_all==0))}, Border(1): {int(np.sum(y_all==1))}, "
          f"Spliced(2): {int(np.sum(y_all==2))}")
    print(f"Nyers blokkok: {len(X_all):,}  →  {X_all.nbytes / 1024**2:.1f} MB (float32)")

    # MEM #2: Pre-balance subsample — X_all korlátoz MAX_TRAIN_BLOCKS-ra
    if len(X_all) > MAX_TRAIN_BLOCKS:
        np.random.seed(42)
        idx_pre = np.random.choice(len(X_all), MAX_TRAIN_BLOCKS, replace=False)
        X_all   = X_all[idx_pre]
        y_all   = y_all[idx_pre]
        del idx_pre
        gc.collect()
        print(f"Pre-balance subsample → {len(X_all):,} minta (MAX_TRAIN_BLOCKS={MAX_TRAIN_BLOCKS:,})")

    X_bal, y_bal = balance_classes(X_all, y_all)
    del X_all, y_all          # MEM #4: X_all már nem kell
    gc.collect()

    if len(X_bal) == 0:
        print("Hiba: Nincs elég adat!"); exit()
    print(f"Kiegyensúlyozott → Uniform: {int(np.sum(y_bal==0))}, Border: {int(np.sum(y_bal==1))}, "
          f"Spliced: {int(np.sum(y_bal==2))}")
    print(f"Balanced méret: {X_bal.nbytes / 1024**2:.1f} MB (float32)")

    # MEM #5: StandardScaler → float32-re visszakonvertálás (scaler float64-et ad)
    print("\nGlobális StandardScaler illesztése...")
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_bal).astype(np.float32)  # MEM #5
    joblib.dump(scaler, "itier_scaler.pkl")
    print(f"Scaler elmentve: itier_scaler.pkl")
    print(f"X_sc méret: {X_sc.nbytes / 1024**2:.1f} MB (float32)")
    del X_bal                 # MEM #6: X_bal már nem kell
    gc.collect()

    results = {}

    # ── MODEL 1: RFC ──────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  [1/7] Random Forest Classifier (GridSearchCV)")
    print(f"  n_jobs={N_JOBS_CV}  (MEM #7: nincs fork-másolat)")
    print("=" * 62)
    # Csökkentett grid: 2×2×2×2 = 16 kombináció (volt: 108)
    # Bővítsd ha N_JOBS_CV > 1 és van elegendő RAM
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
    results["RFC"] = gs_rf.best_score_
    del gs_rf                 # MEM #8
    gc.collect()

    # ── MODEL 2: SVM ──────────────────────────────────────────────────────
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
        X_svm, y_svm = X_sc, y_bal; svm_note = "teljes halmaz"

    gs_svm = GridSearchCV(
        SVC(probability=False, random_state=42),
        param_grid={"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"],
                    "gamma": ["scale", "auto"]},
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
    results["SVM"] = gs_svm.best_score_
    del gs_svm, X_svm, y_svm, calibrated_svm  # MEM #9
    gc.collect()

    # ── MODEL 3: K-Means ──────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  [3/7] K-Means Classifier (k=2)")
    print("=" * 62)
    km_clf = KMeansClassifier(n_clusters=6, n_init=10, max_iter=300, random_state=42)
    km_clf.fit(X_sc, y_bal)
    cv_km = cross_val_score(km_clf, X_sc, y_bal, cv=3, scoring="f1_macro", n_jobs=N_JOBS_CV).mean()
    print(f"Border klaszter index: {km_clf.border_cluster_}")
    print(f"CV F1 (3-fold): {cv_km:.4f}")
    joblib.dump(km_clf, "itier_kmeans_model.pkl")
    print("Mentve: itier_kmeans_model.pkl")
    results["K-Means"] = cv_km
    del km_clf
    gc.collect()

    # ── MODEL 4: MLP ──────────────────────────────────────────────────────
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
        X_mlp, y_mlp = X_sc, y_bal; mlp_note = "teljes halmaz"

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
        n_iter=30, cv=3, n_jobs=N_JOBS_CV, verbose=1, scoring="f1_macro", random_state=42)
    rs_mlp.fit(X_mlp, y_mlp)
    print(f"Legjobb MLP paraméterek: {rs_mlp.best_params_}")
    print(f"CV F1 (3-fold, {mlp_note}): {rs_mlp.best_score_:.4f}")
    calibrated_mlp = CalibratedClassifierCV(rs_mlp.best_estimator_,
                                            method="sigmoid", cv=5)
    calibrated_mlp.fit(X_mlp, y_mlp)
    joblib.dump(calibrated_mlp, "itier_mlp_model.pkl")
    print("Mentve: itier_mlp_model.pkl")
    results["MLP"] = rs_mlp.best_score_
    del rs_mlp, X_mlp, y_mlp, calibrated_mlp  # MEM #9
    gc.collect()

    # ── MODEL 5: Gradient Boosting / XGBoost ─────────────────────────────
    print("\n" + "=" * 62)
    print(f"  [5/7] {'XGBoost' if XGBOOST_AVAILABLE else 'GradientBoosting'} "
          f"(RandomizedSearchCV)")
    print("=" * 62)

    if XGBOOST_AVAILABLE:
        base_boost = XGBClassifier(use_label_encoder=False, eval_metric="logloss",
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
                                  n_iter=20, cv=3, n_jobs=N_JOBS_CV, verbose=1,
                                  scoring="f1_macro", random_state=42)
    rs_boost.fit(X_sc, y_bal)
    best_boost = rs_boost.best_estimator_
    print(f"Legjobb paraméterek: {rs_boost.best_params_}")
    print(f"CV F1 (3-fold): {rs_boost.best_score_:.4f}")
    joblib.dump(best_boost, model_filename)
    print(f"Mentve: {model_filename}")
    results[model_key] = rs_boost.best_score_
    with open("itier_boost_info.txt", "w") as f:
        f.write(f"{model_key}\n{model_filename}")
    del rs_boost, base_boost  # MEM #8
    gc.collect()

    # ── MODEL 6: GMM ─────────────────────────────────────────────────────
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
        X_gmm, y_gmm = X_sc, y_bal; gmm_note = "teljes halmaz"

    gs_gmm = GridSearchCV(
        GMMClassifier(random_state=42),
        param_grid={
            # 3-OSZTÁLYOS SÉMA: min. 3 komponens kell, hogy Uniform/Border/
            # Spliced mindegyike kaphasson saját komponenst (korábban [2,3,4]
            # volt jó a bináris esetre).
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
    results["GMM"] = gs_gmm.best_score_
    del gs_gmm, X_gmm, y_gmm  # MEM #9
    gc.collect()

    # ── MODEL 7: Bayesian EM ──────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  [7/7] Bayesian Gaussian Mixture (Variational EM)")
    print("  Dirichlet-prior → automatikus komponensszám-meghatározás")
    print("=" * 62)

    n_em = min(MAX_BGMM_SAMPLES, len(X_sc))
    if len(X_sc) > n_em:
        np.random.seed(42)
        idx_em    = np.random.choice(len(X_sc), n_em, replace=False)
        X_em, y_em = X_sc[idx_em], y_bal[idx_em]
        del idx_em
        em_note = f"subsample {n_em}/{len(X_sc)}"
        print(f"BayesianEM subsample: {n_em:,} minta.")
    else:
        X_em, y_em = X_sc, y_bal; em_note = "teljes halmaz"

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
    results["BayesianEM"] = rs_em.best_score_
    del rs_em, X_em, y_em  # MEM #9
    gc.collect()

    # ── Összefoglaló ──────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  ÖSSZEFOGLALÓ — tanítóhalmazon (3-fold CV F1)")
    print("=" * 62)
    for name, score in sorted(results.items(), key=lambda x: -x[1]):
        tag = " [kalibrált]" if name in ("SVM", "MLP") else ""
        tag += " [EM/GMM]"    if name in ("GMM", "BayesianEM") else ""
        print(f"  {name:<20}: {score:.4f}  {'#'*int(score*40)}{tag}")
    print(f"\n  Legjobb modell: {max(results, key=results.get)}")

    with open("training_summary_seven.json", "w") as f:
        json.dump({
            "cv_f1_scores":   results,
            "best_model":     max(results, key=results.get),
            "n_train":        len(train_images),
            # 3-OSZTÁLYOS SÉMA: len(y_bal) helyes tetszőleges osztályszámnál,
            # a korábbi "* 2" csak a bináris (Uniform/Border) esetre volt jó.
            "n_balanced":     int(len(y_bal)),
            "n_classes":      int(len(np.unique(y_bal))),
            "max_train_blocks": MAX_TRAIN_BLOCKS,
            "n_jobs_cv":      N_JOBS_CV,
        }, f, indent=4)
    print("JSON export: training_summary_seven.json")

    print("\nPost-training küszöbbecslés indítása (RFC modell, mindkét módszer)...")
    estimate_both_thresholds(best_rf, scaler, data)