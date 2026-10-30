"""
tune_threshold.py

Képszintű optimális küszöb (t_hat) meghatározása — MINDIG MINDKÉT módszerrel:

  ── ROC módszer (cikk Section 3.5, Table 2) ─────────────────────────────────
  Képszintű score = max(P_I): a hőtérkép maximuma (Eq. 4-5).
  A ROC görbe (0,1) pontjától vett L2-távolságot minimalizáljuk → t_hat_roc.
  Mentés: optimal_threshold_roc.txt  +  optimal_threshold_meta_roc.json

  ── KS módszer (cikk Eq. 8, Table 3, Zampoglou et al. [40]) ─────────────────
  Képszintű score = KS-statisztika: max|C(S) - C(H)|
      S = spliced régió pixelei (GT maszk alapján)
      H = háttér / host image pixelei
      Authentic képeknél: S = középső negyed (szintetikus GT)
  A KS score-okon szintén ROC L2-minimalizálással kapjuk t_hat_ks-t.
  Mentés: optimal_threshold_ks.txt   +  optimal_threshold_meta_ks.json

Mindkét módszer eredménye egymás mellett jelenik meg a 2×2 ábrán és a
konzolra írt összefoglalóban. Az evaluatemodel_five.py mindkét fájlt
beolvassa és párhuzamosan mutatja az eredményeket.
"""

import os
import json
import joblib
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.metrics import roc_curve, roc_auc_score
from itier import extract_features_from_image, load_binary_mask


# ---------------------------------------------------------------------------
# Hőtérkép-építő segédfüggvény
# ---------------------------------------------------------------------------

def build_heat_map(img_path, clf, scaler, block_size=32, step=8):
    """
    Visszaadja a (H, W) float32 hőtérképet és a kép méretét.
    Visszatérés: (heat_map, h, w) vagy (None, None, None) hiba esetén.
    """
    feats, _, positions, dims = extract_features_from_image(
        img_path, block_size=block_size, step=step)
    if feats is None or len(feats) == 0 or dims is None:
        return None, None, None

    h, w = dims
    feats_sc = np.nan_to_num(
        scaler.transform(feats), nan=0.0, posinf=0.0, neginf=0.0)
    # 3-OSZTÁLYOS SÉMA (checklist #1): P(nem Uniform) = P(Border)+P(Spliced)
    # = 1 - P(Uniform). 2 osztálynál visszafelé kompatibilis a régi [:, 1]-gyel.
    probs = 1.0 - clf.predict_proba(feats_sc)[:, 0]

    hs = np.zeros((h, w), np.float32)
    hc = np.zeros((h, w), np.float32)
    for p, (y, x) in zip(probs, positions):
        hs[y:y+block_size, x:x+block_size] += p
        hc[y:y+block_size, x:x+block_size] += 1
    hc[hc == 0] = 1
    return hs / hc, h, w


# ---------------------------------------------------------------------------
# Score-számítók
# ---------------------------------------------------------------------------

def score_roc(heat_map):
    """
    ROC módszer: max(P_I) — hőtérkép maximuma (cikk Section 3.5, Eq. 4-5).
    """
    if heat_map is None:
        return 0.0
    val = float(np.nanmax(heat_map))
    return val if np.isfinite(val) else 0.0


def score_ks(heat_map, h, w, mask_binary=None):
    """
    KS módszer: max|C(S) - C(H)| (cikk Eq. 8).

    Spliced kép (mask_binary megadva): S = maszkolt régió, H = háttér.
    Authentic kép (mask_binary=None): S = középső negyed (szintetikus GT).

    Visszatérés: ks_stat ∈ [0, 1]
    """
    if heat_map is None:
        return 0.0
    if mask_binary is not None:
        S_mask = mask_binary > 127
    else:
        cy, cx     = h // 2, w // 2
        sq_h, sq_w = max(1, h // 4), max(1, w // 4)
        S_mask     = np.zeros((h, w), dtype=bool)
        S_mask[cy - sq_h:cy + sq_h, cx - sq_w:cx + sq_w] = True

    S_vals = heat_map[S_mask].flatten()
    H_vals = heat_map[~S_mask].flatten()
    if len(S_vals) < 2 or len(H_vals) < 2:
        return 0.0
    ks_stat, _ = ks_2samp(S_vals, H_vals)
    return float(ks_stat)


def both_scores_from_path(img_path, clf, scaler,
                          mask_path=None, block_size=32, step=8):
    """
    Kiszámítja MINDKÉT score-t egyetlen hőtérkép-felépítéssel.

    mask_path: spliced képeknél GT maszk (KS S vs H szétválasztáshoz).
               None → authentic kép → szintetikus GT.

    Visszatérés: (max_p, ks_stat)
    """
    heat_map, h, w = build_heat_map(img_path, clf, scaler, block_size, step)
    if heat_map is None:
        return 0.0, 0.0

    mp = score_roc(heat_map)

    mask_binary = None
    if mask_path and os.path.exists(mask_path):
        mb = load_binary_mask(mask_path)
        if mb is not None:
            if mb.shape != (h, w):
                mb = cv2.resize(mb, (w, h), cv2.INTER_NEAREST)
            mask_binary = mb

    ks = score_ks(heat_map, h, w, mask_binary)
    return mp, ks


# ---------------------------------------------------------------------------
# ROC L2-optimális küszöb
# ---------------------------------------------------------------------------

def roc_optimal_threshold(y_true, y_scores):
    """
    ROC görbe alapján az (0,1) ponttól vett L2-távolságot minimalizáló küszöb.

    Visszatérés: (t_hat, fpr_opt, tpr_opt, auc, fpr_arr, tpr_arr)
    """
    y_t   = np.array(y_true)
    y_s   = np.array(y_scores, dtype=float)
    valid = np.isfinite(y_s)

    if not np.any(valid) or len(np.unique(y_t[valid])) < 2:
        return 0.5, 0.0, 0.0, 0.5, np.array([0, 1]), np.array([0, 1])

    auc              = roc_auc_score(y_t[valid], y_s[valid])
    fpr, tpr, thrs   = roc_curve(y_t[valid], y_s[valid])
    dist             = np.sqrt(fpr**2 + (tpr - 1)**2)
    best             = np.argmin(dist)
    t_hat            = float(thrs[best])

    return t_hat, float(fpr[best]), float(tpr[best]), float(auc), fpr, tpr


# ---------------------------------------------------------------------------
# Mentési segédfüggvény
# ---------------------------------------------------------------------------

def save_threshold(t_hat, method, auc, fpr_opt, tpr_opt,
                   spliced_count, authentic_count):
    """
    Menti az optimal_threshold_{method}.txt és
    az optimal_threshold_meta_{method}.json fájlokat.
    """
    txt_path  = f"optimal_threshold_{method}.txt"
    meta_path = f"optimal_threshold_meta_{method}.json"

    with open(txt_path, "w") as f:
        f.write(str(t_hat))

    meta = {
        "threshold":       t_hat,
        "method":          method,
        "auc":             auc,
        "fpr_at_optimum":  fpr_opt,
        "tpr_at_optimum":  tpr_opt,
        "spliced_count":   spliced_count,
        "authentic_count": authentic_count,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)

    print(f"    Mentve: {txt_path:<35s}  t_hat={t_hat:.4f}  AUC={auc:.4f}")
    print(f"    Mentve: {meta_path}")


# ---------------------------------------------------------------------------
# Fő küszöbmeghatározó függvény
# ---------------------------------------------------------------------------

def find_optimal_threshold(model_path="itier_rf_model.pkl",
                           scaler_path="itier_scaler.pkl",
                           limit=None):
    """
    Mindkét módszerrel kiszámítja az optimális küszöböt a teszthalmaz képein.

    Kimenet:
      optimal_threshold_roc.txt        — max_p alapú t_hat
      optimal_threshold_ks.txt         — KS alapú t_hat
      optimal_threshold_meta_roc.json
      optimal_threshold_meta_ks.json
      roc_curve_both.png               — 2×2 ábra
    """
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("Hiba: Modell vagy scaler nem található!")
        return

    clf    = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Modell és scaler betöltve.\n")

    try:
        with open("dataset_splits.json") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Hiba: dataset_splits.json nem található!")
        return

    test_images = data["test_images"][:limit]
    test_masks  = data["test_masks"][:limit]

    # ── Spliced képek ──────────────────────────────────────────────────────
    print("=" * 62)
    print("  Spliced képek (test split — modell NEM látta ezeket)")
    print("=" * 62)

    y_true     = []
    scores_roc = []
    scores_ks  = []
    spliced_count = 0

    for img_path, mask_path in zip(test_images, test_masks):
        lower = img_path.lower()
        base  = os.path.basename(img_path)
        if not ("splc" in lower or "\\tp\\" in lower or "/tp/" in lower
                or base.startswith("Tp_")):
            continue

        mp, ks = both_scores_from_path(img_path, clf, scaler, mask_path=mask_path)
        y_true.append(1)
        scores_roc.append(mp)
        scores_ks.append(ks)
        spliced_count += 1
        print(f"  Spliced:   {base:42s}  max_p={mp:.4f}  ks={ks:.4f}")

    if spliced_count == 0:
        print("HIBA: Egyetlen spliced képet sem találtam a test halmazban!")
        return

    # ── Authentic képek ────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  Eredeti képek (max {spliced_count} db, kiegyensúlyozáshoz)")
    print("=" * 62)

    auth_files = []
    for adir in ["4cam_auth", "Au"]:
        if os.path.exists(adir):
            for fn in sorted(os.listdir(adir)):
                if fn.lower().endswith((".jpg", ".png", ".tif")):
                    auth_files.append(os.path.join(adir, fn))

    if not auth_files:
        print("HIBA: Nem találtam authentic képeket (4cam_auth / Au)!")
        return

    np.random.seed(42)
    np.random.shuffle(auth_files)
    auth_files = auth_files[:spliced_count]

    for img_path in auth_files:
        base   = os.path.basename(img_path)
        mp, ks = both_scores_from_path(img_path, clf, scaler, mask_path=None)
        y_true.append(0)
        scores_roc.append(mp)
        scores_ks.append(ks)
        print(f"  Authentic: {base:42s}  max_p={mp:.4f}  ks={ks:.4f}")

    authentic_count = len(auth_files)
    y_true_arr      = np.array(y_true)
    sc_roc          = np.array(scores_roc)
    sc_ks           = np.array(scores_ks)

    # ── Eloszlás debug ─────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("  Score eloszlások")
    print("=" * 62)
    for label, sc in [("max_p   (ROC módszer)", sc_roc),
                      ("ks_stat (KS módszer) ", sc_ks)]:
        spl = sc[y_true_arr == 1]
        aut = sc[y_true_arr == 0]
        print(f"  {label}:")
        print(f"    Spliced  : min={spl.min():.4f}  "
              f"mean={spl.mean():.4f}  max={spl.max():.4f}")
        print(f"    Authentic: min={aut.min():.4f}  "
              f"mean={aut.mean():.4f}  max={aut.max():.4f}")

    # ── ROC számítás — mindkét módszer ────────────────────────────────────
    print(f"\n{'='*62}")
    print("  ROC számítás")
    print("=" * 62)

    t_roc, fpr_roc_opt, tpr_roc_opt, auc_roc, fpr_roc, tpr_roc = \
        roc_optimal_threshold(y_true_arr, sc_roc)
    print(f"  ROC (max_p) :  t_hat={t_roc:.4f}  "
          f"FPR={fpr_roc_opt:.4f}  TPR={tpr_roc_opt:.4f}  AUC={auc_roc:.4f}")
    if t_roc > 0.95:
        print("    FIGYELEM: t_hat_roc > 0.95 → erős átfedés vagy kevés adat.")

    t_ks, fpr_ks_opt, tpr_ks_opt, auc_ks, fpr_ks, tpr_ks = \
        roc_optimal_threshold(y_true_arr, sc_ks)
    print(f"  KS (ks_stat):  t_hat={t_ks:.4f}  "
          f"FPR={fpr_ks_opt:.4f}  TPR={tpr_ks_opt:.4f}  AUC={auc_ks:.4f}")

    # ── Mentés ────────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("  Mentés")
    print("=" * 62)
    save_threshold(t_roc, "roc", auc_roc, fpr_roc_opt, tpr_roc_opt,
                   spliced_count, authentic_count)
    save_threshold(t_ks,  "ks",  auc_ks,  fpr_ks_opt,  tpr_ks_opt,
                   spliced_count, authentic_count)

    # ── 2×2 ábra ──────────────────────────────────────────────────────────
    spl_roc = sc_roc[y_true_arr == 1]; aut_roc = sc_roc[y_true_arr == 0]
    spl_ks  = sc_ks[y_true_arr == 1];  aut_ks  = sc_ks[y_true_arr == 0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Küszöbmeghatározás — ROC módszer (bal) vs KS módszer (jobb)",
        fontsize=13, fontweight="bold")

    roc_configs = [
        (fpr_roc, tpr_roc, auc_roc, t_roc, fpr_roc_opt, tpr_roc_opt,
         f"ROC görbe — max(P_I)\nAUC={auc_roc:.3f}",
         spl_roc, aut_roc, t_roc, "max_p értéke",
         f"max(P_I) eloszlás — t_hat={t_roc:.3f}"),
        (fpr_ks,  tpr_ks,  auc_ks,  t_ks,  fpr_ks_opt,  tpr_ks_opt,
         f"ROC görbe — KS-statisztika\nAUC={auc_ks:.3f}",
         spl_ks,  aut_ks,  t_ks,  "KS-statisztika értéke",
         f"KS-statisztika eloszlás — t_hat={t_ks:.3f}"),
    ]

    for col, (fpr_a, tpr_a, auc_a, t_a, fpr_opt_a, tpr_opt_a, roc_title,
              spl_s, aut_s, t_dist, xlabel, dist_title) in enumerate(roc_configs):
        # Sor 0: ROC görbe
        ax0 = axes[0][col]
        ax0.plot(fpr_a, tpr_a, color="darkorange", lw=2, label=f"AUC={auc_a:.3f}")
        ax0.plot([0, 1], [0, 1], "navy", lw=1, linestyle="--")
        ax0.scatter([fpr_opt_a], [tpr_opt_a], color="red", s=100, zorder=5,
                    label=f"t_hat={t_a:.3f}")
        ax0.set_xlabel("FPR"); ax0.set_ylabel("TPR")
        ax0.set_title(roc_title); ax0.legend(); ax0.grid(True, alpha=0.3)

        # Sor 1: Score eloszlás
        ax1 = axes[1][col]
        all_s = np.concatenate([spl_s, aut_s])
        bins  = np.linspace(0, max(float(all_s.max()), 0.01), 40)
        ax1.hist(spl_s, bins=bins, alpha=0.6, color="red",
                 label=f"Spliced (n={len(spl_s)})")
        ax1.hist(aut_s, bins=bins, alpha=0.6, color="blue",
                 label=f"Authentic (n={len(aut_s)})")
        ax1.axvline(t_dist, color="black", linestyle="--", lw=2,
                    label=f"t_hat={t_dist:.3f}")
        ax1.set_xlabel(xlabel); ax1.set_ylabel("Képek száma")
        ax1.set_title(dist_title); ax1.legend(); ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("roc_curve_both.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("\nÁbra mentve: roc_curve_both.png")

    # ── Összefoglaló ──────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("  ÖSSZEFOGLALÓ")
    print("=" * 62)
    print(f"  {'Módszer':<14} {'t_hat':>8} {'AUC':>8} "
          f"{'FPR':>8} {'TPR':>8}  Fájl")
    print(f"  {'-'*66}")
    print(f"  {'ROC / max_p':<14} {t_roc:>8.4f} {auc_roc:>8.4f} "
          f"{fpr_roc_opt:>8.4f} {tpr_roc_opt:>8.4f}  "
          f"optimal_threshold_roc.txt")
    print(f"  {'KS-stat':<14} {t_ks:>8.4f} {auc_ks:>8.4f} "
          f"{fpr_ks_opt:>8.4f} {tpr_ks_opt:>8.4f}  "
          f"optimal_threshold_ks.txt")


if __name__ == "__main__":
    find_optimal_threshold(limit=None)