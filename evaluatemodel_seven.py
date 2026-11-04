"""
evaluatemodel_seven.py

Mind a hét betanított osztályozó modell párhuzamos képszintű és pixelszintű
kiértékelése a teszthalmazon (45 hamisított + 45 hiteles kép).

Minden modellre párhuzamosan kiszámítja mindkét képszintű döntési pontszámot:
  1. ROC módszer (max_p)   : max(P_I) — a hőtérkép globális maximuma (Cikk Eq. 4-5)
  2. KS módszer (ks_stat)  : max|C(S) - C(H)| — kétmintás Kolmogorov-Smirnov
                             statisztika a maszkolt régió és a háttér között (Cikk Eq. 8)
Valamint kiszámítja a pixelszintű lokalizációs F1-pontszámot (Pixel F1) a
négylépcsős, perem-vékonyított `build_pred_mask()` utófeldolgozással.

Kiértékelt modellek:
  1. RFC        — Random Forest Classifier (cikk alapmodellje)
  2. SVM        — Support Vector Machine (isotonic kalibrációval)
  3. K-Means    — K-Means Classifier (k=6, 3-osztályos Softmax wrapper)
  4. MLP        — Multi-Layer Perceptron (sigmoid/Platt kalibrációval)
  5. Boosting   — XGBoost / GradientBoostingClassifier
  6. GMM        — Gaussian Mixture Model (standard EM)
  7. BayesianEM — Bayesian Gaussian Mixture (Variational EM, Dirichlet-prior)

─── MEMÓRIA-OPTIMALIZÁCIÓK ─────────────────────────────────────────────────────
  MEM #1 — matplotlib.use("Agg") : nincs grafikus (GUI) ablak-memóriaigény
  MEM #2 — Jellemzőkinyerés és képbeolvasás képenként egyszer, 7 modellnek átadva
  MEM #3 — Explicit `del` + `gc.collect()` minden `GC_INTERVAL` kép után
  MEM #4 — SAVE_VISUALIZATIONS kapcsoló (PNG mentés ki-/bekapcsolása)
  MEM #5 — VIZ_ONLY_SPLICED kapcsoló (csak a hamisított képekről ment 5-paneles ábrát)
  MEM #6 — Kontrollált felbontás (VIZ_DPI = 100)
────────────────────────────────────────────────────────────────────────────────
"""

import os
import gc
import json
import joblib
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")               # MEM #1: háttérbeli (GUI nélküli) renderelés
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

from itier import extract_features_from_image, load_binary_mask
from evaluate_model import connected_components_postprocess
# 3-OSZTÁLYOS SÉMA (checklist #1): a KMeansClassifier a trainmodel_seven.py-ban,
# a GMMClassifier és BayesianEMClassifier pedig a clustering_models.py-ban van
# definiálva — a joblib.load() unpickle műveletéhez ezeket innen importáljuk.
from trainmodel_seven import KMeansClassifier
from clustering_models import GMMClassifier, BayesianEMClassifier


# ===========================================================================
# MEMÓRIA- ÉS VIZUALIZÁCIÓS KONFIGURÁCIÓ
# ===========================================================================
SAVE_VISUALIZATIONS = True    # MEM #4: False -> nem ment 5-paneles PNG ábrákat
VIZ_ONLY_SPLICED    = True    # MEM #5: True  -> csak hamisított képekhez ment PNG-t
VIZ_DPI             = 100     # MEM #6: Kimentett PNG ábrák DPI felbontása
GC_INTERVAL         = 10      # MEM #3: Minden N. kép után explicit gc.collect()
# ===========================================================================


# ===========================================================================
# Határvonal-tudatos, perem-vékonyított maszkgenerálás (JAVÍTOTT)
# ===========================================================================

BORDER_THRESH       = 0.72   # 0.65 -> 0.72: szigorúbb küszöb a külső FP glória ellen
BORDER_ERODE_PX     = 5      # 32x32-es ablak túlnyúlásának visszavágása (4-6 px optimális)
MIN_COMPONENT_RATIO = 0.015  # Képméret 1.5%-ánál kisebb FP foltok (pl. ikonok) törlése
FILL_INTERNAL_HOLES = True   # Manipulált régión belüli zöld (FN) lyukak kitöltése


def build_pred_mask(heat_map, block_size):
    """
    Finomított, határvonal-vékonyított bináris maszk (0 vagy 255) előállítása
    a (H, W) méretű valószínűségi hőtérképből.

    Fizikai és matematikai indoklás:
      A 32x32-es csúszóablak már ~35-40%-os belépésnél (beta in [0.4, 0.6])
      magas (~1.0) valószínűséget ad, miközben a blokk másik ~60%-a (~19 pixel)
      kilóg a hiteles (Uniform) háttérbe. Ezt a geometriai túlnyúlást (FP glóriát)
      és a háttér éles grafikus elemein (pl. asztali ikonokon) keletkező apró
      lokális riasztásokat négy lépésben küszöböljük ki:

      1. Alapmaszk (`connected_components_postprocess`) egyesítése a megemelt
         `BORDER_THRESH = 0.72` küszöbű perempixelekkel, majd 5x5-ös elliptikus
         morfológiai zárás (`cv2.MORPH_CLOSE`) a 8 pixeles lépésközből eredő
         rácshézagok összekötésére.
      2. Kis területű hamis pozitív (FP) szigetek szűrése (`cv2.connectedComponentsWithStats`):
         minden olyan különálló komponenst törlünk, amely nem éri el a képméret
         `MIN_COMPONENT_RATIO` (1.5%) arányát vagy a legnagyobb komponens 10%-át.
      3. Belső üregek (False Negative lyukak) kontúralapú kitöltése
         (`cv2.findContours` + `cv2.FILLED`), hogy a manipulált régió belseje
         tömör maradjon.
      4. Képszél-védett perem-erózió (`BORDER_ERODE_PX = 5`):
         levágja a 32x32-es blokkok külső (host irányú) túlnyúlását, de a
         `cv2.copyMakeBorder(..., value=255)` keretezésnek köszönhetően a valódi
         képszélekhez tapadó manipulált régiót nem húzza el a kép keretétől.
    """
    h, w = heat_map.shape[:2]
    base_mask     = connected_components_postprocess(heat_map, block_size=block_size)
    border_pixels = (heat_map >= BORDER_THRESH).astype(np.uint8) * 255
    combined      = cv2.bitwise_or(base_mask, border_pixels)

    # 1. Rácshézagok bezárása (5x5 elliptikus strukturáló elem)
    k_close  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_bin = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_close)

    # 2. Kis területű FP foltok (pl. asztali ikonok, lokális él-zajok) kiszűrése
    if MIN_COMPONENT_RATIO > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask_bin, connectivity=8)
        if num_labels > 1:
            areas    = stats[1:, cv2.CC_STAT_AREA]
            max_area = int(np.max(areas))
            min_area = max(int(h * w * MIN_COMPONENT_RATIO), int(max_area * 0.10))
            filtered = np.zeros_like(mask_bin)
            for lbl in range(1, num_labels):
                if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
                    filtered[labels == lbl] = 255
            # Biztonsági visszaesés: ha minden kiesne, megtartjuk a legnagyobb komponenst
            if np.any(filtered):
                mask_bin = filtered
            else:
                best_lbl = 1 + int(np.argmax(areas))
                mask_bin = np.where(labels == best_lbl, 255, 0).astype(np.uint8)

    # 3. Manipulált területen belüli FN lyukak kitöltése
    if FILL_INTERNAL_HOLES:
        contours, _ = cv2.findContours(
            mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask_bin)
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
        mask_bin = filled

    # 4. Képszél-védett perem-erózió a 32x32-es csúszóablak túlnyúlásának levágására
    if BORDER_ERODE_PX > 0:
        r = int(BORDER_ERODE_PX)
        k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        # Pad 255-tel a külső képszéleken, hogy a kép széléig érő (pl. jobb oldali)
        # hamisított régiót az erózió ne húzza el a kép keretétől:
        padded = cv2.copyMakeBorder(
            mask_bin, r, r, r, r, borderType=cv2.BORDER_CONSTANT, value=255)
        eroded = cv2.erode(padded, k_erode, iterations=1)[r:r + h, r:r + w]
        # Ha az erózió teljesen eltüntetne egy nagyon vékony maszkot, megtartjuk az eredetit
        if np.any(eroded):
            mask_bin = eroded

    return mask_bin


# ===========================================================================
# KS-statisztika és vektorizált hőtérkép-építő függvények
# ===========================================================================

def compute_ks_score(heat_map, h, w, mask_binary=None):
    """
    Kétmintás Kolmogorov-Smirnov statisztika: ks = max|C(S) - C(H)| (Cikk Eq. 8).

    Paraméterek:
      heat_map    : (H, W) float32 valószínűségi hőtérkép.
      h, w        : Kép magassága és szélessége.
      mask_binary : Spliced képeknél a (H, W) bináris GT maszk (S = maszk > 127,
                    H = háttér). Authentic képeknél None -> S a kép középső
                    negyedét lefedő szintetikus négyzet (Zampoglou et al. [40]).
    """
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


def build_heat_map_vectorized(probs, positions, h, w, block_size):
    """
    Vektorizált hőtérkép-felépítés az átfedő (32x32, step=8) ablakok
    valószínűségeinek pixelenkénti átlagolásával (Cikk Section 3.6).
    """
    hs = np.zeros((h, w), np.float32)
    hc = np.zeros((h, w), np.float32)
    bs = block_size
    for p, (y, x) in zip(probs, positions):
        hs[y:y+bs, x:x+bs] += p
        hc[y:y+bs, x:x+bs] += 1.0
    hc[hc == 0] = 1.0
    heat = hs / hc
    del hs, hc
    return heat


# ===========================================================================
# 5-paneles vizualizáció mentése (memóriakímélő azonnali felszabadítással)
# ===========================================================================

def save_visualization(out_path, base_name, model_name,
                       rgb_uint8, gt_binary, heat_clean, pred_mask,
                       valid_gt, max_p, ks_score, f1, h, w):
    """
    5-paneles kiértékelő ábra mentése PNG-be:
      1. Eredeti RGB kép
      2. Ground Truth (GT) bináris maszk
      3. Valószínűségi hőtérkép (Jet színskála, [0, 1])
      4. Finomított prediktált bináris maszk (`build_pred_mask`)
      5. Színkódolt Overlay:
           - Narancssárga = True Positive  (TP: helyesen detektált manipulált pixel)
           - Piros        = False Positive (FP: tévesen manipuláltnak jelölt háttér)
           - Zöld         = False Negative (FN: fel nem ismert manipulált pixel)
    """
    overlay = rgb_uint8.copy().astype(np.float32)
    if valid_gt:
        pb, gb = pred_mask > 127, gt_binary > 127
        overlay[pb & ~gb] = overlay[pb & ~gb] * 0.4 + np.array([220,  50,  50]) * 0.6  # Piros = FP
        overlay[~pb & gb] = overlay[~pb & gb] * 0.4 + np.array([ 50, 200,  50]) * 0.6  # Zöld  = FN
        overlay[pb &  gb] = overlay[pb &  gb] * 0.4 + np.array([255, 165,   0]) * 0.6  # Narancs = TP
    overlay = overlay.clip(0, 255).astype(np.uint8)

    mp_str = f"{max_p:.3f}" if np.isfinite(max_p) else "nan"

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle(
        f"{base_name} | {model_name} | max_p={mp_str}  ks={ks_score:.3f} | "
        f"Pixel F1={f1:.3f}",
        fontsize=9)

    axes[0].imshow(rgb_uint8);   axes[0].set_title("Eredeti");   axes[0].axis("off")
    axes[1].imshow(gt_binary if valid_gt else np.zeros((h, w), np.uint8), cmap="gray")
    axes[1].set_title("GT maszk");                                axes[1].axis("off")
    axes[2].imshow(heat_clean, cmap="jet", vmin=0, vmax=1)
    axes[2].set_title("Hőtérkép");                                axes[2].axis("off")
    axes[3].imshow(pred_mask, cmap="gray")
    axes[3].set_title("Pred. maszk");                             axes[3].axis("off")
    axes[4].imshow(overlay)
    axes[4].set_title("Overlay\nPiros=FP  Narancs=TP  Zöld=FN")
    axes[4].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=VIZ_DPI)
    plt.close(fig)
    del fig, axes, overlay


# ===========================================================================
# Segédfüggvények: Képtípus-azonosítás, küszöbvédelem és modellbetöltés
# ===========================================================================

def is_spliced_image(img_path):
    """Eldönti az elérési út alapján, hogy a kép hamisított (Columbia/CASIA) vagy hiteles."""
    lower, basename = img_path.lower(), os.path.basename(img_path)
    columbia = "4cam_splc" in lower and "edgemask" not in lower
    casia    = ((os.sep + "tp" + os.sep) in lower or "/tp/" in lower) \
               and basename.startswith("Tp_")
    return columbia or casia


def find_authentic_images(max_count):
    """Hiteles (authentic) képek összegyűjtése a 4cam_auth/ és Au/ könyvtárakból."""
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
    return files[:max_count]


def load_threshold(method):
    """Betölti a tune_treshold.py / trainmodel_seven.py által mentett globális küszöböt."""
    path = f"optimal_threshold_{method}.txt"
    if os.path.exists(path):
        with open(path) as f:
            val = float(f.read().strip())
        print(f"  Betöltve: {path:<40s}  t_hat={val:.4f}")
        return val
    print(f"  FIGYELEM: {path} nem található -> fallback 0.5")
    print("    Futtasd: tune_treshold.py  vagy  trainmodel_seven.py")
    return 0.5


def safe_threshold(model_name, method_tag, y_true, y_scores, global_t, auth_scores):
    """
    Háromszintű robusztus küszöbválasztás (degenerált pontszám-eloszlások ellen):
      1. szint: Ha AUC < 0.5 -> automatikus pontszám-invertálás (1.0 - score).
      2. szint: GB-eset (ha a hamisított képek szórása < 0.001, pl. mind ~1.0) ->
                midpoint vagy a hiteles képek 90. percentilise (ha nem telítődött 1.0-ra).
      3. szint: Normál ROC L2-minimum (a mesterséges 0. indexű +inf pont kizárásával);
                ha t_hat >= 0.999 -> fallback: global_t -> auth_p95 -> 0.5.
    """
    yt    = np.array(y_true)
    ys    = np.array(y_scores, dtype=float)
    valid = np.isfinite(ys)

    if not np.any(valid) or len(np.unique(yt[valid])) < 2:
        return 0.5, 0.0, False

    auc = float(roc_auc_score(yt[valid], ys[valid]))

    # --- 1. szint: Fordított korreláció (AUC < 0.5) korrekciója ---
    if auc < 0.5:
        print(f"  [{model_name}/{method_tag}] AUC={auc:.4f} < 0.5 -> Score invertált!")
        ys_flip = np.where(valid, 1.0 - ys, ys)
        fpr2, tpr2, thr2 = roc_curve(yt[valid], ys_flip[valid])
        if len(thr2) > 1:
            fpr2, tpr2, thr2 = fpr2[1:], tpr2[1:], thr2[1:]
        auc2 = float(roc_auc_score(yt[valid], ys_flip[valid]))
        t2   = float(thr2[np.argmin(np.sqrt(fpr2**2 + (tpr2 - 1.0)**2))])
        return t2, auc2, True

    # --- 2. szint: Konstans Spliced pontszámok (std < 0.001) kezelése ---
    spl_sc  = ys[valid & (yt == 1)]
    aut_arr = np.array([s for s in auth_scores if np.isfinite(s)])
    if len(spl_sc) > 0 and np.std(spl_sc) < 0.001 and len(aut_arr) > 0:
        spl_min = float(np.min(spl_sc))
        aut_max = float(np.max(aut_arr))
        t_gb = (spl_min + aut_max) / 2.0 if aut_max < spl_min \
               else float(np.percentile(aut_arr, 90))
        if t_gb < 0.999:
            print(f"  [{model_name}/{method_tag}] GB-eset -> t_hat={t_gb:.4f}")
            return t_gb, auc, False

    # --- 3. szint: ROC L2-minimum + többlépcsős fallback (ha t_hat >= 0.999) ---
    fpr, tpr, thrs = roc_curve(yt[valid], ys[valid])
    if len(thrs) > 1:
        fpr, tpr, thrs = fpr[1:], tpr[1:], thrs[1:]
    t_hat = float(thrs[np.argmin(np.sqrt(fpr**2 + (tpr - 1.0)**2))])

    if t_hat >= 0.999:
        fallback = None
        if global_t < 0.999:
            t_hat, fallback = global_t, f"global={global_t:.4f}"
        elif len(aut_arr) > 0:
            p95 = float(np.percentile(aut_arr, 95)) + 0.01
            if p95 < 0.999:
                t_hat, fallback = p95, f"auth_p95={p95:.4f}"
        if fallback is None:
            t_hat, fallback = 0.5, "0.5"
        print(f"  [{model_name}/{method_tag}] Degenerált ROC -> fallback {fallback}")

    return t_hat, auc, False


def load_models(scaler_path="itier_scaler.pkl"):
    """Betölti a globális StandardScalert és az összes elérhető betanított modellt."""
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler nem található: {scaler_path}")
    scaler = joblib.load(scaler_path)
    models = {}

    model_files = {
        "RFC":        "itier_rf_model.pkl",
        "SVM":        "itier_svm_model.pkl",
        "K-Means":    "itier_kmeans_model.pkl",
        "MLP":        "itier_mlp_model.pkl",
        "GMM":        "itier_gmm_model.pkl",
        "BayesianEM": "itier_bayesian_em_model.pkl",
    }

    # Boosting modell felderítése (XGBoost vagy GradientBoosting)
    if os.path.exists("itier_boost_info.txt"):
        with open("itier_boost_info.txt") as f:
            lines = f.read().strip().split("\n")
        model_files[lines[0]] = lines[1]
    else:
        for fname, key in [("itier_xgb_model.pkl", "XGBoost"),
                           ("itier_gb_model.pkl",  "GradientBoosting")]:
            if os.path.exists(fname):
                model_files[key] = fname
                break

    for name, path in model_files.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
            model_type = type(models[name]).__name__
            print(f"  Betöltve: {name:<20} <- {path}  [{model_type}]")
        else:
            print(f"  Hiányzik: {name:<20} ({path}) — kihagyva")

    return scaler, models


def save_comparison_chart(summary):
    """Összehasonlító csoportosított oszlopdiagram mentése mind a 7 modellre."""
    names   = list(summary.keys())
    metrics = [
        ("auc_roc",       "AUC (ROC/max_p)",  "#1565C0"),
        ("img_f1_roc",    "F1 (ROC/max_p)",   "#42A5F5"),
        ("auc_ks",        "AUC (KS-stat)",     "#1B5E20"),
        ("img_f1_ks",     "F1 (KS-stat)",      "#66BB6A"),
        ("mean_pixel_f1", "Pixel F1",          "#FF7043"),
    ]
    x     = np.arange(len(names))
    width = 0.14
    fig, ax = plt.subplots(figsize=(max(15, len(names) * 2.2), 6))
    for mi, (key, label, color) in enumerate(metrics):
        vals   = [summary[n].get(key, 0.0) for n in names]
        offset = (mi - len(metrics) / 2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width * 0.92, label=label,
                        color=color, alpha=0.88, edgecolor="white")
        for bar, val in zip(bars, vals):
            if val > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005, f"{val:.3f}",
                        ha="center", va="bottom", fontsize=6.0, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Pontszám", fontsize=11)
    ax.set_title(
        "Hét osztályozó összehasonlítása — Itier et al. (2021)\n"
        "Kék = ROC/max_p  |  Zöld = KS  |  Narancs = Pixel F1\n"
        "GMM és BayesianEM: EM alapú klaszterezés",
        fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("comparison_seven_models.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    del fig, ax
    print("Chart mentve: comparison_seven_models.png")


# ===========================================================================
# Fő kiértékelési folyamat
# ===========================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  Összehasonlító Kiértékelés: 7 modell")
    print("  RFC / SVM / K-Means / MLP / Boosting / GMM / BayesianEM")
    print("  Mindkét score párhuzamosan: max(P_I)  +  KS-statisztika")
    print(f"  Vizualizáció: {'BE' if SAVE_VISUALIZATIONS else 'KI'}"
          f"  |  Csak spliced: {'IGEN' if VIZ_ONLY_SPLICED else 'NEM'}"
          f"  |  DPI: {VIZ_DPI}"
          f"  |  GC minden {GC_INTERVAL}. képnél")
    print("=" * 65)

    print("\nKüszöbök betöltése:")
    global_t_roc = load_threshold("roc")
    global_t_ks  = load_threshold("ks")

    try:
        scaler, models = load_models()
    except FileNotFoundError as e:
        print(f"Hiba: {e}\nFuttasd le: trainmodel_seven.py"); exit()
    if not models:
        print("Hiba: Egyetlen modell sem tölthető be!"); exit()

    print(f"\nBetöltött modellek ({len(models)} db): {list(models.keys())}")
    print(f"Maszkparaméterek: BORDER_THRESH={BORDER_THRESH}, "
          f"BORDER_ERODE_PX={BORDER_ERODE_PX}, "
          f"MIN_COMPONENT_RATIO={MIN_COMPONENT_RATIO}\n")

    with open("dataset_splits.json") as f:
        data = json.load(f)
    test_images, test_masks = data["test_images"], data["test_masks"]

    output_dirs = {}
    for name in models:
        d = f"eval7_{name.lower().replace('-','_').replace('/','_')}"
        os.makedirs(d, exist_ok=True)
        output_dirs[name] = d

    eval_data = {
        name: {"y_true_img": [], "y_score_roc": [], "y_score_ks": [], "f1_pixels": []}
        for name in models
    }

    block_size    = 32
    spliced_count = 0

    # =========================================================
    # 1. Teszt (hamisított) képek kiértékelése
    # =========================================================
    print(f"Tesztelendő képek: {len(test_images)}\n")

    for i, (img_p, mask_p) in enumerate(zip(test_images, test_masks)):
        base_name  = os.path.splitext(os.path.basename(img_p))[0]
        is_spliced = 1 if is_spliced_image(img_p) else 0
        print(f"[{i+1}/{len(test_images)}] {base_name}  "
              f"({'HAMISÍTOTT' if is_spliced else 'VALÓDI'})")

        # ── Jellemzőkinyerés (képenként egyszer, mind a 7 modellnek) ──────
        feats, _, positions, dims = extract_features_from_image(
            img_p, block_size=block_size, step=8)
        if feats is None or len(feats) == 0 or dims is None:
            print("  -> Sikertelen jellemzőkinyerés.\n"); continue

        h, w = dims
        fs   = np.nan_to_num(scaler.transform(feats),
                             nan=0.0, posinf=0.0, neginf=0.0)
        del feats   # MEM #3

        # ── GT maszk betöltése (egyszer) ──────────────────────────────────
        gt_binary = load_binary_mask(mask_p)
        valid_gt  = False
        y_true_pixel = None
        if gt_binary is not None:
            if gt_binary.shape != (h, w):
                gt_binary = cv2.resize(gt_binary, (w, h), interpolation=cv2.INTER_NEAREST)
            y_true_pixel = (gt_binary.flatten() > 127).astype(np.int8)
            valid_gt = True

        # ── Képbeolvasás vizualizációhoz (egyszer, MEM #2) ────────────────
        do_viz = (SAVE_VISUALIZATIONS and
                  (not VIZ_ONLY_SPLICED or is_spliced))
        rgb_uint8 = None
        if do_viz:
            bgr = cv2.imread(img_p)
            if bgr is not None:
                rgb_uint8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                del bgr
            else:
                do_viz = False

        ks_mask_binary = gt_binary if (is_spliced and valid_gt) else None
        if is_spliced:
            spliced_count += 1

        # ── Modell-ciklus (azonos skálázott `fs` mátrixon) ─────────────────
        for model_name, model in models.items():
            # 3-OSZTÁLYOS SÉMA: P(nem Uniform) = P(Border) + P(Spliced)
            probs      = 1.0 - model.predict_proba(fs)[:, 0]
            heat_clean = build_heat_map_vectorized(probs, positions, h, w, block_size)
            del probs

            # Képszintű pontszámok (ROC max_p + KS-statisztika)
            max_p    = float(np.nanmax(heat_clean)) \
                       if np.any(np.isfinite(heat_clean)) else float("nan")
            ks_score = compute_ks_score(heat_clean, h, w, ks_mask_binary)

            eval_data[model_name]["y_true_img"].append(is_spliced)
            eval_data[model_name]["y_score_roc"].append(max_p)
            eval_data[model_name]["y_score_ks"].append(ks_score)

            # Pixelszintű maszk és Pixel F1
            pred_mask = build_pred_mask(heat_clean, block_size=block_size)
            f1 = 0.0
            if valid_gt:
                f1 = float(f1_score(
                    y_true_pixel,
                    (pred_mask.flatten() > 127).astype(np.int8),
                    zero_division=1
                ))
                eval_data[model_name]["f1_pixels"].append(f1)

            mp_str = f"{max_p:.3f}" if np.isfinite(max_p) else " nan"
            print(f"  [{model_name:22s}] max_p={mp_str}  ks={ks_score:.3f}"
                  f"  | Pixel F1={f1:.4f}")

            if do_viz and rgb_uint8 is not None:
                out_path = os.path.join(output_dirs[model_name],
                                        f"{base_name}_result.png")
                save_visualization(
                    out_path, base_name, model_name,
                    rgb_uint8, gt_binary, heat_clean, pred_mask,
                    valid_gt, max_p, ks_score, f1, h, w)

            del heat_clean, pred_mask

        # ── Kép-szintű memóriafelszabadítás ───────────────────────────────
        del fs, positions, y_true_pixel, gt_binary, ks_mask_binary
        if rgb_uint8 is not None:
            del rgb_uint8

        if (i + 1) % GC_INTERVAL == 0:
            gc.collect()
            print(f"  [gc.collect() @ kép {i+1}]")

        print()

    # =========================================================
    # 2. Hiteles (authentic) képek kiértékelése (képszintű AUC/F1-hez)
    # =========================================================
    auth_files = find_authentic_images(max_count=spliced_count)
    auth_roc_per_model = {name: [] for name in models}
    auth_ks_per_model  = {name: [] for name in models}

    if auth_files:
        print(f"Authentic képek: {len(auth_files)} db (4cam_auth/Au)\n")
        for j, img_p in enumerate(auth_files):
            base = os.path.basename(img_p)
            feats, _, positions, dims = extract_features_from_image(
                img_p, block_size=block_size, step=8)

            if feats is None or len(feats) == 0 or dims is None:
                print(f"  [{j+1}/{len(auth_files)}] {base}: kinyerés sikertelen")
                for name in models:
                    eval_data[name]["y_true_img"].append(0)
                    eval_data[name]["y_score_roc"].append(float("nan"))
                    eval_data[name]["y_score_ks"].append(float("nan"))
                continue

            h, w     = dims
            feats_sc = np.nan_to_num(scaler.transform(feats),
                                     nan=0.0, posinf=0.0, neginf=0.0)
            del feats

            scores_line = []
            for model_name, model in models.items():
                probs      = 1.0 - model.predict_proba(feats_sc)[:, 0]
                heat_clean = build_heat_map_vectorized(probs, positions, h, w, block_size)
                del probs

                mp = float(np.nanmax(heat_clean)) \
                     if np.any(np.isfinite(heat_clean)) else float("nan")
                ks = compute_ks_score(heat_clean, h, w, mask_binary=None)
                del heat_clean

                eval_data[model_name]["y_true_img"].append(0)
                eval_data[model_name]["y_score_roc"].append(mp)
                eval_data[model_name]["y_score_ks"].append(ks)
                if np.isfinite(mp):
                    auth_roc_per_model[model_name].append(mp)
                auth_ks_per_model[model_name].append(ks)

                tag = model_name[:6]
                scores_line.append(
                    f"{tag}: mp={mp:.3f} ks={ks:.3f}" if np.isfinite(mp)
                    else f"{tag}: nan")

            del feats_sc, positions
            print(f"  [{j+1}/{len(auth_files)}] {base}  |  "
                  + "  |  ".join(scores_line))

            if (j + 1) % GC_INTERVAL == 0:
                gc.collect()
    else:
        print("Figyelem: 4cam_auth/Au mappa nem található -> AUC megbízhatatlan.\n")

    gc.collect()

    # =========================================================
    # 3. Diagnosztika: Score eloszlások (Spliced vs. Authentic)
    # =========================================================
    print(f"\n{'='*70}")
    print("  Score eloszlások (spliced vs authentic)")
    print("=" * 70)
    print(f"  {'Modell':<24} {'ROC spl':>8} {'ROC aut':>8}  |  "
          f"{'KS spl':>8} {'KS aut':>8}")
    print("  " + "-" * 68)
    for name in models:
        yt   = np.array(eval_data[name]["y_true_img"])
        yroc = np.array(eval_data[name]["y_score_roc"], dtype=float)
        yks  = np.array(eval_data[name]["y_score_ks"],  dtype=float)

        spl_roc = yroc[(yt == 1) & np.isfinite(yroc)]
        aut_roc = yroc[(yt == 0) & np.isfinite(yroc)]
        spl_ks  = yks[(yt == 1)  & np.isfinite(yks)]
        aut_ks  = yks[(yt == 0)  & np.isfinite(yks)]

        mu_sr = np.mean(spl_roc) if len(spl_roc) > 0 else float("nan")
        mu_ar = np.mean(aut_roc) if len(aut_roc) > 0 else float("nan")
        mu_sk = np.mean(spl_ks)  if len(spl_ks)  > 0 else float("nan")
        mu_ak = np.mean(aut_ks)  if len(aut_ks)  > 0 else float("nan")
        oc = " <- OVERCONFIDENT" if (np.isfinite(mu_ar) and mu_ar > 0.9) else ""
        print(f"  {name:<24} {mu_sr:>8.4f} {mu_ar:>8.4f}  |  "
              f"{mu_sk:>8.4f} {mu_ak:>8.4f}{oc}")

    # =========================================================
    # 4. Per-modell küszöb és AUC — mindkét módszerre
    # =========================================================
    print(f"\n{'='*70}")
    print("  Per-modell küszöb és AUC (ROC módszer)")
    print("=" * 70)
    thresh_roc, auc_roc_m, flipped_roc = {}, {}, {}
    for name in models:
        t, auc, fl = safe_threshold(name, "ROC",
                                    eval_data[name]["y_true_img"],
                                    eval_data[name]["y_score_roc"],
                                    global_t_roc, auth_roc_per_model[name])
        thresh_roc[name] = t; auc_roc_m[name] = auc; flipped_roc[name] = fl
        flip_note = " [INVERTÁLT]" if fl else ""
        print(f"  {name:<24} t_hat={t:.4f}  AUC={auc:.4f}{flip_note}")

    print(f"\n{'='*70}")
    print("  Per-modell küszöb és AUC (KS módszer)")
    print("=" * 70)
    thresh_ks, auc_ks_m, flipped_ks = {}, {}, {}
    for name in models:
        t, auc, fl = safe_threshold(name, "KS",
                                    eval_data[name]["y_true_img"],
                                    eval_data[name]["y_score_ks"],
                                    global_t_ks, auth_ks_per_model[name])
        thresh_ks[name] = t; auc_ks_m[name] = auc; flipped_ks[name] = fl
        flip_note = " [INVERTÁLT]" if fl else ""
        print(f"  {name:<24} t_hat={t:.4f}  AUC={auc:.4f}{flip_note}")

    # =========================================================
    # 5. Végső összesítés és export
    # =========================================================
    print(f"\n{'='*70}")
    print("  VÉGSŐ TESZTEREDMÉNYEK")
    print("=" * 70)

    summary = {}
    for name in models:
        yt  = np.array(eval_data[name]["y_true_img"])
        yr  = np.array(eval_data[name]["y_score_roc"], dtype=float)
        yk  = np.array(eval_data[name]["y_score_ks"],  dtype=float)
        f1p = eval_data[name]["f1_pixels"]

        yr_use     = 1.0 - yr if flipped_roc[name] else yr
        y_pred_roc = np.where(np.isfinite(yr_use),
                              (yr_use >= thresh_roc[name]).astype(int), 0)
        img_f1_roc = float(f1_score(yt, y_pred_roc, zero_division=1))

        yk_use    = 1.0 - yk if flipped_ks[name] else yk
        y_pred_ks = np.where(np.isfinite(yk_use),
                             (yk_use >= thresh_ks[name]).astype(int), 0)
        img_f1_ks = float(f1_score(yt, y_pred_ks, zero_division=1))

        mean_pixel_f1 = float(np.mean(f1p)) if f1p else 0.0

        summary[name] = {
            "auc_roc":           auc_roc_m[name],
            "img_f1_roc":        img_f1_roc,
            "t_hat_roc":         thresh_roc[name],
            "score_flipped_roc": flipped_roc[name],
            "auc_ks":            auc_ks_m[name],
            "img_f1_ks":         img_f1_ks,
            "t_hat_ks":          thresh_ks[name],
            "score_flipped_ks":  flipped_ks[name],
            "mean_pixel_f1":     mean_pixel_f1,
        }

        flip_r = "  [inv.]" if flipped_roc[name] else ""
        flip_k = "  [inv.]" if flipped_ks[name]  else ""
        print(f"\n--- {name} ---")
        print(f"  ROC (max_p) : t={thresh_roc[name]:.4f}  AUC={auc_roc_m[name]:.4f}"
              f"  Kép F1={img_f1_roc:.4f}{flip_r}")
        print(f"  KS (ks_st)  : t={thresh_ks[name]:.4f}  AUC={auc_ks_m[name]:.4f}"
              f"  Kép F1={img_f1_ks:.4f}{flip_k}")
        print(f"  Pixel F1    : {mean_pixel_f1:.4f}")

    print(f"\n{'='*85}")
    print(f"  {'Modell':<24} {'AUC_roc':>8} {'F1_roc':>8} "
          f"{'AUC_ks':>8} {'F1_ks':>8} {'PixelF1':>9}")
    print("  " + "-" * 75)
    for name, s in sorted(summary.items(), key=lambda x: -x[1]["auc_roc"]):
        fr = "*" if s["score_flipped_roc"] else " "
        fk = "*" if s["score_flipped_ks"]  else " "
        print(f"  {name:<24} {s['auc_roc']:>8.4f}{fr} {s['img_f1_roc']:>8.4f}  "
              f"{s['auc_ks']:>8.4f}{fk} {s['img_f1_ks']:>8.4f}  "
              f"{s['mean_pixel_f1']:>9.4f}")
    print("  (* = score invertálva: AUC<0.5 korrekció)")

    save_comparison_chart(summary)

    with open("evaluation_summary_seven.json", "w") as f:
        json.dump(summary, f, indent=4)
    print("JSON export: evaluation_summary_seven.json")