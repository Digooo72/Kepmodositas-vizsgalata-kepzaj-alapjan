"""
evaluate_robustness.py

CHECKLIST #17 — Képzajalapú képmódosítás-vizsgálatok (Ellen-forenzikus robusztusság)
======================================================================================
Az Itier et al. (2021) cikk Section 4.1 utolsó kísérletének reprodukciója és
kiterjesztése a teljes 7-modelles keretrendszerre:

  "We also performed the classification task after JPEG compression of the
   test dataset (4 folds) in the 5-fold cross-validation. We used standard
   JPEG compression i.e. (4:2:2) sub-sampling. Images from the JPEG test
   dataset are then decompressed in the RGB color space and our proposed
   method is applied on each RGB channel. Using images with a quality
   factor (QF) of 75%, the method still performs an average F1
   classification score of 0.91. For lower JPEG quality, e.g. QF = 50%,
   average F1 classification score decreases to 0.77. [...] This also
   happens with other counter-attacks such as uniform noise addition
   strong enough to cover the intrinsic noise in the spliced image."

─── Vizsgált ellen-forenzikus támadások (Counter-Attacks) ───────────────────────
  1. JPEG újratömörítés csökkenő minőségi faktorral:
     QF = [100, 90, 75, 50, 30] — memóriában (cv2.imencode / cv2.imdecode),
     felesleges lemezre írás nélkül.

  2. Additív egyenletes (uniform) zaj növekvő amplitúdóval:
     amp = [0, 0.5, 1, 2, 3, 5, 10, 20] — két különböző fizikai üzemmódban:
       - "independent" : csatornánként független (H, W, 3) zaj, amely közvetlenül
                         szétrombolja az RGB csatornák közötti korrelációt.
       - "correlated"  : mindhárom csatornán azonos (H, W, 1) zaj, amely megtartja
                         a csatornaközi korrelációt, csak a zajenergiát növeli.
     Tudományos jelentőség: a két zajmód összevetése közvetlenül bizonyítja, hogy
     a modellek nem a zaj intenzitására, hanem az RGB csatornák közötti
     korreláció megbomlására érzékenyek.

─── "Anti-Oracle" rögzített küszöbprotokoll ─────────────────────────────────────
  A képszintű döntési küszöböt (t_roc és t_ks) KIZÁRÓLAG EGYSZER, a torzítatlan
  alapfutáson (QF=100, zaj=0) határozzuk meg és rögzítjük. Minden további
  torzítási szinten ezt a fix küszöböt alkalmazzuk, így valós támadási helyzetet
  szimulálunk (nem kalibráljuk újra a detektort a támadó paramétereire).
────────────────────────────────────────────────────────────────────────────────
"""

import os
import gc
import json
import joblib
import cv2
import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

from itier import extract_features_from_array, load_binary_mask
# 3-OSZTÁLYOS SÉMA (checklist #1): a KMeansClassifier a trainmodel_seven.py-ban,
# a GMMClassifier és BayesianEMClassifier pedig a clustering_models.py-ban van
# definiálva — a pickle-elt modellek betöltéséhez innen importáljuk őket.
from trainmodel_seven import KMeansClassifier
from clustering_models import GMMClassifier, BayesianEMClassifier

# ===========================================================================
# KONFIGURÁCIÓ
# ===========================================================================
BLOCK_SIZE = 32   # Csúszóablak mérete (32x32 pixel, Cikk Section 4.1)
STEP       = 8    # Csúszóablak lépésköze (8 pixel, Cikk Section 4.1)

# None = minden betöltött modellt tesztel; pl. ["RFC"] = gyors, egymodelles futás
MODELS_TO_TEST = None

# JPEG minőségi faktorok (QF=100 a torzítatlan kontroll)
JPEG_QUALITIES = [100, 90, 75, 50, 30]
RUN_JPEG       = True   # False = a JPEG-görbét kihagyja

# Egyenletes zaj [-amplitude, +amplitude] tartományból (0 = torzítatlan kontroll)
NOISE_AMPLITUDES = [0, 0.5, 1, 2, 3, 5, 10, 20]

# Zajmódok:
#   "independent" : csatornánként független zaj (lerombolja az RGB korrelációt)
#   "correlated"  : azonos zaj mindhárom RGB csatornán (megtartja a korrelációt)
NOISE_MODES = ["independent", "correlated"]

MAX_TEST_IMAGES = 60   # None = összes tesztkép; 60 = teljes 45 képes Columbia test split
MAX_AUTH_IMAGES = 20   # Hiteles képek száma a robusztussági görbéhez
RANDOM_SEED     = 42   # Reprodukálhatósági magérték
# ===========================================================================


# ---------------------------------------------------------------------------
# Ellen-forenzikus támadás-szimulátorok (memóriában, lemez-I/O nélkül)
# ---------------------------------------------------------------------------

def apply_jpeg_compression(img_rgb, quality):
    """
    JPEG újratömörítés memóriában (cv2.imencode -> cv2.imdecode).
    quality >= 100 esetén (kontroll) az eredeti képet adja vissza változatlanul.

    Módszertani megjegyzés (OpenCV 4:2:0 vs. Cikk 4:2:2):
      A cikk Section 4.1 fejezete 4:2:2 chroma subsamplinget említ, míg az OpenCV
      cv2.imencode alapértelmezésben 4:2:0 szubmintavételezést alkalmaz (vízszintesen
      és függőlegesen is felezi a Cb/Cr színcsatornák felbontását). Ez szigorúbb
      ellen-forenzikus teszt, mivel a 4:2:0 erősebben mossa össze a magas
      frekvenciás RGB színzaj-korrelációt.
    """
    if quality >= 100:
        return img_rgb.copy()
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        return img_rgb.copy()
    dec_bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec_bgr, cv2.COLOR_BGR2RGB)


def apply_uniform_noise(img_rgb, amplitude, rng, correlated=False):
    """
    Additív egyenletes (uniform) zaj [-amplitude, +amplitude] tartományból.
      - correlated=False ("independent"): (H, W, 3) alakú független zaj minden
        RGB csatornára -> közvetlenül rontja a csatornaközi zajkorrelációt.
      - correlated=True  ("correlated"):  (H, W, 1) alakú zaj, amely mindhárom
        csatornára azonos értékkel adódik hozzá -> a csatornák közötti korreláció
        megmarad, csupán a zaj energiája nő.

    NUMERIKUS HIBAJAVÍTÁS (np.rint):
      A közvetlen `.astype(np.uint8)` csonkolást (truncation towards zero) végez,
      ami miatt kis amplitúdóknál (pl. amp=0.5 és amp=1.0) minden pozitív törtzajból
      0, minden negatív törtzajból -1 lenne (így amp=0.5 és amp=1.0 pontosan
      ugyanazt a uint8 képet adná). Az `np.rint(noisy)` szimmetrikus matematikai
      kerekítést végez a [0, 255] vágás előtt:
        - amp=0.5 esetén [-0.5, +0.5] -> csak a kerekítési határon változtat
        - amp=1.0 esetén [-1.0, +1.0] -> valódi {-1, 0, +1} perturbációt ad.
    """
    if amplitude <= 0:
        return img_rgb.copy()
    shape = img_rgb.shape[:2] + (1,) if correlated else img_rgb.shape
    noise = rng.uniform(-amplitude, amplitude, shape).astype(np.float32)
    noisy = img_rgb.astype(np.float32) + noise
    return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Segédfüggvények (önálló definíciók a modul függetlenségéhez)
# ---------------------------------------------------------------------------

def is_spliced_image(img_path):
    """Azonosítja, hogy a kép hamisított-e (Columbia 4cam_splc vagy CASIA Tp_)."""
    lower, basename = img_path.lower(), os.path.basename(img_path)
    columbia = "4cam_splc" in lower and "edgemask" not in lower
    casia    = ((os.sep + "tp" + os.sep) in lower or "/tp/" in lower) \
               and basename.startswith("Tp_")
    return columbia or casia


def find_authentic_images(max_count=None):
    """Hiteles (authentic) képek összegyűjtése a 4cam_auth/ és Au/ könyvtárakból."""
    files = []
    for d in ["4cam_auth", "Au"]:
        if os.path.exists(d):
            for fn in sorted(os.listdir(d)):
                if fn.lower().endswith((".jpg", ".png", ".tif", ".jpeg")):
                    files.append(os.path.join(d, fn))
    if not files:
        return []
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(files)
    return files[:max_count] if max_count is not None else files


def compute_ks_score(heat_map, h, w, mask_binary=None):
    """
    Kétmintás Kolmogorov-Smirnov statisztika: max|C(S) - C(H)| (Cikk Eq. 8).
    Authentic képnél (mask_binary=None) a középső negyed négyzetet használja
    szintetikus GT maszkként (Zampoglou et al. [40]).
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
    """Vektorizált hőtérkép-felépítés az átfedő (32x32, step=8) ablakokból."""
    hs = np.zeros((h, w), np.float32)
    hc = np.zeros((h, w), np.float32)
    bs = block_size
    for p, (y, x) in zip(probs, positions):
        hs[y:y+bs, x:x+bs] += p
        hc[y:y+bs, x:x+bs] += 1.0
    hc[hc == 0] = 1.0
    return hs / hc


def load_models(scaler_path="itier_scaler.pkl"):
    """Betölti a globális StandardScalert és a 7 betanított modellt."""
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
            print(f"  Betöltve: {name:<20} <- {path}")
        else:
            print(f"  Hiányzik: {name:<20} ({path}) — kihagyva")

    if MODELS_TO_TEST is not None:
        models = {k: v for k, v in models.items() if k in MODELS_TO_TEST}
    return scaler, models


# ---------------------------------------------------------------------------
# Képszintű pontozás és támadási szintek kiértékelése
# ---------------------------------------------------------------------------

def score_image(img_rgb, scaler, models, block_size=32, step=8):
    """
    Egyetlen (memóriában lévő, esetleg már torzított) RGB képre kinyeri a
    blokkjellemzőket (egyszer), majd mind a 7 modellre kiszámítja a hőtérképet
    és a max_p értéket.
    """
    feats, _, positions, dims = extract_features_from_array(
        img_rgb, binary_mask=None, block_size=block_size, step=step)
    if feats is None or len(feats) == 0 or dims is None:
        return None

    h, w = dims
    fs = np.nan_to_num(scaler.transform(feats), nan=0.0, posinf=0.0, neginf=0.0)
    del feats

    out = {}
    for model_name, model in models.items():
        # 3-OSZTÁLYOS SÉMA: P(nem Uniform) = 1 - P(Uniform) = P(Border) + P(Spliced)
        probs = 1.0 - model.predict_proba(fs)[:, 0]
        heat  = build_heat_map_vectorized(probs, positions, h, w, block_size)
        max_p = float(np.nanmax(heat)) if np.any(np.isfinite(heat)) else float("nan")
        out[model_name] = (max_p, heat)
    del fs
    return out, (h, w)


def evaluate_attack_level(attack_fn, label, images_with_masks, auth_images,
                          scaler, models, block_size=32, step=8):
    """
    Végigfut az összes (spliced + authentic) tesztképen egy adott torzítási
    függvénnyel (`attack_fn`), és összegyűjti modellenként a képszintű
    ROC (max_p) és KS pontszámokat.
    """
    print(f"\n  --- Támadás: {label} ---")
    y_true = {name: [] for name in models}
    y_roc  = {name: [] for name in models}
    y_ks   = {name: [] for name in models}

    n_total = len(images_with_masks) + len(auth_images)
    done = 0

    # --- 1. Spliced (hamisított) képek ---
    for img_p, mask_p in images_with_masks:
        bgr = cv2.imread(img_p)
        if bgr is None:
            done += 1
            continue
        img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img_attacked = attack_fn(img_rgb)

        gt = load_binary_mask(mask_p)
        result = score_image(img_attacked, scaler, models, block_size, step)
        done += 1
        if result is None:
            continue
        model_scores, (h, w) = result
        if gt is not None and gt.shape != (h, w):
            gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)

        for name, (max_p, heat) in model_scores.items():
            ks = compute_ks_score(heat, h, w, mask_binary=gt)
            y_true[name].append(1)
            y_roc[name].append(max_p)
            y_ks[name].append(ks)

        del model_scores
        if done % 20 == 0:
            print(f"    feldolgozva: {done}/{n_total}")

    # --- 2. Authentic (hiteles) képek ---
    for img_p in auth_images:
        bgr = cv2.imread(img_p)
        if bgr is None:
            done += 1
            continue
        img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img_attacked = attack_fn(img_rgb)

        result = score_image(img_attacked, scaler, models, block_size, step)
        done += 1
        if result is None:
            continue
        model_scores, (h, w) = result

        for name, (max_p, heat) in model_scores.items():
            ks = compute_ks_score(heat, h, w, mask_binary=None)
            y_true[name].append(0)
            y_roc[name].append(max_p)
            y_ks[name].append(ks)

        del model_scores
        if done % 20 == 0:
            print(f"    feldolgozva: {done}/{n_total}")

    gc.collect()
    return y_true, y_roc, y_ks


def get_optimal_roc_threshold(y_true, y_scores):
    """
    ROC L2-optimális küszöb a tiszta alapfutáson, a mesterséges (0,0)/+inf
    kezdőpont kizárásával és NaN-szűréssel.
    """
    yt = np.asarray(y_true)
    ys = np.asarray(y_scores, dtype=float)
    valid = np.isfinite(ys)
    if not np.any(valid) or len(np.unique(yt[valid])) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(yt[valid], ys[valid])
    if len(thr) > 1:
        fpr, tpr, thr = fpr[1:], tpr[1:], thr[1:]
    dist = np.sqrt(fpr**2 + (1.0 - tpr)**2)
    return float(thr[np.argmin(dist)])


def level_metrics(yt, yroc, yks, t_roc, t_ks):
    """
    Kiszámítja egy modell egy adott torzítási szinthez tartozó F1-értékeit
    a RÖGZÍTETT (alapfutáson kalibrált) t_roc és t_ks küszöbökkel, valamint
    a diagnosztikai átlagokat (spliced vs. authentic átlagos KS és max_p).
    """
    yt = np.asarray(yt)
    yr = np.asarray(yroc, dtype=float)
    yk = np.asarray(yks,  dtype=float)

    pred_roc = np.where(np.isfinite(yr), (yr >= t_roc).astype(int), 0)
    pred_ks  = np.where(np.isfinite(yk), (yk >= t_ks).astype(int), 0)
    spl, aut = (yt == 1), (yt == 0)

    def mean_of(a, mask):
        return float(np.nanmean(a[mask])) if np.any(mask) else float("nan")

    return dict(
        f1_roc=float(f1_score(yt, pred_roc, zero_division=0)),
        f1_ks=float(f1_score(yt, pred_ks, zero_division=0)),
        ks_mean_spliced=mean_of(yk, spl),
        ks_mean_authentic=mean_of(yk, aut),
        maxp_mean_spliced=mean_of(yr, spl),
        maxp_mean_authentic=mean_of(yr, aut),
        flagged_ks=float(pred_ks.mean()),
        flagged_roc=float(pred_roc.mean()),
    )


# ===========================================================================
# Fő futtatás: Alapfutás + JPEG görbe + Független és Korrelált zajgörbék
# ===========================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("  Robusztussági vizsgálat (checklist #17)")
    print("  JPEG-újratömörítés + Egyenletes zaj (independent vs. correlated)")
    print("  Cikk Section 4.1 referencia: QF=75% -> F1~0.91, QF=50% -> F1~0.77")
    print("=" * 72)

    with open("dataset_splits.json") as f:
        data = json.load(f)
    test_images = data["test_images"]
    test_masks  = data["test_masks"]
    if MAX_TEST_IMAGES is not None:
        test_images = test_images[:MAX_TEST_IMAGES]
        test_masks  = test_masks[:MAX_TEST_IMAGES]
    images_with_masks = list(zip(test_images, test_masks))

    auth_images = find_authentic_images(MAX_AUTH_IMAGES)
    print(f"\nTeszt-spliced képek: {len(images_with_masks)}  |  "
          f"Authentic képek: {len(auth_images)}")

    print("\nModellek betöltése:")
    scaler, models = load_models()
    if not models:
        print("Hiba: Egyetlen modell sem tölthető be — futtasd a trainmodel_seven.py-t.")
        exit()
    print(f"Tesztelt modellek ({len(models)} db): {list(models.keys())}")

    # ── 1. LÉPÉS: TISZTA ALAPFUTÁS (QF=100, zaj=0) — Küszöbök rögzítése ────
    print("\n" + "=" * 72)
    print("  1. ALAPFUTÁS (QF=100, zaj=0) — Anti-Oracle küszöbök RÖGZÍTÉSE")
    print("=" * 72)
    yt0, yroc0, yks0 = evaluate_attack_level(
        lambda img: img, "alapfutás (nincs torzítás)",
        images_with_masks, auth_images, scaler, models, BLOCK_SIZE, STEP)

    fixed_thresh_roc, fixed_thresh_ks = {}, {}
    baseline = {}
    for name in models:
        t_roc = get_optimal_roc_threshold(yt0[name], yroc0[name])
        t_ks  = get_optimal_roc_threshold(yt0[name], yks0[name])
        fixed_thresh_roc[name] = t_roc
        fixed_thresh_ks[name]  = t_ks

        diag0    = level_metrics(yt0[name], yroc0[name], yks0[name], t_roc, t_ks)
        auc_roc0 = float(roc_auc_score(yt0[name], yroc0[name])) if len(set(yt0[name])) > 1 else float("nan")
        auc_ks0  = float(roc_auc_score(yt0[name], yks0[name]))  if len(set(yt0[name])) > 1 else float("nan")

        baseline[name] = dict(
            f1_roc=diag0["f1_roc"],
            f1_ks=diag0["f1_ks"],
            auc_roc=auc_roc0,
            auc_ks=auc_ks0,
            t_roc=t_roc,
            t_ks=t_ks,
            ks_mean_spliced=diag0["ks_mean_spliced"],
            ks_mean_authentic=diag0["ks_mean_authentic"],
            flagged_ks=diag0["flagged_ks"],
        )
        print(f"  [{name:<18s}] t_roc={t_roc:.4f}  t_ks={t_ks:.4f}  "
              f"F1_roc={diag0['f1_roc']:.4f}  F1_ks={diag0['f1_ks']:.4f}")

    results = {"baseline": baseline, "jpeg": {}, "noise": {}, "noise_correlated": {}}

    def run_sweep(key, levels, clean_level, make_attack, label):
        """Végigfuttat egy teljes torzítási görbét a RÖGZÍTETT alapküszöbökkel."""
        for lv in levels:
            if lv == clean_level:
                yt, yroc, yks = yt0, yroc0, yks0   # Alapfutás újrahasznosítása
            else:
                yt, yroc, yks = evaluate_attack_level(
                    make_attack(lv), f"{label} = {lv}", images_with_masks,
                    auth_images, scaler, models, BLOCK_SIZE, STEP)
            results[key][lv] = {}
            for name in models:
                met = level_metrics(yt[name], yroc[name], yks[name],
                                    fixed_thresh_roc[name], fixed_thresh_ks[name])
                results[key][lv][name] = met
                print(f"  {label}={str(lv):<5} [{name:16s}] F1_ks={met['f1_ks']:.4f}  "
                      f"KS spliced={met['ks_mean_spliced']:.3f} "
                      f"authentic={met['ks_mean_authentic']:.3f} "
                      f"(t_ks={fixed_thresh_ks[name]:.3f})  "
                      f"jelölt={met['flagged_ks'] * 100:.0f}%")

    # ── 2. LÉPÉS: JPEG-újratömörítési görbe ────────────────────────────────
    if RUN_JPEG:
        print("\n" + "=" * 72)
        print("  2. JPEG ÚJRATÖMÖRÍTÉSI GÖRBE (QF = 100 -> 30)")
        print("=" * 72)
        run_sweep("jpeg", JPEG_QUALITIES, 100,
                  lambda q: (lambda img: apply_jpeg_compression(img, q)), "QF")

    # ── 3. LÉPÉS: Egyenletes zaj görbék (independent vs. correlated) ───────
    for mode in NOISE_MODES:
        key = "noise" if mode == "independent" else "noise_correlated"
        print("\n" + "=" * 72)
        print(f"  3. EGYENLETES ZAJ GÖRBE — mód: {mode}")
        print("=" * 72)

        def make_attack(a, correlated=(mode == "correlated")):
            # Szintenként azonos RandomState magérték: a zaj így skálázottan
            # egymásba ágyazott, ezért a torzítási görbe monoton és reprodukálható.
            rng = np.random.RandomState(RANDOM_SEED)
            return lambda img: apply_uniform_noise(img, a, rng, correlated=correlated)

        run_sweep(key, NOISE_AMPLITUDES, 0, make_attack, f"amp[{mode}]")

    # ── 4. LÉPÉS: Összefoglaló táblázatok és JSON export ───────────────────
    def print_tables(key, levels, title, prefix):
        if not results[key]:
            return
        print("\n" + "=" * 82)
        print(f"  ÖSSZEFOGLALÓ — Képszintű F1 (KS-módszer): {title}")
        print("=" * 82)
        print(f"  {'Modell':<18}" + "".join(f"{prefix}{lv}".ljust(10) for lv in levels))
        print("  " + "-" * (18 + 10 * len(levels)))
        for name in models:
            print(f"  {name:<18}" + "".join(
                f"{results[key][lv][name]['f1_ks']:<10.4f}" for lv in levels))

        print(f"\n  KS-szeparáció (átlagos KS: spliced/authentic) — {title}")
        print(f"  {'Modell':<18}" + "".join(f"{prefix}{lv}".ljust(10) for lv in levels))
        print("  " + "-" * (18 + 10 * len(levels)))
        for name in models:
            print(f"  {name:<18}" + "".join(
                f"{results[key][lv][name]['ks_mean_spliced']:.2f}/"
                f"{results[key][lv][name]['ks_mean_authentic']:.2f}".ljust(10)
                for lv in levels))
        print("  Rögzített t_ks küszöbök: " + ", ".join(
            f"{n}={fixed_thresh_ks[n]:.2f}" for n in models))

    print_tables("jpeg", JPEG_QUALITIES, "JPEG minőségi faktor", "QF=")
    print_tables("noise", NOISE_AMPLITUDES, "Független zaj amplitúdó", "a=")
    print_tables("noise_correlated", NOISE_AMPLITUDES, "Korrelált zaj amplitúdó", "a=")

    with open("robustness_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nJSON export sikeresen mentve: robustness_results.json")