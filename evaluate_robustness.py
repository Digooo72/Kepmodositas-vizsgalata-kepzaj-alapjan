"""
evaluate_robustness.py

CHECKLIST #17 — Képzajalapú képmódosítás-vizsgálatok (robusztusság)
=====================================================================
A cikk (Itier et al., 2021) Section 4.1 utolsó kísérletének reprodukciója:

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

Két támadás-típust szimulál a MÁR BETANÍTOTT modelleken (nem tanít újra):
  1. JPEG újratömörítés csökkenő minőségi faktorral (QF = 100→30)
  2. Egyenletes (uniform) zaj hozzáadása növekvő amplitúdóval
     (a cikk ezt csak kvalitatívan említi, konkrét görbét nem közöl rá —
     itt egy ésszerű amplitúdó-tartományt választottunk kiegészítésként)

MÓDSZERTANI DÖNTÉS (fontos!): a küszöböt EGYETLEN alkalommal, a TISZTA
(QF=100, zaj=0) alapfutáson határozzuk meg és RÖGZÍTJÜK — minden további,
torzított szinten EZT a fix küszöböt alkalmazzuk, nem számoljuk újra.
Ez felel meg a "támadás" (counter-attack) fogalmának: egy már KALIBRÁLT,
éles detektort tesztelünk azzal, hogy mennyire esik vissza a teljesítménye,
ha valaki utólag JPEG-be tömöríti vagy zajjal terheli a képet — nem pedig
azt vizsgáljuk, hogy egy MINDIG ÚJRAKALIBRÁLT detektor meddig tartja magát
(az utóbbi szisztematikusan alábecsülné a valódi teljesítményvesztést).

KIBŐVÍTÉS (az első valós futtatás tanulságai alapján):
  - Finomabb zaj-rács (0.5, 1, 2, 3, 5 ...): az első futtatásban már amp=5-nél
    összeomlott a detektor, a valódi átmenet a 0-5 tartományban van.
  - Zaj-variáns "correlated": UGYANAZ a zaj kerül mindhárom csatornára, így
    megmarad a csatornák közti korreláció — az "independent" (csatornánként
    független) zajjal összevetve szétválasztható, mennyit számít a zaj
    energiája, és mennyit a csatornakorreláció romlása.
  - Diagnosztika: szintenként kiírja az átlagos KS-értéket spliced és
    authentic képekre, valamint a fix küszöb által "hamisítottnak" jelölt
    képek arányát. F1=0 ilyenkor értelmezhető: ha a KS lecsúszik a küszöb
    alá, a detektor "vak" (nem jelez semmit), nem rossz irányba téved.

Ez a szkript ÖNÁLLÓ, nem importál semmit evaluatemodel_seven.py-ból (a
projekt eddigi konvenciója szerint minden evaluate_*.py fájl önmagában
értelmezhető) — a szükséges kis segédfüggvényeket (compute_ks_score,
build_heat_map_vectorized, is_spliced_image, find_authentic_images,
load_models) saját maga tartalmazza, ugyanazzal a logikával, mint
evaluatemodel_seven.py.
"""

import os
import json
import joblib
import cv2
import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

from itier import extract_features_from_array, load_binary_mask
# 3-OSZTÁLYOS SÉMA (checklist #1): a KMeansClassifier trainmodel_seven.py-ban
# van definiálva (önállóan) — a pickle-elt modell unpickle-öléséhez innen
# kell importálni, ugyanúgy, mint evaluatemodel_seven.py-ban.
from trainmodel_seven import KMeansClassifier
from clustering_models import GMMClassifier, BayesianEMClassifier

# ============================================================
# KONFIGURÁCIÓ
# ============================================================
BLOCK_SIZE = 32
STEP       = 8

# None = minden betöltött modellt tesztel; pl. ["RFC"] a cikkel egyező,
# gyorsabb, egy-modelles futtatáshoz.
MODELS_TO_TEST = None

# QF=100 a "kontroll" (nincs újratömörítés) — ehhez képest mérjük a visszaesést.
# A cikk konkrétan a 75%-ot és 50%-ot közli; itt egy teljesebb görbét adunk.
JPEG_QUALITIES = [100, 90, 75, 50, 30]

RUN_JPEG = True    # False = a JPEG-görbét kihagyja (gyorsabb futás)

# Egyenletes zaj [-amplitúdó, +amplitúdó] tartományból, pixelenként.
# 0 = kontroll (nincs zaj hozzáadva). Az első futtatásban 5 felett minden
# modell már 0 volt, ezért a rács most a 0-5 tartományt sűríti.
NOISE_AMPLITUDES = [0, 0.5, 1, 2, 3, 5, 10, 20]

# "independent": csatornánként független zaj (a korrelációt is rontja)
# "correlated" : ugyanaz a zaj mindhárom csatornán (a korrelációt megtartja)
# Futásidő: minden mód + szint egy teljes kép-átfutás; gyorsításhoz
# elég ["independent"], vagy rövidebb NOISE_AMPLITUDES lista.
NOISE_MODES = ["independent", "correlated"]

MAX_TEST_IMAGES = 60   # None = az összes; kis szám = gyors próbafuttatás
MAX_AUTH_IMAGES = 20   # None = az összes elérhető 4cam_auth/Au kép
RANDOM_SEED     = 42


# ---------------------------------------------------------------------------
# Támadás-szimulátorok
# ---------------------------------------------------------------------------

def apply_jpeg_compression(img_rgb, quality):
    """
    JPEG újratömörítés memóriában (lemezre írás nélkül): encode majd decode.
    quality=100 esetén (kontroll) az eredeti képet adja vissza változatlanul
    (nem futtat felesleges tömörítést).
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
    Egyenletes zaj [-amplitude, +amplitude] tartományból, csatornánként
    független mintavétellel (ez FÜGGETLEN zajt ad minden RGB csatornára,
    tehát pontosan azt a fajta zaj-korreláció-rombolást szimulálja, amit a
    cikk módszere elméletileg detektálna — jó "stressz-teszt" arra nézve,
    hogy mekkora külső zaj tudja már elfedni a splicing intrinzik jelét).
    amplitude=0 esetén (kontroll) az eredeti képet adja vissza.

    correlated=True: egyetlen (H, W, 1) zajmintát használ, ami mindhárom
    csatornára UGYANAZ — a csatornák közti korreláció megmarad (csak a
    zaj energiája nő), ellentétben az alapértelmezett független zajjal.
    """
    if amplitude <= 0:
        return img_rgb.copy()
    shape = img_rgb.shape[:2] + (1,) if correlated else img_rgb.shape
    noise = rng.uniform(-amplitude, amplitude, shape).astype(np.float32)
    noisy = img_rgb.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Kisegítő függvények (önálló másolat, ugyanaz a logika, mint
# evaluatemodel_seven.py-ban — a checklist #1 kérésének megfelelően minden
# evaluate_*.py fájl önmagában értelmezhető, nem importál másik evaluate
# fájlból)
# ---------------------------------------------------------------------------

def is_spliced_image(img_path):
    lower, basename = img_path.lower(), os.path.basename(img_path)
    columbia = "4cam_splc" in lower and "edgemask" not in lower
    casia    = ((os.sep + "tp" + os.sep) in lower or "/tp/" in lower) \
               and basename.startswith("Tp_")
    return columbia or casia


def find_authentic_images(max_count=None):
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
    """max|C(S)-C(H)| — ugyanaz, mint evaluatemodel_seven.py compute_ks_score-ja.
    Authentic képnél (mask_binary=None) a cikk szerinti középső-négyzet
    szintetikus GT-t használja."""
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
    hs = np.zeros((h, w), np.float32)
    hc = np.zeros((h, w), np.float32)
    bs = block_size
    for p, (y, x) in zip(probs, positions):
        hs[y:y+bs, x:x+bs] += p
        hc[y:y+bs, x:x+bs] += 1.0
    hc[hc == 0] = 1.0
    return hs / hc


def load_models(scaler_path="itier_scaler.pkl"):
    """Ugyanaz, mint evaluatemodel_seven.py load_models()-je."""
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
                model_files[key] = fname; break

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
# Egy adott támadás-szint (pl. QF=75 vagy zaj-amplitúdó=20) teljes
# kiértékelése minden képre és minden modellre
# ---------------------------------------------------------------------------

def score_image(img_rgb, scaler, models, block_size=32, step=8):
    """
    Egyetlen (már betöltött/torzított) képre kiszámolja mind a max_p, mind
    a KS pontszámot minden modellre. Visszatérés: {model_name: (max_p, heat_map, positions, dims)}
    — a heat_map-et is visszaadjuk, mert a KS-számításhoz a GT maszk (ha
    van) is kell, amit a hívó tud hozzárendelni.
    """
    feats, _, positions, dims = extract_features_from_array(
        img_rgb, binary_mask=None, block_size=block_size, step=step)
    if feats is None or len(feats) == 0 or dims is None:
        return None

    h, w = dims
    fs = np.nan_to_num(scaler.transform(feats), nan=0.0, posinf=0.0, neginf=0.0)

    out = {}
    for model_name, model in models.items():
        probs = 1.0 - model.predict_proba(fs)[:, 0]   # P(nem Uniform), N-osztályos-biztos
        heat  = build_heat_map_vectorized(probs, positions, h, w, block_size)
        max_p = float(np.nanmax(heat)) if np.any(np.isfinite(heat)) else float("nan")
        out[model_name] = (max_p, heat)
    return out, (h, w)


def evaluate_attack_level(attack_fn, label, images_with_masks, auth_images,
                           scaler, models, block_size=32, step=8):
    """
    Végigfut az összes (spliced + authentic) képen EGY adott torzítási
    szinttel, és összegyűjti modellenként a képszintű max_p/ks score-okat +
    a valós (spliced/authentic) címkét.
    """
    print(f"\n  --- Támadás: {label} ---")
    y_true = {name: [] for name in models}
    y_roc  = {name: [] for name in models}
    y_ks   = {name: [] for name in models}

    n_total = len(images_with_masks) + len(auth_images)
    done = 0

    for img_p, mask_p in images_with_masks:
        bgr = cv2.imread(img_p)
        if bgr is None:
            done += 1; continue
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

        if done % 20 == 0:
            print(f"    {done}/{n_total}")

    for img_p in auth_images:
        bgr = cv2.imread(img_p)
        if bgr is None:
            done += 1; continue
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

        if done % 20 == 0:
            print(f"    {done}/{n_total}")

    return y_true, y_roc, y_ks


def get_optimal_roc_threshold(y_true, y_scores):
    """ROC L2-optimum, a mesterséges (0,0)/inf pont kizárva (lásd
    evaluate_filter_ablation.py — ugyanaz a hibajavítás itt is szükséges,
    hogy a küszöb sose ragadjon a triviális "utasíts el mindent" pontra)."""
    fpr, tpr, thr = roc_curve(y_true, y_scores)
    if len(thr) > 1:
        fpr, tpr, thr = fpr[1:], tpr[1:], thr[1:]
    dist = np.sqrt(fpr**2 + (1 - tpr)**2)
    return float(thr[np.argmin(dist)])


# ---------------------------------------------------------------------------
# Fő futtatás
# ---------------------------------------------------------------------------

def level_metrics(yt, yroc, yks, t_roc, t_ks):
    """Egy modell egy torzítási szintjének metrikái a RÖGZÍTETT küszöbökkel,
    plusz diagnosztika (átlagos pontszámok, jelölt képek aránya)."""
    yt  = np.asarray(yt)
    yr  = np.asarray(yroc, dtype=float)
    yk  = np.asarray(yks,  dtype=float)
    pred_roc = (yr >= t_roc).astype(int)
    pred_ks  = (yk >= t_ks).astype(int)
    spl, aut = yt == 1, yt == 0

    def mean_of(a, mask):
        return float(np.nanmean(a[mask])) if mask.any() else float("nan")

    return dict(
        f1_roc=f1_score(yt, pred_roc, zero_division=0),
        f1_ks=f1_score(yt, pred_ks, zero_division=0),
        ks_mean_spliced=mean_of(yk, spl), ks_mean_authentic=mean_of(yk, aut),
        maxp_mean_spliced=mean_of(yr, spl), maxp_mean_authentic=mean_of(yr, aut),
        flagged_ks=float(pred_ks.mean()), flagged_roc=float(pred_roc.mean()),
    )


if __name__ == "__main__":
    print("=" * 70)
    print("  Robusztussági vizsgálat (checklist #17)")
    print("  JPEG-újratömörítés + egyenletes zaj hozzáadás")
    print("  Cikk Section 4.1: QF=75%% -> F1~0.91, QF=50%% -> F1~0.77")
    print("=" * 70)

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
        print("Hiba: egyetlen modell sem tölthető be — futtasd trainmodel_seven.py-t.")
        exit()
    print(f"Tesztelt modellek: {list(models.keys())}")

    # ── 1. LÉPÉS: TISZTA alapfutás — ebből fixáljuk a küszöböket ──────────
    print("\n" + "=" * 70)
    print("  ALAPFUTÁS (QF=100, zaj=0) — a küszöbök innen lesznek RÖGZÍTVE")
    print("=" * 70)
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

        pred_roc = (np.array(yroc0[name]) >= t_roc).astype(int)
        pred_ks  = (np.array(yks0[name])  >= t_ks).astype(int)
        f1_roc0  = f1_score(yt0[name], pred_roc, zero_division=0)
        f1_ks0   = f1_score(yt0[name], pred_ks,  zero_division=0)
        auc_roc0 = roc_auc_score(yt0[name], yroc0[name]) if len(set(yt0[name])) > 1 else float("nan")
        auc_ks0  = roc_auc_score(yt0[name], yks0[name])  if len(set(yt0[name])) > 1 else float("nan")
        diag0 = level_metrics(yt0[name], yroc0[name], yks0[name], t_roc, t_ks)
        baseline[name] = dict(f1_roc=f1_roc0, f1_ks=f1_ks0, auc_roc=auc_roc0, auc_ks=auc_ks0,
                              t_roc=t_roc, t_ks=t_ks,
                              ks_mean_spliced=diag0["ks_mean_spliced"],
                              ks_mean_authentic=diag0["ks_mean_authentic"],
                              flagged_ks=diag0["flagged_ks"])
        print(f"  [{name}] t_roc={t_roc:.4f}  t_ks={t_ks:.4f}  "
              f"F1_roc={f1_roc0:.4f}  F1_ks={f1_ks0:.4f}")

    results = {"baseline": baseline, "jpeg": {}, "noise": {}, "noise_correlated": {}}

    def run_sweep(key, levels, clean_level, make_attack, label):
        """Egy teljes torzítási görbe; minden szinten a RÖGZÍTETT küszöbbel."""
        for lv in levels:
            if lv == clean_level:
                yt, yroc, yks = yt0, yroc0, yks0     # alapfutás újrahasznosítva
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
        print("\n" + "=" * 70)
        print("  JPEG ÚJRATÖMÖRÍTÉSI GÖRBE")
        print("=" * 70)
        run_sweep("jpeg", JPEG_QUALITIES, 100,
                  lambda q: (lambda img: apply_jpeg_compression(img, q)), "QF")

    # ── 3. LÉPÉS: Egyenletes zaj görbék (független / korrelált) ────────────
    for mode in NOISE_MODES:
        key = "noise" if mode == "independent" else "noise_correlated"
        print("\n" + "=" * 70)
        print(f"  EGYENLETES ZAJ GÖRBE — mód: {mode}")
        print("=" * 70)

        def make_attack(a, correlated=(mode == "correlated")):
            # Szintenként ugyanaz a magérték: a zaj így "egymásba ágyazott"
            # (amp=2 zaja az amp=1 zajának kétszerese), a görbe simább.
            rng = np.random.RandomState(RANDOM_SEED)
            return lambda img: apply_uniform_noise(img, a, rng, correlated=correlated)

        run_sweep(key, NOISE_AMPLITUDES, 0, make_attack, f"amp[{mode}]")

    # ── Összefoglaló táblázatok ──────────────────────────────────────────
    def print_tables(key, levels, title, prefix):
        if not results[key]:
            return
        print("\n" + "=" * 78)
        print(f"  ÖSSZEFOGLALÓ — F1 (KS-módszer): {title}")
        print("=" * 78)
        print(f"  {'Modell':<18}" + "".join(f"{prefix}{lv}".ljust(10) for lv in levels))
        for name in models:
            print(f"  {name:<18}" + "".join(
                f"{results[key][lv][name]['f1_ks']:<10.4f}" for lv in levels))

        print(f"\n  KS-szeparáció (átlagos KS: spliced/authentic) — {title}")
        print(f"  {'Modell':<18}" + "".join(f"{prefix}{lv}".ljust(10) for lv in levels))
        for name in models:
            print(f"  {name:<18}" + "".join(
                f"{results[key][lv][name]['ks_mean_spliced']:.2f}/"
                f"{results[key][lv][name]['ks_mean_authentic']:.2f}".ljust(10)
                for lv in levels))
        print("  rögzített t_ks: " + ", ".join(
            f"{n}={fixed_thresh_ks[n]:.2f}" for n in models))

    print_tables("jpeg", JPEG_QUALITIES, "JPEG minőségi faktor", "QF=")
    print_tables("noise", NOISE_AMPLITUDES, "független zaj amplitúdó", "a=")
    print_tables("noise_correlated", NOISE_AMPLITUDES, "korrelált zaj amplitúdó", "a=")

    with open("robustness_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nJSON export: robustness_results.json")