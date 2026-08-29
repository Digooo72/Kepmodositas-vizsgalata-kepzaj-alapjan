"""
evaluatemodel_three.py

Mindhárom modell (RFC, SVM, K-Means) párhuzamos kiértékelése a teszthalmazon.
Kiszámolja a képszintű ROC AUC-t és a pixelszintű F1 pontszámot mindhárom modellre.

Javítások (evaluate_model.py alapján):
  FIX #1 (kritikus): step=16 → step=8 — a cikk (Section 4.1) és a base file is step=8-at ír elő.
  FIX #2: t_hat betöltése optimal_threshold.txt-ből + per-modell adaptív küszöb K-Means-hez.
  FIX #3: Robusztus is_spliced detekció könyvtárnév + fájlnév-prefix alapján.
  FIX #4: Per-képes részletes konzolkiírás.
  FIX #5: Per-modell output könyvtár és vizualizáció.
  FIX #6 (ÚJ): Authentic képek hozzáadása az AUC számításhoz (4cam_auth / Au mappákból),
               mint tune_threshold.py — a prepare_database.py csak spliced képeket tesz
               a splitbe (nincs mask-uk az authentic képeknek), ezért AUC = nan volt.
  FIX #7 (ÚJ): K-Means per-modell t_hat: a t_hat=0.980 az RFC-re van optimalizálva,
               a K-Means más valószínűségi skálán dolgozik. A modell saját score
               eloszlásából ROC-alapú optimális küszöböt számolunk.
"""

import os
import json
import joblib
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from itier import extract_features_from_image, load_binary_mask
from evaluate_model import connected_components_postprocess

# Segédosztály a K-Means modell helyes betöltéséhez
from trainmodel_three import KMeansClassifier


def is_spliced_image(img_path):
    """
    FIX #3: Robusztus spliced/authentic detekció.
    Columbia spliced: "4cam_splc" almappában lévő képek
    CASIA spliced:    "Tp" mappában lévő, "Tp_" prefixű fájlok
    """
    lower    = img_path.lower()
    basename = os.path.basename(img_path)

    columbia_spliced = "4cam_splc" in lower and "edgemask" not in lower
    casia_spliced = (
        (os.sep + "tp" + os.sep) in lower or
        lower.endswith(os.sep + "tp") or
        "/tp/" in lower
    ) and basename.startswith("Tp_")

    return columbia_spliced or casia_spliced


def get_image_max_prob(img_path, model, scaler, block_size=32, step=8):
    """
    Kiszámítja egy kép max(P_I) értékét (Section 3.5).
    Authentic képekhez és AUC-számításhoz használjuk.
    """
    feats, _, positions, dims = extract_features_from_image(
        img_path, block_size=block_size, step=step)

    if feats is None or len(feats) == 0 or dims is None:
        return float("nan")

    h, w = dims
    feats_scaled = np.nan_to_num(
        scaler.transform(feats), nan=0.0, posinf=0.0, neginf=0.0)
    probs = model.predict_proba(feats_scaled)[:, 1]

    if not np.any(np.isfinite(probs)):
        return float("nan")

    heat_sum   = np.zeros((h, w), dtype=np.float32)
    heat_count = np.zeros((h, w), dtype=np.float32)
    for prob, (y, x) in zip(probs, positions):
        heat_sum[y:y + block_size, x:x + block_size]   += prob
        heat_count[y:y + block_size, x:x + block_size] += 1

    heat_count[heat_count == 0] = 1
    return float(np.nanmax(heat_sum / heat_count))


def find_authentic_images(max_count):
    """
    FIX #6: Authentic képek betöltése AUC számításhoz.
    Ugyanazt a logikát követi mint tune_threshold.py.
    """
    auth_dirs  = ["4cam_auth", "Au"]
    auth_files = []
    for adir in auth_dirs:
        if os.path.exists(adir):
            for fn in sorted(os.listdir(adir)):
                if fn.lower().endswith((".jpg", ".png", ".tif")):
                    auth_files.append(os.path.join(adir, fn))

    if not auth_files:
        return []

    np.random.seed(42)
    np.random.shuffle(auth_files)
    return auth_files[:max_count]


def compute_per_model_threshold(y_true, y_scores):
    """
    FIX #7: ROC-alapú L2-optimális küszöb számítása (Section 3.5).
    Csak akkor fut, ha van legalább 2 osztály és érvényes score az adatban.
    """
    y_true_arr   = np.array(y_true)
    y_scores_arr = np.array(y_scores, dtype=float)

    valid = np.isfinite(y_scores_arr)
    if not np.any(valid):
        return 0.5

    y_true_arr   = y_true_arr[valid]
    y_scores_arr = y_scores_arr[valid]

    if len(np.unique(y_true_arr)) < 2:
        return 0.5

    fpr, tpr, thresholds = roc_curve(y_true_arr, y_scores_arr)
    distances = np.sqrt(fpr ** 2 + (tpr - 1) ** 2)
    return float(thresholds[np.argmin(distances)])


if __name__ == "__main__":
    print("=" * 60)
    print("  Összehasonlító Kiértékelés: RFC vs SVM vs K-Means")
    print("=" * 60)

    # --- Modellek és scaler betöltése ---
    try:
        scaler = joblib.load("itier_scaler.pkl")
        models = {
            "RFC":     joblib.load("itier_rf_model.pkl"),
            "SVM":     joblib.load("itier_svm_model.pkl"),
            "K-Means": joblib.load("itier_kmeans_model.pkl")
        }
    except FileNotFoundError as e:
        print(f"Hiba: {e}")
        print("Futtasd le előbb a trainmodel_three.py-t!")
        exit()

    # FIX #2: RFC/SVM-hez az optimalizált t_hat betöltése
    global_t_hat = 0.5
    if os.path.exists("optimal_threshold.txt"):
        with open("optimal_threshold.txt", "r") as f:
            global_t_hat = float(f.read().strip())
        print(f"RFC/SVM ROC küszöb betöltve (t_hat): {global_t_hat:.4f}")
    else:
        print(f"Figyelem: optimal_threshold.txt hiányzik, fallback t_hat={global_t_hat}")
    print("K-Means: saját per-modell küszöb (FIX #7), lásd lentebb.\n")

    with open("dataset_splits.json", "r") as f:
        data = json.load(f)

    test_images = data["test_images"]
    test_masks  = data["test_masks"]

    # FIX #5: Per-modell output könyvtárak
    output_dirs = {}
    for model_name in models:
        d = f"evaluation_results_{model_name.lower().replace('-', '_')}"
        os.makedirs(d, exist_ok=True)
        output_dirs[model_name] = d

    eval_data = {
        name: {
            "y_true_img":  [],
            "y_score_img": [],
            "f1_pixels":   []
        }
        for name in models
    }

    block_size    = 32
    spliced_count = 0

    # =========================================================
    # 1. SPLICED képek kiértékelése (pixel F1 + score gyűjtés)
    # =========================================================
    print(f"Tesztelendő spliced képek: {len(test_images)}\n")

    for i, (img_p, mask_p) in enumerate(zip(test_images, test_masks)):
        base_name  = os.path.splitext(os.path.basename(img_p))[0]
        is_spliced = 1 if is_spliced_image(img_p) else 0
        label_str  = "HAMISÍTOTT" if is_spliced else "VALÓDI"

        print(f"[{i + 1}/{len(test_images)}] {base_name}  ({label_str})")

        # FIX #1: step=8
        feats, _, positions, dims = extract_features_from_image(
            img_p, block_size=block_size, step=8)

        if feats is None or len(feats) == 0 or dims is None:
            print("  -> Sikertelen jellemzőkinyerés.\n")
            continue

        h, w = dims
        feats_scaled = np.nan_to_num(
            scaler.transform(feats), nan=0.0, posinf=0.0, neginf=0.0)

        gt_binary = load_binary_mask(mask_p)
        valid_gt  = False
        if gt_binary is not None:
            if gt_binary.shape != (h, w):
                gt_binary = cv2.resize(
                    gt_binary, (w, h), interpolation=cv2.INTER_NEAREST)
            y_true_pixel = (gt_binary.flatten() > 127).astype(int)
            valid_gt = True

        if is_spliced:
            spliced_count += 1

        for model_name, model in models.items():
            probs = model.predict_proba(feats_scaled)[:, 1]

            heat_sum   = np.zeros((h, w), dtype=np.float32)
            heat_count = np.zeros((h, w), dtype=np.float32)
            for prob, (y, x) in zip(probs, positions):
                heat_sum[y:y + block_size, x:x + block_size]   += prob
                heat_count[y:y + block_size, x:x + block_size] += 1

            heat_count[heat_count == 0] = 1
            heat_map  = heat_sum / heat_count
            max_p     = float(np.nanmax(heat_map)) if np.any(np.isfinite(heat_map)) else float("nan")

            eval_data[model_name]["y_true_img"].append(is_spliced)
            eval_data[model_name]["y_score_img"].append(max_p)

            pred_mask = connected_components_postprocess(
                np.nan_to_num(heat_map), block_size=block_size)

            f1 = 0.0
            if valid_gt:
                y_pred_pixel = (pred_mask.flatten() > 127).astype(int)
                f1 = f1_score(y_true_pixel, y_pred_pixel, zero_division=1)
                eval_data[model_name]["f1_pixels"].append(f1)

            max_p_str = f"{max_p:.3f}" if np.isfinite(max_p) else "nan"
            print(f"  [{model_name:7s}] max_p={max_p_str} | Pixel F1={f1:.4f}")

            # Vizualizáció
            original_bgr = cv2.imread(img_p)
            if original_bgr is None:
                continue
            original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
            overlay = original_rgb.copy().astype(np.float32)
            if valid_gt:
                pred_bool = pred_mask > 127
                gt_bool   = gt_binary > 127
                overlay[pred_bool & ~gt_bool] = (
                    overlay[pred_bool & ~gt_bool] * 0.4 + np.array([220, 50, 50]) * 0.6)
                overlay[~pred_bool & gt_bool] = (
                    overlay[~pred_bool & gt_bool] * 0.4 + np.array([50, 200, 50]) * 0.6)
                overlay[pred_bool & gt_bool] = (
                    overlay[pred_bool & gt_bool] * 0.4 + np.array([255, 165, 0]) * 0.6)
            overlay = overlay.clip(0, 255).astype(np.uint8)

            fig, axes = plt.subplots(1, 5, figsize=(25, 5))
            fig.suptitle(
                f"{base_name} | {model_name} | max_p={max_p_str} | F1={f1:.3f}",
                fontsize=10)
            axes[0].imshow(original_rgb);  axes[0].set_title("Eredeti");              axes[0].axis("off")
            axes[1].imshow(gt_binary if valid_gt else np.zeros((h, w)), cmap="gray")
            axes[1].set_title("GT maszk"); axes[1].axis("off")
            axes[2].imshow(np.nan_to_num(heat_map), cmap="jet", vmin=0, vmax=1)
            axes[2].set_title("Hőtérkép p(Border)"); axes[2].axis("off")
            axes[3].imshow(pred_mask, cmap="gray"); axes[3].set_title("Predikált maszk"); axes[3].axis("off")
            axes[4].imshow(overlay)
            axes[4].set_title("Overlay\nPiros=FP  Narancs=TP  Zöld=FN"); axes[4].axis("off")
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dirs[model_name], f"{base_name}_result.png"),
                bbox_inches="tight", dpi=100)
            plt.close()

        print()

    # =========================================================
    # FIX #6: AUTHENTIC képek hozzáadása AUC-hoz
    # A prepare_database.py nem tartalmaz authentic képeket a splitben
    # (nincs GT mask-juk), ezért a test set csak spliced képeket tartalmaz
    # → AUC = nan. Megoldás: 4cam_auth / Au mappákból töltjük be őket,
    # ahogy tune_threshold.py is teszi.
    # =========================================================
    auth_files = find_authentic_images(max_count=spliced_count)

    if auth_files:
        print(f"Authentic képek hozzáadva AUC-hoz: {len(auth_files)} db "
              f"(4cam_auth / Au)\n")
        for img_p in auth_files:
            for model_name, model in models.items():
                max_p = get_image_max_prob(img_p, model, scaler,
                                           block_size=block_size, step=8)
                eval_data[model_name]["y_true_img"].append(0)
                eval_data[model_name]["y_score_img"].append(max_p)
    else:
        print("Figyelem: Nem találtam '4cam_auth' vagy 'Au' mappát.")
        print("  AUC számítás nem lesz megbízható (csak spliced képek a test splitben).\n")

    # =========================================================
    # FIX #7: Per-modell t_hat számítása
    # Az RFC t_hat-ja a tune_threshold.py-ból jön. Az SVM ezt örökli
    # (közel azonos score skála). A K-Means valószínűségei más skálán
    # vannak (távolság-alapú softmax), ezért saját ROC-optimális küszöböt
    # számolunk a megfigyelt score eloszlásból.
    # =========================================================
    model_thresholds = {}
    for model_name in models:
        if model_name == "K-Means":
            t = compute_per_model_threshold(
                eval_data[model_name]["y_true_img"],
                eval_data[model_name]["y_score_img"]
            )
            model_thresholds[model_name] = t
            print(f"K-Means per-modell t_hat (ROC L2-opt): {t:.4f}")
        else:
            model_thresholds[model_name] = global_t_hat

    # =========================================================
    # VÉGSŐ ÖSSZESÍTÉS
    # =========================================================
    print("\n" + "=" * 60)
    print("  VÉGSŐ TESZTEREDMÉNYEK (Test Set)")
    print("=" * 60)

    summary = {}
    for model_name in models:
        y_true_arr  = np.array(eval_data[model_name]["y_true_img"])
        y_score_arr = np.array(eval_data[model_name]["y_score_img"], dtype=float)
        f1_pixels   = eval_data[model_name]["f1_pixels"]
        t_hat       = model_thresholds[model_name]

        valid = np.isfinite(y_score_arr)
        auc   = 0.0
        if np.sum(valid) >= 2 and len(np.unique(y_true_arr[valid])) == 2:
            try:
                auc = roc_auc_score(y_true_arr[valid], y_score_arr[valid])
            except ValueError:
                auc = 0.0
        elif len(np.unique(y_true_arr)) < 2:
            print(f"  [{model_name}] Figyelem: AUC nem számítható — "
                  f"csak {len(np.unique(y_true_arr))} osztály.")

        y_pred_arr = (y_score_arr > t_hat).astype(int)
        try:
            img_f1 = f1_score(y_true_arr, y_pred_arr, zero_division=1)
        except Exception:
            img_f1 = 0.0

        mean_pixel_f1 = float(np.mean(f1_pixels)) if f1_pixels else 0.0

        summary[model_name] = {
            "auc":           auc,
            "img_f1":        img_f1,
            "mean_pixel_f1": mean_pixel_f1,
            "t_hat":         t_hat
        }

        print(f"\n--- {model_name} (t_hat={t_hat:.4f}) ---")
        print(f"  Képszintű AUC  (ROC):       {auc:.4f}")
        print(f"  Képszintű F1   (t_hat):     {img_f1:.4f}")
        print(f"  Pixelszintű F1 (átlag):     {mean_pixel_f1:.4f}")
        print(f"  Eredmények mappája:         {output_dirs[model_name]}/")

    print("\n" + "=" * 60)
    print(f"{'Modell':<12} {'t_hat':>7} {'AUC':>8} {'Kép F1':>8} {'Pixel F1':>10}")
    print("-" * 50)
    for name, s in sorted(summary.items(), key=lambda x: -x[1]["auc"]):
        print(f"{name:<12} {s['t_hat']:>7.4f} {s['auc']:>8.4f} "
              f"{s['img_f1']:>8.4f} {s['mean_pixel_f1']:>10.4f}")