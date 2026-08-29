import os
import json
import joblib
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from itier import extract_features_from_image, load_binary_mask


# ---------------------------------------------------------------------------
# Post-processing: Szigorú elméleti implementáció (Cikk 3.6 fejezet)
# ---------------------------------------------------------------------------

def strict_paper_postprocess(prob_map, t_hat=0.5, t_low=0.2):
    """
    Iteratív régió-növesztő algoritmus a határvonal rekonstrukciójához.
    """
    h, w = prob_map.shape

    # 1. LÉPÉS: Kezdőpontok (seeds) és az engedélyezett tartomány kijelölése
    border_seeds = (prob_map > t_hat).astype(np.uint8) * 255
    allowed_area = (prob_map > t_low).astype(np.uint8) * 255

    # 2. LÉPÉS: Iteratív Régió-növesztés
    current_border = border_seeds.copy()
    kernel = np.ones((3, 3), np.uint8)  # 8-szomszédság vizsgálata

    while True:
        # Szomszédok tágítása
        dilated = cv2.dilate(current_border, kernel, iterations=1)
        # Csak a t_low feletti szomszédok megtartása
        new_border = cv2.bitwise_and(dilated, allowed_area)

        # Ha nem találtunk új pixelt, az algoritmus megáll
        if np.array_equal(new_border, current_border):
            break
        current_border = new_border

    # 3. LÉPÉS: Gazdakép és Hamisított terület szétválasztása
    # Invertáljuk a határt, hogy megkapjuk a zárt területeket
    non_border = cv2.bitwise_not(current_border)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(non_border, connectivity=8)

    if num_labels <= 1:
        return np.zeros((h, w), dtype=np.uint8)

        # Területek kigyűjtése (a 0-s label a határvonal maga)
    areas = [(label_idx, stats[label_idx, cv2.CC_STAT_AREA]) for label_idx in range(1, num_labels)]
    areas.sort(key=lambda x: x[1], reverse=True)

    # A cikk feltételezése: a legnagyobb terület a gazdakép (host image)
    # A kisebb területek a hamisítottak (spliced areas)
    spliced_mask = np.zeros((h, w), dtype=np.uint8)
    for i in range(1, len(areas)):
        label_idx = areas[i][0]
        spliced_mask[labels == label_idx] = 255

    return spliced_mask


# ---------------------------------------------------------------------------
# Képszintű osztályozás (Cikk 3.5 fejezet, 5. egyenlet)
# ---------------------------------------------------------------------------

def classify_image(prob_map, threshold):
    """ Ha a hőtérkép maximuma nagyobb a küszöbnél, a kép hamisított.  """
    return bool(np.max(prob_map) > threshold)


# ---------------------------------------------------------------------------
# Fő értékelő futtatás
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("--- Modell Kiértékelés (ROC-optimalizált küszöbbel) ---\n")

    # Modell és Scaler betöltése
    try:
        clf = joblib.load("itier_rf_model.pkl")
        scaler = joblib.load("itier_scaler.pkl")
    except FileNotFoundError:
        print("Hiba: Hiányzó modellfájlok!")
        exit()

    # ROC optimalizált küszöb beolvasása (tune_treshold.py kimenete)
    t_hat = 0.5  # Fallback érték
    if os.path.exists("optimal_threshold.txt"):
        with open("optimal_threshold.txt", "r") as f:
            t_hat = float(f.read().strip())
        print(f"-> Betöltött optimális küszöb: {t_hat:.4f}")
    else:
        print("-> Figyelem: optimal_threshold.txt nem található, alapértelmezett 0.5-öt használok.")

    # t_low meghatározása (a cikk szerint t_low < t_hat)
    t_low = t_hat * 0.4

    # Adathalmaz betöltése
    with open("dataset_splits.json", "r") as f:
        data = json.load(f)
    test_images = data["test_images"]
    test_masks = data["test_masks"]

    output_dir = "evaluation_results_roc"
    os.makedirs(output_dir, exist_ok=True)

    f1_scores = []

    for i, (img_p, mask_p) in enumerate(zip(test_images, test_masks)):
        base_name = os.path.splitext(os.path.basename(img_p))[0]
        print(f"[{i + 1}/{len(test_images)}] {base_name}")

        # Jellemzőkinyerés
        feats, _, positions, (h, w) = extract_features_from_image(img_p)
        if feats is None or len(feats) == 0: continue

        # Standardizálás és predikció
        feats_scaled = np.nan_to_num(scaler.transform(feats))
        probs = clf.predict_proba(feats_scaled)[:, 1]

        # Hőtérkép aggregálása átlagolással
        heat_sum = np.zeros((h, w), dtype=np.float32)
        heat_count = np.zeros((h, w), dtype=np.float32)
        for p, (y, x) in zip(probs, positions):
            heat_sum[y:y + 32, x:x + 32] += p
            heat_count[y:y + 32, x:x + 32] += 1
        heat_count[heat_count == 0] = 1
        heat_map = heat_sum / heat_count

        # Képszintű döntés az optimalizált küszöbbel
        is_fake = classify_image(heat_map, t_hat)

        # Pixelszintű lokalizáció
        pred_mask = strict_paper_postprocess(heat_map, t_hat=t_hat, t_low=t_low)

        # Ground Truth betöltése
        gt_binary = load_binary_mask(mask_p)
        if gt_binary is None: continue
        if gt_binary.shape != (h, w):
            gt_binary = cv2.resize(gt_binary, (w, h), interpolation=cv2.INTER_NEAREST)

        # F1 Score számítása
        f1 = f1_score((gt_binary.flatten() > 127), (pred_mask.flatten() > 127), zero_division=1)
        f1_scores.append(f1)
        print(f"   Eredmény: {'HAMISÍTOTT' if is_fake else 'VALÓDI'} | Pixel F1: {f1:.4f}")

    if f1_scores:
        print("\n" + "=" * 30)
        print(f"Átlagos F1: {np.mean(f1_scores):.4f}")
        print("=" * 30)