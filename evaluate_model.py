"""
evaluate_model.py

POST-PROCESSING: Connected Components + ROC optimalizált küszöb

FONTOS ARCHITEKTÚRA:
  t_hat = képszintű döntési küszöb (max(P_I) > t_hat → Fake)
  A pixel-szintű szegmentáció MINDIG saját adaptív küszöböt használ,
  FÜGGETLEN t_hat-tól. Ha t_hat magas (pl. 0.989), az CSAK azt jelenti
  hogy a kép-szintű osztályozó konzervatívabb — a lokalizáció attól
  még helyesen működik.
"""

import os
import json
import joblib
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from itier import extract_features_from_image, load_binary_mask


# ---------------------------------------------------------------------------
# Segédfüggvények
# ---------------------------------------------------------------------------

def mean_prob_under_mask(prob_map, binary_mask):
    """Visszaadja a prob_map átlagát azon a területen ahol a binary_mask fehér."""
    pixels = prob_map[binary_mask > 0]
    return float(np.mean(pixels)) if len(pixels) > 0 else 0.0


def blob_detect(blurred, block_size):
    """
    Fallback: konvex burok kitöltés a legmagasabb p(Border) blobra.
    Nyílt határvonalnál (pl. párhuzamos csíkok) ez adja a legjobb eredményt.
    """
    h, w = blurred.shape
    max_p     = np.max(blurred)
    threshold = max_p * 0.45

    _, blob = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    blob = blob.astype(np.uint8)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (block_size, block_size))
    blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = (block_size * 2) ** 2
    valid    = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not valid:
        return np.zeros((h, w), dtype=np.uint8)

    # Legjobb blob: legmagasabb átlagos p(Border) alatta
    best_cnt, best_score = None, -1.0
    for cnt in valid:
        tmp = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(tmp, [cnt], -1, 255, cv2.FILLED)
        score = mean_prob_under_mask(blurred, tmp)
        if score > best_score:
            best_score, best_cnt = score, cnt

    hull   = cv2.convexHull(best_cnt)
    result = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(result, [hull], -1, 255, cv2.FILLED)

    if cv2.countNonZero(result) > h * w * 0.5:
        result = cv2.bitwise_not(result)
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (block_size // 2, block_size // 2))
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, k_open)

    return result


def touches_image_edge(mask, margin=3):
    return (np.any(mask[:margin, :]) or np.any(mask[-margin:, :]) or
            np.any(mask[:, :margin]) or np.any(mask[:, -margin:]))


def identify_background(num_labels, labels, stats, blurred):
    """
    Háttér azonosítása a cikk Section 3.6 alapján:
      "pixels of the largest area are labeled as belonging to the host image"

    Logika (prioritás sorrendben):
      1. Ha csak egy szél-érintő komponens van → az a háttér (egyértelmű)
      2. Ha több szél-érintő van:
         a. Ha a legnagyobb >2× akkora mint a 2. legnagyobb → méret dönt (háttér = legnagyobb)
         b. Ha méretek közel azonosak (arány < 2) → p(Border) átlag dönt (legkisebb = háttér)
            mert hasonló méretű régióknál a zajkorreláció alapján lehet dönteni
      3. Ha egyik sem érinti a szélét → p(Border) átlag dönt (ritka eset)

    Megjegyzés: a méret-alapú döntés az elsődleges, mert a cikk is ezt írja.
    A p(Border) alapú döntés csak akkor lép be, ha a méretek hasonlóak
    (ilyenkor a méret önmagában nem megbízható kritérium).
    """
    h, w = blurred.shape
    total_pixels = h * w

    edge = []
    for lbl in range(1, num_labels):
        mask  = (labels == lbl).astype(np.uint8) * 255
        area  = stats[lbl, cv2.CC_STAT_AREA]
        score = mean_prob_under_mask(blurred, mask)
        if touches_image_edge(mask):
            edge.append((lbl, area, score))

    if len(edge) == 0:
        # Egyik sem érinti a szélét → legkisebb p(Border) átlag = háttér
        all_lbls = []
        for lbl in range(1, num_labels):
            mask  = (labels == lbl).astype(np.uint8) * 255
            score = mean_prob_under_mask(blurred, mask)
            all_lbls.append((lbl, score))
        return min(all_lbls, key=lambda x: x[1])[0]

    if len(edge) == 1:
        return edge[0][0]

    # Több szél-érintő: méret vs p(Border) döntés
    edge_sorted = sorted(edge, key=lambda x: x[1], reverse=True)  # meret szerint csökkeno
    largest      = edge_sorted[0]   # (lbl, area, score)
    second       = edge_sorted[1]

    size_ratio = largest[1] / max(second[1], 1)

    if size_ratio >= 2.0:
        # Legnagyobb >2× akkora → méret dönt → az a háttér (cikk Section 3.6)
        return largest[0]
    else:
        # Hasonló méret → p(Border) átlag dönt → legkisebb = háttér
        return min(edge, key=lambda x: x[2])[0]


def merge_nearby_blobs(spliced, block_size):
    """
    A nem teljesen zárt határvonal miatt keletkező közeli foltokat összevonja,
    majd visszaszűkíti az eredeti méretre.
    Cél: az egymáshoz közel lévő spliced foltok egyetlen régióvá álljanak össze.
    """
    # Erős tágítás hogy a közeli foltok összeérjenek
    k_merge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (block_size * 2, block_size * 2))
    merged  = cv2.dilate(spliced, k_merge, iterations=2)
    # Visszaszűkítés
    merged  = cv2.erode(merged, k_merge, iterations=2)

    # Lyukkitöltés
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(spliced)
    for cnt in contours:
        cv2.drawContours(filled, [cnt], -1, 255, cv2.FILLED)

    # Csak a legnagyobb blob marad
    num_l, lbl_map, st, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
    if num_l > 2:
        biggest = int(np.argmax(st[1:, cv2.CC_STAT_AREA])) + 1
        filled  = (lbl_map == biggest).astype(np.uint8) * 255

    return filled


def connected_components_postprocess(prob_map, block_size=32):
    """
    Connected Components alapú pixel-szintű szegmentálás (Section 3.6).

    Három fő javítás az előző verzióhoz képest:

    1. Egyszerű rés-tömés (iterations=1): a 3× iteráció "megeszi" a régiót.
       Egy iteráció (~32px) elegendő a tipikus résekhez, a progresszív zárás
       kezeli a nagyobb réseket.

    2. merge_nearby_blobs(): ha a CC eredménye több kis folt, erős morfológiai
       zárással összevonja őket egyetlen régióvá. Ez kezeli a "foltok" problémát.

    3. CC vs blob_detect verseny: ha a CC eredménye kis (a hőtérkép magas
       területének <25%-a), a blob_detect (konvex burok) veszi át.
       Ez kezeli a nyílt határvonal esetet (párhuzamos csíkok).
    """
    h, w = prob_map.shape

    if np.max(prob_map) < 0.05:
        return np.zeros((h, w), dtype=np.uint8)

    blurred   = cv2.GaussianBlur(prob_map, (21, 21), 0)
    max_p     = np.max(blurred)
    mean_p    = np.mean(blurred)
    std_p     = np.std(blurred)
    threshold = max(mean_p + std_p * 1.0, max_p * 0.40)

    _, border_bin = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    border_bin    = border_bin.astype(np.uint8)

    # Referencia: hőtérkép magas területe (a predikció méretének mércéje)
    high_prob_area = max(cv2.countNonZero(border_bin), 1)

    # Rés-tömés: iterations=1 (~32px) — elég a tipikus résekhez
    # (iterations=3 túl agresszív volt, megevte a régiót)
    k_gap         = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (block_size, block_size))
    pregap        = cv2.dilate(border_bin, k_gap, iterations=1)
    pregap        = cv2.erode(pregap,      k_gap, iterations=1)
    border_filled = cv2.bitwise_or(border_bin, pregap)

    # Progresszív zárás
    cc_result = None
    for close_factor in [0.5, 1.0, 1.5, 2.0, 2.5]:
        ksize   = max(3, int(block_size * close_factor))
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        closed  = cv2.morphologyEx(border_filled, cv2.MORPH_CLOSE, k_close)

        non_border = cv2.bitwise_not(closed)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            non_border, connectivity=8)

        if num_labels >= 3:
            bg_lbl  = identify_background(num_labels, labels, stats, blurred)

            spliced = np.zeros((h, w), dtype=np.uint8)
            for lbl in range(1, num_labels):
                if lbl != bg_lbl:
                    spliced[labels == lbl] = 255

            k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (block_size // 2, block_size // 2))
            spliced = cv2.morphologyEx(spliced, cv2.MORPH_OPEN, k_open)

            if cv2.countNonZero(spliced) > 0:
                # Foltok összevonása
                spliced = merge_nearby_blobs(spliced, block_size)
                if cv2.countNonZero(spliced) > 0:
                    cc_result = spliced
                    break

    # Blob detect (konvex burok) — nyílt határvonalhoz jobb
    blob_result = blob_detect(blurred, block_size)

    # Döntés: CC vs blob_detect
    # Ha a CC eredménye elég nagy (>25% a magas prob területhez képest) → CC
    # Különben a blob_detect jobban lokalizál (nyílt határ esetén)
    if cc_result is not None:
        cc_area   = cv2.countNonZero(cc_result)
        blob_area = cv2.countNonZero(blob_result)
        ratio     = cc_area / high_prob_area

        if ratio >= 0.25:
            return cc_result
        elif blob_area > 0:
            # blob_detect vs cc: amelyik jobban fedi a magas prob területet
            cc_overlap   = cv2.countNonZero(cv2.bitwise_and(cc_result,   border_bin))
            blob_overlap = cv2.countNonZero(cv2.bitwise_and(blob_result, border_bin))
            cc_prec   = cc_overlap   / max(cc_area,   1)
            blob_prec = blob_overlap / max(blob_area, 1)
            return cc_result if cc_prec >= blob_prec else blob_result
        else:
            return cc_result

    if cv2.countNonZero(blob_result) > 0:
        return blob_result

    return np.zeros((h, w), dtype=np.uint8)
def classify_image(prob_map, t_hat):
    """
    if max(P_I) > t_hat → Fake, else Authentic  (Eq. 5)
    t_hat a ROC görbe alapján optimalizált képszintű küszöb.
    """
    return bool(np.max(prob_map) > t_hat)


# ---------------------------------------------------------------------------
# Fő értékelés
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("--- Modell Kiértékelés ---\n")

    for p in ["itier_rf_model.pkl", "itier_scaler.pkl"]:
        if not os.path.exists(p):
            print(f"Hiba: '{p}' nem található.")
            exit()

    clf    = joblib.load("itier_rf_model.pkl")
    scaler = joblib.load("itier_scaler.pkl")
    print("Modell és scaler betöltve.")

    # Képszintű küszöb (Section 3.5)
    t_hat = 0.5
    if os.path.exists('optimal_threshold.txt'):
        with open('optimal_threshold.txt', 'r') as f:
            t_hat = float(f.read().strip())
        print(f"ROC küszöb (t_hat): {t_hat:.4f}")
    else:
        print(f"Figyelem: optimal_threshold.txt hiányzik, fallback: {t_hat}")
    print(f"Megjegyzés: t_hat csak képszintű döntéshez használt.")
    print(f"A pixel-szintű szegmentáció saját adaptív küszöböt használ.\n")

    with open("dataset_splits.json", "r") as f:
        data = json.load(f)

    TEST_LIMIT  = None
    test_images = data["test_images"][:TEST_LIMIT]
    test_masks  = data["test_masks"][:TEST_LIMIT]

    output_dir = "evaluation_results_roc_cc"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Tesztelendő képek: {len(test_images)}\n")

    f1_scores = []

    for i, (img_p, mask_p) in enumerate(zip(test_images, test_masks)):
        base_name    = os.path.splitext(os.path.basename(img_p))[0]
        forgery_type = "D-type (splicing)" if "_D_" in base_name else "S-type (copy-move)"
        print(f"({i+1}/{len(test_images)}) {base_name}")
        print(f"  Típus: {forgery_type}")

        feats, _, positions, (h, w) = extract_features_from_image(img_p)
        if feats is None or len(feats) == 0:
            print("  -> Sikertelen jellemzőkinyerés.\n")
            continue

        feats_scaled = scaler.transform(feats)
        feats_scaled = np.nan_to_num(feats_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        # 3-OSZTÁLYOS SÉMA (checklist #1): P(nem Uniform) = P(Border)+P(Spliced)
        # = 1 - P(Uniform). 2 osztálynál ez pontosan a régi [:, 1] oszloppal
        # egyezik meg (visszafelé kompatibilis), 3 osztálynál a Border+Spliced
        # együttes valószínűsége — ez felel meg a GT maszkok szemantikájának
        # (minden ami nem host/background terület).
        probs        = 1.0 - clf.predict_proba(feats_scaled)[:, 0]

        # Hőtérkép összerakása (átlagolt átfedő valószínűségek, Section 3.6)
        heat_sum   = np.zeros((h, w), dtype=np.float32)
        heat_count = np.zeros((h, w), dtype=np.float32)
        block_size = 32

        for prob, (y, x) in zip(probs, positions):
            heat_sum[y:y+block_size, x:x+block_size]   += prob
            heat_count[y:y+block_size, x:x+block_size] += 1

        heat_count[heat_count == 0] = 1
        heat_map = heat_sum / heat_count

        # Képszintű osztályozás (t_hat alapján, Section 3.5)
        is_fake = classify_image(heat_map, t_hat)

        # Pixel-szintű szegmentáció (saját adaptív küszöb, Section 3.6)
        # FÜGGETLEN t_hat-tól — magas t_hat sem okoz üres maszkot
        pred_mask = connected_components_postprocess(heat_map, block_size=block_size)

        gt_binary = load_binary_mask(mask_p)
        if gt_binary is None:
            print("  -> GT maszk nem olvasható.\n")
            continue
        if gt_binary.shape != (h, w):
            gt_binary = cv2.resize(gt_binary, (w, h), interpolation=cv2.INTER_NEAREST)

        y_true = (gt_binary.flatten() > 127).astype(int)
        y_pred  = (pred_mask.flatten()  > 127).astype(int)
        f1      = f1_score(y_true, y_pred, zero_division=1)
        f1_scores.append(f1)

        gt_pct   = 100.0 * np.sum(y_true) / len(y_true)
        pred_pct = 100.0 * np.sum(y_pred) / len(y_pred)
        print(f"  Képszintű: {'HAMISÍTOTT' if is_fake else 'VALÓDI'} (max_p={np.max(heat_map):.3f} vs t_hat={t_hat:.3f})")
        print(f"  GT fehér:  {gt_pct:.1f}%  |  Pred fehér: {pred_pct:.1f}%")
        print(f"  Pixel F1:  {f1:.4f}\n")

        # Vizualizáció
        original_bgr = cv2.imread(img_p)
        if original_bgr is None:
            continue
        original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

        overlay   = original_rgb.copy().astype(np.float32)
        pred_bool = pred_mask  > 127
        gt_bool   = gt_binary  > 127

        overlay[pred_bool & ~gt_bool] = overlay[pred_bool & ~gt_bool] * 0.4 + np.array([220, 50,  50]) * 0.6
        overlay[~pred_bool & gt_bool] = overlay[~pred_bool & gt_bool] * 0.4 + np.array([50,  200, 50]) * 0.6
        overlay[pred_bool & gt_bool]  = overlay[pred_bool & gt_bool]  * 0.4 + np.array([255, 165,  0]) * 0.6
        overlay = overlay.clip(0, 255).astype(np.uint8)

        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        fig.suptitle(f"{base_name} | {forgery_type} | F1: {f1:.3f} | t_hat: {t_hat:.3f}", fontsize=10)

        axes[0].imshow(original_rgb);           axes[0].set_title("Eredeti");              axes[0].axis('off')
        axes[1].imshow(gt_binary, cmap='gray'); axes[1].set_title("GT (Hamisított ter.)"); axes[1].axis('off')
        axes[2].imshow(heat_map, cmap='jet', vmin=0, vmax=1)
        axes[2].set_title("Hőtérkép p(Border)"); axes[2].axis('off')
        axes[3].imshow(pred_mask, cmap='gray'); axes[3].set_title("Predikált maszk (CC)"); axes[3].axis('off')
        axes[4].imshow(overlay)
        axes[4].set_title("Overlay\nPiros=FP  Narancs=TP  Zöld=FN"); axes[4].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base_name}_result.png"), bbox_inches='tight', dpi=120)
        plt.close()

    if f1_scores:
        avg_f1 = np.mean(f1_scores)
        print("=" * 45)
        print(f"KÉSZ! Eredmények: '{output_dir}/'")
        print(f"Átlagos F1:  {avg_f1:.4f}")
        print(f"Min F1:      {np.min(f1_scores):.4f}")
        print(f"Max F1:      {np.max(f1_scores):.4f}")