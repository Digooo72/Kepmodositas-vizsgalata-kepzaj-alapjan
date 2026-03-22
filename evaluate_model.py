import os
import json
import joblib
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from itier import extract_features_from_image

if __name__ == "__main__":
    print("--- Modell Kiértékelése (evaluate_model.py) ---")

    # 1. Betöltjük a korábban betanított modellt
    # _svm_ , _kmeans_
    model_path = "itier_rf_model.pkl"
    if not os.path.exists(model_path):
        print(f"Hiba: Nem található a '{model_path}' fájl! Előbb futtasd le a betanítást.")
        exit()

    clf = joblib.load(model_path)
    print("Modell sikeresen betöltve.")

    # 2. Betöltjük a teszt halmaz elérési útjait
    with open("dataset_splits.json", "r") as f:
        data = json.load(f)

    test_images = data["test_images"]
    test_masks = data["test_masks"]

    # Csak az első 10 képet teszteljük a gyors próba kedvéért, None ha az összesre
    TEST_LIMIT = 10
    test_images = test_images[:TEST_LIMIT]
    test_masks = test_masks[:TEST_LIMIT]

    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Összes tesztelendő kép: {len(test_images)}")
    print(f"Eredmények mentése ide: '{output_dir}/'\n")

    f1_scores = []

    # 3. Végigmegyünk a teszt képeken
    for i, (img_p, mask_p) in enumerate(zip(test_images, test_masks)):
        base_name = os.path.splitext(os.path.basename(img_p))[0]
        print(f"Tesztelés ({i + 1}/{len(test_images)}): {base_name}")

        # Jellemzők kinyerése a tesztképből (maszk nélkül, hiszen itt mi prediktálunk)
        feats, _, positions, (h, w) = extract_features_from_image(img_p)
        if feats is None or len(feats) == 0:
            print("  -> Hiba: Nem sikerült jellemzőket kinyerni a képből.")
            continue

        # Predikció: a modell megmondja, mennyi az esélye, hogy egy blokk "Border" (1-es osztály)
        probs = clf.predict_proba(feats)[:, 1]

        # Hőtérkép rekonstruálása az eredeti képméretre (32x32-es blokkokkal)
        heat_map_sum = np.zeros((h, w), dtype=np.float32)
        heat_map_count = np.zeros((h, w), dtype=np.float32)

        for prob, (y, x) in zip(probs, positions):
            heat_map_sum[y:y + 32, x:x + 32] += prob
            heat_map_count[y:y + 32, x:x + 32] += 1

        heat_map_count[heat_map_count == 0] = 1  # Nullával osztás elkerülése
        final_heat_map = heat_map_sum / heat_map_count

        # Bináris maszk létrehozása (50% feletti valószínűség manipuláltnak számít)
        pred_binary = np.where(final_heat_map > 0.5, 255, 0).astype(np.uint8)

        # Ground Truth (GT) maszk beolvasása az értékeléshez
        gt_mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            print("  -> Hiba: Nem sikerült beolvasni a maszkot.")
            continue
        _, gt_binary = cv2.threshold(gt_mask, 127, 255, cv2.THRESH_BINARY)

        # F1 Score számítása
        y_true = (gt_binary.flatten() / 255).astype(int)
        y_pred = (pred_binary.flatten() / 255).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=1)
        f1_scores.append(f1)
        print(f"  -> F1 Score: {f1:.4f}")

        # 4. Eredményképek vizualizálása és mentése
        original = cv2.cvtColor(cv2.imread(img_p), cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(20, 5))
        plt.subplot(1, 4, 1);
        plt.title("Eredeti Kép")
        plt.imshow(original);
        plt.axis('off')

        plt.subplot(1, 4, 2);
        plt.title("Ground Truth Maszk")
        plt.imshow(gt_binary, cmap='gray');
        plt.axis('off')

        plt.subplot(1, 4, 3);
        plt.title("Prediktált Hőtérkép")
        plt.imshow(final_heat_map, cmap='jet');
        plt.axis('off')

        plt.subplot(1, 4, 4);
        plt.title(f"Prediktált Maszk (F1: {f1:.2f})")
        plt.imshow(pred_binary, cmap='gray');
        plt.axis('off')

        # Mentés a kimeneti mappába
        plt.savefig(os.path.join(output_dir, f"{base_name}_result.png"), bbox_inches='tight')
        plt.close()

    # Végső statisztika kiírása
    if f1_scores:
        avg_f1 = np.mean(f1_scores)
        print(f"\n====================================")
        print(f"KÉSZ! Átlagos F1 Score a teszt halmazon: {avg_f1:.4f}")
        print(f"A kimeneti képeket megtalálod a '{output_dir}' mappában.")