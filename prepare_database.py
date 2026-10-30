"""

Két mód:
  COLUMBIA_ONLY = True  -> csak Columbia adatbázis (cikk eredeti beállítása)
  COLUMBIA_ONLY = False -> Columbia + CASIA Tp_D_* (cross-image splicing)

Columbia GT maszk: edgemask = HATÁRVONAL a két terület között (nem a spliced régió!)
CASIA GT maszk:   _gt = a hamisított terület bináris maszkja
"""

import os
import glob
import json
from sklearn.model_selection import train_test_split

# ============================================================
# KONFIGURÁCIÓ — itt állítsd be
COLUMBIA_ONLY = True   # True = csak Columbia, False = Columbia + CASIA Tp_D_*
# ============================================================


def load_dataset_paths(image_dir, mask_dir, mask_suffixes,
                       image_exts=None, mask_exts=None,
                       only_cross_image=False):
    if image_exts is None:
        image_exts = [".jpg"]
    if mask_exts is None:
        mask_exts = [".png"]

    image_paths, mask_paths = [], []
    found_images = []
    for ext in image_exts:
        found_images.extend(glob.glob(os.path.join(image_dir, f"*{ext}")))

    for img_path in sorted(found_images):
        basename = os.path.basename(img_path)
        # CASIA szures: csak Tp_D_* (different source = kulonbozo kameral hamisitott)
        if only_cross_image and not basename.startswith("Tp_D_"):
            continue

        base_name = os.path.splitext(basename)[0]
        mask_found = False
        for suffix in mask_suffixes:
            for m_ext in mask_exts:
                mask_path = os.path.join(mask_dir, f"{base_name}{suffix}{m_ext}")
                if os.path.exists(mask_path):
                    image_paths.append(img_path)
                    mask_paths.append(mask_path)
                    mask_found = True
                    break
            if mask_found:
                break

    return image_paths, mask_paths


if __name__ == "__main__":
    all_imgs, all_masks = [], []

    # --- Columbia ---
    columbia_imgs, columbia_masks = load_dataset_paths(
        image_dir="4cam_splc",
        mask_dir="4cam_splc/edgemask",
        mask_suffixes=["_edgemask", "_edgemask_2", "_edgemask_3", ""],
        image_exts=[".jpg", ".tif"],
        mask_exts=[".jpg", ".png"],
        only_cross_image=False
    )
    all_imgs  += columbia_imgs
    all_masks += columbia_masks
    print(f"Columbia kepek: {len(columbia_imgs)}")

    # --- CASIA (opcionális) ---
    if not COLUMBIA_ONLY:
        casia_imgs, casia_masks = load_dataset_paths(
            image_dir="Tp",
            mask_dir="CASIA 2 Groundtruth",
            mask_suffixes=["_gt"],
            image_exts=[".jpg", ".tif", ".TIF", ".jpeg"],
            mask_exts=[".png", ".jpg", ".tif"],
            only_cross_image=True   # csak Tp_D_*
        )
        all_imgs  += casia_imgs
        all_masks += casia_masks
        print(f"CASIA Tp_D_* kepek: {len(casia_imgs)}")

    print(f"Osszes: {len(all_imgs)} kep")

    if not all_imgs:
        print("Hiba: Nem talaltam kep-maszk parokat!")
        exit()

    X_train, X_test, y_train, y_test = train_test_split(
        all_imgs, all_masks, test_size=0.25, random_state=67
    )

    dataset_splits = {
        "train_images": X_train,
        "train_masks":  y_train,
        "test_images":  X_test,
        "test_masks":   y_test,
        "stats": {
            "columbia":      len(columbia_imgs),
            "columbia_only": COLUMBIA_ONLY,
            "total":         len(all_imgs),
            "train":         len(X_train),
            "test":          len(X_test)
        }
    }

    with open("dataset_splits.json", "w") as f:
        json.dump(dataset_splits, f, indent=4)

    print(f"Tanitohalmaz: {len(X_train)} kep")
    print(f"Teszthalmaz:  {len(X_test)} kep")
    print("Elmentve: dataset_splits.json")

    if COLUMBIA_ONLY:
        print("\nColumbia mod: GT = elmaszkolt hatar (edgemask).")
        print("A modell a HATART tanulja, nem a hamisitott regiót.")
        print("Ertekelesnél a hőtérkép-alapu szegmentáció adja a regiót.")