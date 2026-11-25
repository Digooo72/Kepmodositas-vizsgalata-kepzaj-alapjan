"""
prepare_database.py

Adatbázis-felderítő, maszkpárosító és reprodukálható tanító/teszt felosztást
(dataset_splits.json) előállító modul.

─── Támogatott adatbázisok és üzemmódok (COLUMBIA_ONLY kapcsoló) ───────────────
  1. COLUMBIA_ONLY = True (alapértelmezett, az Itier et al., 2021 cikk beállítása):
     Kizárólag a Columbia Uncompressed Image Splicing adatbázis (180 hamisított
     kép a 4cam_splc/ mappában, 4 különböző kamerával készítve).

  2. COLUMBIA_ONLY = False (kiterjesztett üzemmód):
     Columbia + CASIA v2.0 adatbázis kereszt-kamera (cross-image splicing)
     manipulációi (Tp_D_* előtagú fájlok).

─── Ground Truth (GT) maszkok forenzikus szerepe (lásd itier.py) ───────────────
  - Columbia edgemask (4cam_splc/edgemask/):
    Háromszínű maszk (piros = befogadó háttér, zöld = beillesztett terület,
    kék = vágási határvonal). Az itier.py load_binary_mask() függvénye HSV
    színtérben detektálja a piros hátteret, majd negálja azt, így a bináris GT
    maszk a teljes manipulált területet (zöld + kék) tartalmazza. Ez teszi
    lehetővé a 3-osztályos tanítást: CLASS_UNIFORM (0), CLASS_BORDER (1) és
    CLASS_SPLICED (2).
  - CASIA GT maszk (CASIA 2 Groundtruth/*_gt.*):
    Szürkeárnyalatos bináris maszk, amely közvetlenül a beillesztett régiót jelöli.

  Megjegyzés a hiteles (authentic) képekhez:
    A dataset_splits.json kizárólag a maszkos (hamisított) képeket osztja fel
    75% tanító (135 kép) és 25% teszt (45 kép) arányban. A tanítás során a
    CLASS_UNIFORM (0) blokkok a hamisított képek érintetlen háttérterületeiről
    származnak, míg a képszintű kiértékeléshez (ROC / KS AUC) a hiteles képeket
    a 4cam_auth/ és Au/ mappákból olvassák be a kiértékelő szkriptek.
────────────────────────────────────────────────────────────────────────────────
"""

import os
import glob
import json
from sklearn.model_selection import train_test_split

# ===========================================================================
# KONFIGURÁCIÓ — adatbázis-összetétel és felosztási arányok
# ===========================================================================
COLUMBIA_ONLY = True   # True  = csak Columbia (cikk szerinti tiszta összevetés)
                       # False = Columbia + CASIA Tp_D_* (vegyes adatbázis)
TEST_SIZE     = 0.25   # Teszthalmaz aránya (Columbia 180 kép -> 135 train / 45 test)
RANDOM_SEED   = 67     # Rögzített magérték a reprodukálható train/test splithez
# ===========================================================================


def load_dataset_paths(image_dir, mask_dir, mask_suffixes,
                       image_exts=None, mask_exts=None,
                       only_cross_image=False):
    """
    Összepárosítja az `image_dir` könyvtárban található manipulált képeket
    a `mask_dir` könyvtárban lévő Ground Truth (GT) maszkjaikkal.
    """
    if image_exts is None:
        image_exts = [".jpg"]
    if mask_exts is None:
        mask_exts = [".png"]

    # --- 1. Képfájlok összegyűjtése duplikáció-szűréssel ---
    # Kis- és nagybetű-érzéketlen fájlrendszereken (pl. Windows) a ".tif" és ".TIF"
    # kiterjesztések ugyanazokat az állományokat adnák vissza; az os.path.normcase
    # halmaz megakadályozza a képek többszörös bekerülését.
    found_images = []
    seen_norm_paths = set()
    for ext in image_exts:
        for p in glob.glob(os.path.join(image_dir, f"*{ext}")):
            norm_p = os.path.normcase(os.path.abspath(p))
            if norm_p not in seen_norm_paths:
                seen_norm_paths.add(norm_p)
                found_images.append(p)

    image_paths, mask_paths = [], []

    # --- 2. Maszkok keresése és párosítása ---
    for img_path in sorted(found_images):
        basename = os.path.basename(img_path)

        # CASIA szűrés: csak Tp_D_* (különböző kamerából származó splicing)
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


# ===========================================================================
# Fő futtatás: Adatbázisok beolvasása és dataset_splits.json mentése
# ===========================================================================

if __name__ == "__main__":
    all_imgs, all_masks = [], []

    # --- 1. Columbia adatbázis betöltése ---
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

    # --- 2. CASIA v2.0 adatbázis betöltése (opcionális) ---
    casia_imgs = []
    if not COLUMBIA_ONLY:
        casia_imgs, casia_masks = load_dataset_paths(
            image_dir="Tp",
            mask_dir="CASIA 2 Groundtruth",
            mask_suffixes=["_gt"],
            image_exts=[".jpg", ".tif", ".TIF", ".jpeg"],
            mask_exts=[".png", ".jpg", ".tif"],
            only_cross_image=True   # Kizárólag Tp_D_* (cross-image splicing)
        )
        all_imgs  += casia_imgs
        all_masks += casia_masks
        print(f"CASIA Tp_D_* kepek: {len(casia_imgs)}")

    print(f"Osszes talalt kep-maszk par: {len(all_imgs)} kep")

    if not all_imgs:
        print("Hiba: Nem talaltam kep-maszk parokat! Ellenorizd a 4cam_splc/ mappat.")
        exit()

    # --- 3. Reprodukálható tanító / teszt felosztás (75% train, 25% test) ---
    X_train, X_test, y_train, y_test = train_test_split(
        all_imgs, all_masks, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    dataset_splits = {
        "train_images": X_train,
        "train_masks":  y_train,
        "test_images":  X_test,
        "test_masks":   y_test,
        "stats": {
            "columbia":      len(columbia_imgs),
            "casia":         len(casia_imgs),
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
        print("\nColumbia mod aktiv:")
        print("  - Az edgemask 3 szinebol (piros=hatter, zold=spliced, kek=hatar) az itier.py")
        print("    HSV-inverzioval nyeri ki a teljes hamisitott regiot (zold + kek).")
        print("  - A 3-osztalyos sema igy a hatarvonalat (Border) ES a belso teruletet (Spliced)")
        print("    is megtanulja, a lokalizaciot pedig a hoterkep morfologiai szegmentalasa adja.")