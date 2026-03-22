import os
import glob
import json
from sklearn.model_selection import train_test_split


def load_dataset_paths(image_dir, mask_dir, mask_suffixes, image_exts=[".jpg"], mask_exts=[".png"]):
    """
    Párosítja a képeket a maszkokkal, támogatva a többféle kiterjesztést és maszk elnevezést.

    Paraméterek:
    - mask_suffixes: LISTA a lehetséges utótagokról (pl. ["_gt", "_edgemask_3"])
    - image_exts: LISTA a képek lehetséges kiterjesztéseiről (pl. [".jpg", ".tif", ".TIF"])
    - mask_exts: LISTA a maszkok lehetséges kiterjesztéseiről (pl. [".png", ".jpg"])
    """
    image_paths = []
    mask_paths = []

    # Képek begyűjtése az összes megadott kiterjesztéssel
    found_images = []
    for ext in image_exts:
        search_pattern = os.path.join(image_dir, f"*{ext}")
        found_images.extend(glob.glob(search_pattern))

    for img_path in found_images:
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        # Végigpróbáljuk a lehetséges maszk utótagokat és kiterjesztéseket
        mask_found = False
        for suffix in mask_suffixes:
            for m_ext in mask_exts:
                mask_path = os.path.join(mask_dir, f"{base_name}{suffix}{m_ext}")

                if os.path.exists(mask_path):
                    image_paths.append(img_path)
                    mask_paths.append(mask_path)
                    mask_found = True
                    break  # Megvan a megfelelő maszk kiterjesztés
            if mask_found:
                break  # Megvan a megfelelő maszk utótag

        if not mask_found:
            # Csak csendben kihagyjuk, de ha akarod, kiírathatod:
            # print(f"Figyelem: Nem találtam maszkot a következő képhez: {base_name}")
            pass

    return image_paths, mask_paths


if __name__ == "__main__":
    # --- MAPPÁK BEÁLLÍTÁSA ---

    # 1. CASIA adatbázis
    # Itt megadjuk a .jpg és .tif kiterjesztéseket is!
    casia_spliced_dir = "Tp"
    casia_masks_dir = "CASIA 2 Groundtruth"

    casia_imgs, casia_masks = load_dataset_paths(
        casia_spliced_dir,
        casia_masks_dir,
        mask_suffixes=["_gt"],
        image_exts=[".jpg", ".tif", ".TIF", ".jpeg"],
        mask_exts=[".png", ".jpg", ".tif"]
    )

    # 2. Columbia adatbázis
    columbia_spliced_dir = "4cam_splc"
    columbia_masks_dir = "4cam_splc/edgemask"

    columbia_imgs, columbia_masks = load_dataset_paths(
        columbia_spliced_dir,
        columbia_masks_dir,
        mask_suffixes=["_edgemask", "_edgemask_2", "_edgemask_3", ""],
        image_exts=[".jpg", ".tif"],
        mask_exts=[".jpg", ".png"]
    )

    # Adatok összefűzése vegyes tanításhoz
    all_imgs = casia_imgs + columbia_imgs
    all_masks = casia_masks + columbia_masks

    print(f"\n--- Adatbázis statisztika ---")
    print(f"Képek a CASIA mappából: {len(casia_imgs)}")
    print(f"Képek a Columbia mappából: {len(columbia_imgs)}")
    print(f"Összes sikeresen párosított kép-maszk: {len(all_imgs)}")

    if len(all_imgs) > 0:
        # 80% Train, 20% Test szétválasztás
        X_train, X_test, y_train, y_test = train_test_split(all_imgs, all_masks, test_size=0.2, random_state=42)

        dataset_splits = {
            "train_images": X_train,
            "train_masks": y_train,
            "test_images": X_test,
            "test_masks": y_test
        }

        with open("dataset_splits.json", "w") as f:
            json.dump(dataset_splits, f, indent=4)

        print("Adatbázis felosztva és elmentve a 'dataset_splits.json' fájlba.")
        print(f"Tanító halmaz: {len(X_train)} kép, Teszt halmaz: {len(X_test)} kép.")
    else:
        print("\nHiba: Nem találtam megfelelő kép-maszk párokat. Ellenőrizd a mappák elérését!")