import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from itier import extract_features_from_image


def balance_classes(X, y):
    """Kiegyensúlyozza az adathalmazt alul-mintavételezéssel (undersampling)."""
    X_arr = np.array(X)
    y_arr = np.array(y)

    border_idx = np.where(y_arr == 1)[0]
    uniform_idx = np.where(y_arr == 0)[0]

    # Megkeressük a kisebb osztály méretét
    min_size = min(len(border_idx), len(uniform_idx))

    if min_size == 0:
        return np.array([]), np.array([])

    # Véletlenszerűen kiválasztunk 'min_size' darabot mindkét osztályból
    np.random.seed(42)
    border_idx_sampled = np.random.choice(border_idx, min_size, replace=False)
    uniform_idx_sampled = np.random.choice(uniform_idx, min_size, replace=False)

    balanced_idx = np.concatenate([border_idx_sampled, uniform_idx_sampled])
    np.random.shuffle(balanced_idx)  # Összekeverjük a sorrendet

    return X_arr[balanced_idx], y_arr[balanced_idx]


if __name__ == "__main__":
    print("--- Modell Betanítása (Random Forest + GridSearch) ---")

    # ==========================================================
    # KONFIGURÁCIÓ
    # Ha a teljes adatbázison akarod futtatni, írd át erre: MAX_IMAGES = None
    MAX_IMAGES = 100
    # ==========================================================

    with open("dataset_splits.json", "r") as f:
        data = json.load(f)

    train_images = data["train_images"]
    train_masks = data["train_masks"]

    if MAX_IMAGES is not None:
        # Keresünk a te valós mappaneveid ("tp", "4cam") alapján is
        casia_pairs = [(img, mask) for img, mask in zip(train_images, train_masks) if
                       "tp" in img.lower() or "casia" in img.lower()]
        columbia_pairs = [(img, mask) for img, mask in zip(train_images, train_masks) if
                          "4cam" in img.lower() or "columbia" in img.lower()]

        half = MAX_IMAGES // 2
        selected_pairs = casia_pairs[:half] + columbia_pairs[:half]

        # Ha valamiért még így sem találna, akkor egyszerűen csak veszi az első 20-at
        if len(selected_pairs) == 0:
            selected_pairs = list(zip(train_images, train_masks))[:MAX_IMAGES]

        train_images = [p[0] for p in selected_pairs]
        train_masks = [p[1] for p in selected_pairs]
        print(f"Figyelem: Csak {len(train_images)} képen tanítunk a gyors teszt miatt!")
    else:
        print(f"Tanítás az ÖSSZES ({len(train_images)}) képen. Ez eltarthat egy darabig!")

    X_train_raw, y_train_raw = [], []

    for i, (img_p, mask_p) in enumerate(zip(train_images, train_masks)):
        print(f"Kép feldolgozása ({i + 1}/{len(train_images)}): {img_p}")
        feats, labels, _, _ = extract_features_from_image(img_p, mask_p)

        if feats is not None and labels is not None and len(feats) > 0:
            X_train_raw.extend(feats)
            y_train_raw.extend(labels)

    print(
        f"\nJellemzők kinyerve. Eredeti eloszlás -> Uniform(0): {y_train_raw.count(0)}, Border(1): {y_train_raw.count(1)}")

    # 1. Osztályok kiegyensúlyozása a cikk iránymutatása alapján
    X_train, y_train = balance_classes(X_train_raw, y_train_raw)

    if len(X_train) == 0:
        print("Hiba: Nincs elég adat a betanításhoz. Ellenőrizd a maszkokat!")
        exit()

    print(f"Kiegyensúlyozott eloszlás -> Uniform(0): {np.sum(y_train == 0)}, Border(1): {np.sum(y_train == 1)}")

    # 2. Random Forest inicializálása és GridSearch beállítása
    print("\nRandom Forest betanítása és hiperparaméterek keresése (GridSearchCV)...")
    base_rf = RandomForestClassifier(random_state=42)

    # A keresendő paraméterek (finomhangolás)
    param_grid = {
        'n_estimators': [50, 100, 200],  # Hány fa legyen az erdőben
        'max_depth': [10, 20, None],  # Milyen mélyek lehessenek a fák
        'criterion': ['gini', 'entropy']  # Döntési kritérium
    }

    # 3-szoros keresztvalidáció, hogy biztosan robusztus legyen a modell
    grid_search = GridSearchCV(estimator=base_rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

    # Betanítás
    grid_search.fit(X_train, y_train)

    print("\nLegjobb megtalált paraméterek:", grid_search.best_params_)

    # A legjobb modell kimentése
    best_model = grid_search.best_estimator_
    model_filename = "itier_rf_model.pkl"
    joblib.dump(best_model, model_filename)

    print(f"\nA legjobb modell sikeresen elmentve: '{model_filename}'")