"""


FIX #1 (kritikus): A StandardScalert egyszer illesztjük a TELJES tanítóhalmazon,
                   majd elmentjük a modellel együtt. Nem képenként külön.

FIX #2: Kompatibilis az itier_fixed.py-val (β küszöb javítva, nincs per-képes skálázás).
"""

import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from itier import extract_features_from_image


def balance_classes(X, y):
    """
    Osztályok kiegyensúlyozása alul-mintavételezéssel (undersampling).
    A cikk (Section 3.4) kiegyensúlyozott Border/Uniform arányt ír elő.
    """
    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.int32)

    border_idx  = np.where(y_arr == 1)[0]
    uniform_idx = np.where(y_arr == 0)[0]

    min_size = min(len(border_idx), len(uniform_idx))
    if min_size == 0:
        return np.array([]), np.array([])

    np.random.seed(42)
    border_sampled  = np.random.choice(border_idx,  min_size, replace=False)
    uniform_sampled = np.random.choice(uniform_idx, min_size, replace=False)

    balanced_idx = np.concatenate([border_sampled, uniform_sampled])
    np.random.shuffle(balanced_idx)

    return X_arr[balanced_idx], y_arr[balanced_idx]


if __name__ == "__main__":
    print("--- Modell Betanítása (Random Forest + GlobalScaler) ---")

    # ==========================================================
    # KONFIGURÁCIÓ
    MAX_IMAGES = None   # None = teljes adatbázis
    # ==========================================================

    with open("dataset_splits.json", "r") as f:
        data = json.load(f)

    train_images = data["train_images"]
    train_masks  = data["train_masks"]

    if MAX_IMAGES is not None:
        casia_pairs    = [(img, msk) for img, msk in zip(train_images, train_masks)
                          if "tp" in img.lower() or "casia" in img.lower()]
        columbia_pairs = [(img, msk) for img, msk in zip(train_images, train_masks)
                          if "4cam" in img.lower() or "columbia" in img.lower()]

        half = MAX_IMAGES // 2
        selected = casia_pairs[:half] + columbia_pairs[:half]
        if not selected:
            selected = list(zip(train_images, train_masks))[:MAX_IMAGES]

        train_images = [p[0] for p in selected]
        train_masks  = [p[1] for p in selected]
        print(f"Gyors teszt: {len(train_images)} képen tanítunk.")
    else:
        print(f"Teljes tanítóhalmaz: {len(train_images)} kép.")

    # --- 1. NYERS JELLEMZŐK ÖSSZEGYŰJTÉSE ---
    X_raw, y_raw = [], []

    for i, (img_p, mask_p) in enumerate(zip(train_images, train_masks)):
        print(f"Feldolgozás ({i+1}/{len(train_images)}): {img_p}")
        feats, labels, _, _ = extract_features_from_image(img_p, mask_p)

        if feats is not None and labels is not None and len(feats) > 0:
            X_raw.append(feats)
            y_raw.append(labels)

    if not X_raw:
        print("Hiba: Nem sikerült jellemzőket kinyerni!")
        exit()

    X_all = np.vstack(X_raw)
    y_all = np.concatenate(y_raw)

    border_count  = int(np.sum(y_all == 1))
    uniform_count = int(np.sum(y_all == 0))
    print(f"\nNyers eloszlás → Uniform(0): {uniform_count}, Border(1): {border_count}")

    # --- 2. OSZTÁLYOK KIEGYENSÚLYOZÁSA ---
    X_bal, y_bal = balance_classes(X_all, y_all)
    if len(X_bal) == 0:
        print("Hiba: Nincs elég adat a kiegyensúlyozáshoz!")
        exit()
    print(f"Kiegyensúlyozott → Uniform: {int(np.sum(y_bal==0))}, Border: {int(np.sum(y_bal==1))}")

    # --- 3. GLOBÁLIS STANDARDIZÁLÁS (FIX #1) ---
    # A scaler-t EGYSZER illesztjük a teljes kiegyensúlyozott tanítóhalmazra,
    # majd elmentjük, hogy a teszteléskor UGYANAZT tudjuk alkalmazni.
    print("\nGlobális StandardScaler illesztése...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_bal)

    # Scaler mentése
    joblib.dump(scaler, "itier_scaler.pkl")
    print("Scaler elmentve: 'itier_scaler.pkl'")

    # --- 4. RANDOM FOREST + GRIDSEARCH ---
    print("\nRandom Forest betanítása (GridSearchCV)...")
    base_rf = RandomForestClassifier(random_state=42)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth':    [10, 20, None],
        'criterion':    ['gini', 'entropy']
    }

    grid_search = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=2
    )
    grid_search.fit(X_scaled, y_bal)

    print("\nLegjobb paraméterek:", grid_search.best_params_)

    best_model = grid_search.best_estimator_
    joblib.dump(best_model, "itier_rf_model.pkl")
    print("Modell elmentve: 'itier_rf_model.pkl'")
    print("\nKész! Mind a modell, mind a scaler el van mentve.")