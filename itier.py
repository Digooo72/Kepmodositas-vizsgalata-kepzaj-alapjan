import cv2
import numpy as np
import warnings
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# 3-OSZTÁLYOS SÉMA — blokk-osztály konstansok (checklist #1)
#   A cikk eredetileg csak Uniform/Border (2 osztály) blokkokat tanít.
#   Itt bevezetjük a harmadik, "Spliced" osztályt: azok a blokkok, amik
#   TELJES EGÉSZÉBEN a beillesztett (hamisított) régión belül vannak.
#   Ez rugalmasabbá teszi a sémát: a modell nemcsak a határvonalat,
#   hanem a hamisított terület belsejét is felismerheti.
# ============================================================
CLASS_UNIFORM = 0   # Homogén terület (host / background, β = 0.0)
CLASS_BORDER  = 1   # Határvonal — host és spliced terület közötti blokk
CLASS_SPLICED = 2   # Teljesen a beillesztett (spliced) régión belül (β = 1.0)

# ============================================================
# JELLEMZŐKINYERÉS SZŰRŐNKÉNTI SZÉTVÁLASZTÁSA (checklist #7)
#   A cikk Table 1-je az 5 SRM szűrőt (HSB, VSB, RF0, RF1, RF2) KÜLÖN-KÜLÖN
#   is kiértékeli, majd a végén konkatenálva ("Concatenated" oszlop) —
#   hogy megmutassa, melyik szűrő önmagában mennyire informatív, és hogy a
#   kombináció jobb-e mindegyiknél külön-külön.
#   Korábban a kód mindig azonnal konkatenálta mind az 5-öt (60 dim), nem
#   volt kapcsoló egyetlen szűrő önálló kiértékelésére. Az alábbi
#   ALL_FILTER_NAMES sorrend rögzíti a névhez tartozó indexet, így bármelyik
#   függvény "filter_names" paraméterrel kérhet tetszőleges RÉSZHALMAZT
#   (akár egyetlen szűrőt is) a névvel hivatkozva.
# ============================================================
ALL_FILTER_NAMES = ["HSB", "VSB", "RF0", "RF1", "RF2"]
N_FEATURES_PER_FILTER = 12   # Eq. 3: 3 variancia + 3 ferdeség + 3 csúcsosság + 3 szög


def get_srm_kernels():
    """
    A cikkben szereplő 5 SRM magas-passzív szűrő (Fig. 2).
    HSB, VBH, RF0, RF1, RF2 — változatlanul helyes.

    JELLEMZŐKINYERÉS SZÉTVÁLASZTÁSA (checklist #7): a szűrők most egy
    NÉV → kernel szótárban vannak, hogy extract_all_noise_maps() és
    extract_features_from_image() név szerint tudjon közülük válogatni
    (pl. csak "HSB", vagy csak ["RF1","RF2"], stb.) — nem csak az
    összes egyszerre konkatenálva.
    """
    hsb = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, -3, 3, -1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32) / 3.0

    vsb = np.array([
        [0, 0, 1, 0, 0],
        [0, 0, -3, 0, 0],
        [0, 0, 3, 0, 0],
        [0, 0, -1, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32) / 3.0

    rf0 = np.array([
        [0, 0, 0, 0, 0],
        [0, -1, 2, -1, 0],
        [0, 2, -4, 2, 0],
        [0, -1, 2, -1, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32) / 4.0

    rf1 = np.array([
        [-1, 2, -2, 2, -1],
        [2, -6, 8, -6, 2],
        [-2, 8, -12, 8, -2],
        [2, -6, 8, -6, 2],
        [-1, 2, -2, 2, -1]
    ], dtype=np.float32) / 12.0

    rf2 = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, -2, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32) / 2.0

    return {"HSB": hsb, "VSB": vsb, "RF0": rf0, "RF1": rf1, "RF2": rf2}


def _resolve_filter_names(filter_names):
    """
    Segédfüggvény: normalizálja a filter_names paramétert.
      None            → mind az 5 szűrő (visszafelé kompatibilis, változatlan
                         viselkedés — a régi kód mindig ezt tette)
      "HSB"           → egyetlen szűrő (str) → [szűrő] listává alakítva
      ["RF1","RF2"]   → tetszőleges részhalmaz, MEGŐRZI a megadott sorrendet
    Érvénytelen névnél ValueError-t dob, hogy elgépelés esetén ne csendben
    fusson le hibás (üres/rossz dimenziójú) jellemzőkinyerés.
    """
    if filter_names is None:
        return list(ALL_FILTER_NAMES)
    if isinstance(filter_names, str):
        filter_names = [filter_names]
    unknown = [n for n in filter_names if n not in ALL_FILTER_NAMES]
    if unknown:
        raise ValueError(
            f"Ismeretlen szűrőnév: {unknown}. Érvényes nevek: {ALL_FILTER_NAMES}")
    return list(filter_names)


def extract_all_noise_maps(image, filter_names=None):
    """
    Kiszámolja a zajképeket a kiválasztott SRM szűrő(k)kel.

    filter_names : None (mind az 5, régi viselkedés) | str (egy szűrő,
                   pl. "HSB") | list[str] (részhalmaz, pl. ["RF1","RF2"])

    Visszatérés: (noise_maps, used_names) — a noise_maps lista sorrendje
    MINDIG megegyezik used_names sorrendjével, hogy compute_block_features
    kimenete és a végső jellemzővektor oszlopai egyértelműen megfeleltethetők
    legyenek az egyes szűrőknek.
    """
    used_names = _resolve_filter_names(filter_names)
    img_float = image.astype(np.float32)
    kernels = get_srm_kernels()
    noise_maps = [cv2.filter2D(img_float, -1, kernels[name]) for name in used_names]
    return noise_maps, used_names


def compute_block_features(noise_block):
    """
    Egy blokkból kinyeri a 12 jellemzőt (Eq. 3 a cikkből):
      - 3 variancia  (σ²(X), σ²(Y), σ²(Z))
      - 3 ferdeség   (γ(X),  γ(Y),  γ(Z))
      - 3 csúcsosság (κ(X),  κ(Y),  κ(Z))
      - 3 szögjellemző (θ(X), θ(Y), θ(Z)) — az első főtengely és az RGB tengelyek szöge

    Visszatérés: 12 elemű numpy vektor (nullvektor ha a blokk konstans)
    """
    pixels = noise_block.reshape(-1, 3)

    if np.std(pixels) == 0:
        return np.zeros(12)

    pca = PCA(n_components=3)
    try:
        pca.fit(pixels)
        transformed = pca.transform(pixels)

        var  = np.var(transformed, axis=0)
        sk   = np.nan_to_num(skew(transformed, axis=0, bias=False))
        kurt = np.nan_to_num(kurtosis(transformed, axis=0, bias=False))

        # θ(d) = arccos(e_d) — az első főkomponens iránya az RGB térben (Eq. 2)
        e_d = np.clip(pca.components_[0], -1.0, 1.0)
        theta = np.arccos(e_d)

        return np.concatenate([var, sk, kurt, theta])
    except Exception:
        return np.zeros(12)


def load_binary_mask(mask_path):
    """
    Betölti a maszkot és bináris képpé alakítja.

    CASIA groundtruth: szürkeárnyalatos bináris maszk → egyszerű küszöbölés.

    Columbia 3-osztályos maszk (piros=háttér, zöld=hamisított, kék=határvonal):
    HSV-alapú piros háttér detekció — robusztusabb az RGB alapú megközelítésnél,
    mert a HSV-ban a piros szín jól elkülöníthető két tartományban (0-10° és 170-180°).
    GT = spliced régió + határvonal = minden ami NEM piros háttér.
    """
    mask_bgr = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    if mask_bgr is None:
        return None

    # Ha bináris maszk (max 4 egyedi szín → fekete-fehér, CASIA típus)
    unique_colors = len(np.unique(mask_bgr.reshape(-1, mask_bgr.shape[2]), axis=0))
    if unique_colors <= 4:
        gray = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return binary

    # Columbia 3-osztályos maszk: HSV alapon azonosítjuk a piros hátteret.
    # A HSV színtér előnye: a piros szín eltolódása megvilágítástól független.
    # Piros HSV tartományok: [0-10°] és [170-180°]
    hsv = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, np.array([0,   50, 50]), np.array([10,  255, 255]))
    red2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
    red_background = cv2.bitwise_or(red1, red2)

    # GT = minden ami NEM piros háttér (spliced régió + határvonal együtt)
    binary = cv2.bitwise_not(red_background)
    return binary
def extract_features_from_image(image_path, mask_path=None, block_size=32, step=8,
                                 filter_names=None):
    """
    Kép jellemzőinek kinyerése csúszó ablakkal.

    FIX #1: A StandardScaler-t KIVETTÜK innen. A skálázást a trainmodel.py-ban
             kell elvégezni a teljes tanítóhalmazon, majd menteni.
             Lásd: trainmodel_fixed.py

    FIX #2: β küszöb javítva [0.3, 0.7] → [0.4, 0.6] a cikk szerint (Section 3.4, 4.1)

    3-OSZTÁLYOS SÉMA: a korábbi bináris (Uniform/Border) címkézés helyett
    három osztályt különböztetünk meg (lásd CLASS_* konstansok):
      CLASS_UNIFORM (0) : β == 0.0            → teljesen host/background
      CLASS_BORDER  (1) : β ∈ [0.4, 0.6]       → határvonal (cikk szerint, változatlan)
      CLASS_SPLICED (2) : β == 1.0            → teljesen a hamisított régión belül
      (minden más β-érték kétértelmű részleges blokk → kihagyjuk, mint eddig)

    JELLEMZŐKINYERÉS SZÉTVÁLASZTÁSA (checklist #7, ÚJ paraméter):
      filter_names : None (alapértelmezés, mind az 5 szűrő konkatenálva,
                     60 dim — 100% visszafelé kompatibilis a régi hívásokkal)
                     | str, pl. "HSB" (EGYETLEN szűrő önállóan, 12 dim)
                     | list[str], pl. ["RF1","RF2"] (részhalmaz, 12×N dim)
                     Ezzel reprodukálható a cikk Table 1-jének ablációja:
                     minden szűrő KÜLÖN-KÜLÖN kiértékelhető, majd a
                     "Concatenated" (mind az 5) esettel összehasonlítható —
                     lásd evaluate_filter_ablation.py.

    Visszatérés:
      features  : (N, 12×len(filter_names)) nyers jellemzőmátrix
                  (alapértelmezésben 60 = 5 szűrő × 12 jellemző)
      labels    : (N,) tömb {0,1,2} értékekkel (Uniform/Border/Spliced), vagy None ha nincs maszk
      positions : lista (y, x) pozíciókkal
      (h, w)    : kép mérete
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, None, None, None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    binary_mask = None
    if mask_path:
        binary_mask = load_binary_mask(mask_path)

    return extract_features_from_array(img, binary_mask=binary_mask,
                                        block_size=block_size, step=step,
                                        filter_names=filter_names)


def extract_features_from_array(img_rgb, binary_mask=None, block_size=32, step=8,
                                 filter_names=None):
    """
    ROBUSZTUSSÁGI VIZSGÁLAT (checklist #17, ÚJ függvény): ugyanaz a blokk-
    feldolgozó logika, mint extract_features_from_image()-ben, de már egy
    MEMÓRIÁBAN LÉVŐ RGB képtömbből indul, fájlba írás/visszaolvasás nélkül.

    Ez teszi lehetővé, hogy pl. egy JPEG-újratömörített vagy zajjal terhelt
    kép variánsát közvetlenül, lemez-kerülővel lehessen feldolgozni — lásd
    evaluate_robustness.py: apply_jpeg_compression()/apply_uniform_noise()
    a cv2.imencode/imdecode-dal memóriában állítja elő a torzított képet,
    majd ezt a függvényt hívja rá.

    extract_features_from_image() maga is ezt hívja a lemezről beolvasott
    kép feldolgozásához — tehát ez a KÖZÖS, egyetlen implementáció, nincs
    kódduplikáció a két belépési pont között.

    Paraméterek:
      img_rgb      : (H, W, 3) uint8 RGB kép (NEM BGR — a hívó felelőssége
                     a színcsatorna-sorrend biztosítása, ahogy
                     extract_features_from_image is cvtColor-oz hívás előtt)
      binary_mask  : (H, W) bináris maszk (255=hamisított) vagy None
                     (None → nincs címkézés, csak jellemzők, mint egy
                     authentic/maszk nélküli képnél)
      block_size, step, filter_names : ugyanaz, mint extract_features_from_image-nél

    Visszatérés: ugyanaz a négyes, mint extract_features_from_image().
    """
    noise_maps, used_names = extract_all_noise_maps(img_rgb, filter_names=filter_names)
    n_dim = N_FEATURES_PER_FILTER * len(used_names)
    h, w = img_rgb.shape[:2]

    features, labels, positions = [], [], []

    for y in range(0, h - block_size + 1, step):
        for x in range(0, w - block_size + 1, step):
            block_feature_vector = []

            for noise_img in noise_maps:
                noise_block = noise_img[y:y + block_size, x:x + block_size]
                feat12 = compute_block_features(noise_block)
                block_feature_vector.extend(feat12)

            features.append(block_feature_vector)
            positions.append((y, x))

            if binary_mask is not None:
                mask_block = binary_mask[y:y + block_size, x:x + block_size]
                beta = np.sum(mask_block == 255) / (block_size * block_size)

                # FIX #2 + 3-OSZTÁLYOS SÉMA: β ∈ [0.4, 0.6] → "Border" (változatlan),
                # β == 0.0 → "Uniform", β == 1.0 → ÚJ "Spliced" osztály.
                if 0.4 <= beta <= 0.6:
                    labels.append(CLASS_BORDER)    # Határvonal
                elif beta == 0.0:
                    labels.append(CLASS_UNIFORM)   # Homogén (host) terület
                elif beta == 1.0:
                    labels.append(CLASS_SPLICED)   # Teljesen a hamisított régión belül
                else:
                    # Részlegesen érintett, de nem egyértelműen egyik osztályba
                    # sem eső blokk — kihagyjuk (mint eddig)
                    features.pop()
                    positions.pop()

    if features:
        features = np.array(features, dtype=np.float32)
    else:
        # Üres kép/nincs blokk esetén is helyes (N_dim oszlopos) alakú tömböt
        # adjunk vissza, ne (0,)-t — így a hívó kód vstack-je nem törik el
        # akkor sem, ha az adott szűrő-részhalmazzal 0 blokk keletkezett.
        features = np.zeros((0, n_dim), dtype=np.float32)
    # NaN/Inf csere nullával (degenerate blokkok: teljesen egyszínű, nulla variancia)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # FONTOS: NEM standardizálunk itt! A scaler a trainmodel_fixed.py-ban kerül
    # illesztésre és mentésre, majd evaluate_model_fixed.py-ban betöltésre.

    return features, np.array(labels, dtype=np.int32) if binary_mask is not None else None, positions, (h, w)