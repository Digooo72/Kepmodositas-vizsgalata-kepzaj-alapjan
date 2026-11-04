"""
itier.py

Központi jellemzőkinyerő és maszkfeldolgozó könyvtár az Itier et al. (2021)
"Color noise correlation-based splicing detection for image forensics"
cikk módszertana alapján, kiegészítve a szakdolgozati továbbfejlesztésekkel:

  1. 3-OSZTÁLYOS SÉMA (checklist #1):
     A cikk eredeti bináris (Uniform vs. Border) blokkcímkézése helyett három
     osztályt különböztetünk meg a GT maszk blokkon belüli aránya (beta) alapján:
       - CLASS_UNIFORM (0): beta == 0.0        (tisztán befogadó / host háttér)
       - CLASS_BORDER  (1): 0.4 <= beta <= 0.6 (vágási határvonal, FIX #2)
       - CLASS_SPLICED (2): beta == 1.0        (teljesen a beillesztett régión belül)

  2. SZÉTVÁLASZTHATÓ SRM SZŰRŐBANK (checklist #7):
     Az 5 darab 5x5-ös SRM magas-passzív szűrő (HSB, VSB, RF0, RF1, RF2)
     tetszőleges részhalmaza lekérhető a `filter_names` paraméterrel (a cikk
     Table 1 szűrő-ablációs vizsgálatához, lásd Evaluate filter ablation.py).

  3. ADATSZIVÁRGÁS-MENTESÍTÉS (FIX #1):
     A jellemzőkinyerés nyers float32 blokkmátrixot ad vissza, nem skáláz
     képenként. A globális StandardScaler a trainmodel_seven.py-ban illeszkedik
     a teljes tanítóhalmazon (itier_scaler.pkl).

  4. MEMÓRIABELI TÖMB-FELDOLGOZÁS (checklist #17):
     Az `extract_features_from_array()` közvetlenül memóriában lévő RGB numpy
     tömbből dolgozik, lehetővé téve a lemezre írás nélküli JPEG és zaj
     robusztussági teszteket (lásd evaluate_robustness.py).
"""

import cv2
import numpy as np
import warnings
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ===========================================================================
# 1. BLOKK-OSZTÁLY KONSTANSOK (3-osztályos séma — checklist #1)
# ===========================================================================
CLASS_UNIFORM = 0   # Homogén terület (host / background, beta = 0.0)
CLASS_BORDER  = 1   # Határvonal — host és spliced terület közötti blokk (beta in [0.4, 0.6])
CLASS_SPLICED = 2   # Teljesen a beillesztett (spliced) régión belül (beta = 1.0)

# ===========================================================================
# 2. SRM SZŰRŐK KONSTANSAI (checklist #7)
# ===========================================================================
ALL_FILTER_NAMES = ["HSB", "VSB", "RF0", "RF1", "RF2"]
N_FEATURES_PER_FILTER = 12   # Cikk Eq. 3: 3 variancia + 3 ferdeség + 3 csúcsosság + 3 szög


def get_srm_kernels():
    """
    Visszaadja az Itier et al. (2021) cikk 2. ábráján (Fig. 2) szereplő
    5 darab 5x5-ös SRM (Spatial Rich Model) magas-passzív szűrőmagot.

    Fizikai szerepük (Cikk Section 3.2, Eq. 1: I_n = I - A):
      Elnyomják a kép alacsony frekvenciás jelenet-tartalmát (A), és kiemelik
      a magas frekvenciás szenzor- és utófeldolgozási zajt (I_n) mindhárom
      RGB csatornán:
        - HSB : 3. rendű vízszintes derivált szűrő (Cozzolino et al. [11])
        - VSB : 3. rendű függőleges derivált szűrő (a cikk Fig. 2b-ben VBH néven)
        - RF0 : 2. rendű, 3x3-as tartójú bidirekcionális szűrő (Zhou et al. [41])
        - RF1 : 4. rendű, 5x5-ös izotróp magas-passzív szűrő (Zhou et al. [41])
        - RF2 : 2. rendű vízszintes Laplace-szűrő (Zhou et al. [41])

    Visszatérés:
      dict[str, np.ndarray] : Név -> (5, 5) float32 konvolúciós mag szótár.
    """
    hsb = np.array([
        [0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0],
        [0,  1, -3,  3, -1],
        [0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0]
    ], dtype=np.float32) / 3.0

    vsb = np.array([
        [0,  0,  1,  0,  0],
        [0,  0, -3,  0,  0],
        [0,  0,  3,  0,  0],
        [0,  0, -1,  0,  0],
        [0,  0,  0,  0,  0]
    ], dtype=np.float32) / 3.0

    rf0 = np.array([
        [0,  0,  0,  0,  0],
        [0, -1,  2, -1,  0],
        [0,  2, -4,  2,  0],
        [0, -1,  2, -1,  0],
        [0,  0,  0,  0,  0]
    ], dtype=np.float32) / 4.0

    rf1 = np.array([
        [-1,  2, -2,  2, -1],
        [ 2, -6,  8, -6,  2],
        [-2,  8, -12, 8, -2],
        [ 2, -6,  8, -6,  2],
        [-1,  2, -2,  2, -1]
    ], dtype=np.float32) / 12.0

    rf2 = np.array([
        [0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0],
        [0,  1, -2,  1,  0],
        [0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0]
    ], dtype=np.float32) / 2.0

    return {"HSB": hsb, "VSB": vsb, "RF0": rf0, "RF1": rf1, "RF2": rf2}


def _resolve_filter_names(filter_names):
    """
    Normalizálja és ellenőrzi a `filter_names` paramétert:
      - None          -> mind az 5 szűrő (["HSB", "VSB", "RF0", "RF1", "RF2"])
      - "HSB" (str)   -> egyelemű lista (["HSB"])
      - ["RF1","RF2"] -> részhalmaz a megadott sorrend megtartásával

    Hibakezelés:
      Üres lista vagy ismeretlen szűrőnév esetén ValueError kivételt dob,
      megelőzve a csendes, hibás dimenziójú jellemzőkinyerést.
    """
    if filter_names is None:
        return list(ALL_FILTER_NAMES)
    if isinstance(filter_names, str):
        filter_names = [filter_names]
    filter_list = list(filter_names)
    if not filter_list:
        raise ValueError("A filter_names lista nem lehet üres!")
    unknown = [n for n in filter_list if n not in ALL_FILTER_NAMES]
    if unknown:
        raise ValueError(
            f"Ismeretlen szűrőnév: {unknown}. Érvényes nevek: {ALL_FILTER_NAMES}")
    return filter_list


def extract_all_noise_maps(image, filter_names=None):
    """
    Kiszámítja a reziduális zajképeket a kiválasztott SRM szűrő(k)kel.
    A cv2.filter2D csatornánként (R, G, B) külön-külön elvégzi a 2D konvolúciót.

    Paraméterek:
      image        : (H, W, 3) RGB képtömb.
      filter_names : None | str | list[str] — használandó SRM szűrők.

    Visszatérés:
      noise_maps : list[np.ndarray] — (H, W, 3) float32 zajtérképek listája.
      used_names : list[str]        — a ténylegesen alkalmazott szűrőnevek.
    """
    used_names = _resolve_filter_names(filter_names)
    img_float = image.astype(np.float32)
    kernels = get_srm_kernels()
    noise_maps = [cv2.filter2D(img_float, -1, kernels[name]) for name in used_names]
    return noise_maps, used_names


def compute_block_features(noise_block):
    """
    Egyetlen (N, N, 3) méretű zajblokkból kinyeri a 12 statisztikai és
    geometriai jellemzőt az Itier et al. (2021) cikk Section 3.3 (Eq. 2-3) alapján:

      1. A blokk (N*N, 3) RGB zajvektorait PCA-val (n_components=3) új,
         ortogonális XYZ koordináta-rendszerbe transzformáljuk.
         - Homogén blokkban (Fig. 3a) az RGB zajok erősen korreláltak ->
           a variancia az első főtengelyre koncentrálódik.
         - Határblokkban (Fig. 3b) két eltérő kamera zajeloszlása keveredik ->
           a PCA szétteríti a két eloszlást, megnőnek a magasabb rendű momentumok.

      2. Kinyert 12 jellemző (Eq. 3):
         - 3 variancia    : sigma^2(X), sigma^2(Y), sigma^2(Z)  (2. centrális momentum)
         - 3 ferdeség     : gamma(X),   gamma(Y),   gamma(Z)    (3. standardizált momentum)
         - 3 csúcsosság   : kappa(X),   kappa(Y),   kappa(Z)    (4. standardizált momentum)
         - 3 forgásszög   : theta(X),   theta(Y),   theta(Z)    (Eq. 2: arccos(e_d), ahol
                            e_d az első főtengely sajátvektora az RGB térben; az sklearn
                            PCA belső svd_flip lépése garantálja a determinisztikus előjelet).

    Visszatérés:
      (12,) alakú float64/float32 numpy vektor (nullvektor, ha a blokk konstans).
    """
    pixels = noise_block.reshape(-1, 3)

    # Ha a blokk teljesen homogén (pl. szaturált fehér/fekete terület, std == 0),
    # a kovarianciamátrix zérus -> nullvektort adunk vissza.
    if np.std(pixels) == 0:
        return np.zeros(N_FEATURES_PER_FILTER)

    pca = PCA(n_components=3)
    try:
        pca.fit(pixels)
        transformed = pca.transform(pixels)

        var  = np.var(transformed, axis=0)
        sk   = np.nan_to_num(skew(transformed, axis=0, bias=False))
        kurt = np.nan_to_num(kurtosis(transformed, axis=0, bias=False))

        # Eq. 2: theta(d) = arccos(e(d)) — az első főkomponens iránya az RGB tengelyekhez képest
        e_d = np.clip(pca.components_[0], -1.0, 1.0)
        theta = np.arccos(e_d)

        return np.concatenate([var, sk, kurt, theta])
    except Exception:
        return np.zeros(N_FEATURES_PER_FILTER)


def load_binary_mask(mask_path):
    """
    Betölti a Ground Truth (GT) maszkot és (H, W) uint8 bináris képpé (0 vagy 255) alakítja.

    Kétféle maszkformátumot kezel automatikusan:
      1. CASIA v2.0 groundtruth (<= 4 egyedi szín):
         Szürkeárnyalatos bináris maszk -> egyszerű küszöbölés 127-nél.
      2. Columbia 3-osztályos színes edgemask (> 4 egyedi szín):
         Piros = befogadó háttér (host), Zöld = beillesztett terület (spliced),
         Kék = határvonal (border).
         A piros hátteret HSV színtérben detektáljuk (két Hue tartományban:
         [0-10°] és [170-180°]), ami robusztus a JPEG-tömörítési árnyalat-eltolódásokra,
         majd negáljuk (cv2.bitwise_not), így a bináris maszkban minden, ami NEM
         piros háttér (azaz a zöld spliced régió + kék határvonal), 255-ös értéket kap.
    """
    mask_bgr = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    if mask_bgr is None:
        return None

    # 1. eset: Fekete-fehér / szürkeárnyalatos bináris maszk (CASIA típus)
    unique_colors = len(np.unique(mask_bgr.reshape(-1, mask_bgr.shape[2]), axis=0))
    if unique_colors <= 4:
        gray = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return binary

    # 2. eset: Columbia 3-osztályos színes maszk -> HSV piros háttér detekció és inverzió
    hsv = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, np.array([0,   50, 50]), np.array([10,  255, 255]))
    red2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
    red_background = cv2.bitwise_or(red1, red2)

    # GT = minden, ami NEM piros háttér (spliced belső régió + határvonal együtt = 255)
    binary = cv2.bitwise_not(red_background)
    return binary


def extract_features_from_image(image_path, mask_path=None, block_size=32, step=8,
                                 filter_names=None):
    """
    Képfájl beolvasása lemezről (BGR -> RGB konverzióval), opcionális GT maszk
    betöltése, majd a csúszóablakos SRM-PCA jellemzőkinyerés meghívása.

    Paraméterek:
      image_path   : A képfájl elérési útja.
      mask_path    : A GT maszk elérési útja (tanításkor megadva; teszteléskor None).
      block_size   : Csúszóablak mérete pixelben (alapértelmezés: 32, cikk Section 4.1).
      step         : Csúszóablak lépésköze pixelben (alapértelmezés: 8, cikk Section 4.1).
      filter_names : Használandó SRM szűrők (None = mind az 5 -> 60 jellemző / blokk).

    Visszatérés:
      features  : (N, 12 * len(filter_names)) float32 nyers jellemzőmátrix.
      labels    : (N,) int32 tömb {0, 1, 2} címkékkel, vagy None ha mask_path=None.
      positions : list[(y, x)] — a blokkok bal felső sarkának koordinátái.
      (h, w)    : A kép magassága és szélessége pixelben.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, None, None, None

    # FONTOS: Az OpenCV BGR sorrendben olvas be, a cikk (és a PCA szögjellemzők)
    # viszont RGB színteret várnak.
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
    Közös csúszóablakos SRM-PCA jellemzőkinyerő függvény, amely közvetlenül
    memóriában lévő (H, W, 3) RGB képtömbből dolgozik (checklist #17).

    Főbb jellemzők és optimalizációk:
      - 3-OSZTÁLYOS CÍMKÉZÉS (checklist #1 + FIX #2):
          beta == 0.0        -> CLASS_UNIFORM (0)
          0.4 <= beta <= 0.6 -> CLASS_BORDER  (1)
          beta == 1.0        -> CLASS_SPLICED (2)
          Minden más köztes beta értékű blokkot kihagyunk a tanításból.
      - FUTÁSIDŐ-OPTIMALIZÁCIÓ:
          Ha `binary_mask` meg van adva (tanítási fázis), először a gyors beta
          arányt számoljuk ki, és a kihagyandó (köztes) blokkoknál azonnal
          továbblépünk (`continue`), így nem fut le rájuk feleslegesen az 5xPCA!
      - ADATSZIVÁRGÁS-MENTESÍTÉS (FIX #1):
          Nem alkalmazunk képenkénti StandardScaler-t; a skálázást globálisan
          a trainmodel_seven.py végzi.
    """
    noise_maps, used_names = extract_all_noise_maps(img_rgb, filter_names=filter_names)
    n_dim = N_FEATURES_PER_FILTER * len(used_names)
    h, w = img_rgb.shape[:2]
    block_area = float(block_size * block_size)

    # Biztonsági méretegyeztetés: ha a maszk mérete eltérne a képétől, átméretezzük
    if binary_mask is not None and binary_mask.shape[:2] != (h, w):
        binary_mask = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    features, labels, positions = [], [], []

    for y in range(0, h - block_size + 1, step):
        for x in range(0, w - block_size + 1, step):
            # --- 1. Ha van GT maszk (tanítás), először címkézzük a blokkot ---
            if binary_mask is not None:
                mask_block = binary_mask[y:y + block_size, x:x + block_size]
                beta = np.sum(mask_block == 255) / block_area

                # FIX #2 + 3-OSZTÁLYOS SÉMA:
                if 0.4 <= beta <= 0.6:
                    lbl = CLASS_BORDER     # Határvonal (host és spliced keveréke)
                elif beta == 0.0:
                    lbl = CLASS_UNIFORM    # Homogén (host) háttérterület
                elif beta == 1.0:
                    lbl = CLASS_SPLICED    # Teljesen a beillesztett régión belül
                else:
                    # Részlegesen érintett, kétértelmű blokk -> PCA nélkül kihagyjuk!
                    continue

                labels.append(lbl)

            # --- 2. PCA alapú 12D jellemzők kinyerése minden kiválasztott szűrőre ---
            block_feature_vector = []
            for noise_img in noise_maps:
                noise_block = noise_img[y:y + block_size, x:x + block_size]
                feat12 = compute_block_features(noise_block)
                block_feature_vector.extend(feat12)

            features.append(block_feature_vector)
            positions.append((y, x))

    if features:
        features = np.array(features, dtype=np.float32)
    else:
        # Üres kép / 0 érvényes blokk esetén is megtartjuk a helyes (0, n_dim) alakot,
        # hogy a hívó oldali np.vstack ne törjön el.
        features = np.zeros((0, n_dim), dtype=np.float32)

    # NaN / Inf csere nullával (degenerált, teljesen egyszínű blokkok védelme)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return (
        features,
        np.array(labels, dtype=np.int32) if binary_mask is not None else None,
        positions,
        (h, w),
    )