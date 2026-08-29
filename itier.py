import cv2
import numpy as np
import warnings
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=RuntimeWarning)


def get_srm_kernels():
    """
    A cikkben szereplő 5 SRM magas-passzív szűrő (Fig. 2).
    HSB, VBH, RF0, RF1, RF2 — változatlanul helyes.
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

    return [hsb, vsb, rf0, rf1, rf2]


def extract_all_noise_maps(image):
    """5 SRM szűrővel kiszámolja a zajképeket."""
    img_float = image.astype(np.float32)
    kernels = get_srm_kernels()
    noise_maps = [cv2.filter2D(img_float, -1, kernel) for kernel in kernels]
    return noise_maps


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
def extract_features_from_image(image_path, mask_path=None, block_size=32, step=8):
    """
    Kép jellemzőinek kinyerése csúszó ablakkal.

    FIX #1: A StandardScaler-t KIVETTÜK innen. A skálázást a trainmodel.py-ban
             kell elvégezni a teljes tanítóhalmazon, majd menteni.
             Lásd: trainmodel_fixed.py

    FIX #2: β küszöb javítva [0.3, 0.7] → [0.4, 0.6] a cikk szerint (Section 3.4, 4.1)

    Visszatérés:
      features  : (N, 60) nyers jellemzőmátrix (5 szűrő × 12 jellemző)
      labels    : (N,) bináris tömb, vagy None ha nincs maszk
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

    noise_maps = extract_all_noise_maps(img)
    h, w, _ = img.shape

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

                # FIX #2: β ∈ [0.4, 0.6] → "Border" blokk (cikk Section 3.4 és 4.1)
                if 0.4 <= beta <= 0.6:
                    labels.append(1)   # Határvonal ("Border")
                elif beta == 0.0:
                    labels.append(0)   # Homogén terület ("Uniform")
                else:
                    # Részlegesen érintett blokk, de nem határon — kihagyjuk
                    features.pop()
                    positions.pop()

    features = np.array(features, dtype=np.float32)
    # NaN/Inf csere nullával (degenerate blokkok: teljesen egyszínű, nulla variancia)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # FONTOS: NEM standardizálunk itt! A scaler a trainmodel_fixed.py-ban kerül
    # illesztésre és mentésre, majd evaluate_model_fixed.py-ban betöltésre.

    return features, np.array(labels, dtype=np.int32) if binary_mask is not None else None, positions, (h, w)