import cv2
import numpy as np
import warnings
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA

# Figyelmeztetések elrejtése (pl. ha a variancia 0)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def get_srm_kernels():
    """Visszaadja a cikkben szereplő 5 SRM magas-passzív szűrőt (HSB, VSB/VBH, RF0, RF1, RF2)."""
    hsb = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, -3, 3, -1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32) / 3.0

    # A cikkben VSB néven említik, az ábrán VBH (Vertical)
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
    """Létrehozza a zajképeket mind az 5 szűrővel a gyorsabb feldolgozás érdekében."""
    img_float = image.astype(np.float32)
    kernels = get_srm_kernels()
    noise_maps = [cv2.filter2D(img_float, -1, kernel) for kernel in kernels]
    return noise_maps


def compute_block_features(noise_block):
    """Kiszámítja a 12-dimenziós PCA alapú jellemzővektort egyetlen blokkra."""
    pixels = noise_block.reshape(-1, 3)

    if np.std(pixels) == 0:
        return np.zeros(12)

    pca = PCA(n_components=3)
    try:
        pca.fit(pixels)
        transformed = pca.transform(pixels)

        var = np.var(transformed, axis=0)
        sk = np.nan_to_num(skew(transformed, axis=0, bias=False))
        kurt = np.nan_to_num(kurtosis(transformed, axis=0, bias=False))

        e_d = np.clip(pca.components_[0], -1.0, 1.0)
        theta = np.arccos(e_d)

        return np.concatenate([var, sk, kurt, theta])
    except Exception:
        # Ha bármi hiba történik a PCA során (pl. túl homogén blokk), nullákkal térünk vissza
        return np.zeros(12)


def extract_features_from_image(image_path, mask_path=None, block_size=32, step=8):
    """
    Kinyeri a blokkok jellemzőit.
    A cikk alapján mind az 5 szűrő jellemzőit összefűzi, így egy blokk = 60 dimenzió.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, None, None, None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = None
    if mask_path:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Kiszámoljuk mind az 5 zajtérképet egyszer a teljes képre
    noise_maps = extract_all_noise_maps(img)
    h, w, _ = img.shape

    features, labels, positions = [], [], []

    for y in range(0, h - block_size + 1, step):
        for x in range(0, w - block_size + 1, step):
            block_feature_vector = []

            # Végigmegyünk az 5 különböző zajtérképen, és mindenhol kiszámoljuk a 12 jellemzőt
            for noise_img in noise_maps:
                noise_block = noise_img[y:y + block_size, x:x + block_size]
                feat12 = compute_block_features(noise_block)
                block_feature_vector.extend(feat12)

            features.append(block_feature_vector)  # Ez itt már 60 elemű lesz!
            positions.append((y, x))

            if mask is not None:
                mask_block = mask[y:y + block_size, x:x + block_size]
                beta = np.sum(mask_block == 255) / (block_size * block_size)
                # Ha a blokk legalább 10%-ban tartalmaz manipulált pixelt, tekintse manipuláltnak
                if beta > 0.1:
                    labels.append(1)  # Tampered (Manipulált)
                else:
                    labels.append(0)  # Authentic (Eredeti)

    return np.array(features), np.array(labels) if mask is not None else None, positions, (h, w)