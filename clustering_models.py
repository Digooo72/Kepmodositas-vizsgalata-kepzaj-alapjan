"""
clustering_models.py

Két klaszterező osztályozó az Itier et al. (2021) keretrendszerbe:

  GMMClassifier        — Gaussian Mixture Model (standard EM illesztés)
  BayesianEMClassifier — Bayesian Gaussian Mixture (Variational EM,
                         automatikus komponensszám-meghatározás)

Mindkét osztály sklearn-kompatibilis (BaseEstimator + ClassifierMixin),
ezért cross_val_score, GridSearchCV, joblib.dump/load mind működik velük.

─── Elméleti háttér ────────────────────────────────────────────────────────────

  K-Means (kemény hozzárendelés, eddigi 3. modell):
    Minden mintát egyetlen klaszterhez rendelünk (argmin L2-távolság).
    A klaszterközépponttól való távolság alapján dönt — minden klaszter
    gömbölyű Voronoi-cellának felel meg.

  GMM / EM (puha hozzárendelés — GMMClassifier):
    Minden minta az összes GMM-komponenshez tartozhat valamilyen
    γ_k(x) = P(z=k | x) valószínűséggel (felelősségi együttható).

    EM két lépése ismétlődik konvergenciáig:
      E-lépés: γ_k(x_i) = π_k · N(x_i; μ_k, Σ_k) / Σ_j π_j · N(...)
      M-lépés: μ_k, Σ_k, π_k frissítése a várható log-likelihood
               maximalizálásával:
               Q(θ) = Σ_i Σ_k γ_k(x_i) · log[π_k · N(x_i; μ_k, Σ_k)]

    Előnye K-Means-szel szemben:
      - Teljes kovariancia-mátrix → elliptikus klaszterformák
      - Puha (valószínűségi) hozzárendelés → közvetlen predict_proba()
      - A kovariancia struktúrája illeszthető (full / tied / diag / spherical)

  Bayesian GMM / Variational EM (BayesianEMClassifier):
    Dirichlet-folyamat prior a komponenssúlyokon:
      π ~ Dirichlet(α_0 / K, ..., α_0 / K)   (K komponens esetén)

    A Variational EM az ELBO-t (Evidence Lower BOund) maximalizálja:
      ELBO = E_q[log p(X, Z, π, μ, Λ)] - E_q[log q(Z, π, μ, Λ)]

    Hatás: a "felesleges" komponensek súlya automatikusan → 0,
    így a modell maga határozza meg a hatékony klaszterszámot.
    Kevésbé hajlamos overfittingre mint a standard GMM.

─── Osztályok hozzárendelése (mind a kettőnél) ─────────────────────────────────
  3-OSZTÁLYOS SÉMA (checklist #1): a leképezés N osztályra általánosítva.
  Eredetileg: "Uniform" (0) vs "Border" (1) blokkok (bináris).
  Most: "Uniform" (0) / "Border" (1) / "Spliced" (2) — de a kód tetszőleges
  osztályszámmal működik (self.classes_ = np.unique(y) alapján), tehát
  visszafelé kompatibilis a 2-osztályos esettel is.

  A GMM/BayesEM önmagában felügyelet nélküli → fit() után minden komponenst
  a TÖBBSÉGI tanítócímke alapján rendelünk egy osztályhoz:
    class_rate_k(c) = (c-osztályú minták száma a k-adik komponensben) /
                       (összes minta a k-adik komponensben)
  A k-adik komponens osztálya = argmax_c class_rate_k(c).
  predict_proba()[:, c] = Σ_{k: k-adik komponens → c osztály} γ_k(x)

  Megjegyzés: legalább annyi komponens kell (n_components >= osztályok száma),
  hogy minden osztálynak juthasson legalább egy komponens — 3 osztálynál
  ezért n_components alsó határa min. 3 legyen (lásd trainmodel_seven.py grid).
────────────────────────────────────────────────────────────────────────────────
"""

import warnings
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ===========================================================================
# 1. GMMClassifier — standard EM illesztés
# ===========================================================================

class GMMClassifier(BaseEstimator, ClassifierMixin):
    """
    Gaussian Mixture Model alapú, sklearn-kompatibilis osztályozó.

    A modell önmagában felügyelet nélküli (GaussianMixture), de fit(X, y)
    meghíváskor a tanítócímkék alapján minden komponenst egy osztályhoz
    rendelünk → predict_proba() valódi Border-valószínűséget ad vissza.

    Paraméterek
    -----------
    n_components : int, default=2
        GMM-komponensek száma. 2 = közvetlen Border/Uniform megfeleltetés.
        Nagyobb értékkel finomabb sűrűségmodell, de lassabb illesztés.
    covariance_type : {'full', 'tied', 'diag', 'spherical'}, default='full'
        'full'      : minden komponensnek saját teljes kovariancia-mátrixa
        'tied'      : közös kovariancia-mátrix minden komponensre (gyorsabb)
        'diag'      : diagonális kovariancia (ált. jobb mint K-Means)
        'spherical' : izotropikus (egyetlen szórás / komponens)
    max_iter : int, default=200
        EM iterációk maximális száma.
    n_init : int, default=10
        Véletlen újraindítások száma; a legjobb log-likelihood-ot tartja meg.
    tol : float, default=1e-4
        Konvergencia-kritérium (log-likelihood-változás küszöbe).
    reg_covar : float, default=1e-6
        Regularizáció a kovariancia-mátrix főátlójához (numerikus stabilitás).
    random_state : int, default=42
        Reprodukálhatóság.
    """

    def __init__(self, n_components=2, covariance_type='full',
                 max_iter=200, n_init=10, tol=1e-4,
                 reg_covar=1e-6, random_state=42):
        self.n_components      = n_components
        self.covariance_type   = covariance_type
        self.max_iter          = max_iter
        self.n_init            = n_init
        self.tol               = tol
        self.reg_covar         = reg_covar
        self.random_state      = random_state

    # ------------------------------------------------------------------
    def fit(self, X, y):
        """
        1. GaussianMixture illesztése X-en (felügyelet nélkül, y nélkül).
        2. Kemény komponens-hozzárendelés a tanítóhalmazon.
        3. Minden komponenshez a TÖBBSÉGI osztály meghatározása (N-osztályos)
           → osztály-leképezés.
        """
        X, y = check_X_y(X, y)
        X    = np.asarray(X, dtype=np.float64)
        y    = np.asarray(y, dtype=np.int32)
        self.classes_ = np.unique(y)          # pl. [0,1] vagy [0,1,2]
        n_classes     = len(self.classes_)

        # ── GMM illesztés ──────────────────────────────────────────────
        self.gmm_ = GaussianMixture(
            n_components    = self.n_components,
            covariance_type = self.covariance_type,
            max_iter        = self.max_iter,
            n_init          = self.n_init,
            tol             = self.tol,
            reg_covar       = self.reg_covar,
            random_state    = self.random_state,
        )
        self.gmm_.fit(X)

        # ── Komponensek → osztályok leképezése (tanítócímkékkel) ────────
        hard_labels = self.gmm_.predict(X)   # argmax(γ_k) minden mintára

        # class_rates[k, c] = a k-adik komponensbe eső minták hány %-a c osztályú
        class_rates = np.zeros((self.n_components, n_classes), dtype=np.float64)
        for k in range(self.n_components):
            mask = hard_labels == k
            if mask.sum() > 0:
                for ci, c in enumerate(self.classes_):
                    class_rates[k, ci] = np.mean(y[mask] == c)
            # Ha üres komponens: minden sor 0 → a biztonsági mentés lentebb korrigál

        # A k-adik komponens osztálya = a többségi (legmagasabb arányú) osztály
        component_to_class_idx = np.argmax(class_rates, axis=1)

        # Biztonsági mentés: minden osztálynak jusson legalább egy komponens
        for ci in range(n_classes):
            if not np.any(component_to_class_idx == ci):
                best_k = int(np.argmax(class_rates[:, ci]))
                component_to_class_idx[best_k] = ci

        self.component_to_class_idx_ = component_to_class_idx  # index self.classes_-be
        self.class_rates_            = class_rates

        # ── Visszafelé-kompatibilis "Border" attribútumok (naplózáshoz) ──
        if 1 in self.classes_:
            border_idx = int(np.where(self.classes_ == 1)[0][0])
            self.border_rates_     = class_rates[:, border_idx]
            self.border_component_ = int(np.argmax(self.border_rates_))

        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X):
        """
        Visszatérés: (N, n_classes) tömb — oszloponként self.classes_ sorrendben
        (2 osztálynál [Uniform, Border], 3-nál [Uniform, Border, Spliced]).

        P(osztály c | x) = Σ_{k: k-adik komponens → c osztály} γ_k(x)
        """
        check_is_fitted(self, ['gmm_', 'component_to_class_idx_'])
        X = check_array(X)
        X = np.asarray(X, dtype=np.float64)

        # GMM felelősségi mátrix: shape (N, n_components)
        resp = self.gmm_.predict_proba(X)

        n_classes = len(self.classes_)
        proba = np.zeros((X.shape[0], n_classes), dtype=np.float64)
        for k in range(self.n_components):
            ci = self.component_to_class_idx_[k]
            proba[:, ci] += resp[:, k]

        row_sums = proba.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return proba / row_sums

    # ------------------------------------------------------------------
    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


# ===========================================================================
# 2. BayesianEMClassifier — Variational EM (Dirichlet-prior)
# ===========================================================================

class BayesianEMClassifier(BaseEstimator, ClassifierMixin):
    """
    Bayesian Gaussian Mixture Model (Variational EM) alapú osztályozó.

    A BayesianGaussianMixture Dirichlet-folyamat prior segítségével
    automatikusan meghatározza a hatékony komponensszámot:
      - n_components = felső korlát (az EM-ből legtöbbször kevesebbet használ)
      - A kis súlyú komponensek 0-ra csökkennek → hatékony modellszelekció

    Standard EM (GMMClassifier) vs. Variational EM (BayesianEMClassifier):
      Standard: L(θ) = log p(X | θ) → hajlamos a legtöbb komponenst kitölteni
      Variational: ELBO = E_q[log p(X,Z,θ)] - E_q[log q(Z,θ)] → regularizált

    Paraméterek
    -----------
    n_components : int, default=8
        Komponensek maximális száma (a tényleges aktívak általában kevesebben).
    covariance_type : {'full', 'tied', 'diag', 'spherical'}, default='full'
    max_iter : int, default=300
        Variational EM iterációk maximuma.
    n_init : int, default=3
        Véletlen újraindítások száma.
    tol : float, default=1e-4
        Konvergencia-küszöb.
    reg_covar : float, default=1e-6
        Regularizáció.
    weight_concentration_prior_type : str, default='dirichlet_process'
        'dirichlet_process'      : stick-breaking process (erős sparsity)
        'dirichlet_distribution' : szimmetrikus Dirichlet (egyenletesebb)
    weight_concentration_prior : float or None, default=None
        Dirichlet koncentráció. None → sklearn default (1/n_components).
        Kis érték (pl. 0.001) → erősebb sparsity → kevesebb aktív komponens.
        Nagy érték (pl. 10.0) → egyenletesebb eloszlás → több aktív komponens.
    random_state : int, default=42
    """

    def __init__(self, n_components=8, covariance_type='full',
                 max_iter=300, n_init=3, tol=1e-4, reg_covar=1e-6,
                 weight_concentration_prior_type='dirichlet_process',
                 weight_concentration_prior=None,
                 random_state=42):
        self.n_components                    = n_components
        self.covariance_type                 = covariance_type
        self.max_iter                        = max_iter
        self.n_init                          = n_init
        self.tol                             = tol
        self.reg_covar                       = reg_covar
        self.weight_concentration_prior_type = weight_concentration_prior_type
        self.weight_concentration_prior      = weight_concentration_prior
        self.random_state                    = random_state

    # ------------------------------------------------------------------
    def fit(self, X, y):
        """
        1. BayesianGaussianMixture illesztése (Variational EM).
        2. Aktív komponensek azonosítása (weight > 1e-3).
        3. Komponensek → osztályok leképezése tanítócímkékkel (N-osztályos,
           ugyanaz az elv, mint GMMClassifier-nél).
        """
        X, y = check_X_y(X, y)
        X    = np.asarray(X, dtype=np.float64)
        y    = np.asarray(y, dtype=np.int32)
        self.classes_ = np.unique(y)          # pl. [0,1] vagy [0,1,2]
        n_classes     = len(self.classes_)

        # ── BayesianGaussianMixture paraméterösszeállítás ──────────────
        kwargs = dict(
            n_components                    = self.n_components,
            covariance_type                 = self.covariance_type,
            max_iter                        = self.max_iter,
            n_init                          = self.n_init,
            tol                             = self.tol,
            reg_covar                       = self.reg_covar,
            weight_concentration_prior_type = self.weight_concentration_prior_type,
            random_state                    = self.random_state,
        )
        if self.weight_concentration_prior is not None:
            kwargs['weight_concentration_prior'] = self.weight_concentration_prior

        # ── Variational EM illesztés ───────────────────────────────────
        self.bgmm_ = BayesianGaussianMixture(**kwargs)
        self.bgmm_.fit(X)

        # ── Aktív komponensek (nem-nulla súlyú) ────────────────────────
        weights = self.bgmm_.weights_
        self.n_active_components_ = int(np.sum(weights > 1e-3))

        # ── Komponensek → osztályok leképezése (többségi szavazat) ─────
        hard_labels = self.bgmm_.predict(X)
        class_rates = np.zeros((self.n_components, n_classes), dtype=np.float64)

        for k in range(self.n_components):
            mask = hard_labels == k
            if mask.sum() > 0:
                for ci, c in enumerate(self.classes_):
                    class_rates[k, ci] = np.mean(y[mask] == c)

        component_to_class_idx = np.argmax(class_rates, axis=1)

        # Biztonsági mentés: minden osztálynak jusson legalább egy komponens
        for ci in range(n_classes):
            if not np.any(component_to_class_idx == ci):
                best_k = int(np.argmax(class_rates[:, ci]))
                component_to_class_idx[best_k] = ci

        self.component_to_class_idx_ = component_to_class_idx
        self.class_rates_            = class_rates

        # ── Visszafelé-kompatibilis "Border" attribútumok (naplózáshoz) ──
        if 1 in self.classes_:
            border_idx = int(np.where(self.classes_ == 1)[0][0])
            self.border_rates_     = class_rates[:, border_idx]
            self.border_component_ = int(np.argmax(self.border_rates_))

        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X):
        """
        Variational EM felelősségi mátrix alapján, N osztályra általánosítva:
          P(osztály c | x) = Σ_{k: k-adik komponens → c osztály} γ_k(x)
        """
        check_is_fitted(self, ['bgmm_', 'component_to_class_idx_'])
        X = check_array(X)
        X = np.asarray(X, dtype=np.float64)

        resp = self.bgmm_.predict_proba(X)

        n_classes = len(self.classes_)
        proba = np.zeros((X.shape[0], n_classes), dtype=np.float64)
        for k in range(self.n_components):
            ci = self.component_to_class_idx_[k]
            proba[:, ci] += resp[:, k]

        row_sums = proba.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return proba / row_sums

    # ------------------------------------------------------------------
    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    # ------------------------------------------------------------------
    @property
    def n_active_components(self):
        """Ténylegesen aktív GMM-komponensek száma (weight > 1e-3)."""
        return getattr(self, 'n_active_components_', self.n_components)