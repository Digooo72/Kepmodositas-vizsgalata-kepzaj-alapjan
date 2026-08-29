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
  Az Itier-féle feladat: "Uniform" (U=0) vs "Border" (B=1) blokkok.
  A GMM/BayesEM önmagában felügyelet nélküli → fit() után a komponenseket
  a tanítócímkék alapján rendeljük "Border" vagy "Uniform" osztályhoz:
    border_rate_k = (Border-minták száma a k-adik komponensben) /
                    (összes minta a k-adik komponensben)
  Ha border_rate_k >= 0.5 → a k-adik komponens "Border".
  predict_proba()[:,1] = Σ_{k: Border-komponens} γ_k(x)
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
        3. Minden komponenshez Border-arány számítás → osztály-leképezés.
        """
        X, y = check_X_y(X, y)
        X    = np.asarray(X, dtype=np.float64)
        y    = np.asarray(y, dtype=np.int32)
        self.classes_ = np.array([0, 1])

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

        border_rates = np.zeros(self.n_components, dtype=np.float64)
        for k in range(self.n_components):
            mask = hard_labels == k
            if mask.sum() > 0:
                border_rates[k] = np.mean(y[mask] == 1)
            # Ha üres komponens: border_rate=0 → Uniform

        # Border = border_rate >= 0.5
        self.component_to_class_ = (border_rates >= 0.5).astype(np.int32)

        # Biztonsági mentés: ha véletlenül minden komponens azonos osztályra esne
        if self.component_to_class_.sum() == 0:
            # Nincs Border-komponens → a leginkább Border a legmagasabb arányú
            self.component_to_class_[np.argmax(border_rates)] = 1
        if self.component_to_class_.sum() == self.n_components:
            # Minden Border → a legkevésbé Border legyen Uniform
            self.component_to_class_[np.argmin(border_rates)] = 0

        # Fő Border-komponens indexe (naplózáshoz / inspectionhoz)
        self.border_component_ = int(np.argmax(border_rates))
        self.border_rates_     = border_rates

        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X):
        """
        Visszatérés: (N, 2) tömb — [:, 0] = Uniform valószínűség,
                                   [:, 1] = Border valószínűség.

        Border-valószínűség = az összes Border-komponens felelősségének összege:
          P(Border | x) = Σ_{k: Border-komponens} γ_k(x)
        """
        check_is_fitted(self, ['gmm_', 'component_to_class_'])
        X = check_array(X)
        X = np.asarray(X, dtype=np.float64)

        # GMM felelősségi mátrix: shape (N, n_components)
        resp = self.gmm_.predict_proba(X)

        # Összegzés a Border-komponensekre
        border_mask  = (self.component_to_class_ == 1)
        border_prob  = np.clip(resp[:, border_mask].sum(axis=1), 0.0, 1.0)

        return np.column_stack([1.0 - border_prob, border_prob])

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
        3. Komponensek → osztályok leképezése tanítócímkékkel.
        """
        X, y = check_X_y(X, y)
        X    = np.asarray(X, dtype=np.float64)
        y    = np.asarray(y, dtype=np.int32)
        self.classes_ = np.array([0, 1])

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

        # ── Komponensek → osztályok leképezése ─────────────────────────
        hard_labels  = self.bgmm_.predict(X)
        border_rates = np.zeros(self.n_components, dtype=np.float64)

        for k in range(self.n_components):
            mask = hard_labels == k
            if mask.sum() > 0:
                border_rates[k] = np.mean(y[mask] == 1)

        self.component_to_class_ = (border_rates >= 0.5).astype(np.int32)

        if self.component_to_class_.sum() == 0:
            self.component_to_class_[np.argmax(border_rates)] = 1
        if self.component_to_class_.sum() == self.n_components:
            self.component_to_class_[np.argmin(border_rates)] = 0

        self.border_component_ = int(np.argmax(border_rates))
        self.border_rates_     = border_rates

        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X):
        """
        Variational EM felelősségi mátrix alapján:
          P(Border | x) = Σ_{k: Border-komponens} γ_k(x)
        """
        check_is_fitted(self, ['bgmm_', 'component_to_class_'])
        X = check_array(X)
        X = np.asarray(X, dtype=np.float64)

        resp         = self.bgmm_.predict_proba(X)
        border_mask  = (self.component_to_class_ == 1)
        border_prob  = np.clip(resp[:, border_mask].sum(axis=1), 0.0, 1.0)

        return np.column_stack([1.0 - border_prob, border_prob])

    # ------------------------------------------------------------------
    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    # ------------------------------------------------------------------
    @property
    def n_active_components(self):
        """Ténylegesen aktív GMM-komponensek száma (weight > 1e-3)."""
        return getattr(self, 'n_active_components_', self.n_components)