"""
clustering_models.py

Két generatív, EM-alapú klaszterező osztályozó az Itier et al. (2021)
keretrendszerbe illesztve:

  1. GMMClassifier        — Gaussian Mixture Model (standard EM illesztés)
  2. BayesianEMClassifier — Bayesian Gaussian Mixture (Variational EM,
                            Dirichlet-prioros automatikus komponensszám-becslés)

Mindkét osztály teljes mértékben scikit-learn kompatibilis (BaseEstimator +
ClassifierMixin), így a GridSearchCV, RandomizedSearchCV, cross_val_score és
a joblib.dump / joblib.load közvetlenül működik velük.

─── Matematikai és elméleti háttér ─────────────────────────────────────────────

  1. K-Means (kemény hozzárendelés — viszonyítási alap):
     Minden mintát egyetlen klaszterközépponthoz rendel (argmin L2-távolság).
     Minden klaszter izotróp (gömbölyű) Voronoi-cellának felel meg, így a
     60 dimenziós SRM-PCA jellemzőtér eltérő irányú korrelációit nem tudja
     rugalmasan követni.

  2. Standard GMM / EM (puha hozzárendelés — GMMClassifier):
     A jellemzőtér sűrűségfüggvényét K darab többváltozós Gauss-eloszlás
     súlyozott keverékeként modellezi:
       p(x) = Σ_{k=1}^K π_k · N(x | μ_k, Σ_k)

     Az Expectation-Maximization (EM) algoritmus két lépést ismétel:
       - E-lépés (Expectation): Kiszámítja a γ_k(x_i) = P(z=k | x_i) utólagos
         felelősségi súlyokat (responsibilities) Bayes-tétellel:
           γ_k(x_i) = [π_k · N(x_i; μ_k, Σ_k)] / [Σ_{j=1}^K π_j · N(x_i; μ_j, Σ_j)]
       - M-lépés (Maximization): Frissíti a (π_k, μ_k, Σ_k) paramétereket a
         várható teljes log-likelihood maximalizálásával:
           Q(θ, θ^{old}) = Σ_i Σ_k γ_k(x_i) · log[π_k · N(x_i; μ_k, Σ_k)]

     Előnye a K-Means-szel szemben:
       - Teljes ('full'), közös ('tied') vagy diagonális ('diag') kovariancia-
         mátrix (Σ_k) -> elliptikus klaszterformák a 60D térben.
       - Valódi valószínűségi sűrűségbecslés -> közvetlen predict_proba() kimenet.

  3. Bayesian GMM / Variational EM (BayesianEMClassifier):
     A komponenssúlyokra Dirichlet-folyamat (stick-breaking) priort helyez:
       π ~ Dirichlet(α_0 / K, ..., α_0 / K)
     A paramétereket Variational Bayes (Variational EM) eljárással becsüli az
     ELBO (Evidence Lower Bound) célfüggvény maximalizálásán keresztül:
       ELBO(q) = E_q[log p(X, Z, π, μ, Λ)] - E_q[log q(Z, π, μ, Λ)]

     Forenzikus előnye:
       A felesleges komponensek π_k súlya automatikusan 0-ra zsugorodik
       (weights <= 1e-3), így a modell önállóan határozza meg a ténylegesen
       aktív komponensek számát (n_active_components_), és sokkal tisztább,
       zajmentesebb hőtérkép-hátteret ad a pixelszintű lokalizációhoz.

─── Felügyelt N-osztályos leképezés (3-osztályos séma) ─────────────────────────
  A GMM és a BayesianGMM önmagában felügyelet nélküli sűrűségbecslő. A fit(X, y)
  hívás során a tanítóhalmaz ismert y in {0, 1, 2} (Uniform / Border / Spliced)
  címkéi alapján minden k-adik komponenst a benne lévő minták TÖBBSÉGI
  osztályához rendelünk:
    class_rates[k, c] = P(y = c | z = k)
    comp_to_class[k]  = argmax_c class_rates[k, c]

  Előrejelzéskor (predict_proba) egy x blokk c osztályba tartozási valószínűsége
  az adott c osztályhoz rendelt összes GMM-komponens felelősségi súlyának összege:
    P(y = c | x) = Σ_{k : comp_to_class[k] == c} γ_k(x)
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
# Közös segédfüggvény: Komponensek -> Osztályok ütközésmentes leképezése
# ===========================================================================

def _assign_components_to_classes(hard_labels, y, classes, n_components):
    """
    Meghatározza minden GMM/BGMM komponenshez a többségi tanítóosztályt,
    és ütközésmentes biztonsági hozzárendeléssel (fallback) garantálja, hogy
    minden osztályhoz tartozzon legalább egy komponens.

    Paraméterek:
      hard_labels  : (N,) int tömb — a komponens-indexek (argmax_k gamma_k).
      y            : (N,) int tömb — a valódi tanítócímkék (pl. {0, 1, 2}).
      classes      : (n_classes,) tömb — az egyedi osztályok (np.unique(y)).
      n_components : int — a keverékmodell komponenseinek száma.

    Visszatérés:
      component_to_class_idx : (n_components,) int tömb — index a `classes` tömbbe.
      class_rates            : (n_components, n_classes) float64 mátrix.
    """
    n_classes = len(classes)
    class_rates = np.zeros((n_components, n_classes), dtype=np.float64)

    # 1. Kiszámítjuk minden k komponensre az egyes osztályok relatív gyakoriságát.
    #    Ha egy komponens üres marad (mask.sum() == 0, pl. BayesianEM-ben lenullázott
    #    súlyú komponens), a sora csupa 0 marad -> argmax a 0. indexet (Uniform) adja,
    #    ami biztonságos, mert a γ_k(x) felelősségi súlya ~0.
    for k in range(n_components):
        mask = (hard_labels == k)
        if np.any(mask):
            y_k = y[mask]
            for ci, c in enumerate(classes):
                class_rates[k, ci] = np.mean(y_k == c)

    # 2. Alapértelmezett hozzárendelés: minden komponens a többségi osztályát kapja
    component_to_class_idx = np.argmax(class_rates, axis=1)

    # 3. Ütközésmentes biztonsági hozzárendelés (fallback):
    #    Ha valamelyik ci osztály egyetlen komponensben sem szerzett többséget,
    #    megkeressük azt a komponenst, ahol a ci osztály aránya a legmagasabb,
    #    ügyelve arra, hogy ne írjunk felül már lefoglalt tartalékot, és ne vegyük
    #    el egy másik osztály egyetlen megmaradt komponensét sem.
    reserved_components = set()
    for ci in range(n_classes):
        if not np.any(component_to_class_idx == ci):
            candidate_order = np.argsort(-class_rates[:, ci])
            assigned = False
            for cand_k in candidate_order:
                cand_k = int(cand_k)
                if cand_k in reserved_components:
                    continue
                current_owner = component_to_class_idx[cand_k]
                if np.sum(component_to_class_idx == current_owner) > 1:
                    component_to_class_idx[cand_k] = ci
                    reserved_components.add(cand_k)
                    assigned = True
                    break
            if not assigned:
                for cand_k in candidate_order:
                    cand_k = int(cand_k)
                    if cand_k not in reserved_components:
                        component_to_class_idx[cand_k] = ci
                        reserved_components.add(cand_k)
                        break

    return component_to_class_idx, class_rates


# ===========================================================================
# 1. GMMClassifier — Standard EM illesztés
# ===========================================================================

class GMMClassifier(BaseEstimator, ClassifierMixin):
    """
    Gaussian Mixture Model (standard EM) alapú, scikit-learn kompatibilis osztályozó.

    A modell a fit(X, y) során felügyelet nélkül illeszti a Gauss-keveréket X-re,
    majd a tanítócímkék (y) többségi aránya alapján minden komponenst hozzárendel
    a {Uniform (0), Border (1), Spliced (2)} osztályok egyikéhez.

    Paraméterek
    -----------
    n_components : int, default=3
        GMM-komponensek száma. 3-osztályos sémánál minimum 3 komponens szükséges,
        hogy minden osztálynak juthasson legalább egy saját sűrűségkomponens.
    covariance_type : {'full', 'tied', 'diag', 'spherical'}, default='full'
        'full'      : minden komponensnek saját, teljes (60x60) kovarianciamátrixa van
        'tied'      : közös kovarianciamátrix minden komponensre (gyorsabb, regularizált)
        'diag'      : tengelyek menti diagonális kovariancia
        'spherical' : izotróp (egyetlen szórásérték komponensenként)
    max_iter : int, default=200
        EM iterációk maximális száma.
    n_init : int, default=10
        Véletlen (k-means alapú) újraindítások száma; a legjobb log-likelihoodot tartja meg.
    tol : float, default=1e-4
        Konvergencia-küszöb a log-likelihood javulására.
    reg_covar : float, default=1e-6
        Nemnegatív regularizáció a kovarianciamátrix főátlójához (szingularitás ellen).
    random_state : int, default=42
        Magérték a reprodukálható inicializációhoz.
    """

    def __init__(self, n_components=3, covariance_type='full',
                 max_iter=200, n_init=10, tol=1e-4,
                 reg_covar=1e-6, random_state=42):
        self.n_components    = n_components
        self.covariance_type = covariance_type
        self.max_iter        = max_iter
        self.n_init          = n_init
        self.tol             = tol
        self.reg_covar       = reg_covar
        self.random_state    = random_state

    # ------------------------------------------------------------------
    def fit(self, X, y):
        """
        1. Ellenőrzi az X, y tömböket és meghatározza az osztályokat (self.classes_).
        2. Illeszti a GaussianMixture modellt X-re (standard EM algoritmussal).
        3. Többségi szavazattal minden komponenst hozzárendel egy osztályhoz.
        """
        X, y = check_X_y(X, y)
        X    = np.asarray(X, dtype=np.float64)
        y    = np.asarray(y, dtype=np.int32)
        self.classes_ = np.unique(y)
        n_classes     = len(self.classes_)

        # Legalább annyi komponens illesztése, ahány osztály szerepel a címkékben
        self.effective_n_components_ = max(int(self.n_components), n_classes)

        # ── 1. Standard EM illesztés ───────────────────────────────────
        self.gmm_ = GaussianMixture(
            n_components    = self.effective_n_components_,
            covariance_type = self.covariance_type,
            max_iter        = self.max_iter,
            n_init          = self.n_init,
            tol             = self.tol,
            reg_covar       = self.reg_covar,
            random_state    = self.random_state,
        )
        self.gmm_.fit(X)

        # ── 2. Komponensek -> osztályok leképezése tanítócímkékkel ─────
        hard_labels = self.gmm_.predict(X)
        self.component_to_class_idx_, self.class_rates_ = _assign_components_to_classes(
            hard_labels, y, self.classes_, self.effective_n_components_
        )

        # ── 3. Naplózási attribútumok (Border komponens azonosítása) ───
        if 1 in self.classes_:
            border_idx = int(np.where(self.classes_ == 1)[0][0])
            self.border_rates_     = self.class_rates_[:, border_idx]
            self.border_component_ = int(np.argmax(self.border_rates_))
        else:
            self.border_rates_     = np.zeros(self.effective_n_components_, dtype=np.float64)
            self.border_component_ = 0

        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X):
        """
        Kiszámítja az osztály-valószínűségeket minden mintára:
          P(osztály c | x) = Σ_{k : k-adik komponens -> c osztály} γ_k(x)

        Visszatérés:
          proba : (N, n_classes) float64 tömb, oszlopai self.classes_ sorrendben
                  (3 osztálynál: [P(Uniform), P(Border), P(Spliced)]).
        """
        check_is_fitted(self, ['gmm_', 'component_to_class_idx_', 'effective_n_components_'])
        X = check_array(X)
        X = np.asarray(X, dtype=np.float64)

        # GMM utólagos felelősségi mátrix (responsibilities): shape (N, n_components)
        resp = self.gmm_.predict_proba(X)

        n_classes = len(self.classes_)
        proba = np.zeros((X.shape[0], n_classes), dtype=np.float64)
        for k in range(self.effective_n_components_):
            ci = self.component_to_class_idx_[k]
            proba[:, ci] += resp[:, k]

        # Numerikus normálás (hogy a sorösszegek pontosan 1.0-t adjanak)
        row_sums = proba.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return proba / row_sums

    # ------------------------------------------------------------------
    def predict(self, X):
        """Kemény osztálypredikció: argmax_c P(osztály c | x)."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


# ===========================================================================
# 2. BayesianEMClassifier — Variational EM (Dirichlet-prior)
# ===========================================================================

class BayesianEMClassifier(BaseEstimator, ClassifierMixin):
    """
    Bayesian Gaussian Mixture Model (Variational EM) alapú osztályozó.

    A modell Dirichlet-folyamat (stick-breaking) priort helyez a komponenssúlyokra,
    és az ELBO maximalizálásával automatikusan lenullázza a felesleges komponensek
    súlyát (automatikus modellszelekció és regularizáció).

    Paraméterek
    -----------
    n_components : int, default=8
        Komponensek maximális száma (felső korlát; a ténylegesen aktív komponensek
        száma ennél általában kisebb, lásd `n_active_components_`).
    covariance_type : {'full', 'tied', 'diag', 'spherical'}, default='full'
        A komponensek kovarianciamátrixának szerkezete.
    max_iter : int, default=300
        Variational EM iterációk maximális száma.
    n_init : int, default=3
        Véletlen újraindítások száma.
    tol : float, default=1e-4
        Konvergencia-küszöb az ELBO változására.
    reg_covar : float, default=1e-6
        Főátló-regularizáció a numerikus stabilitásért.
    weight_concentration_prior_type : str, default='dirichlet_process'
        'dirichlet_process'      : végtelen keverékmodell (stick-breaking, erős ritkítás)
        'dirichlet_distribution' : véges szimmetrikus Dirichlet-eloszlás
    weight_concentration_prior : float or None, default=None
        Dirichlet koncentrációs hiperparaméter (alpha_0).
        Kis érték (pl. 0.001) -> kevesebb aktív komponens (erős regularizáció).
        Nagy érték (pl. 1.0 - 10.0) -> több aktív komponens.
    random_state : int, default=42
        Reprodukálhatósági magérték.
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
        1. Variational EM (BayesianGaussianMixture) illesztése X-en.
        2. Ténylegesen aktív komponensek megszámolása (weight > 1e-3).
        3. Komponensek -> osztályok leképezése a tanítócímkék többsége alapján.
        """
        X, y = check_X_y(X, y)
        X    = np.asarray(X, dtype=np.float64)
        y    = np.asarray(y, dtype=np.int32)
        self.classes_ = np.unique(y)
        n_classes     = len(self.classes_)

        self.effective_n_components_ = max(int(self.n_components), n_classes)

        # ── 1. Variational EM paraméterek összeállítása és illesztés ───
        kwargs = dict(
            n_components                    = self.effective_n_components_,
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

        self.bgmm_ = BayesianGaussianMixture(**kwargs)
        self.bgmm_.fit(X)

        # ── 2. Aktív (nem-nulla súlyú) komponensek száma ───────────────
        weights = self.bgmm_.weights_
        self.n_active_components_ = int(np.sum(weights > 1e-3))

        # ── 3. Komponensek -> osztályok leképezése (többségi szavazat) ─
        hard_labels = self.bgmm_.predict(X)
        self.component_to_class_idx_, self.class_rates_ = _assign_components_to_classes(
            hard_labels, y, self.classes_, self.effective_n_components_
        )

        # ── 4. Naplózási attribútumok (Border komponens azonosítása) ───
        if 1 in self.classes_:
            border_idx = int(np.where(self.classes_ == 1)[0][0])
            self.border_rates_     = self.class_rates_[:, border_idx]
            self.border_component_ = int(np.argmax(self.border_rates_))
        else:
            self.border_rates_     = np.zeros(self.effective_n_components_, dtype=np.float64)
            self.border_component_ = 0

        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X):
        """
        Variational EM felelősségi mátrix alapján számított osztály-valószínűségek:
          P(osztály c | x) = Σ_{k : k-adik komponens -> c osztály} γ_k(x)
        """
        check_is_fitted(self, ['bgmm_', 'component_to_class_idx_', 'effective_n_components_'])
        X = check_array(X)
        X = np.asarray(X, dtype=np.float64)

        resp = self.bgmm_.predict_proba(X)

        n_classes = len(self.classes_)
        proba = np.zeros((X.shape[0], n_classes), dtype=np.float64)
        for k in range(self.effective_n_components_):
            ci = self.component_to_class_idx_[k]
            proba[:, ci] += resp[:, k]

        row_sums = proba.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return proba / row_sums

    # ------------------------------------------------------------------
    def predict(self, X):
        """Kemény osztálypredikció: argmax_c P(osztály c | x)."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    # ------------------------------------------------------------------
    @property
    def n_active_components(self):
        """Ténylegesen aktív GMM-komponensek száma (weight > 1e-3)."""
        return getattr(self, 'n_active_components_', self.n_components)