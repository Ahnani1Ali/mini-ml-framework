"""
models/kernel.py
================
Kernel Ridge Regression (KRR) avec noyau RBF.

Théorie (RKHS) :
    Minimise : min_{f ∈ H_k} 1/n ‖y - f(X)‖² + λ‖f‖²_{H_k}

    Par le Théorème du Représentant (Kimeldorf & Wahba, 1971) :
        f*(x) = Σᵢ αᵢ* k(xᵢ, x)

    Solution analytique :
        α* = (K + nλI)⁻¹ y
        K[i,j] = k(xᵢ, xⱼ)  ← matrice de Gram

Noyau RBF (Gaussien) :
    k(x, x') = exp(-γ ‖x - x'‖²)

    Construction efficace de ‖x - x'‖² :
        ‖xᵢ - xⱼ‖² = ‖xᵢ‖² + ‖xⱼ‖² - 2 xᵢᵀxⱼ
    → un seul produit matriciel X₁ @ X₂ᵀ   (O(n₁ n₂ d))

Complexité :
    - fit       : O(n²d) construction K  +  O(n³) inversion  → O(n³) total
    - predict   : O(n_test · n · d)
    - mémoire   : O(n²)

Auteur : AHNANI Ali
"""

import numpy as np
import time


class KernelRidgeRegression:
    """
    Kernel Ridge Regression avec noyau RBF.

    Paramètres
    ----------
    gamma : float — largeur de bande du noyau RBF γ
                    grand γ → noyau étroit → modèle complexe
    lam   : float — coefficient de régularisation λ
                    grand λ → lissage fort → underfitting
    """

    def __init__(self, gamma: float = 1.0, lam: float = 1.0):
        self.gamma = gamma
        self.lam = lam
        self.alpha = None       # coefficients duaux α*
        self.X_train = None     # exemples d'entraînement (stockés pour predict)
        self.fit_time = None    # temps de fit (benchmark)

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Calcule la matrice de Gram K[i,j] = exp(-γ ‖xᵢ - xⱼ‖²).

        Utilise l'identité ‖x - x'‖² = ‖x‖² + ‖x'‖² - 2 xᵀx'
        pour éviter une double boucle Python.

        Complexité : O(n₁ n₂ d) — un seul matmul.
        """
        sq1 = np.sum(X1 ** 2, axis=1, keepdims=True)   # (n1, 1)
        sq2 = np.sum(X2 ** 2, axis=1, keepdims=True)   # (n2, 1)
        cross = X1 @ X2.T                               # (n1, n2)
        dist_sq = sq1 + sq2.T - 2.0 * cross            # (n1, n2)
        # Clip pour éviter des valeurs légèrement négatives dues aux erreurs flottantes
        dist_sq = np.maximum(dist_sq, 0.0)
        return np.exp(-self.gamma * dist_sq)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KernelRidgeRegression':
        """
        Entraîne le modèle : résout (K + nλI)α = y.

        np.linalg.solve utilise la factorisation LU → O(n³).
        On préfère solve à inv car plus stable numériquement.

        Paramètres
        ----------
        X : np.ndarray, shape (n, d)
        y : np.ndarray, shape (n,) ou (n, 1)
        """
        self.X_train = X.copy()
        n = X.shape[0]
        t0 = time.time()

        K = self._rbf_kernel(X, X)                          # (n, n)
        self.alpha = np.linalg.solve(
            K + self.lam * np.eye(n),                        # (n, n)
            y.ravel()                                         # (n,)
        )

        self.fit_time = time.time() - t0
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Prédit f*(x) = K_test @ α* pour chaque point de test.

        Complexité : O(n_test · n · d)
        """
        if self.alpha is None:
            raise RuntimeError("Le modèle n'est pas encore entraîné. Appelez fit() d'abord.")
        K_test = self._rbf_kernel(X_test, self.X_train)     # (n_test, n)
        return K_test @ self.alpha

    def __repr__(self):
        return f"KernelRidgeRegression(gamma={self.gamma}, lam={self.lam})"


class KernelClassifier:
    """
    Classification multiclasse par KRR One-vs-Rest.

    Entraîne K modèles KRR indépendants (un par classe).
    Prédiction : argmax des K scores.

    Paramètres
    ----------
    gamma : float — paramètre du noyau RBF
    lam   : float — régularisation
    """

    def __init__(self, gamma: float = 1.0, lam: float = 0.1):
        self.gamma = gamma
        self.lam = lam
        self.models = None
        self.n_classes = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KernelClassifier':
        """
        Entraîne les K classifieurs KRR binaires.

        Paramètres
        ----------
        X : np.ndarray, shape (n, d)
        y : np.ndarray, shape (n,)  — labels entiers {0, 1, ..., K-1}
        """
        self.n_classes = len(np.unique(y))
        self.models = []
        for k in range(self.n_classes):
            yk = (y == k).astype(float)                        # labels binaires
            m = KernelRidgeRegression(self.gamma, self.lam)
            m.fit(X, yk)
            self.models.append(m)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Retourne argmax des scores KRR."""
        scores = np.column_stack([m.predict(X) for m in self.models])
        return np.argmax(scores, axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y.astype(int)))

    def __repr__(self):
        return f"KernelClassifier(gamma={self.gamma}, lam={self.lam}, n_classes={self.n_classes})"
