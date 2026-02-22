"""
models/logistic.py
==================
Régression Logistique — modèle discriminatif linéaire.

Modèle :
    P(y=1 | x ; w, b) = σ(wᵀx + b)    où σ est la sigmoïde

Loss (Binary Cross-Entropy) :
    L(w) = -1/n Σ [y log σ(wᵀxᵢ) + (1-y) log(1 - σ(wᵀxᵢ))]

Convexité :
    Hessienne H = (1/n) Xᵀ diag(p(1-p)) X ⪰ 0
    → minimum global unique si X est de plein rang colonne.

Gradient analytique :
    ∂L/∂w = (1/n) Xᵀ (σ(Xw) - y)    [calculé par autodiff]

Auteur : AHNANI Ali
"""

import numpy as np
from autograd.tensor import Tensor
from models.losses import bce_loss
from optim.adam import Adam
from optim.sgd import SGD


class LogisticRegression:
    """
    Régression logistique binaire entraînée par gradient descent.

    Paramètres
    ----------
    n_features   : int   — dimension des entrées
    lr           : float — learning rate
    weight_decay : float — régularisation L2 (λ)
    optimizer    : str   — 'sgd' ou 'adam'
    """

    def __init__(self, n_features: int, lr: float = 0.1,
                 weight_decay: float = 0.0, optimizer: str = 'adam'):
        scale = np.sqrt(1.0 / n_features)
        self.W = Tensor(np.random.randn(n_features, 1) * scale)
        self.b = Tensor(np.zeros((1, 1)))
        self.params = [self.W, self.b]

        if optimizer == 'adam':
            self.opt = Adam(self.params, lr=lr, weight_decay=weight_decay)
        else:
            self.opt = SGD(self.params, lr=lr, weight_decay=weight_decay)

    def forward(self, X: Tensor) -> Tensor:
        """
        Calcule σ(XW + b) — probabilités de classe 1.

        Paramètres
        ----------
        X : Tensor, shape (n, d)

        Retourne
        --------
        Tensor shape (n, 1), valeurs dans (0, 1)
        """
        logits = X @ self.W + self.b
        # Sigmoïde via l'autodiff
        return Tensor(1.0) / (Tensor(1.0) + (logits * -1).exp())

    def train_step(self, X: Tensor, y: Tensor) -> float:
        """
        Effectue un pas de descente de gradient.

        Retourne la loss BCE du pas courant.
        """
        self.opt.zero_grad()
        probs = self.forward(X)
        loss = bce_loss(probs, y)
        loss.backward()
        self.opt.step()
        return float(loss.data)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retourne P(y=1|X) ∈ (0,1)."""
        return self.forward(Tensor(X)).data

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Retourne les classes prédites {0, 1}."""
        return (self.predict_proba(X) >= threshold).astype(int).ravel()

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y.ravel().astype(int)))


class MulticlassLogistic:
    """
    Régression logistique multiclasse via stratégie One-vs-Rest (OvR).

    Entraîne K classifieurs binaires indépendants, un par classe.
    Prédiction : argmax des K scores.

    Paramètres
    ----------
    n_features : int — dimension des entrées
    n_classes  : int — nombre de classes K
    lr         : float
    """

    def __init__(self, n_features: int, n_classes: int, lr: float = 0.1):
        self.n_classes = n_classes
        self.classifiers = [
            LogisticRegression(n_features, lr=lr)
            for _ in range(n_classes)
        ]

    def train(self, X: np.ndarray, y: np.ndarray, n_epochs: int = 200):
        """Entraîne les K classifieurs binaires."""
        Xt = Tensor(X)
        for k, clf in enumerate(self.classifiers):
            yk = Tensor((y == k).astype(float).reshape(-1, 1))
            for _ in range(n_epochs):
                clf.train_step(Xt, yk)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xt = Tensor(X)
        scores = np.column_stack([
            clf.forward(Xt).data.ravel()
            for clf in self.classifiers
        ])
        return np.argmax(scores, axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y.astype(int)))
