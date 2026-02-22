"""
models/linear.py
================
Couche linéaire (Dense) : y = x @ W + b

Initialisation selon l'activation cible :
  - He / Kaiming  (σ² = 2/d_in)            pour ReLU, GELU
  - Xavier / Glorot (σ² = 2/(d_in + d_out)) pour Tanh, Sigmoid, None

Théorème He : avec ReLU, σ² = 2/d_in préserve Var[z^(ℓ)] ≈ Var[z^(0)]
car E[ReLU'(z)²] = 1/2 (moitié des neurones actifs).

Complexité :
  - Forward  : O(n · d_in · d_out)
  - Backward : O(n · d_in · d_out)  — deux produits matriciels

Auteur : AHNANI Ali
"""

import numpy as np
from autograd.tensor import Tensor


class Linear:
    """
    Couche linéaire : out = x @ W + b

    Paramètres
    ----------
    in_features  : int  — dimension d'entrée
    out_features : int  — dimension de sortie
    activation   : str  — activation suivante ('relu', 'gelu', 'none', ...)
                          utilisée pour choisir l'initialisation
    bias         : bool — inclure le biais
    """

    def __init__(self, in_features: int, out_features: int,
                 activation: str = 'relu', bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features

        # ── Choix de l'initialisation ──────────────────────────────────────
        if activation in ('relu', 'gelu'):
            # Initialisation He / Kaiming
            # σ = √(2 / d_in)  —  préserve la variance avec ReLU
            scale = np.sqrt(2.0 / in_features)
        else:
            # Initialisation Xavier / Glorot
            # σ = √(2 / (d_in + d_out))  —  équilibre forward/backward
            scale = np.sqrt(2.0 / (in_features + out_features))

        self.W = Tensor(np.random.randn(in_features, out_features) * scale)
        self.b = Tensor(np.zeros((1, out_features))) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        """Passe forward : out = x @ W + b"""
        out = x @ self.W
        if self.b is not None:
            out = out + self.b
        return out

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    @property
    def parameters(self) -> list:
        """Retourne la liste des paramètres entraînables."""
        params = [self.W]
        if self.b is not None:
            params.append(self.b)
        return params

    def __repr__(self):
        return f"Linear(in={self.in_features}, out={self.out_features}, bias={self.b is not None})"
