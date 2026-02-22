"""
optim/rmsprop.py
================
RMSProp — Root Mean Square Propagation (Tieleman & Hinton, 2012).

Idée : normaliser chaque gradient par la racine de la moyenne mobile
de son carré, ce qui adapte le learning rate par paramètre.

Règles de mise à jour :
    v_t   = ρ · v_{t-1} + (1 - ρ) · g_t²
    θ_{t+1} = θ_t − η / (√v_t + ε) · g_t

Intuition : paramètres avec grands gradients historiques reçoivent un
lr plus petit (stabilité), paramètres peu actifs reçoivent un lr plus
grand (exploration).

Auteur : AHNANI Ali
"""

import numpy as np
from .base import Optimizer


class RMSProp(Optimizer):
    """
    RMSProp.

    Paramètres
    ----------
    params       : list de Tensor
    lr           : float  — taux d'apprentissage η
    rho          : float  — taux de décroissance ρ (typiquement 0.9)
    eps          : float  — terme de stabilisation numérique ε
    weight_decay : float  — régularisation L2
    """

    def __init__(self, params, lr: float = 1e-3, rho: float = 0.9,
                 eps: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(params, lr)
        self.rho = rho
        self.eps = eps
        self.weight_decay = weight_decay
        # Moyenne mobile du carré des gradients
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            g = p.grad + self.weight_decay * p.data

            # Mise à jour de la moyenne mobile
            self.v[i] = self.rho * self.v[i] + (1.0 - self.rho) * g ** 2

            # Mise à jour des paramètres
            p.data -= self.lr / (np.sqrt(self.v[i]) + self.eps) * g
