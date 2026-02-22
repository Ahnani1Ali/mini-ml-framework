"""
optim/sgd.py
============
Stochastic Gradient Descent (SGD) avec Momentum optionnel (Polyak, 1964).

Règles de mise à jour :

  SGD classique :
      θ ← θ − η · ∇L

  SGD avec Momentum :
      v ← β · v + ∇L
      θ ← θ − η · v

Convergence (fonctions L-lisses μ-fortement convexes) :
  - Sans momentum : O(1/T) pour GD, O(1/√T) pour SGD stochastique
  - Avec momentum : O((L/μ)^{1/2}) itérations (vs O(L/μ) sans momentum)

Auteur : AHNANI Ali
"""

import numpy as np
from .base import Optimizer


class SGD(Optimizer):
    """
    SGD avec Momentum et Weight Decay (L2 régularisation).

    Paramètres
    ----------
    params       : list de Tensor
    lr           : float  — taux d'apprentissage
    momentum     : float  — coefficient de momentum β ∈ [0, 1)  (0 = pas de momentum)
    weight_decay : float  — coefficient de régularisation L2 λ
    nesterov     : bool   — utiliser le momentum de Nesterov
    """

    def __init__(self, params, lr: float = 0.01, momentum: float = 0.0,
                 weight_decay: float = 0.0, nesterov: bool = False):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        # Initialisation des vélocités à zéro
        self.velocities = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            # Gradient avec régularisation L2
            g = p.grad + self.weight_decay * p.data

            if self.momentum > 0:
                self.velocities[i] = self.momentum * self.velocities[i] + g

                if self.nesterov:
                    # Momentum de Nesterov : regarder "en avant"
                    update = self.momentum * self.velocities[i] + g
                else:
                    update = self.velocities[i]
            else:
                update = g

            p.data -= self.lr * update
