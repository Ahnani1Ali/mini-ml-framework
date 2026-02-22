"""
optim/adam.py
=============
Adam — Adaptive Moment Estimation (Kingma & Ba, ICLR 2015).

Combine le momentum du premier ordre (m) et l'adaptation du second
ordre (v) avec correction du biais pour les deux moments.

Règles de mise à jour :
    m_t   = β₁ · m_{t-1} + (1 - β₁) · g_t          ← premier moment
    v_t   = β₂ · v_{t-1} + (1 - β₂) · g_t²         ← second moment
    m̂_t  = m_t / (1 - β₁ᵗ)                         ← correction biais
    v̂_t  = v_t / (1 - β₂ᵗ)                         ← correction biais
    θ_{t+1} = θ_t − η · m̂_t / (√v̂_t + ε)

Propriété : invariance à l'échelle des gradients.
Paramètres recommandés : lr=1e-3, β₁=0.9, β₂=0.999, ε=1e-8.

Auteur : AHNANI Ali
"""

import numpy as np
from .base import Optimizer


class Adam(Optimizer):
    """
    Adam avec AdamW optionnel (weight decay découplé).

    Paramètres
    ----------
    params       : list de Tensor
    lr           : float  — taux d'apprentissage η
    beta1        : float  — taux de décroissance du premier moment β₁
    beta2        : float  — taux de décroissance du second moment β₂
    eps          : float  — terme de stabilisation ε
    weight_decay : float  — régularisation L2 (couplée au gradient)
    amsgrad      : bool   — variante AMSGrad (utilise max(v̂) pour garantir convergence)
    """

    def __init__(self, params, lr: float = 1e-3, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8,
                 weight_decay: float = 0.0, amsgrad: bool = False):
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.t = 0  # compteur de pas

        # Initialisation des moments à zéro
        self.m = [np.zeros_like(p.data) for p in self.params]   # premier moment
        self.v = [np.zeros_like(p.data) for p in self.params]   # second moment
        if amsgrad:
            self.v_max = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        self.t += 1

        # Correction du biais (précomputation pour éviter repetition)
        bias_correction1 = 1.0 - self.beta1 ** self.t
        bias_correction2 = 1.0 - self.beta2 ** self.t

        for i, p in enumerate(self.params):
            g = p.grad + self.weight_decay * p.data

            # Mise à jour des moments
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g ** 2

            # Correction du biais
            m_hat = self.m[i] / bias_correction1
            v_hat = self.v[i] / bias_correction2

            if self.amsgrad:
                self.v_max[i] = np.maximum(self.v_max[i], v_hat)
                denom = np.sqrt(self.v_max[i]) + self.eps
            else:
                denom = np.sqrt(v_hat) + self.eps

            p.data -= self.lr * m_hat / denom
