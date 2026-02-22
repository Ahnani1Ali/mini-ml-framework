"""
training/scheduler.py
=====================
Schedulers de learning rate.

  - CosineAnnealingScheduler : lr cosinus entre lr_max et lr_min
  - StepScheduler            : décroissance par paliers
  - ExponentialScheduler     : décroissance exponentielle

Formule Cosine Annealing (Loshchilov & Hutter, ICLR 2017) :
    lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π · t / T))

Auteur : AHNANI Ali
"""

import numpy as np


class CosineAnnealingScheduler:
    """
    Cosine Annealing Learning Rate Scheduler.

    lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π · t / T_max))

    Paramètres
    ----------
    optimizer : objet avec attribut .lr
    T_max     : int   — période complète (nombre de steps)
    lr_min    : float — lr minimum à la fin du cycle
    """

    def __init__(self, optimizer, T_max: int, lr_min: float = 1e-6):
        self.optimizer = optimizer
        self.T_max = T_max
        self.lr_min = lr_min
        self.lr_max = optimizer.lr
        self.t = 0

    def step(self) -> float:
        """Effectue un pas du scheduler et retourne le nouveau lr."""
        self.t += 1
        lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
            1.0 + np.cos(np.pi * self.t / self.T_max)
        )
        self.optimizer.lr = lr
        return lr

    def reset(self):
        """Remet le scheduler à zéro (pour SGDR)."""
        self.t = 0
        self.optimizer.lr = self.lr_max


class StepScheduler:
    """
    Décroissance du lr par paliers.

    lr(t) = lr_0 * gamma^{floor(t / step_size)}

    Paramètres
    ----------
    optimizer  : objet avec attribut .lr
    step_size  : int   — tous les combien d'epochs on multiplie par gamma
    gamma      : float — facteur de décroissance (< 1)
    """

    def __init__(self, optimizer, step_size: int = 10, gamma: float = 0.5):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.lr_init = optimizer.lr
        self.t = 0

    def step(self) -> float:
        self.t += 1
        if self.t % self.step_size == 0:
            self.optimizer.lr *= self.gamma
        return self.optimizer.lr


class ExponentialScheduler:
    """
    Décroissance exponentielle : lr(t) = lr_0 * gamma^t

    Paramètres
    ----------
    optimizer : objet avec attribut .lr
    gamma     : float — taux de décroissance par epoch
    """

    def __init__(self, optimizer, gamma: float = 0.99):
        self.optimizer = optimizer
        self.gamma = gamma

    def step(self) -> float:
        self.optimizer.lr *= self.gamma
        return self.optimizer.lr
