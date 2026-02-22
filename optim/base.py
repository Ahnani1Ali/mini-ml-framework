"""
optim/base.py
=============
Classe de base abstraite pour tous les optimiseurs.

Auteur : AHNANI Ali
"""

from autograd.tensor import Tensor


class Optimizer:
    """
    Interface commune à tous les optimiseurs.

    Paramètres
    ----------
    params : liste de Tensor — paramètres à optimiser
    lr     : float           — taux d'apprentissage (learning rate)
    """

    def __init__(self, params: list, lr: float = 0.01):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        """Remet tous les gradients à zéro avant un nouveau forward."""
        import numpy as np
        for p in self.params:
            p.grad = np.zeros_like(p.data)

    def step(self):
        """
        Met à jour les paramètres à partir de leurs gradients.
        À implémenter dans chaque sous-classe.
        """
        raise NotImplementedError("step() doit être implémenté dans la sous-classe")
