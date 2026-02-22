"""
utils/grad_clip.py
==================
Gradient Clipping — borne la norme globale des gradients.

Algorithme (Pascanu et al., 2013) :
    total_norm = sqrt(Σ_p ||grad_p||²)
    if total_norm > max_norm:
        clip_coef = max_norm / total_norm
        grad_p   *= clip_coef  pour tout p

Propriété :
    - Préserve la direction du gradient (contrairement au clip par valeur)
    - Borne la magnitude sans distordre la direction de descente
    - Essentiel pour les RNNs où le produit de T Jacobiennes peut exploser

Auteur : AHNANI Ali
"""

import numpy as np


def gradient_clipping(params: list, max_norm: float = 1.0) -> float:
    """
    Clippe les gradients par la norme L2 globale.

    Paramètres
    ----------
    params   : list de Tensor — paramètres dont on clippe les gradients
    max_norm : float          — norme maximale autorisée

    Retourne
    --------
    float — norme totale avant clipping (utile pour le monitoring)
    """
    # Calcul de la norme globale : sqrt(Σ ||grad||²)
    total_norm = np.sqrt(sum(
        np.sum(p.grad ** 2)
        for p in params
        if p.grad is not None
    ))

    # Coefficient de clipping
    clip_coef = max_norm / max(total_norm, max_norm)

    # Application
    if clip_coef < 1.0:
        for p in params:
            if p.grad is not None:
                p.grad *= clip_coef

    return float(total_norm)


def gradient_clipping_by_value(params: list, clip_value: float = 1.0):
    """
    Clippe les gradients composante par composante dans [-clip_value, clip_value].

    Moins recommandé que le clipping par norme car distord la direction.

    Paramètres
    ----------
    params     : list de Tensor
    clip_value : float — valeur absolue maximale de chaque composante
    """
    for p in params:
        if p.grad is not None:
            np.clip(p.grad, -clip_value, clip_value, out=p.grad)
