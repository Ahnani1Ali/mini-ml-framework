"""
autograd/tensor.py
==================
Moteur de différentiation automatique en mode reverse (reverse-mode autodiff).

Implémente un graphe computationnel dynamique (define-by-run) :
chaque opération crée un nœud dans le graphe et enregistre sa fonction
backward locale. L'appel à .backward() déclenche le tri topologique
puis la propagation des gradients de la sortie vers les entrées.

Complexité :
  - Forward  : O(|V| + |E|)   — proportionnel à la taille du graphe
  - Backward : O(|V| + |E|)   — même ordre
  - Mémoire  : O(|V|)         — stockage des activations intermédiaires

Auteur : AHNANI Ali
"""

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
#  Utilitaire : gestion du broadcasting
# ──────────────────────────────────────────────────────────────────────────────

def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Réduit `grad` pour qu'il corresponde à `shape` d'origine.

    Lors d'une opération broadcastée (ex. add biais (1,d) à activation (n,d)),
    le gradient doit être sommé sur les axes qui ont été étendus.

    Paramètres
    ----------
    grad  : gradient à réduire
    shape : forme cible (forme du tenseur original)

    Retourne
    --------
    Gradient réduit de forme `shape`
    """
    # Supprimer les dimensions ajoutées en tête
    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)
    # Sommer sur les axes de taille 1
    for i, (gs, ss) in enumerate(zip(grad.shape, shape)):
        if ss == 1 and gs != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


# ──────────────────────────────────────────────────────────────────────────────
#  Classe principale : Tensor
# ──────────────────────────────────────────────────────────────────────────────

class Tensor:
    """
    Tenseur avec support du graphe computationnel et de l'autodiff.

    Attributs
    ---------
    data          : np.ndarray — valeur numérique du tenseur
    grad          : np.ndarray — gradient accumulé (∂L/∂self)
    requires_grad : bool       — True si ce tenseur participe au calcul du gradient
    _backward     : callable   — fonction backward locale (accumule les gradients parents)
    _prev         : set        — nœuds parents dans le graphe (inputs de cette opération)
    _op           : str        — nom de l'opération (debug)
    """

    def __init__(self, data, requires_grad: bool = True, _children=(), _op: str = ''):
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    # ── Opérations arithmétiques ───────────────────────────────────────────────

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, _children=(self, other), _op='add')

        def _backward():
            self.grad  += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, _children=(self, other), _op='mul')

        def _backward():
            self.grad  += _unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += _unbroadcast(self.data  * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other):
        """
        Produit matriciel C = A @ B.

        Gradients :
          dL/dA = dL/dC @ Bᵀ
          dL/dB = Aᵀ @ dL/dC
        """
        out = Tensor(self.data @ other.data, _children=(self, other), _op='matmul')

        def _backward():
            self.grad  += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def __pow__(self, exp):
        assert isinstance(exp, (int, float)), "L'exposant doit être un scalaire"
        out = Tensor(self.data ** exp, _children=(self,), _op=f'pow{exp}')

        def _backward():
            self.grad += exp * (self.data ** (exp - 1)) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):        return self * -1
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other ** -1
    def __rtruediv__(self, other): return other * self ** -1

    # ── Opérations sur les axes ────────────────────────────────────────────────

    def sum(self, axis=None, keepdims: bool = False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims),
                     _children=(self,), _op='sum')

        def _backward():
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis=axis)
            self.grad += np.broadcast_to(grad, self.data.shape)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims: bool = False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # ── Transformations de forme ───────────────────────────────────────────────

    def transpose(self):
        """Transpose 2D : (n, d) → (d, n)."""
        out = Tensor(self.data.T, _children=(self,), _op='transpose')

        def _backward():
            self.grad += out.grad.T

        out._backward = _backward
        return out

    def reshape(self, *shape):
        original = self.data.shape
        out = Tensor(self.data.reshape(*shape), _children=(self,), _op='reshape')

        def _backward():
            self.grad += out.grad.reshape(original)

        out._backward = _backward
        return out

    # ── Activations ───────────────────────────────────────────────────────────

    def exp(self):
        """
        Exponentielle numérique stable (clip à [-500, 500]).
        Backward : d(eˣ)/dx = eˣ
        """
        e = np.exp(np.clip(self.data, -500, 500))
        out = Tensor(e, _children=(self,), _op='exp')

        def _backward():
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self):
        """
        Logarithme naturel avec stabilisation numérique (+ε).
        Backward : d(log x)/dx = 1/x
        """
        out = Tensor(np.log(self.data + 1e-15), _children=(self,), _op='log')

        def _backward():
            self.grad += (1.0 / (self.data + 1e-15)) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        """
        ReLU : max(0, x)
        Backward : 1 si x > 0, 0 sinon
        """
        out = Tensor(np.maximum(0, self.data), _children=(self,), _op='relu')

        def _backward():
            self.grad += (self.data > 0).astype(float) * out.grad

        out._backward = _backward
        return out

    def gelu(self):
        """
        GELU approchée (Hendrycks & Gimpel, 2016) :
            GELU(x) ≈ x · σ(1.702 · x)
        où σ est la sigmoïde.

        Backward :
            d/dx [x · σ(1.702x)] = σ(1.702x) + x · 1.702 · σ(1.702x)(1 - σ(1.702x))
        """
        sig = 1.0 / (1.0 + np.exp(-1.702 * self.data))
        out = Tensor(self.data * sig, _children=(self,), _op='gelu')

        def _backward():
            d_sig = sig * (1.0 - sig)
            d_gelu = sig + self.data * 1.702 * d_sig
            self.grad += d_gelu * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        """σ(x) = 1 / (1 + e^{-x})"""
        s = 1.0 / (1.0 + np.exp(-np.clip(self.data, -500, 500)))
        out = Tensor(s, _children=(self,), _op='sigmoid')

        def _backward():
            self.grad += s * (1.0 - s) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, _children=(self,), _op='tanh')

        def _backward():
            self.grad += (1.0 - t ** 2) * out.grad

        out._backward = _backward
        return out

    # ── Algorithme principal : backward ───────────────────────────────────────

    def backward(self):
        """
        Lance la backpropagation depuis ce tenseur (doit être un scalaire).

        Algorithme :
        1. Tri topologique du graphe (DFS depuis self)
        2. Initialisation : self.grad = 1  (dL/dL = 1)
        3. Itération en ordre topologique renversé :
             pour chaque nœud v, appelle v._backward()
             qui accumule les gradients dans les parents de v

        Complexité : O(|V| + |E|)
        """
        # ── Phase 1 : construction de l'ordre topologique ──
        topo = []
        visited = set()

        def build_topo(v):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # ── Phase 2 : propagation backward ──
        self.grad = np.ones_like(self.data)   # dL/dL = 1
        for node in reversed(topo):
            node._backward()

    def zero_grad(self):
        """Remet le gradient à zéro."""
        self.grad = np.zeros_like(self.data)

    # ── Utilitaires ───────────────────────────────────────────────────────────

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def __repr__(self):
        return (f"Tensor(shape={self.data.shape}, op='{self._op}', "
                f"grad_fn={self._backward.__name__ if hasattr(self._backward, '__name__') else 'None'})")

    def item(self):
        """Retourne la valeur scalaire (si tenseur 0-d ou 1 élément)."""
        return float(self.data)
