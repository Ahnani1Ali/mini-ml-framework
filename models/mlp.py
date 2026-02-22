"""
models/mlp.py
=============
Multi-Layer Perceptron (MLP) profond.

Architecture :
    x → [Dense → Activation] × L → Dense → logits

Caractéristiques :
  - Activations : ReLU ou GELU
  - Initialisation : He (ReLU/GELU) ou Xavier (autre)
  - Softmax stable via log-sum-exp trick
  - Cross-Entropy stable
  - Mini-batch training
  - Analyse du gradient flow par couche

Complexité (L couches, largeur d, batch n) :
  - Forward  : O(L · n · d²)
  - Backward : O(L · n · d²)
  - Mémoire  : O(L · n · d)

Auteur : AHNANI Ali
"""

import numpy as np
from autograd.tensor import Tensor
from models.linear import Linear
from models.activations import get_activation, log_softmax
from models.losses import cross_entropy_loss
from optim.adam import Adam
from optim.sgd import SGD
from optim.rmsprop import RMSProp


def to_onehot(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Convertit un vecteur de labels en matrice one-hot."""
    oh = np.zeros((len(y), n_classes))
    oh[np.arange(len(y)), y.astype(int)] = 1.0
    return oh


class MLP:
    """
    Réseau de neurones profond fully-connected.

    Paramètres
    ----------
    layer_dims   : list[int] — dimensions des couches [d0, d1, ..., dL]
                               ex. [784, 256, 128, 10] pour MNIST
    activation   : str       — 'relu' ou 'gelu'
    weight_decay : float     — régularisation L2
    optimizer    : str       — 'adam', 'sgd', 'momentum', 'rmsprop'
    lr           : float     — learning rate

    Exemple
    -------
    >>> model = MLP([784, 256, 128, 10], activation='relu', lr=1e-3)
    >>> loss = model.train_step(Tensor(X_batch), Tensor(Y_batch_onehot))
    """

    def __init__(self, layer_dims: list, activation: str = 'relu',
                 weight_decay: float = 0.0, optimizer: str = 'adam',
                 lr: float = 1e-3):
        assert len(layer_dims) >= 2, "Il faut au moins une couche d'entrée et une de sortie"
        assert activation in ('relu', 'gelu'), f"Activation '{activation}' non supportée"

        self.activation_name = activation
        self.activation_fn = get_activation(activation)
        self.weight_decay = weight_decay
        self.n_layers = len(layer_dims) - 1

        # Construction des couches linéaires
        # La dernière couche n'a pas d'activation → initialisation Xavier
        self.layers = []
        for i in range(self.n_layers):
            act = activation if i < self.n_layers - 1 else 'none'
            self.layers.append(Linear(layer_dims[i], layer_dims[i + 1], activation=act))

        # Collecte de tous les paramètres
        all_params = []
        for layer in self.layers:
            all_params.extend(layer.parameters)

        # Création de l'optimiseur
        opt_kwargs = dict(lr=lr, weight_decay=weight_decay)
        if optimizer == 'adam':
            self.opt = Adam(all_params, **opt_kwargs)
        elif optimizer == 'sgd':
            self.opt = SGD(all_params, **opt_kwargs)
        elif optimizer == 'momentum':
            self.opt = SGD(all_params, lr=lr, momentum=0.9, weight_decay=weight_decay)
        elif optimizer == 'rmsprop':
            self.opt = RMSProp(all_params, **opt_kwargs)
        else:
            raise ValueError(f"Optimiseur '{optimizer}' inconnu")

    def forward(self, x: Tensor) -> Tensor:
        """
        Passe forward complète.

        Paramètres
        ----------
        x : Tensor, shape (n, d0)

        Retourne
        --------
        Tensor de logits, shape (n, n_classes)
        """
        h = x
        for i, layer in enumerate(self.layers):
            h = layer.forward(h)
            # Activation sur toutes les couches sauf la dernière
            if i < self.n_layers - 1:
                h = self.activation_fn(h)
        return h   # logits

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def train_step(self, X: Tensor, y_onehot: Tensor) -> float:
        """
        Effectue un pas d'entraînement sur un batch.

        Paramètres
        ----------
        X        : Tensor (batch_size, n_features)
        y_onehot : Tensor (batch_size, n_classes) — labels one-hot

        Retourne
        --------
        float — valeur de la loss Cross-Entropy
        """
        self.opt.zero_grad()
        logits = self.forward(X)
        loss = cross_entropy_loss(logits, y_onehot)
        loss.backward()
        self.opt.step()
        return float(loss.data)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Retourne les classes prédites (argmax des logits)."""
        logits = self.forward(Tensor(X))
        return np.argmax(logits.data, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retourne les probabilités softmax."""
        logits = self.forward(Tensor(X))
        ls = log_softmax(logits, axis=1)
        return np.exp(ls.data)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calcule l'accuracy sur l'ensemble X, y."""
        return float(np.mean(self.predict(X) == y.astype(int)))

    def get_grad_norms(self) -> list:
        """
        Retourne la norme de Frobenius du gradient de chaque couche.
        Utile pour analyser le vanishing/exploding gradient.
        """
        norms = []
        for layer in self.layers:
            norms.append(float(np.linalg.norm(layer.W.grad)))
        return norms

    @property
    def parameters(self) -> list:
        params = []
        for layer in self.layers:
            params.extend(layer.parameters)
        return params

    def __repr__(self):
        dims = [self.layers[0].in_features] + [l.out_features for l in self.layers]
        return f"MLP({dims}, activation='{self.activation_name}')"
