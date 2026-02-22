"""
training/trainer.py
===================
Boucle d'entraînement générique avec mini-batch SGD.

Gère :
  - Shuffle des données à chaque epoch
  - Mini-batches de taille configurable
  - Calcul périodique des métriques (train + validation)
  - Intégration du scheduler LR et de l'early stopping
  - Historique complet pour analyse post-entraînement

Auteur : AHNANI Ali
"""

import numpy as np
import time
from autograd.tensor import Tensor
from models.mlp import to_onehot


class Trainer:
    """
    Boucle d'entraînement pour les modèles MLP et Logistique.

    Paramètres
    ----------
    model      : modèle avec .train_step(X, y) et .accuracy(X, y)
    scheduler  : optionnel — CosineAnnealingScheduler ou similaire
    early_stop : optionnel — EarlyStopping
    verbose    : int — 0=silencieux, 1=epoch, 10=tous les 10 epochs
    """

    def __init__(self, model, scheduler=None, early_stop=None,
                 verbose: int = 10):
        self.model = model
        self.scheduler = scheduler
        self.early_stop = early_stop
        self.verbose = verbose

        # Historique
        self.history = {
            'train_loss': [],
            'val_loss':   [],
            'train_acc':  [],
            'val_acc':    [],
            'lr':         [],
            'grad_norms': [],
        }

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            n_epochs: int = 100, batch_size: int = 64,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            n_classes: int = None) -> dict:
        """
        Lance l'entraînement.

        Paramètres
        ----------
        X_train    : np.ndarray (n, d)
        y_train    : np.ndarray (n,) — labels entiers
        n_epochs   : int
        batch_size : int
        X_val      : np.ndarray — données de validation (optionnel)
        y_val      : np.ndarray — labels de validation
        n_classes  : int — nombre de classes (déduit si None)

        Retourne
        --------
        dict — historique complet
        """
        if n_classes is None:
            n_classes = len(np.unique(y_train))

        Y_train_oh = to_onehot(y_train, n_classes)
        n = len(X_train)
        start_time = time.time()

        for epoch in range(1, n_epochs + 1):
            # ── Shuffle des données ──
            perm = np.random.permutation(n)
            X_shuf = X_train[perm]
            Y_shuf = Y_train_oh[perm]

            # ── Mini-batches ──
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n, batch_size):
                Xb = Tensor(X_shuf[start:start + batch_size])
                Yb = Tensor(Y_shuf[start:start + batch_size])
                loss = self.model.train_step(Xb, Yb)
                epoch_loss += loss
                n_batches += 1

            epoch_loss /= n_batches
            self.history['train_loss'].append(epoch_loss)

            # ── Métriques ──
            train_acc = self.model.accuracy(X_train, y_train)
            self.history['train_acc'].append(train_acc)

            if X_val is not None:
                val_acc = self.model.accuracy(X_val, y_val)
                self.history['val_acc'].append(val_acc)
            else:
                val_acc = None

            # ── Gradient norms ──
            if hasattr(self.model, 'get_grad_norms'):
                self.history['grad_norms'].append(self.model.get_grad_norms())

            # ── Scheduler LR ──
            if self.scheduler is not None:
                lr = self.scheduler.step()
                self.history['lr'].append(lr)

            # ── Early Stopping ──
            if self.early_stop is not None:
                if self.early_stop(epoch_loss):
                    if self.verbose > 0:
                        print(f"Early stopping à l'epoch {epoch}")
                    break

            # ── Logging ──
            if self.verbose > 0 and (epoch % self.verbose == 0 or epoch == 1):
                elapsed = time.time() - start_time
                msg = (f"Epoch {epoch:4d}/{n_epochs} | "
                       f"Loss={epoch_loss:.4f} | Train Acc={train_acc:.4f}")
                if val_acc is not None:
                    msg += f" | Val Acc={val_acc:.4f}"
                msg += f" | {elapsed:.1f}s"
                print(msg)

        return self.history
