"""
training/early_stopping.py
===========================
Early Stopping — arrêt anticipé de l'entraînement.

Critère de Prechelt (1998) :
    Generalization Loss GL(t) = 100 * (L_val(t) / min_{s<=t} L_val(s) - 1)
    Stop si GL(t) > alpha

Implémentation simplifiée (patience + delta) :
    - Si la loss de validation ne s'améliore pas de plus de `delta`
      pendant `patience` epochs consécutives → stop.

Connexion théorique :
    Early stopping avec GD sur fonction convexe quadratique ≡ régularisation L2
    Le nombre d'itérations joue le rôle de 1/lambda (Yao et al., 2007).

Auteur : AHNANI Ali
"""


class EarlyStopping:
    """
    Arrêt anticipé basé sur la surveillance de la loss de validation.

    Paramètres
    ----------
    patience : int   — nombre d'epochs sans amélioration avant l'arrêt
    delta    : float — amélioration minimale considérée comme significative
    mode     : str   — 'min' (surveille une loss) ou 'max' (surveille accuracy)
    """

    def __init__(self, patience: int = 10, delta: float = 1e-4,
                 mode: str = 'min'):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best = None
        self.should_stop = False

    def __call__(self, metric: float) -> bool:
        """
        Appeler à chaque epoch avec la métrique surveillée.

        Retourne True si l'entraînement doit s'arrêter.
        """
        if self.best is None:
            self.best = metric
            return False

        if self.mode == 'min':
            improved = metric < self.best - self.delta
        else:
            improved = metric > self.best + self.delta

        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True

        return self.should_stop

    def reset(self):
        """Remet l'état à zéro."""
        self.counter = 0
        self.best = None
        self.should_stop = False

    @property
    def status(self) -> str:
        return f"EarlyStopping(patience={self.patience}, counter={self.counter}/{self.patience}, best={self.best:.6f})"
