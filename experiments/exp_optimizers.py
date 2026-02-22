"""
experiments/exp_optimizers.py
==============================
Expérience 1 : Comparaison des optimiseurs.

Compare SGD, Momentum, RMSProp et Adam sur un dataset synthétique.
Trace la loss et l'accuracy test par epoch pour chaque optimiseur.

Résultat attendu : Adam converge le plus vite, SGD le plus lentement.

Lancer : python experiments/exp_optimizers.py

Auteur : AHNANI Ali
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from autograd.tensor import Tensor
from models.mlp import MLP, to_onehot
from data.synthetic import make_mnist_like

# ── Config ──────────────────────────────────────────────────────────────────
np.random.seed(42)
N_EPOCHS   = 100
BATCH_SIZE = 64
ARCH       = [16, 64, 5]      # [input, hidden, n_classes]

# ── Données ──────────────────────────────────────────────────────────────────
X_tr, y_tr, X_te, y_te = make_mnist_like(n_samples=1000, n_classes=5,
                                          n_features=16, seed=42)
mu, sg = X_tr.mean(0), X_tr.std(0) + 1e-8
X_tr = (X_tr - mu) / sg
X_te = (X_te - mu) / sg
Y_tr_oh = to_onehot(y_tr, 5)

# ── Configs des optimiseurs ───────────────────────────────────────────────────
configs = [
    ('SGD  lr=0.10',   'sgd',      0.10),
    ('Mom  lr=0.05',   'momentum', 0.05),
    ('Adam lr=1e-3',   'adam',     1e-3),
    ('Adam lr=1e-4',   'adam',     1e-4),
]
COLORS = ['#e74c3c', '#f39c12', '#2563eb', '#16a34a']

results = {}
for name, opt, lr in configs:
    print(f"  Entraînement : {name}...")
    np.random.seed(42)
    model = MLP(ARCH, activation='relu', optimizer=opt, lr=lr)
    losses, accs = [], []
    n = len(X_tr)
    for ep in range(N_EPOCHS):
        perm = np.random.permutation(n)
        el = 0; nb = 0
        for st in range(0, n, BATCH_SIZE):
            ib = perm[st:st + BATCH_SIZE]
            el += model.train_step(Tensor(X_tr[ib]), Tensor(Y_tr_oh[ib]))
            nb += 1
        losses.append(el / nb)
        accs.append(model.accuracy(X_te, y_te))
    results[name] = (losses, accs)
    print(f"    → Acc finale = {accs[-1]:.4f}")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
for (name, (ls, ac)), c in zip(results.items(), COLORS):
    a1.plot(ls, label=name, color=c, lw=2)
    a2.plot(ac, label=name, color=c, lw=2)

a1.set_title("Cross-Entropy Loss par epoch", fontweight='bold')
a1.set_xlabel("Epoch"); a1.set_ylabel("Loss"); a1.legend()

a2.set_title("Accuracy Test par epoch", fontweight='bold')
a2.set_xlabel("Epoch"); a2.set_ylabel("Accuracy")
a2.set_ylim(0, 1.05); a2.legend()

plt.suptitle("Comparaison des optimiseurs", fontsize=14, fontweight='bold')
plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/fig_exp_optimizers.png', dpi=150, bbox_inches='tight')
print("Figure sauvegardée : figures/fig_exp_optimizers.png")
plt.close()
