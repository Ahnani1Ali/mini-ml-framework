"""
experiments/exp_lr.py
=====================
Expérience 2 : Impact du learning rate.

Teste 6 valeurs de lr avec Adam sur la même architecture.
Visualise la convergence en échelle linéaire et log.

Résultat attendu :
  - lr trop petit (1e-5) → convergence très lente
  - lr optimal (1e-3 à 1e-2) → convergence rapide et stable
  - lr trop grand (0.5) → oscillations et divergence

Lancer : python experiments/exp_lr.py

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
N_EPOCHS   = 80
BATCH_SIZE = 64
ARCH       = [16, 64, 5]

X_tr, y_tr, X_te, y_te = make_mnist_like(n_samples=1000, n_classes=5,
                                          n_features=16, seed=42)
mu, sg = X_tr.mean(0), X_tr.std(0) + 1e-8
X_tr = (X_tr - mu) / sg
X_te = (X_te - mu) / sg
Y_tr_oh = to_onehot(y_tr, 5)

# ── Valeurs de lr testées ─────────────────────────────────────────────────────
lr_values = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5]
cm = plt.cm.plasma(np.linspace(0.1, 0.9, len(lr_values)))

results = {}
for lr in lr_values:
    print(f"  lr = {lr:.0e}...")
    np.random.seed(42)
    model = MLP(ARCH, activation='relu', optimizer='adam', lr=lr)
    losses = []
    n = len(X_tr)
    for ep in range(N_EPOCHS):
        perm = np.random.permutation(n)
        el = 0; nb = 0
        for st in range(0, n, BATCH_SIZE):
            ib = perm[st:st + BATCH_SIZE]
            el += model.train_step(Tensor(X_tr[ib]), Tensor(Y_tr_oh[ib]))
            nb += 1
        loss_ep = el / nb
        # Cap pour la visualisation (divergence)
        losses.append(min(loss_ep, 20.0))
    results[lr] = losses
    print(f"    → Loss finale = {losses[-1]:.4f}")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
for (lr, ls), c in zip(results.items(), cm):
    a1.plot(ls, label=f'lr={lr:.0e}', color=c, lw=2)
    a2.plot(ls, label=f'lr={lr:.0e}', color=c, lw=2)

a1.set_title("Convergence (échelle linéaire)", fontweight='bold')
a1.set_xlabel("Epoch"); a1.set_ylabel("Loss")
a1.set_ylim(0, 8); a1.legend()

a2.set_yscale('log')
a2.set_title("Convergence (échelle log)", fontweight='bold')
a2.set_xlabel("Epoch"); a2.set_ylabel("Loss (log)")
a2.legend()

plt.suptitle("Impact du Learning Rate — Adam", fontsize=14, fontweight='bold')
plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/fig_lr_impact.png', dpi=150, bbox_inches='tight')
print("Figure sauvegardée : figures/fig_lr_impact.png")
plt.close()
