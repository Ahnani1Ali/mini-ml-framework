"""
experiments/exp_regularization.py
===================================
Expérience 3 : Régularisation L2 et overfitting.

Entraîne un MLP sur un petit dataset (n=80) pour forcer l'overfitting,
puis compare 6 valeurs de λ (weight decay).

Visualise :
  - Accuracy train vs test
  - Gap d'overfitting (train - test) en fonction de λ

Résultat attendu :
  - λ=0     → train acc ≈ 1.0, test acc faible (overfitting)
  - λ≈1e-4  → meilleur compromis biais-variance
  - λ grand  → underfitting (train et test acc baissent)

Lancer : python experiments/exp_regularization.py

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
N_EPOCHS   = 120
BATCH_SIZE = 32
ARCH       = [16, 128, 128, 5]
N_SMALL    = 80    # petit dataset → overfitting facile

X_tr_full, y_tr_full, X_te, y_te = make_mnist_like(
    n_samples=1000, n_classes=5, n_features=16, seed=42
)
mu, sg = X_tr_full.mean(0), X_tr_full.std(0) + 1e-8
X_tr_full = (X_tr_full - mu) / sg
X_te      = (X_te - mu) / sg

X_small   = X_tr_full[:N_SMALL]
y_small   = y_tr_full[:N_SMALL]
Y_small_oh = to_onehot(y_small, 5)

# ── Sweep de λ ───────────────────────────────────────────────────────────────
lambda_values = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1]
cm = plt.cm.viridis(np.linspace(0.1, 0.95, len(lambda_values)))

results = {}
for lam in lambda_values:
    print(f"  λ = {lam:.0e}...")
    np.random.seed(42)
    model = MLP(ARCH, activation='relu', optimizer='adam',
                lr=1e-3, weight_decay=lam)
    tr_accs, te_accs = [], []
    for ep in range(N_EPOCHS):
        perm = np.random.permutation(N_SMALL)
        for st in range(0, N_SMALL, BATCH_SIZE):
            ib = perm[st:st + BATCH_SIZE]
            model.train_step(Tensor(X_small[ib]), Tensor(Y_small_oh[ib]))
        tr_accs.append(model.accuracy(X_small, y_small))
        te_accs.append(model.accuracy(X_te, y_te))
    results[lam] = (tr_accs, te_accs)
    print(f"    → Train={tr_accs[-1]:.4f}  Test={te_accs[-1]:.4f}  Gap={tr_accs[-1]-te_accs[-1]:.4f}")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for (lam, (tr, te)), c in zip(results.items(), cm):
    label = f'λ={lam:.0e}'
    axes[0].plot(tr, color=c, lw=2, label=label)
    axes[1].plot(te, color=c, lw=2, label=label)
    axes[2].plot([t - e for t, e in zip(tr, te)], color=c, lw=2, label=label)

for ax, title in zip(axes, ["Accuracy Train", "Accuracy Test", "Gap Overfitting (train−test)"]):
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.legend(fontsize=8)

axes[2].axhline(y=0, color='k', ls='--', alpha=0.4, lw=1)

plt.suptitle("Impact de la Régularisation L2 (Weight Decay)",
             fontsize=14, fontweight='bold')
plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/fig_regularization.png', dpi=150, bbox_inches='tight')
print("Figure sauvegardée : figures/fig_regularization.png")
plt.close()
