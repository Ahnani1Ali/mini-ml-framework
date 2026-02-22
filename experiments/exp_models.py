"""
experiments/exp_models.py
==========================
Expérience 4 : Comparaison des modèles sur dataset spirale.

Compare les frontières de décision de :
  - Régression Logistique OvR (frontières linéaires)
  - MLP [2→64→64→3]            (frontières non-linéaires)
  - Kernel Ridge Regression OvR (frontières RKHS)

Résultat attendu :
  - Logistique incapable de séparer les spirales (problème non-linéaire)
  - MLP et KRR obtiennent des frontières courbes précises

Lancer : python experiments/exp_models.py

Auteur : AHNANI Ali
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from autograd.tensor import Tensor
from models.mlp import MLP, to_onehot
from models.logistic import MulticlassLogistic
from models.kernel import KernelClassifier
from data.synthetic import make_spirals, train_test_split

# ── Données ──────────────────────────────────────────────────────────────────
np.random.seed(42)
X, y = make_spirals(n_per_class=100, n_classes=3, noise=0.2, seed=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, seed=42)
Y_tr_oh = to_onehot(y_tr, 3)

# ── Entraînement ──────────────────────────────────────────────────────────────
print("Entraînement Logistique OvR...")
t0 = time.time()
log_model = MulticlassLogistic(n_features=2, n_classes=3, lr=0.3)
log_model.train(X_tr, y_tr, n_epochs=150)
t_log = time.time() - t0
acc_log = log_model.accuracy(X_te, y_te)
print(f"  Acc={acc_log:.4f}  Temps={t_log:.2f}s")

print("Entraînement MLP [2→64→64→3]...")
np.random.seed(42)
t0 = time.time()
mlp = MLP([2, 64, 64, 3], activation='relu', optimizer='adam', lr=5e-3)
n = len(X_tr)
for ep in range(300):
    perm = np.random.permutation(n)
    for st in range(0, n, 64):
        ib = perm[st:st + 64]
        mlp.train_step(Tensor(X_tr[ib]), Tensor(Y_tr_oh[ib]))
t_mlp = time.time() - t0
acc_mlp = mlp.accuracy(X_te, y_te)
print(f"  Acc={acc_mlp:.4f}  Temps={t_mlp:.2f}s")

print("Entraînement KRR OvR (gamma=2, lambda=0.05)...")
t0 = time.time()
krr = KernelClassifier(gamma=2.0, lam=0.05)
krr.fit(X_tr, y_tr)
t_krr = time.time() - t0
acc_krr = krr.accuracy(X_te, y_te)
print(f"  Acc={acc_krr:.4f}  Temps={t_krr:.2f}s")

# ── Grille de décision ────────────────────────────────────────────────────────
res = 200
gx, gy = np.meshgrid(np.linspace(-1.5, 1.5, res), np.linspace(-1.5, 1.5, res))
grid = np.c_[gx.ravel(), gy.ravel()]

pg_log = log_model.predict(grid).reshape(gx.shape)
pg_mlp = mlp.predict(grid).reshape(gx.shape)
pg_krr = krr.predict(grid).reshape(gx.shape)

# ── Figure ────────────────────────────────────────────────────────────────────
cls_colors = ['#e74c3c', '#3498db', '#2ecc71']
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

models_info = [
    ('Logistique OvR',     pg_log, acc_log, t_log),
    ('MLP [2→64→64→3]',    pg_mlp, acc_mlp, t_mlp),
    ('KRR RBF OvR',        pg_krr, acc_krr, t_krr),
]
for ax, (name, pg, acc, t) in zip(axes, models_info):
    ax.contourf(gx, gy, pg, alpha=0.25, cmap='Set2',
                levels=[-0.5, 0.5, 1.5, 2.5])
    ax.contour(gx, gy, pg, colors='white', linewidths=0.5,
               levels=[0.5, 1.5], alpha=0.6)
    for k in range(3):
        m = y_te == k
        ax.scatter(X_te[m, 0], X_te[m, 1], c=cls_colors[k], s=35,
                   edgecolors='white', linewidths=0.5, zorder=2, alpha=0.9)
    ax.set_title(f'{name}\nAcc={acc:.3f}  t={t:.2f}s',
                 fontweight='bold', fontsize=11)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('x₁'); ax.set_ylabel('x₂')

plt.suptitle("Comparaison des modèles — Dataset Spirale (3 classes)",
             fontsize=14, fontweight='bold')
plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/fig_model_comparison.png', dpi=150, bbox_inches='tight')
print("Figure sauvegardée : figures/fig_model_comparison.png")
plt.close()

# ── Tableau récapitulatif ─────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"{'Modèle':<22} {'Acc Test':>10} {'Temps (s)':>12} {'Frontière'}")
print("-"*60)
print(f"{'Logistique OvR':<22} {acc_log:>10.4f} {t_log:>12.3f}   Linéaire")
print(f"{'MLP [2→64→64→3]':<22} {acc_mlp:>10.4f} {t_mlp:>12.3f}   Non-linéaire")
print(f"{'KRR RBF OvR':<22} {acc_krr:>10.4f} {t_krr:>12.3f}   RKHS")
print("="*60)
