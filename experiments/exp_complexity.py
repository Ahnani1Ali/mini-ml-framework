"""
experiments/exp_complexity.py
==============================
Expérience 5 : Validation empirique des complexités algorithmiques.

Mesure les temps d'exécution en fonction de la taille des données
et vérifie les pentes en log-log pour valider les O théoriques.

Tests :
  1. Autodiff : temps backward vs profondeur → O(profondeur)
  2. MLP forward : temps vs n (taille batch)  → O(n)
  3. KRR fit : temps vs n                      → O(n³) (pente log-log ≈ 3)

Lancer : python experiments/exp_complexity.py

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
from models.kernel import KernelRidgeRegression

np.random.seed(42)

# ── 1. Autodiff : profondeur → temps backward ─────────────────────────────────
print("Test 1 : Autodiff — temps backward vs profondeur...")
depths     = [2, 4, 6, 8, 10, 12, 15, 20]
times_ad   = []
n_reps     = 10

for depth in depths:
    times = []
    for _ in range(n_reps):
        x = Tensor(np.random.randn(32, 64))
        h = x
        W_list = [Tensor(np.random.randn(64, 64) * 0.1) for _ in range(depth)]
        for W in W_list:
            h = (h @ W).relu()
        loss = h.mean()
        t0 = time.perf_counter()
        loss.backward()
        times.append(time.perf_counter() - t0)
    times_ad.append(np.mean(times))
    print(f"  depth={depth:3d} → {times_ad[-1]*1000:.3f} ms")

# ── 2. MLP forward : n → temps ────────────────────────────────────────────────
print("\nTest 2 : MLP forward — temps vs batch size n...")
n_values   = [50, 100, 200, 500, 1000, 2000]
times_mlp  = []
model_cx   = MLP([64, 128, 64, 10], activation='relu', lr=1e-3)

for n in n_values:
    X_cx = np.random.randn(n, 64)
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        _ = model_cx.forward(Tensor(X_cx))
        times.append(time.perf_counter() - t0)
    times_mlp.append(np.mean(times))
    print(f"  n={n:5d} → {times_mlp[-1]*1000:.3f} ms")

# ── 3. KRR : n → temps fit ────────────────────────────────────────────────────
print("\nTest 3 : KRR fit — temps vs n (attendu : O(n³))...")
n_values_krr = [50, 100, 200, 300, 400, 500]
times_krr    = []

for n in n_values_krr:
    X_k = np.random.randn(n, 4)
    y_k = np.sin(X_k[:, 0])
    times = []
    for _ in range(3):
        krr = KernelRidgeRegression(gamma=1.0, lam=0.1)
        t0 = time.perf_counter()
        krr.fit(X_k, y_k)
        times.append(time.perf_counter() - t0)
    times_krr.append(np.mean(times))
    print(f"  n={n:5d} → {times_krr[-1]*1000:.3f} ms")

# ── Régression log-log pour estimer les pentes ────────────────────────────────
log_n_krr  = np.log(n_values_krr)
log_t_krr  = np.log(times_krr)
slope_krr, intercept_krr = np.polyfit(log_n_krr, log_t_krr, 1)
print(f"\nPente log-log KRR = {slope_krr:.2f}  (théorique : 3.0)")

log_n_ad = np.log(depths)
log_t_ad = np.log(times_ad)
slope_ad, _ = np.polyfit(log_n_ad, log_t_ad, 1)
print(f"Pente log-log Autodiff = {slope_ad:.2f}  (théorique : 1.0)")

log_n_mlp = np.log(n_values)
log_t_mlp = np.log(times_mlp)
slope_mlp, _ = np.polyfit(log_n_mlp, log_t_mlp, 1)
print(f"Pente log-log MLP = {slope_mlp:.2f}  (théorique : 1.0)")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Autodiff
axes[0].plot(depths, [t * 1000 for t in times_ad],
             'o-', color='#2563eb', lw=2, ms=6)
axes[0].set_title(f'Autodiff backward\npente log-log = {slope_ad:.2f}',
                  fontweight='bold')
axes[0].set_xlabel('Profondeur du graphe'); axes[0].set_ylabel('Temps (ms)')

# MLP
axes[1].plot(n_values, [t * 1000 for t in times_mlp],
             's-', color='#16a34a', lw=2, ms=6)
# Fit linéaire pour visualisation
coeffs = np.polyfit(n_values, times_mlp, 1)
n_fit = np.array(n_values)
axes[1].plot(n_fit, np.poly1d(coeffs)(n_fit) * 1000,
             '--', color='gray', alpha=0.7, label='Fit linéaire')
axes[1].set_title(f'MLP forward\npente log-log = {slope_mlp:.2f}',
                  fontweight='bold')
axes[1].set_xlabel('Taille batch n'); axes[1].set_ylabel('Temps (ms)')
axes[1].legend()

# KRR log-log
axes[2].loglog(n_values_krr, [t * 1000 for t in times_krr],
               '^-', color='#e74c3c', lw=2, ms=6, label='KRR fit')
# Ligne de référence O(n³)
n_ref = np.array(n_values_krr)
ref   = np.exp(intercept_krr) * n_ref ** slope_krr * 1000
axes[2].loglog(n_ref, ref, '--', color='gray', alpha=0.7,
               label=f'pente={slope_krr:.2f}')
axes[2].set_title(f'KRR fit (log-log)\npente = {slope_krr:.2f} ≈ O(n³)',
                  fontweight='bold')
axes[2].set_xlabel('n'); axes[2].set_ylabel('Temps (ms)')
axes[2].legend()

plt.suptitle("Validation empirique des complexités algorithmiques",
             fontsize=14, fontweight='bold')
plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/fig_complexity.png', dpi=150, bbox_inches='tight')
print("\nFigure sauvegardée : figures/fig_complexity.png")
plt.close()
