#  Mini ML Framework — From Scratch

> **Autodiff · Optimiseurs · MLP · Kernel Ridge · MNIST**  
> Implémentation complète d'un framework de Machine Learning en **Python + NumPy uniquement** — sans PyTorch, TensorFlow ni scikit-learn.

**Auteur :** AHNANI Ali

---

## Vue d'ensemble

Ce projet construit un pipeline ML moderne entièrement from scratch, de la différentiation automatique jusqu'à l'entraînement sur données réelles. Chaque brique algorithmique est implémentée à partir des mathématiques, avec une analyse de complexité empiriquement validée.

```
Tensor (autodiff) → Optimiseur → Modèle → Données → Expériences → Rapport
```

---

## Architecture

```
mini-ml-framework/
│
├── autograd/
│   └── tensor.py            # Classe Tensor + graphe computationnel + backward
│
├── optim/
│   ├── sgd.py               # SGD + Momentum
│   ├── rmsprop.py           # RMSProp
│   └── adam.py              # Adam (correction biais)
│
├── models/
│   ├── linear.py            # Couche Dense
│   ├── activations.py       # ReLU, GELU, Softmax stable
│   ├── losses.py            # BCE, CrossEntropy (log-sum-exp)
│   ├── logistic.py          # Régression Logistique
│   ├── mlp.py               # MLP Profond (Xavier/He init)
│   └── kernel.py            # Kernel Ridge Regression (RBF, Gram O(n³))
│
├── data/
│   ├── mnist_loader.py      # Chargement MNIST format IDX binaire
│   └── synthetic.py         # Datasets synthétiques (spirale, gaussiennes)
│
├── training/
│   ├── trainer.py           # Boucle d'entraînement mini-batch
│   ├── scheduler.py         # Cosine Annealing LR
│   └── early_stopping.py    # Early Stopping (critère Prechelt)
│
├── utils/
│   ├── init.py              # Xavier / He initialization
│   ├── grad_clip.py         # Gradient Clipping
│   └── metrics.py           # Accuracy, MSE
│
├── experiments/             # Scripts de reproduction des expériences
├── notebooks/               # Notebook interactif complet
├── figures/                 # Figures générées
└── report/                  # Rapport LaTeX + PDF
```

---

## Contenu détaillé

### Partie 1 — Moteur Autodiff

Moteur de différentiation automatique en **mode reverse** (backpropagation généralisée).

- Classe `Tensor` avec graphe computationnel **dynamique** (define-by-run)
- **Tri topologique** + propagation backward automatique
- Accumulation de gradients (gestion des nœuds réutilisés)
- Gestion du **broadcasting NumPy** dans les gradients

**Opérations supportées :**

| Opération | Forward | Backward |
|-----------|---------|----------|
| `add`, `mul` | `a+b`, `a*b` | règle produit |
| `matmul` | `A @ B` | `G @ Bᵀ`, `Aᵀ @ G` |
| `exp`, `log` | `eˣ`, `log x` | `eˣ`, `1/x` |
| `relu`, `gelu` | `max(0,x)`, `x·σ(1.702x)` | dérivées analytiques |
| `sum`, `mean` | réductions | broadcast inverse |
| `pow`, `reshape`, `transpose` | — | règle puissance, reshape inverse |

**Complexité :** Temps `O(|graph|)` · Mémoire `O(|graph|)`

```python
from autograd.tensor import Tensor

x = Tensor([[1.0, 2.0], [3.0, 4.0]])
W = Tensor([[0.5, -1.0], [1.0, 0.5]])
b = Tensor([[0.1, 0.1]])

out = (x @ W + b).relu().mean()
out.backward()

print(W.grad)  # ∂loss/∂W calculé automatiquement
```

---

### Partie 2 — Optimiseurs

| Optimiseur | Mise à jour | Mémoire | Adaptatif |
|-----------|------------|---------|-----------|
| SGD | `θ ← θ − η∇L` | O(p) | Non |
| Momentum | `v ← βv + ∇L`, `θ ← θ − ηv` | O(2p) | Non |
| RMSProp | normalisation par `√v` | O(2p) | Oui |
| Adam | moments 1 et 2 + correction biais | O(3p) | Oui |

```python
from optim.adam import Adam

optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

### Partie 3 — Modèles

#### Régression Logistique
- Loss **Binary Cross-Entropy** (dérivée du maximum de vraisemblance)
- Fonction **convexe** → convergence vers le minimum global garantie
- Gradient via autodiff : `∂L/∂w = (1/n) Xᵀ(σ(Xw) − y)`

#### MLP Profond
- Couches **Dense** enchaînées avec activations non-linéaires
- Activations : **ReLU** et **GELU** (approximation sigmoïde)
- Initialisation **He** (ReLU) et **Xavier/Glorot** (autre)
- **Softmax numérique stable** via log-sum-exp trick
- Analyse du **vanishing/exploding gradient** par couche

```python
from models.mlp import MLP

model = MLP(
    layer_dims=[784, 256, 128, 10],
    activation='relu',
    weight_decay=1e-4
)
```

#### Kernel Ridge Regression
- Noyau **RBF** : `k(x, x') = exp(−γ‖x − x'‖²)`
- **Théorème du représentant** → solution `α = (K + λI)⁻¹y`
- Construction efficace de la **matrice de Gram** via identité `‖x−x'‖² = ‖x‖² + ‖x'‖² − 2xᵀx'`
- Complexité : **O(n³)** fit · **O(n²)** mémoire

```python
from models.kernel import KernelRidgeRegression

krr = KernelRidgeRegression(gamma=2.0, lam=0.1)
krr.fit(X_train, y_train)
y_pred = krr.predict(X_test)
```

---

### Partie 4 — Dataset MNIST

Chargement des fichiers MNIST au **format binaire IDX** sans aucune dépendance externe :

```python
from data.mnist_loader import load_mnist

X_train, y_train, X_test, y_test = load_mnist('path/to/mnist/')
# X_train : (60000, 784), valeurs normalisées dans [0, 1]
```

Format IDX : `[magic 4B] [dims 4B×n] [données uint8 big-endian]`

---

### Partie 5 — Expériences

Toutes les expériences sont reproductibles avec un seed fixé (`np.random.seed(42)`).

| Expérience | Script | Résultat |
|-----------|--------|---------|
| Comparaison optimiseurs | `experiments/exp_optimizers.py` | Adam converge ~3× plus vite que SGD |
| Impact learning rate | `experiments/exp_lr.py` | Fenêtre optimale `η ∈ [1e-3, 1e-2]` pour Adam |
| Régularisation L2 | `experiments/exp_regularization.py` | `λ ≈ 1e-4` réduit le gap train-test |
| Logistic vs MLP vs KRR | `experiments/exp_models.py` | MLP et KRR dominent sur non-linéaire |
| Benchmark complexité | `experiments/exp_complexity.py` | Pente log-log KRR ≈ 2.9 ≈ O(n³) |

---

### Partie 6 — Analyse de Complexité

| Composant | Temps | Mémoire |
|-----------|-------|---------|
| Autodiff (forward + backward) | O(\|graph\|) | O(\|graph\|) |
| MLP — L couches, largeur d, batch n | O(L·n·d²) | O(L·n·d) |
| Adam — p paramètres | O(p) / step | O(3p) |
| KRR — fit | **O(n³)** | O(n²) |
| KRR — predict | O(n\_test · n) | O(n\_test · n) |

---

### Fonctionnalités Avancées

#### Gradient Clipping
```python
from utils.grad_clip import gradient_clipping

total_norm = gradient_clipping(model.parameters(), max_norm=1.0)
```

#### Cosine Annealing LR Scheduler
```python
from training.scheduler import CosineAnnealingScheduler

scheduler = CosineAnnealingScheduler(optimizer, T_max=100, lr_min=1e-6)
# lr(t) = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(π·t/T))
scheduler.step()
```

#### Early Stopping
```python
from training.early_stopping import EarlyStopping

es = EarlyStopping(patience=15, delta=1e-4)
if es(val_loss):
    print("Arrêt anticipé")
    break
```

---

## Démarrage rapide

### Prérequis

```
Python >= 3.8
numpy >= 1.21
matplotlib >= 3.4
```

### Installation

```bash
git clone https://github.com/ton-username/mini-ml-framework.git
cd mini-ml-framework
pip install -r requirements.txt
```

### Lancer le notebook

```bash
jupyter notebook notebooks/mini_framework_ml_AHNANI_Ali.ipynb
```

### Reproduire une expérience

```bash
python experiments/exp_optimizers.py
python experiments/exp_models.py
python experiments/exp_complexity.py
```

### Exemple minimal end-to-end

```python
import numpy as np
from autograd.tensor import Tensor
from models.mlp import MLP
from optim.adam import Adam
from utils.metrics import accuracy

# Données
X = np.random.randn(200, 4)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Modèle
model = MLP([4, 32, 2], activation='relu')
optimizer = Adam(model.parameters(), lr=1e-3)

# Entraînement
for epoch in range(100):
    optimizer.zero_grad()
    logits = model.forward(Tensor(X))
    loss = model.ce_loss(logits, Tensor(to_onehot(y, 2)))
    loss.backward()
    optimizer.step()

print(f"Accuracy : {accuracy(model.predict(X), y):.4f}")
```

---

## Résultats

| Modèle | Dataset Spirale (3 cls) | Temps fit |
|--------|------------------------|-----------|
| Logistique (OvR) | ~60% | < 1s |
| MLP [2→64→64→3] | ~97% | ~3s |
| KRR RBF (OvR) | ~95% | ~0.1s |

Convergence optimiseurs sur Rosenbrock : Adam > RMSProp > Momentum > SGD

---

## Structure mathématique

Le rapport complet (`report/Rapport_ML_AHNANI_Ali.pdf`) couvre :

- **Autodiff :** preuve de correction du backward par tri topologique, complexité `O(|graph|)`
- **SGD :** théorème de convergence `O(DG/√T)` pour fonctions convexes Lipschitz
- **Adam :** invariance à l'échelle, justification de la correction du biais
- **BCE :** preuve de convexité via Hessienne `Xᵀ diag(p(1-p)) X ⪰ 0`
- **He init :** préservation de la variance avec ReLU (`σ² = 2/d_in`)
- **RKHS :** théorème du représentant (Kimeldorf & Wahba 1971), décomposition biais-variance
- **NTK :** régime lazy training et connexion MLP infini ↔ KRR (Jacot et al. 2018)

---

## `requirements.txt`

```
numpy>=1.21
matplotlib>=3.4
```

---

## Licence

MIT — libre d'utilisation, de modification et de distribution.

---

*AHNANI Ali — Mini ML Framework From Scratch*
