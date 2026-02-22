"""
data/mnist_loader.py
====================
Chargement du dataset MNIST depuis les fichiers binaires au format IDX.

Format IDX (sans bibliothèque externe, seulement struct + gzip) :
    [magic 4B] [dim1 4B] [dim2 4B] ... [données uint8 big-endian]

Le magic number encode :
    - octets 0-1 : toujours 0x00 0x00
    - octet  2   : type de données (0x08 = uint8)
    - octet  3   : nombre de dimensions

Fichiers MNIST attendus :
    train-images-idx3-ubyte.gz   (60000 images 28×28)
    train-labels-idx1-ubyte.gz   (60000 labels)
    t10k-images-idx3-ubyte.gz    (10000 images 28×28)
    t10k-labels-idx1-ubyte.gz    (10000 labels)

Téléchargement : http://yann.lecun.com/exdb/mnist/

Auteur : AHNANI Ali
"""

import struct
import gzip
import os
import numpy as np


def read_idx(filepath: str) -> np.ndarray:
    """
    Lit un fichier IDX (compressé .gz ou non) et retourne un np.ndarray.

    Paramètres
    ----------
    filepath : str — chemin vers le fichier .gz ou brut

    Retourne
    --------
    np.ndarray de dtype uint8, forme selon les dimensions du fichier
    """
    opener = gzip.open if filepath.endswith('.gz') else open

    with opener(filepath, 'rb') as f:
        # Magic number (4 octets, big-endian)
        magic = struct.unpack('>I', f.read(4))[0]
        n_dims = magic & 0xFF          # nombre de dimensions (octet de poids faible)
        dtype_code = (magic >> 8) & 0xFF

        # Vérification du type de données
        dtype_map = {0x08: np.uint8, 0x09: np.int8, 0x0B: np.int16,
                     0x0C: np.int32, 0x0D: np.float32, 0x0E: np.float64}
        if dtype_code not in dtype_map:
            raise ValueError(f"Type de données non reconnu : 0x{dtype_code:02X}")

        # Dimensions (4 octets chacune, big-endian)
        dims = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(n_dims))

        # Données brutes
        data = np.frombuffer(f.read(), dtype=dtype_map[dtype_code])

    return data.reshape(dims)


def load_mnist(data_dir: str = '.', normalize: bool = True,
               flatten: bool = True) -> tuple:
    """
    Charge les 4 fichiers MNIST et retourne (X_train, y_train, X_test, y_test).

    Paramètres
    ----------
    data_dir  : str  — répertoire contenant les fichiers MNIST
    normalize : bool — diviser les pixels par 255 pour avoir des valeurs ∈ [0, 1]
    flatten   : bool — aplatir les images 28×28 en vecteurs de 784

    Retourne
    --------
    X_train : np.ndarray, shape (60000, 784) si flatten else (60000, 28, 28)
    y_train : np.ndarray, shape (60000,)  — labels entiers 0-9
    X_test  : np.ndarray, shape (10000, 784) si flatten else (10000, 28, 28)
    y_test  : np.ndarray, shape (10000,)
    """
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images':  't10k-images-idx3-ubyte.gz',
        'test_labels':  't10k-labels-idx1-ubyte.gz',
    }

    # Vérification de l'existence des fichiers
    for key, fname in files.items():
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            # Chercher aussi la version non compressée
            path_raw = path.replace('.gz', '')
            if not os.path.exists(path_raw):
                raise FileNotFoundError(
                    f"Fichier MNIST introuvable : {path}\n"
                    f"Téléchargez depuis : http://yann.lecun.com/exdb/mnist/"
                )
            files[key] = os.path.basename(path_raw)

    def _load(fname):
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path) and os.path.exists(path + '.gz'):
            path = path + '.gz'
        return read_idx(path)

    X_train = _load(files['train_images']).astype(np.float64)
    y_train = _load(files['train_labels']).astype(np.int64)
    X_test  = _load(files['test_images']).astype(np.float64)
    y_test  = _load(files['test_labels']).astype(np.int64)

    if normalize:
        X_train /= 255.0
        X_test  /= 255.0

    if flatten:
        X_train = X_train.reshape(len(X_train), -1)   # (60000, 784)
        X_test  = X_test.reshape(len(X_test),  -1)    # (10000, 784)

    print(f"MNIST chargé : X_train={X_train.shape}, X_test={X_test.shape}")
    print(f"              Classes : {np.unique(y_train)}")
    return X_train, y_train, X_test, y_test
