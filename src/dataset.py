
"""
dataset.py
Carga y preparación del conjunto de datos CIFAR-10.
Evaluación de CNN - Universidad Andina del Cusco.
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split


# ============================================================
# 1. CONFIGURACIÓN Y SEMILLAS
# ============================================================

SEED = 73

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DATA_DIR = "data"
BATCH_SIZE = 64

CLASSES = (
    "avión", "automóvil", "pájaro", "gato", "ciervo",
    "perro", "rana", "caballo", "barco", "camión"
)


# ============================================================
# 2. TRANSFORMACIONES DE LAS IMÁGENES
# ============================================================

# Transformaciones para entrenamiento:
# - RandomCrop: recorta aleatoriamente la imagen.
# - RandomHorizontalFlip: voltea horizontalmente la imagen.
# - ToTensor: convierte la imagen en tensor.
# - Normalize: normaliza los tres canales RGB.

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])

# Para validación y prueba no aplicamos aumento aleatorio.

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])


# ============================================================
# 3. VISUALIZACIÓN DE 12 IMÁGENES
# ============================================================

def show_images(dataset, number=12):
    """Muestra imágenes originales con sus etiquetas."""

    figure, axes = plt.subplots(3, 4, figsize=(10, 7))

    for i, ax in enumerate(axes.flat):
        image, label = dataset[i]
        ax.imshow(image)
        ax.set_title(CLASSES[label])
        ax.axis("off")

    plt.suptitle("Ejemplos del conjunto CIFAR-10")
    plt.tight_layout()
    plt.show()


# ============================================================
# 4. CARGA Y DIVISIÓN ESTRATIFICADA
# ============================================================

def get_loaders():
    """
    Descarga CIFAR-10 y divide el conjunto original de
    entrenamiento en entrenamiento (80 %) y validación (20 %).

    La división es estratificada para conservar la proporción
    de las diez clases en ambos conjuntos.
    """

    os.makedirs(DATA_DIR, exist_ok=True)

    # Dataset original para visualizar imágenes sin transformar.
    raw_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=None
    )

    # Dataset con transformaciones para entrenamiento.
    train_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=False,
        transform=train_transform
    )

    # Dataset con transformaciones para validación.
    val_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=False,
        transform=test_transform
    )

    # Etiquetas de las 50 000 imágenes de entrenamiento.
    labels = np.array(raw_dataset.targets)
    indices = np.arange(len(raw_dataset))

    # División estratificada: 40 000 para entrenar y
    # 10 000 para validar.
    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.20,
        random_state=SEED,
        stratify=labels
    )

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    # Conjunto de prueba oficial de CIFAR-10.
    test_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=test_transform
    )

    # DataLoaders para entregar datos por lotes.
    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # Mostrar las dimensiones de cada conjunto.
    print("\n===== DIMENSIONES DE CIFAR-10 =====")
    print(f"Imágenes originales de entrenamiento: {len(raw_dataset)}")
    print(f"Imágenes para entrenamiento: {len(train_subset)}")
    print(f"Imágenes para validación: {len(val_subset)}")
    print(f"Imágenes para prueba: {len(test_dataset)}")

    print("\n===== DIMENSIONES DE LOS LOTES =====")
    images, labels_batch = next(iter(train_loader))
    print(f"Forma de las imágenes: {images.shape}")
    print(f"Forma de las etiquetas: {labels_batch.shape}")

    return train_loader, val_loader, test_loader


# ============================================================
# 5. EJECUCIÓN DE PRUEBA
# ============================================================

if __name__ == "__main__":

    # Mostrar 12 imágenes originales con sus etiquetas.
    os.makedirs("outputs", exist_ok=True)

    original_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=None
    )

    show_images(original_dataset)

    # Crear los DataLoaders e imprimir dimensiones.
    train_loader, val_loader, test_loader = get_loaders()

    print("\nDataset preparado correctamente.")
