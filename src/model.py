
"""
model.py
Arquitectura de la red neuronal convolucional para CIFAR-10.
"""

import torch
import torch.nn as nn


SEED = 73
torch.manual_seed(SEED)


class CNN(nn.Module):
    """CNN para clasificar imágenes RGB de 32 x 32 en 10 clases."""

    def __init__(self):
        super().__init__()

        # Primer bloque convolucional:
        # Entrada: 3 canales RGB.
        # Salida: 32 mapas de características.
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Segundo bloque convolucional.
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Después de los dos MaxPool:
        # 32 x 32 -> 16 x 16 -> 8 x 8.
        # Hay 64 mapas de características de 8 x 8.
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.relu3 = nn.ReLU()

        # Diez salidas, una por cada clase de CIFAR-10.
        # No se aplica Softmax porque CrossEntropyLoss
        # ya incorpora la operación necesaria.
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """Define el flujo de datos por la red."""

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)

        return x


if __name__ == "__main__":

    # Simulación de un lote de cuatro imágenes RGB.
    x = torch.randn(4, 3, 32, 32)

    model = CNN()

    print("===== VERIFICACIÓN DE DIMENSIONES =====")
    print("Entrada:", x.shape)

    x = model.conv1(x)
    print("Después de Conv2d 1:", x.shape)

    x = model.relu1(x)
    x = model.pool1(x)
    print("Después de MaxPool 1:", x.shape)

    x = model.conv2(x)
    print("Después de Conv2d 2:", x.shape)

    x = model.relu2(x)
    x = model.pool2(x)
    print("Después de MaxPool 2:", x.shape)

    x = model.flatten(x)
    print("Después de Flatten:", x.shape)

    x = model.fc1(x)
    x = model.relu3(x)
    print("Después de Linear 1:", x.shape)

    x = model.fc2(x)
    print("Salida final:", x.shape)

    # Comprobar que la salida tiene 10 clases por imagen.
    assert x.shape == (4, 10), "La dimensión de salida es incorrecta."

    print("Prueba del modelo superada correctamente.")
