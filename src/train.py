
"""
train.py
Entrenamiento, evaluación y generación de gráficos para CIFAR-10.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from dataset import get_loaders, CLASSES
from model import CNN


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

SEED = 73
EPOCHS = 5
LEARNING_RATE = 0.001

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Usar GPU si está disponible; de lo contrario, CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("outputs", exist_ok=True)

print("Dispositivo de entrenamiento:", device)
print("Semilla:", SEED)


# ============================================================
# 2. FUNCIONES DE ENTRENAMIENTO Y EVALUACIÓN
# ============================================================

def run_epoch(model, loader, criterion, optimizer=None):
    """
    Ejecuta una época de entrenamiento o validación.

    Si se recibe optimizer, se entrena.
    Si no se recibe optimizer, se evalúa.
    """

    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad()

        # Calcular predicciones con o sin gradientes.
        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += batch_size

    average_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return average_loss, accuracy


# ============================================================
# 3. ENTRENAMIENTO DEL MODELO
# ============================================================

def main():

    # Cargar datos.
    train_loader, val_loader, test_loader = get_loaders()

    # Crear el modelo y enviarlo al dispositivo.
    model = CNN().to(device)

    # Función de pérdida para clasificación multiclase.
    criterion = nn.CrossEntropyLoss()

    # Optimizador Adam.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # Historial de métricas.
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    print("\n===== INICIO DEL ENTRENAMIENTO =====")

    for epoch in range(EPOCHS):

        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            criterion
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(
            f"Época [{epoch + 1}/{EPOCHS}] | "
            f"Loss entrenamiento: {train_loss:.4f} | "
            f"Accuracy entrenamiento: {train_acc:.2f}% | "
            f"Loss validación: {val_loss:.4f} | "
            f"Accuracy validación: {val_acc:.2f}%"
        )

    # Guardar el modelo entrenado.
    torch.save(model.state_dict(), "outputs/cnn_cifar10.pth")

    # ========================================================
    # 4. GRÁFICOS DE PÉRDIDA Y PRECISIÓN
    # ========================================================

    epochs_range = range(1, EPOCHS + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Gráfico de pérdida.
    axes[0].plot(
        epochs_range, train_losses,
        marker="o", label="Entrenamiento"
    )
    axes[0].plot(
        epochs_range, val_losses,
        marker="o", label="Validación"
    )
    axes[0].set_title("Pérdida por época")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("CrossEntropyLoss")
    axes[0].legend()
    axes[0].grid(True)

    # Gráfico de precisión.
    axes[1].plot(
        epochs_range, train_accuracies,
        marker="o", label="Entrenamiento"
    )
    axes[1].plot(
        epochs_range, val_accuracies,
        marker="o", label="Validación"
    )
    axes[1].set_title("Precisión por época")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("outputs/loss_accuracy.png", dpi=150)
    plt.close()

    print("\nGráfico guardado en outputs/loss_accuracy.png")

    # ========================================================
    # 5. EVALUACIÓN FINAL CON EL CONJUNTO DE PRUEBA
    # ========================================================

    test_loss, test_acc = run_epoch(
        model,
        test_loader,
        criterion
    )

    print("\n===== RESULTADOS DEL CONJUNTO DE PRUEBA =====")
    print(f"Pérdida de prueba: {test_loss:.4f}")
    print(f"Precisión de prueba: {test_acc:.2f}%")

    # Obtener predicciones para la matriz de confusión.
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(len(CLASSES)))
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASSES
    )

    fig, ax = plt.subplots(figsize=(11, 10))

    display.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        colorbar=False
    )

    ax.set_title("Matriz de confusión - CIFAR-10")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=150)
    plt.close()

    print("Matriz guardada en outputs/confusion_matrix.png")
    print("Modelo guardado en outputs/cnn_cifar10.pth")
    print("\nEntrenamiento y evaluación finalizados.")


if __name__ == "__main__":
    main()
