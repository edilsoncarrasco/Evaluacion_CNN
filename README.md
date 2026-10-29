# Evaluacion CNN - CIFAR-10

## 1. Descripcion
Proyecto de Inteligencia Artificial para clasificar imagenes mediante una red neuronal convolucional (CNN) utilizando PyTorch y el conjunto de datos CIFAR-10.

## 2. Objetivo
Preparar datos de imagenes, construir una CNN, entrenar el modelo y evaluar su rendimiento mediante metricas de clasificacion.

## 3. Tecnologias
- Python
- PyTorch y Torchvision
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## 4. Estructura del proyecto
```text
Evaluacion_CNN/
├── src/
│   ├── dataset.py
│   ├── model.py
│   └── train.py
├── notebooks/
│   └── experiment.ipynb
├── outputs/
│   ├── cnn_cifar10.pth
│   ├── confusion_matrix.png
│   └── loss_accuracy.png
├── requirements.txt
├── README.md
└── .gitignore
```

## 5. Datos
CIFAR-10 contiene 60 000 imagenes RGB de 32 x 32 pixeles, distribuidas en diez clases.
- Entrenamiento: 40 000 imagenes.
- Validacion: 10 000 imagenes.
- Prueba: 10 000 imagenes.

## 6. Modelo
La CNN utiliza dos bloques convolucionales con activacion ReLU y MaxPooling, seguidos de capas totalmente conectadas para clasificar las imagenes en diez categorias.

## 7. Reproducibilidad
- Semilla aleatoria: 73.
- Tamano de lote: 64.
- Optimizador: Adam.
- Funcion de perdida: CrossEntropyLoss.

## 8. Ejecucion
Instalar las dependencias:

```bash
pip install -r requirements.txt
```

Preparar el dataset:

```bash
python src/dataset.py
```

Entrenar y evaluar el modelo:

```bash
python src/train.py
```

## 9. Resultados
Las graficas de perdida, exactitud y matriz de confusion se guardan en la carpeta `outputs/`. Las metricas finales deben interpretarse a partir de los resultados reales de la ejecucion.

## 10. Autor
Edilson Huillca Carrasco
Universidad Andina del Cusco
