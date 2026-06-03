# Evaluamos las métricas de nuestra red con el dataset reservado para test.
# Nos centraremos en ROC-AUC y en la Matriz de confusión
# Impotamos librerías
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from dataset import SensoresDataset
from modelo import MLPDetectorFallos
from torch.utils.data import DataLoader

# En el caso de que tengamos acceso a GPU si no seguimos con CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

modelo = MLPDetectorFallos(input_dim=4, dropout_rate=0.3).to(device)

# Cargamos nuestro mejor modelo
modelo.load_state_dict(torch.load('mejor_modelo.pt', map_location=device))
modelo.eval()

# Cargamos nuestro tensor de datos
datos = torch.load('tensores.pt')

# Cargamos los datos del dataset test
test_loader = DataLoader(
    SensoresDataset(datos['X_test'], datos['y_test']),
    batch_size=32, shuffle=False
)

# Guardado de las probabilidades de fallo, predicciones y etiquetas reales del dateset de test
todas_probs = []
todas_preds = []
todas_labels = []


with torch.no_grad():
    for X_batch, y_batch in test_loader:
        # Cargamos datos de features al dispositivo
        X_batch = X_batch.to(device)

        # Calculamos las probabilidades de fallo del sensor
        logits = modelo(X_batch)
        probs = torch.sigmoid(logits)

        # Comenzamos con umbral de 0.5
        preds = (probs >= 0.5).float()

        # Guardamos probabilidades, predicciones y etiqueta real de la muestra en las listas creadas anteriormente
        todas_probs.append(probs.cpu().numpy())
        todas_preds.append(preds.cpu().numpy())
        todas_labels.append(y_batch.numpy())

# Concatenamos los resultados en arrays planos
y_probs = np.concatenate(todas_probs).flatten()
y_pred = np.concatenate(todas_preds).flatten().astype(int)
y_true = np.concatenate(todas_labels).flatten().astype(int)

# Evaluación del test
print("Informe de clasificación:")
print(classification_report(y_true, y_pred,
                             target_names=['Normal', 'Fallo'],
                             digits=4))

roc_auc = roc_auc_score(y_true, y_probs)
print(f"ROC-AUC: {roc_auc:.4f}")

# Ploteamos matriz de confusión y curva ROC
fig = plt.figure(figsize=(13, 5))
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

# Matriz de confusión
ax1 = fig.add_subplot(gs[0])
cm = confusion_matrix(y_true, y_pred)

# Colocamos en el subplot
im = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
plt.colorbar(im, ax=ax1)

# Etiquetas, títulos y labels
clases = ['Normal', 'Fallo']
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.set_xticklabels(clases)
ax1.set_yticklabels(clases)
ax1.set_xlabel('Predicción', fontsize=11)
ax1.set_ylabel('Etiqueta real', fontsize=11)
ax1.set_title('Matriz de confusión', fontsize=12)

# Valores de la confusion matrix en cada celda
for i in range(2):
    for j in range(2):
        ax1.text(j, i, str(cm[i, j]),
                 ha='center', va='center',
                 fontsize=14, fontweight='bold',
                 color='white' if cm[i, j] > cm.max() / 2 else 'black')

# Ploteamos la curva ROC
ax2 = fig.add_subplot(gs[1])
fpr, tpr, thres = roc_curve(y_true, y_probs)

ax2.plot(fpr, tpr, color='steelblue', linewidth=2,
         label=f'ROC-AUC = {roc_auc:.4f}')
ax2.plot([0, 1], [0, 1], color='gray', linestyle='--',
         linewidth=1, label='Clasificador aleatorio')
ax2.set_xlabel('Tasa de falsos positivos (FPR)', fontsize=11)
ax2.set_ylabel('Tasa de verdaderos positivos (TPR)', fontsize=11)
ax2.set_title('Curva ROC', fontsize=12)
ax2.legend(fontsize=10)

plt.suptitle('Evaluación del modelo — Test set', fontsize=13, y=1.01)
plt.savefig('../imagenes/matriz_confusion_y_curva_roc.png', dpi=150, bbox_inches='tight')

#NOTA: El modelo predice a la perfección los resultados, algo de esperar debido al pequeño solapamiento entre las
# distribuciones de los sensores normales y con fallos, así como la alta correlación entre las features y la etiqueta
# fallo.