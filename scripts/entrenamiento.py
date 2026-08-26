# Entrenamiento de nuestra nn. Hemos aplicado un módulo de paciencia que termina el entrenamiento si no encuentra
# mejoras en las pérdidas durante 10 epochs
# Importamos librerías
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ServidoresDataset
from modelo import MLPDetectorFallos
import matplotlib.pyplot as plt
torch.manual_seed(42)
# En el hipotético caso de que se pueda ejecutar en GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# Cargamos tensor de preprocesamiento.py
datos = torch.load('tensores.pt')

# Cargamos los datasets de train
train_loader = DataLoader(
    ServidoresDataset(datos['X_train'], datos['y_train']),
    batch_size=32, shuffle=True
)
# Cargamos los datasets de validación
val_loader = DataLoader(
    ServidoresDataset(datos['X_val'], datos['y_val']),
    batch_size=32, shuffle=False
)

# Cargamos nuestra MLP
modelo = MLPDetectorFallos(input_dim=4, dropout_rate=0.3).to(device)

# Elegimos como función de pérdida BCEWithLogitsLoss
# Aplicamos el mayor peso a servidores que presentan fallo calculado en preprocesamiento.py
pos_weight = datos['pos_weight'].to(device)
criterio = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# usamos Adam como optimizador
optimizador = torch.optim.Adam(modelo.parameters(), lr=0.001)

# reducimos lr si la pérdida en validación no mejora durante 5 epochs
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizador,
    mode='min',
    patience=5,
    factor=0.5
)

# definimos epochs de entrenamiento devolviendo media de la pérdida sobre todos los batches
def train_epoch(modelo, loader, criterio, optimizador, device):

    # Modo entrenamiento
    modelo.train()
    perdida_total = 0.0

    for X_batch, y_batch in loader:
        # Movemos cada batch
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Corregimos acumulación de gradientes
        optimizador.zero_grad()
        # Forward pass
        predicciones = modelo(X_batch)
        # Calculamos pérdida
        perdida = criterio(predicciones, y_batch)
        # Backward pass: calcula gradientes
        perdida.backward()
        # Actualizamos los pesos
        optimizador.step()
        # Sumamos pérdida total
        perdida_total += perdida.item()

    # Devolvemos la media de la pérdida
    return perdida_total / len(loader)

# definimos epochs de validación devolviendo media de la pérdida sobre los batches
def val_epoch(modelo, loader, criterio, device):

    # modelo evaluación
    modelo.eval()
    perdida_total = 0.0

    # Abandonamos cálculo de gradientes
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            predicciones = modelo(X_batch)
            perdida = criterio(predicciones, y_batch)
            perdida_total += perdida.item()

    # Devolvemos media de la pérdida total
    return perdida_total / len(loader)


# Entrenamiento del modelo
n_epochs = 50
historial = {'train': [], 'val': []}

# Paramos si durante max_paciencia la pérdida en validación no mejora
mejor_perdida_val = float('inf')
max_paciencia = 10
paciencia_actual = 0

# Definimos una tabla para los valores de epoch, y las pérdidas en train y validación
print(f"{'Epoch':>6} | {'Train Pérdida':>14} | {'Val Pérdida':>14} | {'lr actual':>14}")

for epoch in range(1, n_epochs + 1):
    # Pérdida en train
    perdida_train = train_epoch(modelo, train_loader, criterio, optimizador, device)
    # Pérdida en val
    perdida_val = val_epoch(modelo, val_loader, criterio, device)

    # Adjuntamos pérdidas al diccionario creado anteriormente
    historial['train'].append(perdida_train)
    historial['val'].append(perdida_val)

    # Pasamos la pérdida de validación al scheduler para reducir lr
    scheduler.step(perdida_val)
    lr_actual = optimizador.param_groups[0]['lr']
    # Representamos valores de pérdidas de la tabla mencionada anteriormente
    print(f"{epoch:>6} | {perdida_train:>14.4f} | {perdida_val:>14.4f} | {lr_actual:>14.6f}")

    # Paramos antes de tiempo si en 10 epochs la pérdida no mejora. Guardamos modelo con mejor pérdida en
    # la fase de validación
    if perdida_val < mejor_perdida_val:
        mejor_perdida_val = perdida_val
        paciencia_actual  = 0
        # Guardamos mejor modelo
        torch.save(modelo.state_dict(), 'mejor_modelo.pt')
    else:
        paciencia_actual += 1
        if paciencia_actual >= max_paciencia:
            print(f"Paramos antes en epoch {epoch}, pérdida no ha mejorado en los últimos 10 epochs.")
            break

print(f"Mejor pérdida de validación: {mejor_perdida_val:.4f}")

# Representamos curvas de aprendizaje
fig, ax = plt.subplots(figsize=(9, 5))
# Curva train
ax.plot(historial['train'], label='Train loss', color='steelblue', linewidth=2)
# Curva validación
ax.plot(historial['val'],   label='Val loss',   color='tomato',    linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('BCEWithLogitsLoss')
ax.set_title('Curvas de aprendizaje')
ax.legend()
plt.tight_layout()
plt.savefig('../imagenes/curvas_aprendizaje.png', dpi=150, bbox_inches='tight')