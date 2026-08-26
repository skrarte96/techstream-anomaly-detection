# Preprocesamiento de los datos. Dividimos el DataFrame en los datasets de entrenamiento, validación y test
# Normalizamos los datos con StandardScaler

# Importamos las librerías
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

# Cargamos la DataFrame
df = pd.read_csv('../datos/datos_servidores.csv')

# Conocemos las features, si no columnas = [col for col in df.columns if col != 'Fallo']
features = ['cpu_uso', 'temperatura', 'memoria_uso', 'trafico_red']

# Separamos features (X) de target (y)
X = df[features].values   # numpy array shape (5000, 4)
y = df['Fallo'].values     # numpy array shape (5000,)

# Como los datos son sintéticos, corregiremos el desbalanceo con el pos_weight en Pytorch en vez de generar más datos
# sintéticos hasta tener un 50% de valores con Fallo y normales
# Miramos el desbalanceo entre valores de Fallo
n_normales = (y == 0).sum()
n_fallos = (y == 1).sum()
ratio = n_normales / n_fallos

print(f"Clase 0, servidores normales: {n_normales}")
print(f"Clase 1, servidores con fallo: {n_fallos}")
print(f"Ratio 0/1: {ratio:.2f}")

# Peso para corregir el desbalanceo en Pytorch, usaremos el ratio
pos_weight = torch.FloatTensor([ratio])

# Creamos los dataset de train, validación y test. 70% de los datos son Train
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Dividimos el 30% que queda en validación (15%) y test (15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# Test si el numero de muestras son las correctas
print(f"Train: {X_train.shape[0]} muestras")
print(f"Validación: {X_val.shape[0]} muestras")
print(f"Test: {X_test.shape[0]} muestras")

# Normalizamos los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# Convertimos los set de datos a tensores
X_train_t = torch.FloatTensor(X_train)
X_val_t   = torch.FloatTensor(X_val)
X_test_t  = torch.FloatTensor(X_test)

# y a una columna
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
y_val_t   = torch.FloatTensor(y_val).unsqueeze(1)
y_test_t  = torch.FloatTensor(y_test).unsqueeze(1)

# Miramos los shapes para ver si son correctos
print(f"Shape X_train tensor: {X_train_t.shape}")
print(f"Shape y_train tensor: {y_train_t.shape}")

# Guardado para entrenamiento
torch.save({
    'X_train':    X_train_t,
    'X_val':      X_val_t,
    'X_test':     X_test_t,
    'y_train':    y_train_t,
    'y_val':      y_val_t,
    'y_test':     y_test_t,
    'pos_weight': pos_weight,
}, 'tensores.pt')