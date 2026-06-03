# Guardado del modelo y demostración de inferencia.
# Reconstruimos el Scaler para los datos de test
# Importamos librerías
import torch
import numpy as np
import pandas as pd
from modelo import MLPDetectorFallos
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.model_selection import train_test_split

# Cargamos nuestros datos generados
datos_raw = pd.read_csv('../datos/datos_sensores.csv')
features = ['cpu_uso', 'temperatura', 'memoria_uso', 'trafico_red']

X = datos_raw[features].values
y = datos_raw['Fallo'].values

# Solo nos quedamos con el set de train (Mantenemos mismo ratio que antes)
X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Scaler con los mismos datos de nuestro dataset de Train
scaler = StandardScaler()
scaler.fit(X_train)
joblib.dump(scaler, 'scaler.pkl')

# Cargamos modelo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
modelo = MLPDetectorFallos(input_dim=4, dropout_rate=0.3).to(device)
modelo.load_state_dict(torch.load('mejor_modelo.pt', map_location=device))
modelo.eval()

# Función de predicción de nuevos resultados con nuestra nn
def predecir(cpu, temperatura, memoria, trafico):
    # Cargamos nuevos datos
    X_nuevo = np.array([[cpu, temperatura, memoria, trafico]],
                        dtype=np.float32)
    # Normalizamos
    X_scaled = scaler.transform(X_nuevo)
    # Pasamos a tensor
    X_tensor = torch.FloatTensor(X_scaled).to(device)

    with torch.no_grad():
        logit = modelo(X_tensor)
        prob = torch.sigmoid(logit).item()

    # Clasificamos en Fallo (1) o Normal (0)
    pred = 1 if prob >= 0.5 else 0
    return pred, prob


# Ejemplo
print('Ejemplo de usos')
# Casos inventados para distintos estados de sensores
casos = [
    # (cpu,  temp, mem,  red,   descripción)
    (28.0,  43.0, 52.0,  95.0, "Servidor en reposo"),
    (45.0,  55.0, 70.0, 150.0, "Carga moderada"),
    (85.0,  79.0, 90.0, 440.0, "Fallo claro"),
    (60.0,  65.0, 75.0, 280.0, "Zona intermedia — ambiguo"),
    (92.0,  82.0, 95.0, 510.0, "Fallo severo"),
]
# Testeo de los datos, printeamos resultados
for cpu, temp, mem, red, descripcion in casos:
    pred, prob = predecir(cpu, temp, mem, red)
    estado = "FALLO" if pred == 1 else "NORMAL"
    print(f"{descripcion}")
    print(f"Sensores con CPU: {cpu}%, Temp: {temp}°C, Mem: {mem}% y Red: {red} Mbps")
    print(f"Predicción: {estado} y Probabilidad de fallo: {prob:.4f}")