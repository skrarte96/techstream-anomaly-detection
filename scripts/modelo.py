# Red neuronal de 4 | 64 | 32 | 16 | 1 nodos con normalización BatchNorm1d y un dropout del 30% en sus dos
# primeras capas ocultas. Debajo de la arquitectura de la red neuronal colocamos un breve script de testeo
# para comprobar su arquitectura.
# Importamos las librerías
import torch
import torch.nn as nn

# Definimos la MLP
class MLPDetectorFallos(nn.Module):
    # Aplicamos una tasa de drop out del 30%
    def __init__(self, input_dim=4, dropout_rate=0.3):

        super(MLPDetectorFallos, self).__init__()

        # Hacemos una red secuencial
        self.red = nn.Sequential(
            # Primera Capa Oculta de 64 nodos
            nn.Linear(input_dim, 64),
            # BatchNorm1d para normalizar
            nn.BatchNorm1d(64),
            # Usamos ReLU(inplace=True) para la activación y modificamos el tensor
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),

            # Pasamos de 64 a 32 nodos
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),

            # Última capa oculta
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),

            # Capa de salida
            nn.Linear(16, 1)
        )
    # Flujo de información a traves de la nn
    def forward(self, x):
        return self.red(x)


# Check de la arquitectura de la nn si cargamos desde aquí
if __name__ == '__main__':
    modelo = MLPDetectorFallos(input_dim=4)

    # Vemos los detalles de nuestra nn
    print(modelo)
    print()

    # Conteo de número de parámetros entrenables
    total_params = sum(p.numel() for p in modelo.parameters()
                       if p.requires_grad)
    print(f"Parámetros entrenables: {total_params:,}")

    # Test con una muestra random de 8 muestras y 4 features
    x_prueba = torch.randn(8, 4)
    salida = modelo(x_prueba)
    print(f"Shape entrada: {x_prueba.shape}")
    print(f"Shape salida:  {salida.shape}")