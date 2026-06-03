# Uso de workers para el procesado de los datos usando DataSet y DataLoader
# Dividiremos los datos en batches de 32 muestras cada uno
# Importamos las librerías
import torch
from torch.utils.data import Dataset, DataLoader

# Creamos una clase de dataset personalizada
class SensoresDataset(Dataset):

    def __init__(self, X, y):
        # Accedemos a los tensores X e y de preprocesamiento.py
        self.X = X
        self.y = y

    def __len__(self):
        # Muestras en el dataset
        return len(self.X)

    def __getitem__(self, idx):
        # Obtenciones de la muestra idx
        return self.X[idx], self.y[idx]


# Checkeo del dataset si cargamos desde aquí
if __name__ == '__main__':

    # Cargamos los tensores que guardamos en preprocesamiento.py
    datos = torch.load('tensores.pt')

    # Creamos los datasets train, validación y test
    train_dataset = SensoresDataset(datos['X_train'], datos['y_train'])
    val_dataset = SensoresDataset(datos['X_val'], datos['y_val'])
    test_dataset = SensoresDataset(datos['X_test'], datos['y_test'])

    # Vemos si las length de nuestros datasets se corresponden con las esperadas
    print(f"Número muestras en train: {len(train_dataset)}")
    print(f"Número muestras en val: {len(val_dataset)}")
    print(f"Número muestras en test: {len(test_dataset)}")

    # testeo de nuestra clase ServidoresDataset para la primera muestra
    x_muestra, y_muestra = train_dataset[0]
    print(f"Primera muestra:")
    print(f"Shape de las features: {x_muestra.shape}")  # son 4 features
    print(f"Shape del target: {y_muestra.shape}")  # solo una columna, (Fallo)
    print(f"Valores del tensor de las features: {x_muestra}")
    print(f"Valores del tensor del target: {y_muestra}")

    # Cargamos datos con batches de 32 muestras. orden aleatorio para train
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset,   batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset,  batch_size=32, shuffle=False)

    # Número de batches en train
    print(f"Batches en train: {len(train_loader)}")

    # testeo del primer batch
    X_batch, y_batch = next(iter(train_loader))

    print(f"Primer batch:")
    print(f"Shape de las features: {X_batch.shape}")  # 32 muestras con 4 features
    print(f"Shape de la target: {y_batch.shape}")  # 32 muestras con la target (Fallo)