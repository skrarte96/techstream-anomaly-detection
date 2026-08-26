# TechStream — Detección de Anomalías en Servidores

Hemos creado una red neuronal densa (MLP) para predecir si qué servidores de una empresa
ficticia TechStream producen fallos o actúan de forma norma. Para ello hemos creado
un generador de datos sintéticos con las features: cpu_uso, temperatura, memoria_uso y tráfico de red.
El modelo ha sido entrenado con dichos datos y evaluado con parámetros de distintos niveles de
actuación de servidores, tanto en fallo como en condiciones normales

---

## Descripción del problema

Los servidores anteriormente mencionados presentan las siguientes features con sus respectivas unidades
- **CPU uso** (%)
- **Temperatura** (°C)
- **Memoria uso** (%)
- **Tráfico de red** (Mbps)

Incorporamos además la etiqueta 'Fallo' que indica si el servidor presenta fallos (1) o actúa normal (0),
puesto que se trata de un supervised model

---
De nuevo discuplen la errata, servidores -> servidores  
## Estructura del repositorio  
TechStreamPrueba/  
├── scripts/  
│   ├── generador_datos_servidores.py # Generador de datos sintéticos  
│   ├── eda.py                      # Análisis exploratorio y visualizaciones  
│   ├── preprocesamiento.py         # Normalización, split y conversión a tensores  
│   ├── dataset.py                  # Clase Dataset y DataLoader de PyTorch  
│   ├── modelo.py                   # Arquitectura MLP (nn.Module)  
│   ├── entrenamiento.py            # Bucle de entrenamiento con early stopping  
│   ├── evaluacion.py               # Métricas, matriz de confusión y curva ROC  
│   └── inferencia.py               # Función de predicción sobre datos nuevos  
├── datos/  
│   ├── datos_servidores.csv  
│   └── datos_servidores.xlsx  
└── imagenes/  
├── eda_histogramas.png  
├── eda_boxplots.png  
├── eda_correlaciones_features.png  
├── eda_correlacion_target.png  
├── curvas_aprendizaje.png  
└── matriz_confusion_y_curva_roc.png  

---

## Requisitos
- torch
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- joblib
- pathlib

---

## Reproducción del proyecto

Ejecutar los scripts en el siguiente orden

1. Generar los datos sintéticos  
python generador_datos_servidores.py 

2. Exploratory Data Analisis de los datos generados  
python eda.py  

3. Preprocesamiento y conversión a tensores  
python preprocesamiento.py  

4. Verificación del Dataset y DataLoader  
python dataset.py  

5. Verificación de la arquitectura de la nn  
python modelo.py  

6. Entrenamiento de la red neuronal  
python entrenamiento.py  

7. Evaluación sobre el dataset de test  
python evaluacion.py  

8. Inferencia sobre datos nuevos  
python inferencia.py  

---
## EDA de los datos sintéticos

![eda_histogramas](imagenes/eda_histogramas.png)

> Hay buena separación en la distribución de valores en las cuatro features respecto a los dos tipos de servidores

![eda_boxplots](imagenes/eda_boxplots.png)

> Vemos, al igual que en los histogramas que los valores de los servidores que presentan fallos son mucho más altos
> que los servidores normales. Todas las medianas de los servidores con fallos son más altas que los normales.
> Podemos ver que hay más diferencia entre las medianas de tráfico de red y cpu_uso con respecto a las otras features.
> De aquí ya tenemos una idea de qué features van a tener más peso a la hora de clasificar servidores entre normales
> o con fallos

![eda_correlaciones_features](imagenes/eda_correlaciones_features.png)

> Como era de esperar por la distribución de cada feature, hay mucha correlación positiva entre cada una de las
> features. Si una aumenta de valor las otras también. Vemos que existe más correlación entre unas que en otras.
> Como ejemplo, las que más correlación presentan son cpu_uso - tráfico de red, seguida de temperatura - tráfico de red

![correlacion_target](imagenes/eda_correlacion_target.png)

> Ahora podemos ver qué features tienen más relación con el nivel de fallo de los servidores, de mayor a menor
> relación siguen el siguiente orden Tráfico de Red > Cpu Uso > Temperatura > Memoria Uso

---

## Arquitectura de la red

Red neuronal densa (MLP) presenta la siguiente arquitectura

Entrada (4 features)  

4 nodos input que pasan a 64 nodos en una capa oculta con 
normalización BatchNorm, función ReLu y un Dropout del 30%  
Linear(4 → 64) + BatchNorm + ReLU + Dropout(0.3)  

Los 64 nodos pasan a una siguiente capa oculta de 32 nodos también con 
normalización BatchNorm, función ReLu y un Dropout del 30%  
Linear(64 → 32) + BatchNorm + ReLU + Dropout(0.3)  

Paso a la última capa oculta con solo función ReLU  
Linear(32 → 16) + ReLU  

Nodo de salida  
Linear(16 → 1)  

Escogimos la función de pérdida: `BCEWithLogitsLoss` para poder incorporar `pos_weight=7.33` y corregir
el desbalanceo de clases (88% normal / 12% fallo), en vez de crear más datos sinteticos.

Optimizador: `Adam` con `lr=0.001` y scheduler `ReduceLROnPlateau`. La red deja de entrenarse si a los 
10 epochs no ha tenido mejoria en su funcion de perdidas

---

## Resultados

| Métrica | Valor |
|---|---|
| Accuracy | 1.0000 |
| Precision (Fallo) | 1.0000 |
| Recall (Fallo) | 1.0000 |
| F1-score (Fallo) | 1.0000 |
| ROC-AUC | 1.0000 |

![matriz_confusion_y_curva_roc](imagenes/matriz_confusion_y_curva_roc.png)

> **Nota:** El modelo predice a la perfección los resultados, algo de esperar debido al pequeño solapamiento entre las
> distribuciones de los sensores normales y con fallos como puede verse en la fase EDA, así como la alta correlación entre las features y la etiqueta fallo.

---

## Ejemplo de uso para datos nuevos

```python
from inferencia import predecir

pred, prob = predecir(cpu=85.0, temperatura=79.0, memoria=90.0, trafico=440.0)
print(f"Predicción: {'FALLO' if pred == 1 else 'NORMAL'}")
print(f"Probabilidad de fallo: {prob:.4f}")
```

---

## Autor

**Óscar Jover Arrate**  
[linkedin.com/in/oscar-jover-arrate-050135163](https://linkedin.com/in/oscar-jover-arrate-050135163) · [github.com/skrarte96](https://github.com/skrarte96)
