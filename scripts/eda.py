# Exploratory Data Analysis de los datos generados en el programa generador_datos_servidores.py
#
# Importamos las librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path # Guardado de imágenes

# Crea la carpeta 'imagenes' si no existe
ruta_datos = Path(__file__).resolve().parent.parent / 'imagenes'
ruta_datos.mkdir(exist_ok=True)

# Carga el csv de los anteriores datos generados
df = pd.read_csv('../datos/datos_servidores.csv')

# Breve exploración de la DataFrame
# Información de la DataFrame
print('Información de la DataFrame:')
print(df.info())
# Las 5000 filas presentan un valor y no debe ser valores NaN o nulos puesto que en el dtype aparece float y no object
# Salvo en la columna Fallo que es int, 1 si fallo 0 si normal
print('Datos de la DataFrame:')
print(df.describe())
# Todos los datos presentan 5000 cuentas, como antes, están todos rellenos y por las estadísticas numéricas todos son
# valores numéricos. Atendiendo a los valores máximos y mínimos no existen valores muy desorbitados de la realidad
# física

# Lista de las features del archivo menos la etiqueta 'Fallo'
features = ['cpu_uso', 'temperatura', 'memoria_uso', 'trafico_red']

# Diccionario de títulos por cada valor es el título de la feature para su ploteo en las figuras
titulos = {f: f.replace('_', ' ').title() for f in features}
# Colores para los dos tipos de servidores
# Fallo = 0 (Normal - Azul) | Fallo = 1 (Fallo - Rojo)
colores = {0: 'steelblue', 1: 'tomato'}

# Figura 1: Histograma de features de servidores por tipo (Fallo | Normal)
# Histograma de cada una de las features agrupadas por las dos clases de servidores. Asi vemos cuanto solapan los
# valores de las features.
# subplot de 2x2 al ser 4 features
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Distribución de cada feature por tipo de servidor (Fallo | Normal)', fontsize=14)

# Iteramos sobre los 4 subgráficos, uno para cada feature
for ax, feat in zip(axes.flatten(), features):
    for clase, color in colores.items():
        # DataFrame de la columna sobre la que estemos iterando
        subset = df[df['Fallo'] == clase][feat]
        # Histograma para cada feature (los histogramas están normalizados, density = True)
        ax.hist(subset, bins=50, alpha=0.55, color=color,
                label='Normal' if clase == 0 else 'Fallo',
                density=True)

    # Títulos y subtítulos
    ax.set_title(titulos[feat], fontsize=11)
    ax.set_xlabel('Valor', fontsize=9)
    ax.set_ylabel('Densidad', fontsize=9)
    ax.legend(fontsize=9)

# Buena separación entre subgráficos
plt.tight_layout()
# Guarda la figura en la carpeta imagenes
plt.savefig('../imagenes/eda_histogramas.png', dpi=150, bbox_inches='tight')

# NOTA: hay buena separación en la distribución de valores en las cuatro features respecto a los dos tipos de servidores

#Figura 2: Boxplots de las dos clases de servidores para cada feature.

# De nuevo 4 subgráficos para cada feature
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Boxplots de cada clase de servidor para cada feature', fontsize=14)

# Iteramos para cada subgráfico
for ax, feat in zip(axes.flatten(), features):
    data_plot = [
        df[df['Fallo'] == 0][feat].values,
        df[df['Fallo'] == 1][feat].values
    ]

    # Creamos el boxplot
    bp = ax.boxplot(
        data_plot,
        patch_artist=True,
        labels=['Normal', 'Fallo'],
        widths=0.5,
        medianprops=dict(color='black', linewidth=2)
    )

    # Coloreamos las cada boxplot
    for patch, color in zip(bp['boxes'], ['steelblue', 'tomato']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_title(titulos[feat], fontsize=11)
    ax.set_ylabel('Valor', fontsize=9)

plt.tight_layout()
plt.savefig('../imagenes/eda_boxplots.png', dpi=150, bbox_inches='tight')

# NOTA: Vemos, al igual que en los histogramas que los valores de los servidores que presentan fallos son mucho más altos
# que los servidores normales. Todas las medianas de los servidores con fallos son más altas que los normales.
# Podemos ver que hay más diferencia entre las medianas de tráfico de red y cpu_uso con respecto a las otras features.
# De aquí ya tenemos una idea de qué features van a tener más peso a la hora de clasificar servidores entre normales
# o con fallos

#Figura 3: Heatmap correlaciones entre features
# Queremos ver la relación que hay entre las cuatro features así como su coeficiente de Pearson
# Matriz de correlación
corr_features = df[features].corr()

fig, ax = plt.subplots(figsize=(7, 5))
# Heatmap de la matriz de correlación
sns.heatmap(
    corr_features,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0,
    ax=ax,
    linewidths=0.5,
    xticklabels=[titulos[f] for f in features],
    yticklabels=[titulos[f] for f in features]
)
ax.set_title('Correlaciones entre features', fontsize=13)
plt.tight_layout()
# Guardamos el heatmap en la carpeta imagenes
plt.savefig('../imagenes/eda_correlaciones_features.png', dpi=150, bbox_inches='tight')

# NOTA: Como era de esperar por la distribución de cada feature, hay mucha correlación positiva entre cada una de las
# features. Si una aumenta de valor las otras también. Vemos que existe más correlación entre unas que en otras.
# Como ejemplo, las que más correlación presentan son cpu_uso - tráfico de red, seguida de temperatura - tráfico de red

# Figura 4: Correlación de cada feature con Fallo
# Vemos el nivel de correlación de cada feature con la target fallo
# Calculaos la correlación de cada feature con fallo y las ordenamos en orden ascendente
corr_target = df[features].corrwith(df['Fallo']).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(7, 4))

# Coloreamos barras según correlación, rojo si positiva, azul si negativa
colores_barras = ['tomato' if v > 0 else 'steelblue' for v in corr_target.values]

# Ploteo de las barras horizontales
ax.barh(
    [titulos[f] for f in corr_target.index],
    corr_target.values,
    color=colores_barras,
    alpha=0.8,
    edgecolor='white'
)

ax.set_xlabel('Correlación Pearson con columna Fallo', fontsize=10)
ax.set_title('Importancia individual de cada feature', fontsize=13)
ax.set_xlim(0, 1.0)
# Añadimos valor de la correlación al final de cada barra
for i, v in enumerate(corr_target.values):
    ax.text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=10)

plt.tight_layout()
# Guardamos la figura en la carpeta imagenes
plt.savefig('../imagenes/eda_correlacion_target.png', dpi=150, bbox_inches='tight')

# NOTA: Ahora podemos ver qué features tienen más relación con el nivel de fallo de los servidores, de mayor a menor
# relación siguen el siguiente orden Tráfico de Red > Cpu Uso > Temperatura > Memoria Uso