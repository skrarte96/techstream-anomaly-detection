# Archivo que genera de forma aleatoria n_total datos de servidores con un ratio_fallo (%) de servidores
# que presentan fallos.

# Importamos las librerías
import numpy as np
import pandas as pd
from pathlib import Path
# Para guardar los archivos excel se debe tener instlado openpyxl

# Crea la carpeta 'datos' si no existe
ruta_datos = Path(__file__).resolve().parent.parent / 'datos'
ruta_datos.mkdir(exist_ok=True)

# Fijamos una semilla aleatoria para mantener la reproducibilidad de los resultados.
np.random.seed(42)

# Parámetros para la generación de datos
n_total = 5000     # Número total de servidores en el csv
ratio_fallo = 0.12 # El % de fallo que vamos a tener en los n_total servidores. Elegimos un 12% para mantener el
                   # número de servidores que fallan en un valor habitual, no el 50% por ejemplo.

# Número de servidores que presentan fallos
n_fallos = int(n_total * ratio_fallo)
# Número de servidores que NO presentan fallos
n_normales = n_total - n_fallos

# Features a generar:
# cpu: uso de cpu en %
# temp: temperatura del servidor en ºC
# mem: memoria de uso en %
# red: tráfico de red en Mbps


# Función generador de datos de un servidor: distribución gaussiana del número de valores especificados arriba (n_fallos
# y n_normales) centradas en valores medios y distribuciones típicas elegidos posteriormente.
# Evitamos salirnos de rangos físicos etremos con np.clip
# Argumentos: n = numero de servidores a generar. params = diccionario con las claves cpu, temp, mem y red
# cada valor de params es una tupla (media, desviación estándar)
def generar_clase(n, params):
    # generación de distribuciones para los cuatro valores cpu, temp, mem y red
    cpu = np.random.normal(*params['cpu'], n)
    temp = np.random.normal(*params['temp'], n)
    mem = np.random.normal(*params['mem'], n)
    red = np.random.normal(*params['red'], n)

    # Con np.clip nos mantenemos en rangos físicamente posibles
    cpu  = np.clip(cpu,  0, 100)
    temp = np.clip(temp, 20, 110)
    mem  = np.clip(mem,  0, 100)
    red  = np.clip(red,  0, 1000)

    return cpu, temp, mem, red

# Parámetros escogidos para un servidor normal (media, desviación típica)
params_normal = {
    'cpu':  (30, 10),    # %
    'temp': (45,  7),    # ºC
    'mem':  (55, 12),    # %
    'red':  (100, 40),   # Mbps
}
# Parámetros escogidos para un servidor que presenta fallos (media, desviación típica)
# Mucho más solape
params_fallo = {
    'cpu':  (40, 8),     # %
    'temp': (55, 7),     # ºC
    'mem':  (75, 8),     # %
    'red':  (150, 70),   # Mbps
}

# Muy Separados
#params_fallo = {
#    'cpu':  (85, 8),     # %
#    'temp': (78, 7),     # ºC
#    'mem':  (88, 8),     # %
#    'red':  (430, 70),   # Mbps
#}

# Generamos los datos para ambos tipos de servidores
# Servidores normales (con sufijo '_n')
cpu_n, temp_n, mem_n, red_n = generar_clase(n_normales, params_normal)
# Servidores con fallos (con sufijo '_f')
cpu_f, temp_f, mem_f, red_f = generar_clase(n_fallos,   params_fallo)

# Creamos el DataFrame
df = pd.DataFrame({
    'cpu_uso':      np.concatenate([cpu_n,  cpu_f]),
    'temperatura':  np.concatenate([temp_n, temp_f]),
    'memoria_uso':  np.concatenate([mem_n,  mem_f]),
    'trafico_red':  np.concatenate([red_n,  red_f]),
    'Fallo':        np.concatenate([np.zeros(n_normales), np.ones(n_fallos)])
})

# Para simular una obtención de datos más realista mezclamos aleatoriamente todas las filas de la DataFrame
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Cambio de la columna 'Fallo' de floats a integers
df['Fallo'] = df['Fallo'].astype(int)

# Guardamos el DataFrame a un archivo csv en la carpeta de datos
df.to_csv('../datos/datos_servidores.csv', index=False)

# Nota: como la prueba indica que se deberán entregar los datos generados en csv o excel incluyo el guardado del
# la DataFrame en xlsx en caso de que el receptor prefiera ver la tabla en excel
df.to_excel('../datos/datos_servidores.xlsx', index=False)

# Verificación de la tabla
# Shape de la tabla
print('Shape de la tabla de datos:') # Deben haber n_total = 5000 filas y 5 columnas (5000, 5)
print(df.shape)

# Distribución de servidores normales y con fallos
print('Distribución de clases:')
print(df['Fallo'].value_counts())                   # Deben haber 600 fallos y 4400 normales
print(f"Ratio de fallos: {df['Fallo'].mean():.1%}") # El ratio debe ser el 12% indicado (ratio_fallo)

# Tabla de medias separadas por servidores normales y con fallos
print('Tabla de medias separadas por servidores y con fallos:')
print(df.groupby('Fallo').mean().round(2)) # Deben ser las mismas medias que las de params_normal (Fallo = 0) y
                                           # params_fallo (Fallo = 1)

# Tabla de desviaciones estándar separadas por servidores normales y con fallos
print('Tabla de desviaciones estándar separadas por servidores y con fallos:')
print(df.groupby('Fallo').std().round(2)) # Deben ser las mismas desviaciones estandar que las de params_normal
                                          # (Fallo = 0) y params_fallo (Fallo = 1)