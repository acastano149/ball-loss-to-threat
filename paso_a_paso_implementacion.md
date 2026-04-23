# Implementación Paso a Paso: Dashboard de Recuperaciones con StatsBomb

Este documento detalla los pasos necesarios para construir la aplicación en Shiny for Python. La aplicación cargará datos de fútbol de StatsBomb, permitirá seleccionar competiciones, temporadas y equipos (selección múltiple), y calculará qué recuperaciones en el campo (dividido en 12x8) terminan en un tiro durante la misma transición.

## Paso 1: Configuración del Entorno y Dependencias

Primero, debemos asegurarnos de instalar todas las dependencias necesarias. Esto se realiza ejecutando el siguiente comando en la terminal:

```bash
pip install shiny statsbombpy pandas mplsoccer matplotlib
```

## Paso 2: Creación de la Estructura de la Aplicación

Crearemos un archivo principal llamado `app.py`. Usaremos **Shiny Express** (`from shiny.express import input, render, ui`), que facilita mucho la creación de interfaces.

La estructura básica consistirá en:
1.  **Barra lateral (`ui.sidebar`):** Para alojar los selectores.
2.  **Área principal:** Para mostrar la métrica principal y el mapa de calor (gráfico de `mplsoccer`).

## Paso 3: Carga Inicial de Datos (Competiciones y Temporadas)

Para no sobrecargar la aplicación, al arrancar, solo pediremos a la API de StatsBomb la lista de competiciones y temporadas.

```python
from statsbombpy import sb
import pandas as pd

# Extraemos todas las competiciones
competitions = sb.competitions()
# Preparamos un diccionario para el UI
comp_dict = {row['competition_id']: row['competition_name'] for idx, row in competitions.iterrows()}
```

## Paso 4: Lógica de la Interfaz (Selectores)

En la barra lateral incluiremos los `ui.input_selectize` para permitir selección múltiple:

1.  **Selector de Competición:** Utilizará el `comp_dict`.
2.  **Selector de Temporada:** Se actualizará dinámicamente cuando el usuario elija una competición.
3.  **Selector de Equipo:** Se actualizará dinámicamente cuando el usuario elija temporada y competición.
4.  **Interruptor (Switch) de Datos 360:** `ui.input_switch("use_360", "Incluir Datos 360", False)`.
5.  **Botón de Acción:** `ui.input_action_button("analyze", "Analizar Datos")` para que las descargas pesadas de datos ocurran solo al hacer clic.

## Paso 5: Cálculo del "Indicador de Peligrosidad" (Transiciones)

Al hacer clic en "Analizar", ejecutaremos una función que hace lo siguiente:

1.  Descarga los eventos de los partidos de los equipos seleccionados (`sb.events(match_id)`).
2.  Si el usuario activa los datos 360, cruzamos esos eventos con `sb.frames(match_id)`.
3.  **Cálculo del Indicador:**
    - Filtramos los eventos de tipo tiro (`type == 'Shot'`) y obtenemos sus `possession_id`.
    - Filtramos los eventos de tipo recuperación (`type == 'Ball Recovery'`).
    - Comprobamos si el `possession_id` de la recuperación coincide con el de un tiro. Si es así, significa que **esa recuperación terminó en un tiro dentro de la misma transición**.
    - Marcamos esas recuperaciones en el DataFrame.

## Paso 6: Visualización Espacial con mplsoccer (Grid 12x8)

Finalmente, usamos la librería `mplsoccer` para dibujar el mapa de calor de las recuperaciones.
La solicitud específica es una cuadrícula de 8 de alto y 12 de largo.

```python
from mplsoccer import Pitch
import matplotlib.pyplot as plt

# Creamos el campo de StatsBomb
pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc')
fig, ax = pitch.draw(figsize=(10, 7))

# Extraemos coordenadas X e Y de las recuperaciones
x = df_recoveries['location_x']
y = df_recoveries['location_y']

# df_recoveries['shot_transition'] es 1 si terminó en tiro, 0 si no
z = df_recoveries['shot_transition'] 

# pitch.bin_statistic agrupa los puntos en el grid y calcula una estadística.
# En este caso 'mean' para sacar el porcentaje de tiros (promedio de 1s y 0s).
stats = pitch.bin_statistic(x, y, values=z, statistic='mean', bins=(12, 8))

# Dibujamos el mapa de calor
# Se mapean los porcentajes al color de las celdas
pcm = pitch.heatmap(stats, ax=ax, cmap='hot', edgecolors='#22312b', lw=1, alpha=0.7)

# (Opcional) Podemos añadir etiquetas para mostrar el % numérico dentro de cada bloque de la cuadrícula de 12x8
pitch.label_heatmap(stats, color='white', fontsize=9, ax=ax, ha='center', va='center')

plt.colorbar(pcm, ax=ax, label='% que terminan en Tiro')
```

## Paso 7: Ejecución

Para iniciar la aplicación localmente y ver los cambios en vivo, se usa el comando:

```bash
# Si 'shiny' no se reconoce como comando (común en Windows), usa:
python -m shiny run app.py --reload

# O si el comando 'shiny' está en tu PATH:
python -m shiny run app.py --reload

```
