# Práctica Visualización y Comunicación de Datos Deportivos

## Proyecto

**BeyondPlay: Transiciones y Pérdidas Peligrosas** es un dashboard interactivo desarrollado con Shiny for Python para analizar cómo las recuperaciones y las pérdidas de balón se relacionan con la generación o concesión de peligro en fútbol.

La aplicación permite explorar datos abiertos de StatsBomb por competición, temporada, equipo y patrones de juego. El foco no está solo en contar acciones, sino en comunicar qué zonas y contextos transforman una acción de cambio de posesión en una situación peligrosa.

## Objetivos

- Construir un dashboard interactivo en Shiny.
- Visualizar eventos deportivos de fútbol sobre mapas de campo.
- Analizar transiciones ofensivas tras recuperación.
- Analizar pérdidas propias que derivan en peligro rival.
- Comparar equipos mediante un cuadrante transicional.
- Explicar los resultados con métricas simples y visualizaciones legibles.

## Dashboard y funcionalidades

El dashboard usa una estructura con **sidebar** de filtros y **pestañas** de análisis.

Filtros disponibles:

- Competiciones.
- Ligas / temporadas.
- Equipos.
- Patrones de juego.
- Criterios de peligro: tiro, entrada al área y From Counter.
- Combinación de criterios: cualquiera de ellos (OR) o todos ellos (AND).
- Alcance del análisis: posesión completa o posesión más ventana temporal.

Pestañas principales:

- **Transiciones**: muestra mapas de volumen de inicios de transición y porcentaje de transiciones peligrosas.
- **Pérdidas peligrosas**: muestra mapas de volumen de pérdidas y riesgo empírico de que una pérdida derive en peligro rival.
- **Cuadrante transicional**: compara equipos según el porcentaje de recuperaciones que generan peligro y el porcentaje de pérdidas que conceden peligro.

Además, incluye una sección de metodología y un resumen con KPIs de transiciones, pérdidas, tiros, goles y zonas de mayor riesgo.

## Datos utilizados

Los datos proceden de **StatsBomb Open Data** y se consultan desde Python con la librería `statsbombpy`.

La aplicación utiliza principalmente:

- Competiciones y temporadas.
- Partidos disponibles.
- Eventos de partido.
- Equipos en posesión.
- Localización de acciones.
- Tiros y goles.
- Patrones de juego.
- Recuperaciones, intercepciones, tackles y acciones candidatas a pérdida.

## Metodología

La aplicación define una misma idea configurable de peligro para dos perspectivas:

- **Transiciones ofensivas**: acciones propias que recuperan la posesión y pueden iniciar una secuencia peligrosa.
- **Pérdidas defensivas**: acciones propias que terminan en posesión rival y pueden conceder peligro.

Para las transiciones ofensivas se consideran:

- `Ball Recovery` exitosa.
- `Interception` exitosa.
- `Duel - Tackle` ganado.

Para las pérdidas se identifica la última acción ofensiva propia antes de que la posesión pase al rival. Después se analiza la siguiente posesión rival.

El peligro puede definirse como:

- Tiro.
- Entrada al área.
- Secuencia marcada como `From Counter`.
- Combinaciones OR / AND de los criterios anteriores.

Las acciones se agregan en una cuadrícula espacial sobre el campo StatsBomb para generar mapas de calor y porcentajes por zona.

## Fundamentos de visualización aplicados

El dashboard aplica varios principios de visualización de datos deportivos:

- Uso del campo de fútbol como referencia espacial natural.
- Separación entre volumen y eficacia para evitar interpretaciones confusas.
- Mapas de calor para detectar concentraciones y zonas de riesgo.
- Etiquetas dentro de las celdas para combinar lectura cuantitativa y espacial.
- KPIs de resumen para facilitar una lectura rápida.
- Cuadrante comparativo para posicionar equipos según amenaza generada y vulnerabilidad concedida.
- Filtros reactivos para que el usuario pueda construir comparaciones relevantes.

## Insights esperados

El dashboard permite responder preguntas como:

- Qué zonas generan más transiciones peligrosas tras recuperación.
- Dónde se pierden balones que terminan generando peligro rival.
- Qué equipos son más productivos tras recuperar.
- Qué equipos son más vulnerables tras perder.
- Cómo cambia el análisis al exigir tiro, entrada al área o contraataque como criterio de peligro.
- Qué diferencias aparecen entre competiciones, temporadas y patrones de juego.

## Despliegue

La aplicación está desplegada en ShinyApps.io:

https://99f1p5-acastano149.shinyapps.io/ball-loss-to-threat/

## Ejecución local

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar la aplicación:

```bash
python -m shiny run app.py --reload
```

## Estructura de archivos

```text
PracticaVisualizacionyComunicaciondeDatosDeportivos/
- app.py
- requirements.txt
- README.md
- .gitignore
```

## Distribución del trabajo

El trabajo se ha desarrollado de forma conjunta. Ambos integrantes participamos en la exploración del problema, la definición de las métricas de análisis y la revisión general del dashboard.

La limpieza de datos, el procesamiento de eventos y la lógica ETL fueron trabajados principalmente por Juan Muñoz. Esta parte incluye la preparación de los datos de StatsBomb, la identificación de recuperaciones, pérdidas y posesiones relevantes, y la construcción de las variables necesarias para medir peligro.

Adrián Castaño se encargó principalmente del desarrollo visual del dashboard, la organización de la interfaz, la mejora de la experiencia de usuario y el despliegue de la aplicación en ShinyApps.io.

La documentación final, la preparación del repositorio de entrega y la revisión de los requisitos del enunciado fueron realizadas por Juan Muñoz.

## Conclusiones y mejoras futuras

El dashboard muestra que el análisis de recuperaciones y pérdidas mejora cuando se combina localización, contexto de posesión y resultado posterior. No todas las recuperaciones tienen el mismo valor ofensivo y no todas las pérdidas implican el mismo riesgo defensivo.

Como mejoras futuras se podrían incorporar:

- Datos 360 para incluir presión, rivales cercanos y contexto espacial.
- Exportación de tablas y gráficos.
- Comparativas predefinidas entre equipos.
- Modelos probabilísticos de amenaza.
- Filtros por marcador, minuto o fase del partido.
