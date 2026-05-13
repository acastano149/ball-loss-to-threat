# Practica Visualizacion y Comunicacion de Datos Deportivos

## Proyecto

**BeyondPlay: Transiciones y Perdidas Peligrosas** es un dashboard interactivo desarrollado con Shiny for Python para analizar como las recuperaciones y las perdidas de balon se relacionan con la generacion o concesion de peligro en futbol.

La aplicacion permite explorar datos abiertos de StatsBomb por competicion, temporada, equipo y patrones de juego. El foco no esta solo en contar acciones, sino en comunicar que zonas y contextos transforman una accion de cambio de posesion en una situacion peligrosa.

## Objetivos

- Construir un dashboard interactivo en Shiny.
- Visualizar eventos deportivos de futbol sobre mapas de campo.
- Analizar transiciones ofensivas tras recuperacion.
- Analizar perdidas propias que derivan en peligro rival.
- Comparar equipos mediante un cuadrante transicional.
- Explicar los resultados con metricas simples y visualizaciones legibles.

## Dashboard y funcionalidades

El dashboard usa una estructura con **sidebar** de filtros y **pestanas** de analisis.

Filtros disponibles:

- Competiciones.
- Ligas / temporadas.
- Equipos.
- Patrones de juego.
- Criterios de peligro: tiro, entrada al area y From Counter.
- Combinacion de criterios: cualquiera de ellos (OR) o todos ellos (AND).
- Alcance del analisis: posesion completa o posesion mas ventana temporal.

Pestanas principales:

- **Transiciones**: muestra mapas de volumen de inicios de transicion y porcentaje de transiciones peligrosas.
- **Perdidas peligrosas**: muestra mapas de volumen de perdidas y riesgo empirico de que una perdida derive en peligro rival.
- **Cuadrante transicional**: compara equipos segun el porcentaje de recuperaciones que generan peligro y el porcentaje de perdidas que conceden peligro.

Ademas, incluye una seccion de metodologia y un resumen con KPIs de transiciones, perdidas, tiros, goles y zonas de mayor riesgo.

## Datos utilizados

Los datos proceden de **StatsBomb Open Data** y se consultan desde Python con la libreria `statsbombpy`.

La aplicacion utiliza principalmente:

- Competiciones y temporadas.
- Partidos disponibles.
- Eventos de partido.
- Equipos en posesion.
- Localizacion de acciones.
- Tiros y goles.
- Patrones de juego.
- Recuperaciones, intercepciones, tackles y acciones candidatas a perdida.

## Metodologia

La aplicacion define una misma idea configurable de peligro para dos perspectivas:

- **Transiciones ofensivas**: acciones propias que recuperan la posesion y pueden iniciar una secuencia peligrosa.
- **Perdidas defensivas**: acciones propias que terminan en posesion rival y pueden conceder peligro.

Para las transiciones ofensivas se consideran:

- `Ball Recovery` exitosa.
- `Interception` exitosa.
- `Duel - Tackle` ganado.

Para las perdidas se identifica la ultima accion ofensiva propia antes de que la posesion pase al rival. Despues se analiza la siguiente posesion rival.

El peligro puede definirse como:

- Tiro.
- Entrada al area.
- Secuencia marcada como `From Counter`.
- Combinaciones OR / AND de los criterios anteriores.

Las acciones se agregan en una cuadricula espacial sobre el campo StatsBomb para generar mapas de calor y porcentajes por zona.

## Fundamentos de visualizacion aplicados

El dashboard aplica varios principios de visualizacion de datos deportivos:

- Uso del campo de futbol como referencia espacial natural.
- Separacion entre volumen y eficacia para evitar interpretaciones confusas.
- Mapas de calor para detectar concentraciones y zonas de riesgo.
- Etiquetas dentro de las celdas para combinar lectura cuantitativa y espacial.
- KPIs de resumen para facilitar una lectura rapida.
- Cuadrante comparativo para posicionar equipos segun amenaza generada y vulnerabilidad concedida.
- Filtros reactivos para que el usuario pueda construir comparaciones relevantes.

## Insights esperados

El dashboard permite responder preguntas como:

- Que zonas generan mas transiciones peligrosas tras recuperacion.
- Donde se pierden balones que terminan generando peligro rival.
- Que equipos son mas productivos tras recuperar.
- Que equipos son mas vulnerables tras perder.
- Como cambia el analisis al exigir tiro, entrada al area o contraataque como criterio de peligro.
- Que diferencias aparecen entre competiciones, temporadas y patrones de juego.

## Despliegue

La aplicacion esta desplegada en ShinyApps.io:

https://99f1p5-acastano149.shinyapps.io/ball-loss-to-threat/

Tambien se conserva la configuracion de despliegue en `rsconnect-python/shiny.json`.

## Ejecucion local

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar la aplicacion:

```bash
python -m shiny run app.py --reload
```

## Estructura de archivos

```text
PracticaVisualizacionyComunicaciondeDatosDeportivos/
├── app.py
├── requirements.txt
├── README.md
├── rsconnect-python/
│   └── shiny.json
└── .gitignore
```

## Distribucion del trabajo

Completar con los integrantes del grupo antes de la entrega final. Propuesta de reparto para documentar:

- Desarrollo de la aplicacion Shiny y filtros interactivos.
- Procesamiento de eventos StatsBomb y definicion de metricas de peligro.
- Visualizaciones, README y despliegue en ShinyApps.io.

## Conclusiones y mejoras futuras

El dashboard muestra que el analisis de recuperaciones y perdidas mejora cuando se combina localizacion, contexto de posesion y resultado posterior. No todas las recuperaciones tienen el mismo valor ofensivo y no todas las perdidas implican el mismo riesgo defensivo.

Como mejoras futuras se podrian incorporar:

- Datos 360 para incluir presion, rivales cercanos y contexto espacial.
- Exportacion de tablas y graficos.
- Comparativas predefinidas entre equipos.
- Modelos probabilisticos de amenaza.
- Filtros por marcador, minuto o fase del partido.
