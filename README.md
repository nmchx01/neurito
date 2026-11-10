# Neurito 🧠✨

Neurito es un pequeño juego educativo hecho con Python y Pygame. Cambia el hambre, la energía y el aburrimiento con las barras o los botones y observa cómo una red neuronal elige entre **comer**, **dormir** y **jugar**. La interfaz usa colores suaves, una tipografía redondeada (Comic Sans MS cuando está instalada) y la mascota animada.

## Ejecutar

Necesitas Python 3.12 o posterior. Desde la carpeta del proyecto:

```bash
uv sync
uv run python main.py
```

También puedes instalar las dependencias con `pip install numpy pygame-ce` y ejecutar `python main.py`.

## Cómo jugar

- Arrastra el círculo de una barra, gira la rueda del ratón sobre ella o pulsa **− / +** para cambiar una necesidad.
- **Azar** genera un estado al azar; **Reiniciar** devuelve el estado tranquilo.
- Teclas **1 / 2 / 3** suben hambre, energía y aburrimiento; **Q / W / E** las bajan. **R** genera un estado al azar, **N** reinicia y **Esc** cierra.
- Las barras de la derecha muestran la probabilidad que Softmax asigna a cada acción. La acción con el porcentaje mayor es la predicción.

## Apartado visual: Neurito busca comida en un laberinto 12 × 12

En la ventana principal, pulsa **Laberinto 12×12** o la tecla **L**. Neurito debe llegar desde la esquina superior izquierda hasta la comida de la esquina inferior derecha. El tablero muestra las celdas exploradas y anima la ruta final.

Al abrir el apartado, el tablero queda quieto y muestra la leyenda y la comparación de resultados. La búsqueda solo comienza al hacer clic en **BFS**, **DFS** o **A***. Usa **Espacio** para pausar o continuar, **R** para reiniciar la búsqueda seleccionada y **Esc** para volver al laboratorio. También puedes abrir este apartado directamente con:

```bash
uv run laberinto.py
```

BFS, DFS y A* son algoritmos de **búsqueda**; no son métodos de entrenamiento de la red neuronal. El mapa se genera de forma reproducible con semilla 16319 y los tres algoritmos recorren el mismo tablero.

### Comparación y recomendación

En este tablero, cada movimiento cuesta lo mismo:

| Algoritmo | Celdas expandidas | Pasos de la ruta | Frontera máxima |
|---|---:|---:|---:|
| BFS | 109 | 24 | 11 |
| DFS | 80 | 66 | 29 |
| A* | 59 | 24 | 18 |

**A* es la mejor opción en este laberinto:** encuentra una ruta óptima de 24 pasos y expande menos celdas que BFS y DFS. BFS también garantiza una ruta más corta cuando todos los movimientos cuestan lo mismo, pero aquí explora más. DFS explora menos que BFS en este mapa, aunque su ruta tiene 42 pasos adicionales; en general no garantiza encontrar la ruta más corta. La recomendación depende del objetivo y del mapa: para este caso, A* combina ruta óptima con menos expansiones.

## Regla y datos del experimento

Para enseñar el concepto sin datos personales, el proyecto inventa estados con tres números entre 0 y 1. La etiqueta correcta se crea con esta regla:

```text
hambre      = hambre
cansancio   = 1 - energía
aburrimiento = aburrimiento
acción correcta = la necesidad con el valor más alto
```

Si hay empate, `argmax` elige la primera opción en este orden: comer, dormir, jugar. Se entrenan 3.000 ejemplos y se mide el resultado en 1.000 estados nuevos. Se mantienen las semillas aleatorias (42 para los datos y 7 para los pesos), la arquitectura de 6 neuronas ocultas y una tasa de aprendizaje de 0,1, salvo en el experimento que cambia ese parámetro. Cada fila parte de pesos iniciales reproducibles. La pérdida de la tabla es la entropía cruzada media en los datos de prueba; la precisión es la proporción cuya acción más probable coincide con la etiqueta de la regla. La etiqueta elegida por la red aparece sobre la mascota. Si las necesidades están tranquilas (hambre y aburrimiento menores que 40%, energía mayor que 60%), el dibujo muestra además el estado visual «Modo calma»; las tres probabilidades siguen visibles.

### Cambiar dos necesidades y variar la tercera

Fijando hambre en **50%** y energía en **80%**, el cansancio es **20%**. Al aumentar el aburrimiento, la regla compara `50% hambre`, `20% cansancio` y aburrimiento. Predice **comer** por debajo de 50%; con aburrimiento exactamente en 50% hay empate y también gana comer por el orden de las etiquetas; por encima de 50% predice **jugar**. La red entrenada aprende una aproximación a este límite, por lo que su cruce puede quedar unas centésimas a un lado del 50%.

### Prioridades parecidas y Softmax

Softmax hace más grande la probabilidad del puntaje más alto, pero si los puntajes están muy juntos, las probabilidades también quedan parecidas. Por ejemplo, para puntajes `[0.51, 0.50, 0.49]` da aproximadamente **[33,7%, 33,3%, 33,0%]**. Para `[0.80, 0.10, 0.10]` da aproximadamente **[50,2%, 24,9%, 24,9%]**. Aun así, la primera clase gana en los dos ejemplos. Esas cifras ilustran Softmax; no son entradas directas de las barras de necesidades.

### Tamaño de la capa oculta

| Neuronas ocultas | Pérdida de prueba ↓ | Precisión de prueba ↑ |
|---:|---:|---:|
| 3 | 0,199 | 94,5% |
| 6 (base) | 0,140 | 98,0% |
| 10 | 0,154 | 98,3% |
| 20 | 0,121 | 99,2% |

Veinte neuronas representaron mejor esta regla en esta corrida. Diez acertaron un poco más que seis, pero su pérdida fue mayor: más neuronas pueden ayudar, aunque no mejoran todas las medidas a la vez. El resultado también depende del problema.

### Épocas de entrenamiento

| Épocas | Pérdida de prueba ↓ | Precisión de prueba ↑ |
|---:|---:|---:|
| 100 | 0,914 | 81,0% |
| 400 | 0,315 | 95,1% |
| 1.200 (base) | 0,140 | 98,0% |

Con pocas épocas la red aún no había aprendido bien la regla. En este rango, más entrenamiento mejoró la pérdida y la precisión. Entrenar más no garantiza que siempre mejore: con demasiadas épocas una red también puede memorizar ejemplos y rendir peor en datos nuevos.

### Tasa de aprendizaje

| Tasa | Pérdida de prueba ↓ | Precisión de prueba ↑ |
|---:|---:|---:|
| 0,01 | 0,858 | 83,6% |
| 0,1 (base) | 0,140 | 98,0% |
| 0,5 | 0,101 | 95,3% |
| 1,0 | 0,062 | 97,1% |

Una tasa demasiado pequeña avanzó poco en las 1.200 épocas. Con 0,5 la precisión fue menor que con la tasa base, aunque la pérdida siguió razonable. A 1,0 la pérdida bajó mucho, pero la precisión no recuperó el nivel de la base. En esta muestra el aprendizaje no explotó; tasas grandes pueden hacer que los pasos se salten el mejor ajuste y vuelvan inestable el entrenamiento.

Los números son de una sola configuración reproducible, no una ley universal. La pérdida y la precisión miden aspectos distintos: la pérdida también penaliza cuánta confianza asigna el modelo a una respuesta equivocada.

## ¿Qué necesitaría una cuarta acción?

Haría falta definir una acción nueva y recopilar ejemplos etiquetados para ella. Por ejemplo, para añadir **buscar compañía**: registrar estados de hambre, energía y aburrimiento, junto con una etiqueta decidida por una persona que indique cuándo esa acción es la adecuada. Conviene incluir muchos ejemplos variados, en especial estados cercanos al límite entre acciones, y acordar qué hacer cuando dos necesidades empatan. Después habría que añadir una cuarta salida en la capa final y evaluar con estados reservados que no participaron en el entrenamiento.

Los datos sintéticos actuales no enseñan cuándo una mascota real debe buscar compañía. Para ese caso se necesitarían observaciones adecuadas y etiquetas fiables, además de revisar que la acción sea útil y segura.

## Preguntas de comprensión

1. **¿En qué se diferencian pesos y sesgos?** Los pesos dicen cuánto influye cada entrada o neurona en la siguiente. Los sesgos desplazan el punto en que una neurona empieza a activarse, como ajustar el cero de una regla. Ambos se aprenden durante el entrenamiento.
2. **¿Por qué el cansancio es `1 - energía`?** Energía alta significa poco cansancio y energía baja significa mucho. Restar de 1 invierte la escala: energía 1 da cansancio 0; energía 0 da cansancio 1.
3. **¿Qué hacen ReLU y Softmax?** ReLU cambia cada valor negativo por cero y deja pasar los positivos; así la capa oculta puede aprender patrones por tramos. Softmax transforma los puntajes finales en probabilidades entre 0 y 1 que suman 1.
4. **¿Por qué una precisión alta puede engañar?** Porque solo cuenta cuántas predicciones ganadoras acertaron. Puede ocultar errores en una acción poco frecuente, confianza desmedida, ejemplos repetidos o una regla sintética demasiado fácil. Una prueba separada ayuda, pero no demuestra que el modelo entienda las necesidades reales de un animal.

## Archivos

- `main.py`: juego, datos sintéticos, red neuronal y controles.
- `assets/`: ilustraciones de Neurito para sus animaciones.

## Neurito aprende a saltar

Este experimento complementario usa una red neuronal supervisada para decidir cuándo saltar obstáculos. La red recibe distancia, velocidad y tiempo estimado hasta el obstáculo. Aprende con ejemplos sintéticos etiquetados con una regla temporal; no aprende directamente de las colisiones.

### Ejecutar

Con Python 3.12 o posterior y las dependencias del proyecto instaladas:

```bash
uv sync
uv run saltar.py
```

### Controles

| Tecla | Acción |
|---|---|
| `1` | Entrenar una época |
| `2` | Entrenar cien épocas |
| `3` | Activar o pausar el entrenamiento continuo |
| `A` | Alternar control automático y manual |
| `Espacio` | Saltar manualmente |
| `C` | Reiniciar la pista y los contadores |
| `R` | Reiniciar la red y los contadores |
| `Esc` | Salir |

### Experimentos

Registra los resultados de cada prueba. Reinicia el juego y la red cuando corresponda para comparar las configuraciones.

1. Prueba el juego sin entrenar. Anota colisiones y obstáculos esquivados.
2. Entrena de cien en cien épocas; compara la pérdida, la precisión y el comportamiento.
3. Cambia `ocultas=12` por `ocultas=3` o `ocultas=24`; compara la generalización.
4. Modifica la tasa de aprendizaje de `0.8` a `0.01` o `1.2` y estudia la convergencia.
5. Cambia la ventana de entrenamiento en `etiqueta()` y en la generación de ejemplos positivos. Comprueba si la política resultante es compatible con la física.

| Prueba | Épocas | Pérdida | Precisión de prueba | Choques | Esquivados | Observaciones |
|---|---:|---:|---:|---:|---:|---|
| Sin entrenar | 0 | | | | | |
| Entrenamiento | 100 | | | | | |

### Pregunta central: ¿por qué una red puede clasificar ejemplos con buena precisión y aun así chocar?

La precisión se calcula sobre ejemplos sintéticos de prueba y mide si la red reproduce las etiquetas de la regla, no si evita choques durante la simulación. La regla actual marca como positivo saltar cuando el tiempo estimado hasta el obstáculo está entre 7 y 13 fotogramas. La física depende además del impulso, la gravedad, el tamaño del jugador y del obstáculo, la velocidad y el momento en que empieza el salto. Una predicción correcta según las etiquetas puede, por tanto, iniciar el salto demasiado pronto o demasiado tarde.

La precisión tampoco distingue por sí sola el costo de cada error: decir «esperar» cuando ya era necesario saltar puede acabar en choque, aunque la mayoría de las demás predicciones sean correctas. La política automática solo inicia un salto cuando Neurito está en el suelo. Por eso hay que comparar las métricas de prueba con el comportamiento real y contar choques y obstáculos esquivados. Una precisión alta en ejemplos sintéticos no garantiza una política compatible con la física.

Archivos del experimento: [`saltar.py`](saltar.py), [`salto_modelo.py`](salto_modelo.py), [`salto_juego.py`](salto_juego.py) y [`GUIA_SALTO.md`](GUIA_SALTO.md).

### Resultados de las pruebas

Corrida reproducible con semilla 42 para los pesos, 5.000 ejemplos de entrenamiento, 1.500 ejemplos de prueba separados y semilla 19 para la secuencia de obstáculos. Salvo cuando se varía una ventana, las comparaciones usan los mismos conjuntos de datos. Los contadores se obtuvieron ejecutando la política automática junto con `JuegoSalto.actualizar()`; no se abrió la ventana gráfica de Pygame en esta corrida.

#### 1. Sin entrenar y entrenamiento gradual

La prueba inicial terminó en **100 colisiones y 0 obstáculos esquivados** antes de completar 100 encuentros. Para cada checkpoint entrenado se simularon 30 encuentros desde la misma pista inicial.

| Épocas | Pérdida de entrenamiento | Pérdida de prueba | Precisión de prueba | Choques / 30 | Esquivados / 30 |
|---:|---:|---:|---:|---:|---:|
| 100 | 0,25917 | 0,26463 | 91,20% | 1 | 29 |
| 200 | 0,23010 | 0,23504 | 92,00% | 0 | 30 |
| 300 | 0,20704 | 0,21073 | 92,27% | 0 | 30 |
| 400 | 0,17769 | 0,17964 | 92,47% | 0 | 30 |
| 500 | 0,14096 | 0,14102 | 93,93% | 0 | 30 |
| 600 | 0,10957 | 0,10851 | 96,13% | 0 | 30 |
| 700 | 0,08895 | 0,08749 | 97,20% | 0 | 30 |
| 800 | 0,07553 | 0,07393 | 97,60% | 0 | 30 |
| 900 | 0,06616 | 0,06454 | 97,93% | 0 | 30 |
| 1.000 | 0,05921 | 0,05761 | 98,00% | 0 | 30 |
| 1.100 | 0,05380 | 0,05223 | 98,20% | 0 | 30 |
| 1.200 | 0,04945 | 0,04791 | 98,33% | 0 | 30 |

En esta corrida, 100 épocas ya mejoraron mucho el juego frente a la red sin entrenar; desde 200 épocas no hubo choques en los bloques de 30 encuentros.

#### 2. Tamaño de la capa oculta

Cada configuración se entrenó durante 1.200 épocas con tasa 0,8 y se evaluó en 100 encuentros.

| Neuronas ocultas | Pérdida de prueba | Precisión de prueba | Choques / 100 | Esquivados / 100 |
|---:|---:|---:|---:|---:|
| 3 | 0,20440 | 93,60% | 0 | 100 |
| 12 | 0,04791 | 98,33% | 0 | 100 |
| 24 | 0,04291 | 98,67% | 0 | 100 |

Con 3 neuronas la red también esquivó los 100 obstáculos de esta prueba, pero clasificó peor el conjunto reservado. Las capas de 12 y 24 generalizaron mejor en estos datos; la diferencia entre ellas fue pequeña.

#### 3. Tasa de aprendizaje

Pérdida y precisión de prueba cada 100 épocas. La última columna muestra el resultado del juego en 100 encuentros al completar 1.200 épocas.

| Tasa | Épocas | Pérdida de prueba | Precisión de prueba | Choques / 100 al final | Esquivados / 100 al final |
|---:|---:|---:|---:|---:|---:|
| 0,01 | 100 | 0,64401 | 90,67% | | |
| 0,01 | 600 | 0,49385 | 86,67% | | |
| 0,01 | 1.200 | 0,40934 | 87,13% | 100 | 0 |
| 0,8 | 100 | 0,26463 | 91,20% | | |
| 0,8 | 600 | 0,10851 | 96,13% | | |
| 0,8 | 1.200 | 0,04791 | 98,33% | 0 | 100 |
| 1,2 | 100 | 0,27804 | 89,47% | | |
| 1,2 | 600 | 0,06872 | 97,07% | | |
| 1,2 | 1.200 | 0,03276 | 98,93% | 0 | 100 |

La tasa 0,01 convergió lentamente y, tras 1.200 épocas, aún chocó en todos los encuentros simulados pese a una precisión de 87,13%. Las tasas 0,8 y 1,2 aprendieron más rápido en esta corrida; 1,2 no mostró inestabilidad en el rango medido.

#### 4. Compatibilidad de la ventana de etiquetas con la física

Se cambió a la vez el intervalo positivo de generación y la condición de `etiqueta()`. Cada política entrenada se midió en 100 encuentros.

| Ventana (fotogramas hasta el obstáculo) | Pérdida de prueba | Precisión de prueba | Choques / 100 | Esquivados / 100 |
|---:|---:|---:|---:|---:|
| 7–13 (base) | 0,04791 | 98,33% | 0 | 100 |
| 18–24 (salto demasiado temprano) | 0,04520 | 98,00% | 100 | 0 |
| 25–31 (salto aún más temprano) | 0,04627 | 98,33% | 100 | 0 |

Las ventanas tempranas mantuvieron una precisión cercana al 98%, pero todas las partidas chocaron: la etiqueta se reproducía bien, aunque el salto se iniciaba demasiado pronto para la trayectoria física. Este resultado responde la pregunta central con un ejemplo directo: la precisión no mide compatibilidad física.

Estos resultados corresponden a una corrida determinista con una secuencia de obstáculos y no garantizan el mismo resultado en todas las semillas o configuraciones. La evaluación sí muestra por qué conviene registrar simultáneamente métricas de clasificación y choques del juego.
