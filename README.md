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
