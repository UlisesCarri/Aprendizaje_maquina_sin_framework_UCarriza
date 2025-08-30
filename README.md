# Implementacion_tecnica_aprendizaje_maquina_sin_framework
**Auto** : Ulises Orlando Carrizalez Lerín
**Fecha**: 30/08/2025

## Informacion de Dataset
Predicción de la edad del abulón mediante mediciones físicas. La edad del abulón se determina cortando la concha a través del cono, tiñéndola y contando el número de anillos con un microscopio, una tarea tediosa y laboriosa. Se utilizan otras mediciones, más fáciles de obtener, para predecir la edad. Para resolver el problema, puede ser necesaria información adicional, como los patrones climáticos y la ubicación (y, por consiguiente, la disponibilidad de alimento).

## Modelo usado 
El modelo que se eligió originalmente fue una ***regresión logística***, con el objetivo de predecir el sexo del abulón a partir de los demás datos del conjunto. Sin embargo, este intento resultó fallido, ya que previamente no se realizó un análisis de correlación. Una vez efectuado dicho análisis, se descubrió que las variables disponibles en el dataset no presentaban relación significativa con el sexo del abulón.

Tras este resultado, se decidió cambiar de enfoque y aplicar un modelo de ***regresión lineal*** con el propósito de predecir la altura del abulón. En este caso, se llevó a cabo el análisis de correlación de manera anticipada, identificando que las variables Length, Diameter, Whole y Shell tenían una correlación superior al 80% con la altura, por lo que se seleccionaron como variables independientes del modelo.

Posteriormente, se programó un modelo de regresión lineal que emplea ***Gradient Descent*** para optimizar los hiperparámetros, utilizando el ***Mean Squared Error*** como métrica de evaluación. Para evitar que el entrenamiento se prolongara indefinidamente, se estableció como criterio de detención alcanzar un error inferior a 0.001. Una vez concluido el entrenamiento, los parámetros obtenidos se emplearon para predecir nuevos valores de altura, los cuales se compararon con los reales, obteniendo un error promedio de 0.07%, lo que indica un desempeño satisfactorio del modelo.

## Documentos
Reg_Lineal.py : Implementación de modelo de regresión lineal exitoso
Reg_Log.py    : Implementación de modelo de regresión logística fallido
