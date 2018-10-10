import numpy as np
import sklearn
from sklearn.datasets import load_iris, load_boston #Ejemplo de diccionario de datos para testear que brinda la librería
from sklearn.model_selection import train_test_split #Librería para entrenamiento de la muestra

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, Ridge

#Cargamos la data 
floresIris = load_iris() #Tipos de flores de iris
boston = load_boston() #Diferentes precios de casas

#type(floresIris) -- El tipo de dato de la variable (un diccionario)
#floresIris.keys() -- Los elementos del array que contiene floresIris
#floresIris['data'] -- Toda la muestra de datos que contiene el diccionario
#floresIris['target'] -- El valor numérico que corresponde a cada fila de la data
#floresIris['target_names'] -- Los nombres que representan los números del target
#floresIris['feature_names'] --  Los nombres de cada parámetro de medición en la data

#Definimos variables que se van a utilizar para entrenar y para testear (como entrada mandamos la data y los resultados de la misma)
#X_train, X_test, Y_train, Y_test = train_test_split(floresIris['data'], floresIris['target']) o
X_train, X_test, Y_train, Y_test = train_test_split(floresIris.data, floresIris.target)
#X_train.shape -- Observamos La cantidad de data y mediciones
#Y_train.shape -- Etiquetas correspondientes a cada elemento de X (las clasificaciones)

#Clasificador de vecinos cercanos
#En un plano, se consideran los vecinos cercanos de un punto X dado
#VECINOS=imágenes de puntos utilizados como muestra (entrenamiento) y con una clasificación determinada
#X=Nuevo punto sin clasificación
#Se consideran N vecinos cercanos (según los que se especifiquen en el clasificador)
#PROCESO
#En el punto X se buscan los N vecinos más cercanos del mismo
#De esos N vecinos, se contabilizan las apariciones de cada clasificación (EJ: 4 vecinos son patos negros y 2 patos blancos)
#Se dice que el elemento X es parte de la clasificación que tiene más apariciones en ese número N (EJ ídem anterior: el punto X es un pato negro)
print("CLASIFICACIÓN - VECINOS CERCANOS")
knn = KNeighborsClassifier(n_neighbors=2) #Instanciamos el clasificador con los parametros que necesitemos para hacer más efectivo nuestro algoritmo
print(knn.fit(X_train, Y_train)) #Le pasamos la data para entrenar
print(knn.score(X_test, Y_test)) #Le pasamos nueva data sin clasificación para que pueda reconocer de qué tipo es. Nos devuelve el rango de efectividad de 0 a 1 del modelo
print(knn.predict([[1.3, 3.4, 5.6, 1.1]])) #Dato X ingresado para 

#Regresión lineal
#Se tienen un conjunto de datos con los que se ha entrenado a la maquina
#En y=mx+b se buscan los valores de m y b que generen la recta de menor distancia entre todos los puntos y la misma recta
#Idealmente la sumatoria de las distancias tiene que ser un mínimo absoluto(en caso de aplicar derivación)
print("REGRESION - LINEAL")
X_train, X_test, Y_train, Y_test = train_test_split(boston.data, boston.target)
rl = LinearRegression()
print(rl.fit(X_train, Y_train))
print(rl.score(X_test, Y_test))
#Cuando la data presenta una especie de curva en la que no aplicaría de forma óptima una regresión lineal simple
#Cuando los elementos no se distribuyen formando una curva, se puede observar que Ridge y Lineal Regression tienen una eficacia similar
print("REGRESION - RIDGE")
ridge=Ridge()
print(ridge.fit(X_train, Y_train))
print(ridge.score(X_test, Y_test))

#Arbol de decisiones
#Arbol binario en el cual se tienen una serie de condiciones con sus bifurcaciones correspondientes
#Las condiciones refieren a valores de X e Y en el plano para determinar la pertenencia a una clasificación

print ("termino")
