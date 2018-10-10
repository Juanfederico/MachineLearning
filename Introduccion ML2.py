from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import graphviz
import numpy as np

iris = load_iris()
X_entrenamiento, X_test, Y_entrenamiento, Y_test=train_test_split(iris.data, iris.target)
arbol = DecisionTreeClassifier()
arbol.fit(X_entrenamiento, Y_entrenamiento)
print("PRUEBA 1: Arbol de decision sin restriccion de profundidad")
print(arbol.score(X_test, Y_test)) #Muestra la efectividad
print(arbol.score(X_entrenamiento, Y_entrenamiento)) #Al mandarle la misma data de entrenamiento se comprueba que reconoce cada uno de sus ítems (efectividad 1.0)

""" #Se muestra el conjunto de datos en un arbol (solo disponible para ejecutar en Anaconda)
export_graphviz(arbol, out_file='arbol.dot', class_names=iris.target_names, feature_names=iris.feature_names, impurity=False, filled=True)

with open('arbol.dot') as f:
	dot_graph=f.read()
graphviz.Source(dot_graph)"""

arbol = DecisionTreeClassifier(max_depth=3) #Defino el arbol de decisión pero ésta vez definiendo la máxima profundidad
arbol.fit(X_entrenamiento, Y_entrenamiento)
print("PRUEBA 2: Arbol de decision con profundidad maxima 3")
print(arbol.score(X_test, Y_test)) #Muestra la efectividad
print(arbol.score(X_entrenamiento, Y_entrenamiento)) #En éste caso la data de entrenamiento no puede asegurarse con un 100% de efectividad debido a la profundidad acotada

