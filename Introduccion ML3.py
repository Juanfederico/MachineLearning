from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn import svm #Support Vector Machine / Máquina de Vectores de Soporte

#Algoritmo de aprendizaje supervisado
#Con éste algoritmo se traza una línea o hiperplano que divida la data de forma efectiva
#Los vectores de soporte se forman con los puntos de cada conjunto más cercanos al hiperplano
#Según la distancia del punto con el hiperplano se determina a qué conjunto pertenece el elemento
#Se considera la probabilidad de cada elemento de pertenecer a un conjunto según la distancia mencionada

iris=load_iris()
X_entrenamiento, X_test, Y_entrenamiento, Y_test=train_test_split(iris.data, iris.target)
algoritmo=svm.SVC(probability=True) #Se declara el algoritmo
algoritmo.fit(X_entrenamiento, Y_entrenamiento) #Se entrena para generar el hiperplano que separe la data en base al entrenamiento

algoritmo.decision_function_shape="ovr"
print("Muestra en valores numericos a que conjunto se acerca mas la X")
print(algoritmo.decision_function(X_entrenamiento)[:10]) #Los números reflejan la cercanía con respecto al hiperplano. El más lejano es al que pertenece el elemento (o el más probable)
print("Muestra en porcentaje a que conjunto se acerca mas la X")
print(algoritmo.predict_proba(X_entrenamiento)[:10]) #Los números reflejan lo mismo que el ejemplo anterior pero expresado en porcentajes (todas las probabilidades suman 100% en cada elemento)
print("Muestra en un array a que conjunto pertenece cada X")
print(algoritmo.predict(X_entrenamiento)[:10]) #Se clasifica a cada elemento en un conjunto según los criterios anteriores


