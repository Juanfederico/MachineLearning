#Piedra, papel o tijera con redes neuronales

from sklearn.neural_network import MLPClassifier
from random import randint
from time import sleep
import numpy as np

opciones = [1, 2, 3, 4, 5, 6, 7, 8, 9] #Posibles casillas para completar
victoria = ["111000000", "000111000", "000000111", "100100100", "010010010", "001001001", "100010001"] #Todas las posibilidades de ganar

class Jugador:
	def __init__(self, nombre):
		self.nombre = nombre
		self.estadoActual = "000000000"

	def pintar(self, posicion):
		self.estadoActual = self.estadoActual[0:posicion-1] + "1" + self.estadoActual[posicion:]

class Tablero:
	def __init__(self):
		self.estadoActual = "000000000"

	def pintar(self, posicion):
		if self.estadoActual[posicion-1] == "0":
			self.estadoActual = self.estadoActual[0:posicion-1] + "1" + self.estadoActual[posicion:]
			return True
		else:
			return False

#Acciones que pueden realizar los diversos

def decidir(jugador1, jugador2, tablero, debug=False):
	
	if predict[0] >= 0.95:
		player2 = options[0]
	elif predict[1] >= 0.95:
		player2 = options[1]
	elif predict[2] >= 0.95:
		player2 = options[2]
	else:
		player2 = get_choice()

	if debug==True:
		print("Jugador1: "+player1+ ", Jugador2 (modelo): " +str(predict)+ " -----> " +player2)

	winner = search_winner(player1, player2)
	if debug==True:
		print("Comprobamos: player1 VS player2: " +str(winner))

	if winner==2:
		data_X.append(str_to_list(player1))
		data_Y.append(str_to_list(player2))
		score["win"]+=1
	else:
		score["loose"]+=1

	return score, data_X, data_Y



def comprobarVictoria(jugador1, jugador2, tablero):
	ganador = False
	for posibilidad in victoria:
		if(jugador1.estadoActual == posibilidad):
			ganador = jugador1
		elif(jugador2.estadoActual == posibilidad):
			ganador = jugador2
	if(tablero.estadoActual == "111111111"): ganador="Empate"
	return ganador

def jugar(jugador1, jugador2):
	tablero = Tablero()
	jugadorTurno = jugador1 #Por defecto el que comienza es el jugador 1
	sinFinalizar = True #Queda en true hasta que termine la partida
	while sinFinalizar:
		while True:
			casillaRandom = randint(1,9)
			flag = tablero.pintar(casillaRandom)
			if(flag):
				print("El tablero esta asi: " + tablero.estadoActual)
				jugadorTurno.pintar(casillaRandom)
				print("El jugador 1 esta asi: " + jugador1.estadoActual)
				print("El jugador 2 esta asi: " + jugador2.estadoActual)
				if(jugadorTurno==jugador1): jugadorTurno=jugador2
				else: jugadorTurno=jugador1
				break
		if(comprobarVictoria(jugador1, jugador2, tablero) != False):
			ganador = comprobarVictoria(jugador1, jugador2, tablero)
			if(ganador=="Empate"): print ("No hubo ganador")
			else: print("ha ganado " + ganador.nombre + " con la jugada " + ganador.estadoActual)
			sinFinalizar = False

jugador1 = Jugador("Juan")
jugador2 = Jugador("Pedro")

#Inicializando datos necesarios para mi red neuronal
score = {"win": 0, "loose": 0}
#Creamos los input necesarios para que se produzca una victoria por parte de algun jugador (de izq a der)
victoria1 = [1, 1, 1, 0, 0, 0, 0, 0, 0]
victoria2 = [0, 0, 0, 1, 1, 1, 0, 0, 0]
victoria3 = [0, 0, 0, 0, 0, 0, 1, 1, 1]
victoria4 = [1, 0, 0, 1, 0, 0, 1, 0, 0]
victoria5 = [0, 1, 0, 0, 1, 0, 0, 1, 0]
victoria6 = [0, 0, 1, 0, 0, 1, 0, 0, 1]
victoria7 = [1, 0, 0, 0, 1, 0, 0, 0, 1]
victoria8 = [0, 0, 1, 0, 1, 0, 1, 0, 0]
novictoria1 = [0, 0, 1, 0, 0, 0, 0, 0, 0]
novictoria2 = [0, 0, 0, 1, 0, 1, 0, 0, 0]
#Data de entrenamiento
data_X = []
data_X.append(victoria1)
data_X.append(victoria2)
data_X.append(victoria3)
data_X.append(victoria4)
data_X.append(victoria5)
data_X.append(victoria6)
data_X.append(victoria7)
data_X.append(victoria8)
data_X.append(novictoria1)
data_X.append(novictoria2)

data_X = [victoria1, victoria2, victoria3, victoria4, victoria5, victoria6, victoria7, victoria8, novictoria1, novictoria2]
data_Y = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0] #1=Ganador , 0=No ganador

redNeuronal = MLPClassifier(hidden_layer_sizes=(10), verbose=False) #En este caso, perceptron multicapa

redNeuronal.fit(data_X, data_Y) #Entrenando la red por primera vez

jugar(jugador1, jugador2)