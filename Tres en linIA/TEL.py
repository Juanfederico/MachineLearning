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

jugador1 = Jugador("Juan")
jugador2 = Jugador("Pedro")

#Inicializando datos necesarios para mi red neuronal
score = {"win": 0, "loose": 0}
data_X = []
data_Y = []
predict = modelo.predict_proba([str_to_list(player1)])[0]
clf = MLPClassifier(verbose=False, warm_start=True) #En este caso, perceptron multicapa


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

jugar(jugador1, jugador2)



#respuesta = comprobarVictoria(jugador, jugador2)
#print(respuesta)


#search_winner("papel", "tijera") // devuelve 2


"""


test = [
	["piedra", "piedra", 0],
	["piedra", "tijera", 1],
	["piedra", "papel", 2]
]

def get_choice():
	return choice(options)

def str_to_list(option):
	if option=="piedra":
		res = [1, 0, 0]
	elif option=="papel":
		res = [0, 1, 0]
	else:
		res = [0, 0, 1]
	return res

data_X = list(map(str_to_list, ["piedra", "tijera", "papel"]))
data_Y = list(map(str_to_list, ["papel", "piedra", "tijera"]))

clf = MLPClassifier(verbose=False, warm_start=True)

modelo = clf.fit([data_X[0]], [data_Y[0]]) #Se entrena con un solo resultado (Si el otro elige piedra, debe elegir papel)

def play_and_learn(iters=10, debug=False):
	score = {"win": 0, "loose": 0}
	data_X = []
	data_Y = []

	for i in range(iters):
		player1 = get_choice()
		predict = modelo.predict_proba([str_to_list(player1)])[0]
		
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

#score, data_X, data_Y = play_and_learn(1, debug=True)
#print(data_X)
#print(data_Y)
#print("Score: " +str(score))
#if len(data_X):
#	modelo = modelo.partial_fit(data_X, data_Y)

i = 0
historic_pct = []
while True:
	i+=1
	score, data_X, data_Y = play_and_learn(iters=1000, debug=False)
	pct = (score["win"]*100/(score["win"]+score["loose"]))
	historic_pct.append(pct)
	print("Iteracion: "+str(i)+" - Score: " +str(score)+ ", %" +str(pct))

	if len(data_X):
		modelo = modelo.partial_fit(data_X, data_Y)

	if sum(historic_pct[-9:])==900: #Ultimos 9 valores
		break

#Ejemplo de partidas random con ganadores en cada caso

#for i in range (10):
#	jugador1 = get_choice()
#	jugador2 = get_choice()
#	print ("jugador1: " +jugador1+ ", jugador2: " +jugador2+ ", ganador: " + str(search_winner(jugador1, jugador2)))  


"""