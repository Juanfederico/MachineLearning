#Piedra, papel o tijera con redes neuronales
#Aprendizaje por refuerzo - Pasivo

from sklearn.neural_network import MLPClassifier
from random import choice

options = ["piedra", "papel", "tijera"]

def search_winner(jug1, jug2):
	if jug1==jug2:
		resultado = 0
	elif jug1=="piedra" and jug2=="tijera":
		resultado = 1
	elif jug1=="piedra" and jug2=="papel":
		resultado = 2
	elif jug1=="tijera" and jug2=="piedra":
		resultado = 2
	elif jug1=="tijera" and jug2=="papel":
		resultado = 1
	elif jug1=="papel" and jug2=="piedra":
		resultado = 1
	elif jug1=="papel" and jug2=="tijera":
		resultado = 2

	return resultado

#search_winner("papel", "tijera") // devuelve 2

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