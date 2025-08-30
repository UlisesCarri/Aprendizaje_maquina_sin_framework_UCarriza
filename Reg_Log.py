#============================
#
#  Author: Ulises Orlando Carrizalez Lerín
#    Date: 26/08/2025
#  Proyect: 
#
# Note: Esta presentando el mismo error de clase, para reducir la candidad de error solo esta prediciendo 1,
# de esta manera tiene un 50/50% de exito.
#============================ 

##############################
# Imports
##############################
import pandas as pd
import math
import os
import matplotlib.pyplot as plt

##############################
# Functions
##############################
def hypothesis(θ, x):
	acum = 0
	for i in range(len(θ)):
		acum = acum + θ[i]*x[i]
	acum = acum*(-1)
	acum = 1/(1+ math.exp (acum))
	return acum

def show_errors(θ, Data, y):
	acum = 0
	error      = 0
	for i in range(len(Data)):
		hyp = hypothesis(θ,Data[i])
		
		if(y[i] == 1):
			if(hyp == 0): hyp = 0.0001
			error = (-1) * math.log(hyp)
		elif y[i] == 0:
			if(hyp == 1): hyp = 0.9999
			error = (-1) * math.log(1 - hyp)
		acum += error
	return acum/len(Data)

def update(θ, Data, y, alfa):
	temp = list(θ)
	for j in range(len(θ)):
		acum =0
		for i in range(len(Data)):
			error = hypothesis(θ,Data[i]) - y[i]
			acum = acum + error*Data[i][j]  #Sumatory part of the Gradient Descent formula for linear Regression.
		temp[j] = θ[j] - alfa*(1/len(Data))*acum  #Subtraction of original value with learning rate included.
	return temp

def Log_Reg(θ,Data,y, alfa, epochs):
	histo_error = []
	for i in range(len(Data)):
		Data[i]=  [1]+Data[i]
	#Data = scaling(Data)
	for _ in range(epochs):
		oldparams = list(θ)
		θ = update(θ, Data,y,alfa)	
		error = show_errors(θ, Data, y) # only used to show errors, it is not used in calculation
		histo_error.append(error)
		if(oldparams == θ or error < 0.0001): break
	print(θ)
	plt.plot(histo_error)
	plt.show()
	print(histo_error[len(histo_error)-1])
	return θ

def Predic(θ, test):
	pre = []
	for i in range(len(test)):
		test[i]=  [1]+test[i]
	for i in range(len(test)):
		hyp = hypothesis(θ,test[i])
		if hyp >= 0.5: pre.append(1)
		elif hyp < 0.5: pre.append(0)
	return pre
##############################
# Main
##############################
def main():
	# === Importar Datos ===
	columns   = ["Sex","Length","Diameter","Height", "Whole", "Shucked", "Viscera", "Shell","Rings"]
	df        = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Abalone","abalone.data"),names = columns).dropna().sample(frac=1)

	# === Limipar Datos ===
	df        = df[df["Sex"] != "I"] #Eliminar los Infantes
	df["Sex"] = df["Sex"].replace("M", 1).replace("F", 0) #Combertir los generos en numeros binarios
	cut       = int(0.75 * len(df))
	df_train  = df.iloc[:cut]
	df_test   = df.iloc[cut:]

    # === Preparar los Datos ===
	θ    = [0,0,0]   # bias + 2 features
	y         = df_train["Sex"].tolist()
	Data   = df_train[["Length","Diameter"]].values.tolist()

	# === Entrenar el modelo ===
	θ = Log_Reg(θ,Data,y, 0.01, 5000)

	# === Hacer prediccion ===
	test = df_test[["Length","Diameter"]].values.tolist()
	preds = Predic(θ,test)

	df_test = df_test.copy()  # evitar el warning
	df_test["Predicted"] = preds
	print(df_test[["Sex","Predicted"]])


if __name__ == "__main__":
    main()