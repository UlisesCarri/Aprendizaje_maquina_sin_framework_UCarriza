#============================
#
#   Author: Ulises Orlando Carrizalez Lerín
#     Date: 30/08/2025
#  Proyect: Implementación de una técnica de aprendizaje máquina sin el uso de un framework
#
# Note: None
#============================ 

##############################
# Imports
##############################
import pandas as pd
import os
import matplotlib.pyplot as plt

##############################
# Functions
##############################
def HYP(x,θ,b):                                 #Hacer una hipotesis en base a los parametros
  y = 0
  for i in range(len(x)):
      y += θ[i]*x[i]
  y += b
  return y

def MSE(Data,θ,b,Y,PRI=False):                  #Calcular el error con Mean squared error
  lost = 0
  for i in range(len(Data)):
    if PRI:print(Data[i])
    lost += (HYP(Data[i],θ,b)-Y[i])**2
    if PRI:print( "hyp  %f  y %f " % (HYP(Data[i],θ,b),  Y[i]))
  return lost/len(Data)

def update(Data,θ,b,Y,alfa):                    #Optimizar los parametros con GD
  θnew = θ
  for j in range(len(θ)):
    grad = 0
    for i in range(len(Data)):
      grad += (HYP(Data[i],θ,b)-Y[i])*Data[i][j]
    θnew[j] = θ[j] - (alfa/len(Data)) * grad
  grad = 0
  for i in range(len(Data)):
    grad += HYP(Data[i],θ,b)-Y[i]
  bnew = b - (alfa/len(Data)) * grad
  return θnew,bnew

def train(Data,θ,b,Y,PRI=False,ploter=False):   #Sacar los mejores valores de parametros para minimisar error
    error  = []
    epochs = 0
    while True:
        error.append(MSE(Data,θ,b,Y))
        if PRI:print(error[epochs])
        if error[epochs] < 0.001:
            print(f"Ultimo Error train: {error[epochs]*100}%")
            break
        θ,b = update(Data,θ,b,Y,0.01)
        epochs += 1
    if ploter:
        print(epochs)
        plt.plot(error)
        plt.show()
    return θ,b

def predict(θ,b,Data,Real,ploter=False):        #Hacer predicciones en base a los parametros introduicidos
    error= []
    pre = []
    for i in range(len(Data)):
        error.append(MSE(Data,θ,b,Real))
        pre.append(HYP(Data[i],θ,b))
    if ploter:
        plt.plot(error)
        plt.show()
    print(f"Promedio Error test: {(sum(error) / len(error))*100}%")
    return pre
   
##############################
# Main
##############################
def main(Correlacion = False):
    # === Importar Datos ===
    columns   = ["Sex","Length","Diameter","Height", "Whole", "Shucked", "Viscera", "Shell","Rings"]
    df        = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Abalone","abalone.data"),names = columns).dropna().sample(frac=1)
 
    # === Correlación ===
    if Correlacion:
        df["Sex"] = df["Sex"].replace("M", 1).replace("F", 0).replace("I", 2) #Combertir los generos en numeros binarios
        print(df.corr())

    # === Limpieza de Datos ===
    cut       = int(0.75 * len(df))
    df_train  = df.iloc[:cut]
    df_test   = df.iloc[cut:]

    # === Preparado de Datos ===
    independiente = ["Length","Diameter", "Whole", "Shell"]
    θ             = [0] * (len(independiente))
    b             = 0

    # === Entrenamiento ===
    Y    = df_train["Height"].tolist()
    Data = df_train[independiente].values.tolist()
    θ,b  = train(Data,θ,b,Y,PRI=False,ploter=True)

    # === Predecirt ===
    Real = df_test["Height"].tolist()
    Data = df_test[independiente].values.tolist()
    pred = predict(θ,b,Data,Real,False)
    
    # === Graficar prediccion contra real (Solo 40 muestras)===
    inst = range(40)
    plt.scatter(inst, pred[:40])
    plt.scatter(inst, Real[:40])
    plt.show()
    
if __name__ == "__main__":
    main()