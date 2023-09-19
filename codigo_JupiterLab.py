# Cargamos a nuestro entorno de trabajo las librerias necesarias.

import matplotlib.pyplot as plt
from sklearn import linear_model # usando sklear para saber los valores optimos
import seaborn as sns
import numpy as np
import pandas as pd


# Cargamos la información del CSV en una variable llamada data

data = pd.read_csv("C:/Users/LEO/Desktop/cars.csv", sep=",")

# Exploración de los datos
data.columns
data.info
data.describe


# Visualizamos los datos obtenidos del archivo CSV
data

# Agrupamos por tipo de inmueble
data.groupby(['Tipo de Inmueble']).count()['Ciudad']


#Realizo la grafica de dispersión
data.plot.scatter(x="year", y="priceUSD")
plt.show()


# Ejemplo Regresión Lineal
 
regresion = linear_model.LinearRegression()

#Agrego los datos en un array o vector
years = data["year"].values.reshape((-1,1))

#Ahora si creamos el modelo
modelo = regresion.fit(years, data["priceUSD"])

print("Interseccion (b)", modelo.intercept_)
#imprimos la pendiente
print("Pendiente (m)", modelo.coef_)


entrada= [[1980],[1990],[2000],[2010]]
predicciones = modelo.predict(entrada)
print(predicciones)

data.plot.scatter(x="year", y="priceUSD", label='Datos originales')
plt.scatter(entrada, predicciones, color='red')
plt.plot(entrada, predicciones, color='black', label='Línea de regresión')
plt.xlabel('year')
plt.ylabel('priceUSD')
plt.legend()
plt.show()