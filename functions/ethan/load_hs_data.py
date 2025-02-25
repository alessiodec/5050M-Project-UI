# Modulised DEAP
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.metrics import r2_score
from deap import gp
import warnings

import Engine
import config

f = open("Latin_Hypercube_Heatsink_1000_samples.txt", "r") # "heatsink.txt" -> needs '\t', Latin_Hypercube_Heatsink_1000_samples
text = f.read()

data = [x.split(' ') for x in text.split('\n')] #If heatsink.txt seporate with /t otherwise tis a space
df = pd.DataFrame(data, columns=['Geometric1', 'Geometric2', 'Thermal_Resistance', 'Pressure_Drop'])
df = df.apply(pd.to_numeric)

X = df[['Geometric1', 'Geometric2']].values
y = df['Pressure_Drop'].values.reshape(-1,) #Thermal_Resistance, Pressure_Drop

mean_y = np.mean(y)
std_y = np.std(y)

config.mean_y = mean_y
config.std_y = std_y

print(mean_y)
print(std_y)

#Adding some scaling (Y): 
standardised_y = (y - mean_y) / std_y
standardised_y

df
