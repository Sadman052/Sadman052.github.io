#Program to develop a linear model for speed in mph [30, 33, 41,51]
# vs density in veh/mile [500, 650, 800, 1100] of a road traffic using Numpy and Matplotlib.

import numpy as np
import matplotlib.pyplot as plt

density = np.array([500, 650, 800, 1100])
speed = np.array([30, 33, 41,51])
A = np.vstack([density, np.ones(len(density))]).T
speed = speed[:, np.newaxis]
param = np.dot((np.dot(np.linalg.inv(np.dot(A.T,A)),A.T)), speed)
slope = np.round(param[0],3)
intercept = np.round(param[1],3)

speed_pred = slope * density + intercept
speed_pred = np.vstack(speed_pred)
err= ( speed- speed_pred)**2
sse = np.sum(err)
mean = np.mean(speed)
ss = (speed-mean) **2
sst = np.sum(ss)
r_square = np.round(1 - sse/sst,2)
print(f"r_square = {r_square}, slope ={slope} and intercept = {intercept}")
plt.figure(figsize=(10,8))
plt.plot(density, speed,'r.')
plt.plot(density, speed_pred,'y')
plt.xlabel('density in veh/mile')
plt.ylabel('speed in mile/hr')
plt.show()