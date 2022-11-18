import scipy.interpolate as interp
import numpy as np
import matplotlib.pyplot as plt
# Reference for Q values:
# Distance from Sun to Jupiter: 740.97 million km, so if it's that or more just return -1
# Measure the default max distance if we didn't do anything, make that return 0
# The closer the distance was to 0 the closer the return value to 1
# Tanh from -4 to 4 has value -0.99 to 0.99
# Or we can use some interpolating polynomial maybe to get the values for below 740


# Data to provide: positions of astronomical objects, position of spacecraft (might be included in the previous already)
#                  velocities of both, current fuel left, current fuel flow, current timestamp


X_values = np.flip(np.array([650000000000,400000000000,300000000000, 200000000000, 120708690477, 80075144865,45075144865, 20075144865, 10000000000, 384400000, 0]))
Y_values = np.flip(np.array([-1, -0.8, -0.5, -0.25, 0, 0.1, 0.25, 0.5, 0.7, 0.8, 1]))


spline = interp.UnivariateSpline(X_values,Y_values, s=0.5)


# TODO: Add model that takes as input the data from the simulation, an action, and returns predicted reward (So input 1+ larger than what Simulation returns)
def score(distance):
    value = spline(distance)
    if(value<-1):
        return -1
    elif(value>1):
        return 1
    return value


def visualize_spline():
    x = np.linspace(0, 800000000000, 1000000)
    y = spline(x)
    plt.figure(figsize=(30, 24))
    plt.plot(x, y, 'ro')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    for x in np.nditer(X_values):
        print("Value at ", x, ": ", score(x))
#visualize_spline()