import scipy.interpolate as interp
import numpy as np

# Reference for Q values:
# Distance from Sun to Jupiter: 740.97 million km, so if it's that or more just return -1
# Measure the default max distance if we didn't do anything, make that return 0
# The closer the distance was to 0 the closer the return value to 1
# Tanh from -4 to 4 has value -0.99 to 0.99
# Or we can use some interpolating polynomial maybe to get the values for below 740


# Data to provide: positions of astronomical objects, position of spacecraft (might be included in the previous already)
#                  velocities of both, current fuel left, current fuel flow, current timestamp


X_values = np.array([])




def score(distance):
    if distance> 740000000:
        return -1.0
    return 0


#spline = interp.CubicSpline