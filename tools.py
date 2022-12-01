import math

import numpy as np
import random

def getPerpendicularVector(input_vector):
    return np.array([-1*input_vector[1],input_vector[0]],dtype=float)

def normalize_vector(a):
    return (a / np.sqrt(np.sum(a ** 2)))

def vector_length(a):
    return np.sqrt(np.sum(a ** 2))

def random_normal_vector():
    a = random.randint(-100,100)
    b = random.randint(-100,100)
    return normalize_vector(np.array([a,b],dtype=float))

def custom_log(base, value):
    try:
        if value==0:
            return 0
        elif value > 0:
            return math.log(value, base)
        else:
            return math.log(value * -1, base) * -1
    except:
        print("Value was: "+str(value))
        raise ValueError

#Normalizes to [-1.0,1.0]
def normalize(x, max_x, min_x):
    return ((2*((x-min_x)/(max_x-min_x)))-1)

def angle_between_vectors(a,b):
    temp = np.dot(a,b)
    temp = temp/(vector_length(a)*vector_length(b))
    return np.arccos(temp)