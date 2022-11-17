import numpy as np
import random

def getPerpendicularVector(input_vector):
    return np.array([-1*input_vector[1],input_vector[0]],dtype=float)

def normalize_vector(a):
    return (a / np.sqrt(np.sum(a ** 2)))

def random_normal_vector():
    a = random.randint(-100,100)
    b = random.randint(-100,100)
    return normalize_vector(np.array([a,b],dtype=float))