# import pandas as pd
import phys
import tools
# import scipy.constants as spc
import math
import csv


SUN_MASS = 1.9891 * (10 ** 30)
G = 6.673 * (10 ** -11)  # Had to remove scipy as it's not supported in web pygame


# Semi major axis is half the sum of perihelion and aphelion
def vis_viva(r, perihelion, aphelion):  # r is the current distance from the sun
    a = (perihelion + aphelion) / 2
    return math.sqrt((G * SUN_MASS) * ((2 / r) - (1 / a)))


# Dataset from https://www.kaggle.com/datasets/iamsouravbanerjee/planet-dataset
# Based on NASA data



def load_data():
    with open('solar_system_dataset.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        #dt["Planet"] = dt["Planet"].astype("string").apply(lambda x: x.strip())
        list_of_objects = {}
        #for index, row in dt.iterrows():
        for row in csv_reader:
            normal = tools.random_normal_vector()
            mass = float(row['Mass (10^24kg)']) * (float(10) ** float(24))
            perihelion = float(row['Perihelion (10^6 km)']) * (float(10) ** float(9))
            aphelion = float(row['Aphelion (10^6 km)']) * (float(10) ** float(9))
            pos = normal * perihelion
            velocityNumber = vis_viva(perihelion, perihelion, aphelion)
            velocityNormal = tools.getPerpendicularVector(normal)
            velocity = velocityNumber * velocityNormal
            Planet = phys.MyPhysObject(pos[0], pos[1], mass, velocity[0], velocity[1])
            list_of_objects[row["Planet"].strip()] = Planet
        return list_of_objects


# This caused circular orbits because we only used the average velocity and distance
"""
def load_data_old():
    list_of_objects = {}
    for index, row in dt.iterrows():
        normal = tools.random_normal_vector()
        mass = row['Mass (10^24kg)']*(10**24)
        distance = row['Distance from Sun (10^6 km)']*(10**9)
        pos = normal*distance
        velocityNumber = row['Orbital Velocity (km/s)']*1000
        velocityNormal = tools.getPerpendicularVector(normal)
        velocity = velocityNumber*velocityNormal

        Planet = phys.MyPhysObject(pos[0],pos[1],mass,velocity[0],velocity[1])
        list_of_objects[index]=Planet
    return list_of_objects
"""
