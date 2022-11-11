# A planet, a spacecraft, a comet, whatever
import scipy.constants as spc
import numpy as np

def normalize_vector(a):
    return (a / np.sqrt(np.sum(a ** 2)))

class MyPhysObject:
    position = np.array([0, 0], dtype=float)
    mass = 1
    velocity = np.array([0, 0], dtype=float)

    def __init__(self, x, y, mass, velocity_x, velocity_y):
        self.position = np.array([x, y], dtype=float)
        self.mass = mass
        self.velocity = np.array([velocity_x, velocity_y], dtype=float)

    def use_force(self, force): #TODO: Make the fucking force actually act in the right direction, because right now it repels stuff
        self.velocity += (force / self.mass)

    def update(self, time_multiplier):
        self.position += self.velocity*time_multiplier

    def set_velocity(self, direction, speed):
        normalized_dir = direction / np.sqrt(np.sum(direction ** 2))
        self.velocity = speed*normalized_dir

def calculate_gravity(object1, object2):
    distance = np.linalg.norm(object1.position-object2.position)
    return (spc.G * object1.mass * object2.mass)/(distance**2)


class Simulation:
    list_of_objects = {}
    multiplier = 1  # A multiplier for the forces (and inverse time)

    def update(self):
        for key1, object1 in self.list_of_objects.items():
            for key2, object2 in self.list_of_objects.items():
                if object1 is not object2:
                    force = calculate_gravity(object1, object2)*self.multiplier
                    direction = object2.position-object1.position
                    normalized = normalize_vector(direction)
                    object1.use_force(force*normalized)
        for key1, object1 in self.list_of_objects.items():
            object1.update(self.multiplier)