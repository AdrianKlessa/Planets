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

    def use_force(self, force):
        self.velocity += (force / self.mass)

    def update(self, time_multiplier):
        self.position += self.velocity * time_multiplier

    def set_velocity(self, direction, speed):
        normalized_dir = direction / np.sqrt(np.sum(direction ** 2))
        self.velocity = speed * normalized_dir


# Thanks https://space.stackexchange.com/questions/30497
class Spaceship(MyPhysObject):  # Default based on Falcon 9, hope I'm not messing up the data here
    fuel_mass = 96670  # kg
    specific_impulse = 340  # s
    max_flow_rate = 2960  # kg/s
    current_flow_rate = 0  # kg/s
    direction = np.array([0, 0], dtype=float)  # Facing direction, vector, hopefully normalized but let's assume not

    def calculate_thrust(self):
        return self.current_flow_rate * self.specific_impulse

    def update(self, time_multiplier):
        if self.fuel_mass>0:
            force = self.calculate_thrust() * time_multiplier
            fuel_used = self.current_flow_rate * time_multiplier
            self.mass -= fuel_used
            self.fuel_mass -= fuel_used
            normalized = normalize_vector(self.direction)
            self.use_force(force * normalized)
        self.position += self.velocity * time_multiplier


def calculate_gravity(object1, object2):
    distance = np.linalg.norm(object1.position - object2.position)
    return (spc.G * object1.mass * object2.mass) / (distance ** 2)


class Simulation:
    list_of_objects = {}
    multiplier = 1  # A multiplier for the forces (and inverse time)

    def update(self):
        for key1, object1 in self.list_of_objects.items():
            for key2, object2 in self.list_of_objects.items():
                if object1 is not object2:
                    force = calculate_gravity(object1, object2) * self.multiplier
                    direction = object2.position - object1.position
                    normalized = normalize_vector(direction)
                    object1.use_force(force * normalized)
        for key1, object1 in self.list_of_objects.items():
            object1.update(self.multiplier)
