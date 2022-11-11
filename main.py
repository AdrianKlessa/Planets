import numpy as np
import pygame
import sys
import phys
pygame.init()

def normalize_vector(a):
    return (a / np.sqrt(np.sum(a ** 2)))


DISPLAY_WIDTH = 1920
DISPLAY_HEIGHT = 1080

EARTH_DISTANCE_FROM_SUN = 148.1*(10**9)
EARTH_VELOCITY = 29.72*10**3  # m/s

EARTH_POS = np.array(normalize_vector(np.array([1,1])),dtype=float)*EARTH_DISTANCE_FROM_SUN

Sun = phys.MyPhysObject(DISPLAY_WIDTH//2, DISPLAY_HEIGHT//2, 1.9891*(10**30), 0, 0)
Earth = phys.MyPhysObject(EARTH_POS[0],EARTH_POS[1], 5.972*(10**24), 0, 0)
Earth.set_velocity(np.array([0,-1], dtype=float), EARTH_VELOCITY)


Simulation = phys.Simulation()
Simulation.list_of_objects["Earth"] = Earth
Simulation.list_of_objects["Sun"] = Sun

display = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
display.fill((255, 255, 255))
pygame.draw.line(display, (0, 0, 255),
                 (10, 10), (100, 100), width=2)

space_factor = 150*(10**7) #Since SI unit of distance is m we would have planets really far away in space, not visible on screen


def draw_body(display, body, color, r):
    pygame.draw.circle(display, color,
                       center=(body.position[0]//space_factor+DISPLAY_WIDTH//3, body.position[1]//space_factor+DISPLAY_HEIGHT//3), radius=r, width=0)
    #print(str(body.position[0]//space_factor+DISPLAY_WIDTH//2)+", "+str(body.position[1]//space_factor+DISPLAY_HEIGHT//2))

Simulation.multiplier=365*24

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    display.fill((255, 255, 255))
    Simulation.update()
    for key, planet in Simulation.list_of_objects.items():
        if key == "Earth":
            #print("Drawing Earth")
            draw_body(display,planet, pygame.Color('forestgreen'), 10)
            #draw_body(display, planet, (125,0,125), 50)
        if key == "Sun":
            #print("Drawing Sun")
            draw_body(display, planet, pygame.Color('goldenrod1'), 10)
            #draw_body(display, planet, (125,0,125), 100)
    pygame.display.update()
