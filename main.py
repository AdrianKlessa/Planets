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
SPACESHIP_POS = np.array(normalize_vector(np.array([2,1])),dtype=float)*EARTH_DISTANCE_FROM_SUN

# TODO: Fix the positions so that sun is actually in the middle of the screen
# TODO: Add more planets, change their render size


#Planetary bodies
Sun = phys.MyPhysObject(DISPLAY_WIDTH//2, DISPLAY_HEIGHT//2, 1.9891*(10**30), 0, 0)
Earth = phys.MyPhysObject(EARTH_POS[0],EARTH_POS[1], 5.972*(10**24), 0, 0)
Earth.set_velocity(np.array([0,-1], dtype=float), EARTH_VELOCITY)

#The spaceship
#Spaceship = phys.Spaceship(900,900,96570,0,0,92670,340,2960,0,np.array([1, 0], dtype=float))
Spaceship = phys.Spaceship(SPACESHIP_POS[0],SPACESHIP_POS[1],96570,0,0,92670,340*1000,2960,0,np.array([0, 1], dtype=float))


Simulation = phys.Simulation()
Simulation.list_of_objects["Earth"] = Earth
Simulation.list_of_objects["Sun"] = Sun
Simulation.list_of_objects["Spaceship"] = Spaceship

display = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
display.fill((0, 0, 0))

space_factor = 150*(10**7) #Since SI unit of distance is m we would have planets really far away in space, not visible on screen

def sim_position_to_screen_position(body_pos_vector):
    return (body_pos_vector[0]//space_factor+DISPLAY_WIDTH//3, body_pos_vector[1]//space_factor+DISPLAY_HEIGHT//3)

def draw_body(display, body, color, r):
    screen_pos = sim_position_to_screen_position(body.position)
    pygame.draw.circle(display, color,
                       center=(screen_pos[0], screen_pos[1]), radius=r, width=0)
    #print(str(body.position[0]//space_factor+DISPLAY_WIDTH//2)+", "+str(body.position[1]//space_factor+DISPLAY_HEIGHT//2))

def draw_spaceship(display, ship):
    pos = ship.position
    SPACESHIP_SIZE = 5*space_factor
    #Draw a triangle representing the ship, a "spear point" pointing in the ship's current orientation
    normalized = normalize_vector(ship.direction)
    point_1 = sim_position_to_screen_position((normalized*3*SPACESHIP_SIZE)+pos) #The tip
    point_2 = sim_position_to_screen_position(np.array([normalized[1],-1*normalized[0]], dtype=float)*SPACESHIP_SIZE+pos)
    point_3 = sim_position_to_screen_position(np.array([-1*normalized[1], normalized[0]], dtype=float)*SPACESHIP_SIZE+pos)

    pygame.draw.line(display, (255, 0, 0),
                     point_1, point_2, width=2)
    pygame.draw.line(display, (255, 0, 0),
                     point_1, point_3, width=2)
    pygame.draw.line(display, (255, 255, 255),
                     point_2, point_3, width=2)


Simulation.multiplier=365


rotation_angle=30

while True:
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                Spaceship.rotate_anticlockwise(rotation_angle)
                print("Rotating")
            if event.key == pygame.K_RIGHT:
                Spaceship.rotate_anticlockwise(360-rotation_angle)
                print("Rotating")
            if event.key == pygame.K_UP:
                Spaceship.current_flow_rate=Spaceship.max_flow_rate
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    display.fill((0, 0, 0))
    Simulation.update()
    print(Spaceship.fuel_mass)
    for key, obj in Simulation.list_of_objects.items():
        if key == "Earth":
            #print("Drawing Earth")
            draw_body(display,obj, pygame.Color('forestgreen'), 10)
            #draw_body(display, planet, (125,0,125), 50)
        if key == "Sun":
            #print("Drawing Sun")
            draw_body(display, obj, pygame.Color('goldenrod1'), 10)
            #draw_body(display, planet, (125,0,125), 100)
        if key == "Spaceship":
            #print("Drawing Spaceship")
            draw_spaceship(display, obj)
    pygame.display.update()
