import asyncio

import numpy as np
import pygame
import sys
import phys
import data_laoder
import tools
import datetime
import math

pygame.init()
clock = pygame.time.Clock()

DISPLAY_WIDTH = 1920
DISPLAY_HEIGHT = 1080

EARTH_DISTANCE_FROM_SUN = 148.1 * (10 ** 9)  # m
EARTH_VELOCITY = 29.72 * 10 ** 3  # m/s

EARTH_POS = np.array(tools.normalize_vector(np.array([1, 1])), dtype=float) * EARTH_DISTANCE_FROM_SUN
SPACESHIP_POS = np.array(tools.normalize_vector(np.array([2, 1])), dtype=float) * EARTH_DISTANCE_FROM_SUN

FRAMERATE = 30

Screen_pos = np.array([0, 0])
SCREEN_MOVE_FACTOR = 100  # How much the screen moves with each keypress

font = pygame.font.SysFont(None, 24)

Simulation = phys.Simulation()
display = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
display.fill((0, 0, 0))

space_factor = 150 * (
        10 ** 7)  # Since SI unit of distance is m we would have planets really far away in space, not visible on screen

# The spaceship
Spaceship = phys.Spaceship(SPACESHIP_POS[0], SPACESHIP_POS[1], 96570, 0, 0, 92670, 340 * 1000, 2960, 0,
                           np.array([0, 1], dtype=float))
Spaceship.set_velocity(np.array([0, 1], dtype=float), EARTH_VELOCITY)
# Planetary bodies
Sun = phys.MyPhysObject(0, 0, 1.9891 * (10 ** 30), 0, 0)
Simulation.list_of_objects = data_laoder.load_data()
Simulation.list_of_objects["Sun"] = Sun
Simulation.list_of_objects["Spaceship"] = Spaceship


def sim_position_to_screen_position(body_pos_vector):
    return (body_pos_vector[0] // space_factor + DISPLAY_WIDTH // 3 + Screen_pos[0],
            body_pos_vector[1] // space_factor + DISPLAY_HEIGHT // 3 + Screen_pos[1])


def draw_body(display, body, color, r):
    screen_pos = sim_position_to_screen_position(body.position)
    pygame.draw.circle(display, color,
                       center=(screen_pos[0], screen_pos[1]), radius=r, width=0)


def draw_spaceship(display, ship):
    pos = ship.position
    SPACESHIP_SIZE = 5 * space_factor
    # Draw a triangle representing the ship, a "spear point" pointing in the ship's current orientation
    normalized = tools.normalize_vector(ship.direction)
    point_1 = sim_position_to_screen_position((normalized * 3 * SPACESHIP_SIZE) + pos)  # The tip
    point_2 = sim_position_to_screen_position(
        np.array([normalized[1], -1 * normalized[0]], dtype=float) * SPACESHIP_SIZE + pos)
    point_3 = sim_position_to_screen_position(
        np.array([-1 * normalized[1], normalized[0]], dtype=float) * SPACESHIP_SIZE + pos)

    pygame.draw.line(display, (255, 0, 0),
                     point_1, point_2, width=2)
    pygame.draw.line(display, (255, 0, 0),
                     point_1, point_3, width=2)
    pygame.draw.line(display, (255, 255, 255),
                     point_2, point_3, width=2)


def draw_HUD(fuel_left, current_throttle, current_time_multiplier):
    UI_string_1 = "Fuel left: " + str(fuel_left)
    UI_string_2 = "Throttle: " + str(current_throttle)
    UI_string_3 = "Time multiplier: " + str(current_time_multiplier)
    img1 = font.render(UI_string_1, True, 'aliceblue')
    img2 = font.render(UI_string_2, True, 'aliceblue')
    img3 = font.render(UI_string_3, True, 'aliceblue')
    display.blit(img1, (20, 20))
    display.blit(img2, (20, 60))
    display.blit(img3, (20, 100))


# 1/FRAMERATE should make it 1 second in simulation = 1 real second
Simulation.multiplier = (1 / FRAMERATE)

clock.tick(FRAMERATE)
ROTATION_ANGLE = 15
current_time = 0


async def main():
    global Simulation
    global Spaceship
    global Screen_pos
    global current_time
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    Spaceship.rotate_anticlockwise(ROTATION_ANGLE)
                elif event.key == pygame.K_RIGHT:
                    Spaceship.rotate_anticlockwise(360 - ROTATION_ANGLE)
                elif event.key == pygame.K_UP:
                    Spaceship.current_flow_rate = min(Spaceship.max_flow_rate, Spaceship.current_flow_rate + 10)
                elif event.key == pygame.K_DOWN:
                    Spaceship.current_flow_rate = max(0, Spaceship.current_flow_rate - 10)
                elif event.key == pygame.K_KP_8:
                    Screen_pos[1] += SCREEN_MOVE_FACTOR
                elif event.key == pygame.K_KP_2:
                    Screen_pos[1] -= SCREEN_MOVE_FACTOR
                elif event.key == pygame.K_KP_4:
                    Screen_pos[0] += SCREEN_MOVE_FACTOR
                elif event.key == pygame.K_KP_6:
                    Screen_pos[0] -= SCREEN_MOVE_FACTOR
                elif event.key == pygame.K_KP_PLUS:
                    space_factor *= 0.1
                elif event.key == pygame.K_KP_MINUS:
                    space_factor *= 10
                elif event.key == pygame.K_z:
                    Simulation.multiplier *= 0.1
                elif event.key == pygame.K_c:
                    Simulation.multiplier *= 10
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        display.fill((0, 0, 0))
        Simulation.update()
        current_time += Simulation.multiplier
        td = datetime.timedelta(seconds=current_time)
        print("Current time in seconds: ", current_time)
        print("Current time in hh:mm:ss", td)
        print("Distance from earth to the sun: ", Simulation.get_distance_from_earth_to_sun())
        draw_HUD(math.floor(Spaceship.fuel_mass), math.floor(Spaceship.current_flow_rate),
                 round(Simulation.multiplier * FRAMERATE))
        for key, obj in Simulation.list_of_objects.items():
            if key == "Earth":
                draw_body(display, obj, pygame.Color('forestgreen'), 10)
            if key == "Sun":
                draw_body(display, obj, pygame.Color('goldenrod1'), 10)
            if key == "Mercury":
                draw_body(display, obj, pygame.Color('darkgrey'), 10)
            if key == "Venus":
                draw_body(display, obj, pygame.Color('saddlebrown'), 10)
            if key == "Mars":
                draw_body(display, obj, pygame.Color('firebrick3'), 10)
            if key == "Jupiter":
                draw_body(display, obj, pygame.Color('orange1'), 10)
            if key == "Saturn":
                draw_body(display, obj, pygame.Color('palegoldenrod'), 10)
            if key == "Uranus":
                draw_body(display, obj, pygame.Color('paleturquoise'), 10)
            if key == "Neptune":
                draw_body(display, obj, pygame.Color('royalblue'), 10)
            if key == "Spaceship":
                draw_spaceship(display, obj)
        pygame.display.update()
        await asyncio.sleep(0)


asyncio.run(main())