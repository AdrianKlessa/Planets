# Planets

## A simple planets/space simulation (game?) and my first attempt at creating a Q-Learning agent.

Planets/Sun in the UI not to scale.

# The simulation can be run in-browser [here](https://adrianklessa.github.io/Planets/). 

## Controls on the bottom of the readme. Left-click on the screen when "ready to start" is displayed.

Attempted to add an AI that tries to get as close as possible to Mars. The agent is a small neural network made with Keras.

As of 2022.12.1 the AI scoring function was based on the distance from target, with score=1 for distance of 0 meters and decreasing from there. Fully random play resulted in average score of -0.3, the current version of the AI averages at around 0.5. 

## Things I learned:

- MSE worked much better than MAE in this problem
- Normalizing the data helped a lot, batch normalization didn't work too well (possibly due to differences between batch averages/variations as single-batch data was correlated).
- Adding intermediate rewards by punishing the AI for rotating more than 60 deg. away from Mars helped due to (slightly) sparser rewards. Tweaking the discount factor would probably help a lot here as well, though.
- Graphing the scores with a moving average helps greatly, especially when the scores are very unstable.

## Issues:

- When adding a limit to the number of rotations the AI can perform (to prevent it from just waiting all the time due to using thrusters usually providing a high negative reward until it grasps how to face the target) I thought I overwrote the data of Neptune - no big deal since it wasn't too relevant for the task. It turned out to overwrite the spaceship's velocity vector component y, possibly causing the massively decreased performance that can be seen right now. Better than random play, but far, far from ideal.

- Rather than saving the training to dataframes (in CSV form on the HDD!) each round it would probably be better to save them to numpy and pass around directly. Or pass a dataframe between functions and only save to HDD once in a while.

- I didn't feel like naming the columns of the created dataframes sensibly because there were ~50 of them. Big mistake - lots of issues later on due to providing the AI with the wrong data series and difficulties with finding which data is where.

## TL;DR : A physics simulation of the solar system turned out well, I'd say. There is a deep Q-learning agent trained based on it, but it's far from perfect - would require a refactor of the code providing it with data and retraining with other/less paramters to improve.

## Libraries used:

- Numpy
- Pandas
- Keras + a bit of direct operations on Tensorflow
- Scipy for constants and interpolation
- Pathos (running the simulation in multiple threads for faster data gathering)

# Controls for the spaceship (the red triangle):

Left/Right arrows - rotate

Up/Down arrows - increase/decrease thrust

Numpad arrows (2/4/6/8) - move the camera

Z/C - accelerate/deccelerate time (C - 10x faster, Z - 10x slower). 

Make sure to use throttle at lower time multipliers to not run out of fuel super fast and zip out of the solar system.

Very high time multipliers (10^8 and above) suffer from numerical instability. 

(Since the velocity is simply multiplied by the time multiplier on each loop - this could be fixed by changing the "speedup" logic to actually do more calculations per frame on higher time multipliers)


## Planetary dataset found on [Kaggle](https://www.kaggle.com/datasets/iamsouravbanerjee/planet-dataset) based on NASA-provided information. 

(Obligatory note: I'm not affiliated with either the creator of the dataset or NASA).
