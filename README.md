# Planets

## A simple planets/space simulation (game?).

Attempted to add an AI that tries to get as close as possible to Mars.

## Things I learned:

- MSE worked much better than MAE in this problem
- Normalizing the data helped a lot, batch normalization didn't work too well.
- Adding intermediate rewards by punishing the AI for rotating more than 60 deg. away from Mars helped due to (slightly) sparse rewards. Tweaking the discount factor would probably help a lot here as well, though.
- Graphing the scores with a moving average helps greatly, especially when the scores are very unstable.

## Issues:

- When adding a limit to the number of rotations the AI can perform (to prevent it from just waiting all the time due to using thrusters usually providing a high negative reward until it grasps how to face the target) I thought I overwrote the data of Neptune - no big deal since it wasn't too relevant for the task. It turned out to overwrite the spaceship's velocity vector component y, possibly causing the massively decreased performance that can be seen right now. Better than random play, but far, far from ideal.

- Rather than saving the training to dataframes (in CSV form on the HDD!) each round it would probably be better to save them to numpy and pass around directly. Or pass a dataframe between functions and only save to HDD once in a while.

- I didn't feel like naming the columns of the created dataframes sensibly because there were ~50 of them. Big mistake - lots of issues later on due to providing the AI with the wrong data series and difficulties with finding which data is where.
