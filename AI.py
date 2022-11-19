import scipy.interpolate as interp
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import SGD

import phys
import data_laoder
import tools
import pandas as pd
import random
import datetime
import tensorflow as tf
from tensorflow import keras
from keras import layers


# Reference for Q values:
# Distance from Sun to Jupiter: 740.97 million km, so if it's that or more just return -1
# Measure the default max distance if we didn't do anything, make that return 0
# The closer the distance was to 0 the closer the return value to 1
# Tanh from -4 to 4 has value -0.99 to 0.99
# Or we can use some interpolating polynomial maybe to get the values for below 740


# Data to provide: positions of astronomical objects, position of spacecraft (might be included in the previous already)
#                  velocities of both, current fuel left, current fuel flow, current timestamp


X_values = np.flip(np.array(
    [650000000000, 400000000000, 300000000000, 200000000000, 120708690477, 80075144865, 45075144865, 20075144865,
     10000000000, 384400000, 0]))
Y_values = np.flip(np.array([-1, -0.8, -0.5, -0.25, 0, 0.1, 0.25, 0.5, 0.7, 0.8, 1]))
EARTH_DISTANCE_FROM_SUN = 148.1 * (10 ** 9)  # m
EARTH_VELOCITY = 29.72 * 10 ** 3  # m/s

EARTH_POS = np.array(tools.normalize_vector(np.array([1, 1])), dtype=float) * EARTH_DISTANCE_FROM_SUN
SPACESHIP_POS = np.array(tools.normalize_vector(np.array([2, 1])), dtype=float) * EARTH_DISTANCE_FROM_SUN

ROTATION_ANGLE = 15  # Has to be the same as in main
SAMPLING_INTERVAL = 10  # We take measurements and make decisions every SAMPLING_INTERVAL rounds
spline = interp.UnivariateSpline(X_values, Y_values, s=0.5)


# 60423640 seems like enough time (Simulation.current time)
# 3333.3333333333326 as time multiplier seems ok
# If we end up having perf problems, let's query the AI every 10th update?

def score_from_distance(distance):
    value = spline(distance)
    if (value < -1):
        return -1
    elif (value > 1):
        return 1
    return value


class MyModel:
    loaded_model = None
    def load_model(self):
        self.loaded_model = keras.models.load_model('my_model')
    def save(self):
        self.loaded_model.save("my_model")
    def create_model(self):
        inputs = layers.Input(shape=(47,1))
        layer1 = layers.Dense(512, activation="relu")(inputs)
        layer2 = layers.Dense(256, activation="relu")(layer1)
        layer3 = layers.Dense(128, activation="relu")(layer2)
        output = layers.Dense(1,activation="tanh")(layer3)
        return keras.Model(inputs=inputs, outputs=output)
    def train_model(self,file,lr=0.1, batch_size=128):
        dataframe = pd.read_csv(file)
        #number_of_rows = dataframe.shape[0]
        #y = np.zeros((number_of_rows,))
        #x = np.zeros((number_of_rows,47))
        x = dataframe.iloc[:,1:48].to_numpy()
        y = dataframe.iloc[:,49].to_numpy()

        opt = SGD(lr=lr)
        self.loaded_model.compile(loss='mse', optimizer=opt)

        self.loaded_model.fit(
            tf.expand_dims(x,axis=-1), y,
            batch_size=batch_size,
            epochs=1)

    # TODO: ValueError: Input 0 of layer "dense" is incompatible with the layer: expected min_ndim=2, found ndim=1. Full shape received: (47,)
    #       https://github.com/mrdbourke/tensorflow-deep-learning/discussions/278
    def approximate_reward(self, data, action):
        return self.loaded_model.call(tf.expand_dims(tf.convert_to_tensor((np.append(data, [action]))),axis=1)).numpy()[0]
        #return random.uniform(-1, 1)

# Takes AI data from the physics simulation as input and performs some normalization on it
#TODO: Normalize data, https://stackoverflow.com/questions/61710791/should-i-use-tf-keras-utils-normalize-to-normalize-my-targets
#                       https://keras.io/api/layers/preprocessing_layers/numerical/normalization/
    # https://machinelearningmastery.com/using-normalization-layers-to-improve-deep-learning-models/
    # https://stackoverflow.com/questions/46771939/batch-normalization-instead-of-input-normalization
    # https://keras.io/api/layers/normalization_layers/layer_normalization/
    # https://www.pinecone.io/learn/batch-layer-normalization/
    # Generally layer normalization seems more fitting here imho
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization?hl=en
    # https://www.youtube.com/watch?v=AFzmpEAMNp4
def normalize_data(AI_data):
    pass


# CSV data structure:
# index, AI_data (from simulation), action, round, result = 50 columns
# TODO: Measure inference time over a single simulation to check if a
#  bigger network might actually be cheap and/or helpful
def simulate(rounds, epsilon):
    dataframe_created = False
    dataframe = None
    int_column_names = [x for x in range(50)]
    string_column_names = [str(x) for x in int_column_names]
    for current_round in range(rounds):
        model = MyModel()
        model.load_model()
        Simulation = phys.Simulation()
        Simulation.multiplier = 3333.3333333333326
        index = 0
        # The spaceship
        Spaceship = phys.Spaceship(SPACESHIP_POS[0], SPACESHIP_POS[1], 96570, 0, 0, 92670, 340 * 1000, 2960, 0,
                                   np.array([0, 1], dtype=float))
        Spaceship.set_velocity(np.array([0, 1], dtype=float), EARTH_VELOCITY)
        # Planetary bodies
        Sun = phys.MyPhysObject(0, 0, 1.9891 * (10 ** 30), 0, 0)
        Simulation.list_of_objects = data_laoder.load_data()
        Simulation.list_of_objects["Sun"] = Sun
        Simulation.list_of_objects["Spaceship"] = Spaceship
        frame_number = 0
        right_side = np.array([0, 0, 0], dtype=float)  # action,round,result
        data = Simulation.get_AI_data()
        min_distance_from_mars_to_spaceship = Simulation.get_distance_from_mars_to_spaceship()
        choice_dictionary = {}
        while Simulation.current_time <= 60423640:
            Simulation.update()
            distance_to_mars = Simulation.get_distance_from_mars_to_spaceship()
            if (distance_to_mars < min_distance_from_mars_to_spaceship):
                min_distance_from_mars_to_spaceship = distance_to_mars
            if (frame_number % SAMPLING_INTERVAL == 0):
                # Getting the decision
                data = Simulation.get_AI_data()

                random_value = random.uniform(0, 1)
                if(random_value<epsilon):
                    choice = random.randint(0,4)
                else:
                    for i in range(5):
                        a = model.approximate_reward(data, i)# i is the action, a is the reword
                        choice_dictionary[i] = a
                    choice = max(choice_dictionary, key=choice_dictionary.get) # choose the best action according to approx. reward

                # If choice==0 then we wait
                if choice == 1:
                    Spaceship.current_flow_rate = min(Spaceship.max_flow_rate, Spaceship.current_flow_rate + 10)
                elif choice == 2:
                    Spaceship.current_flow_rate = max(0, Spaceship.current_flow_rate - 10)
                elif choice == 3:
                    Spaceship.rotate_anticlockwise(ROTATION_ANGLE)
                elif choice == 4:
                    Spaceship.rotate_anticlockwise(360 - ROTATION_ANGLE)
                # Adding data to the dataset
                data = np.append(index, data)
                right_side[0] = choice
                right_side[1] = current_round
                data = np.append(data, right_side)
                if (not dataframe_created):
                    dataframe_created = True
                    dataframe = pd.DataFrame([data], columns=string_column_names)
                else:
                    dataframe = pd.concat([dataframe, pd.DataFrame([data], columns=string_column_names)])# TODO: Check performance implications
                index += 1
            frame_number += 1
        query = "@dataframe['48']=="+str(current_round)
        final_score = score_from_distance(min_distance_from_mars_to_spaceship)
        dataframe.loc[dataframe.eval(query), '49'] = final_score
        print("Finished simulation round: ", current_round, ", final score: ", final_score)
    filename = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))+".csv"
    # print("Dataframe memory usage in bytes: ",dataframe.memory_usage(index=True).sum())
    # print(dataframe.columns)
    dataframe.to_csv(filename)
    return filename

def train():
    model = MyModel()
    model.load_model()
    model.train_model("20221119-171116.csv")
    model.save()

# TODO: print the current iteration and epsilon
# TODO: Make the epsilon (0.5 in this case) a parameter
# TODO: save intermediate models, don't overwrite (or make it an option to save backups)
# TODO: Loss started to become NaN after ~80 rounds, check out this: https://stackoverflow.com/questions/40050397/deep-learning-nan-loss-reasons
#   assert not np.any(np.isnan(x)) on input data before running it maybe
def simulate_and_train(rounds):
    model = MyModel()
    model.load_model()
    for i in range(rounds):
        filename = simulate(5,0.5-(i/rounds)*0.5)
        model.train_model(filename)
        model.save()


def visualize_spline():
    x = np.linspace(0, 800000000000, 1000000)
    y = spline(x)
    plt.figure(figsize=(30, 24))
    plt.plot(x, y, 'ro')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    for x in np.nditer(X_values):
        print("Value at ", x, ": ", score_from_distance(x))

#model = MyModel().create_model().save("my_model")
#train()
#simulate(2, 0.5)
simulate_and_train(100)