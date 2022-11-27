import scipy.interpolate as interp
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers

import phys
import data_laoder
import tools
import pandas as pd
import random
import datetime
import tensorflow as tf
from tensorflow import keras
from keras import layers
import time
import os
import threading
from os.path import join
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
SAMPLING_INTERVAL = 5  # We take measurements and make decisions every SAMPLING_INTERVAL rounds
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


# TODO: One-hot encoding the actions
# TODO: Add an option to append new training data to an existing "training_file.csv" and train on that
class MyModel:
    loaded_model = None

    def load_model(self):
        self.loaded_model = keras.models.load_model('my_model')

    def save(self):
        self.loaded_model.save("my_model")

    def create_model(self):
        inputs = layers.Input(shape=(47,))
        layer_normalization = layers.BatchNormalization()(inputs)
        layer1 = layers.Dense(512, activation="tanh")(layer_normalization)
        layer2 = layers.Dense(256, activation="tanh")(layer1)
        layer3 = layers.Dense(256, activation="tanh")(layer2)
        layer4 = layers.Dense(128, activation="tanh")(layer3)
        layer5 = layers.Dense(64, activation="tanh")(layer4)
        output = layers.Dense(1, activation="tanh")(layer5)
        return keras.Model(inputs=inputs, outputs=output)

    def train_model(self, file, lr=0.00005, batch_size=512, epochs=300, stop_early=True, visualize_only=False):
        dataframe = pd.read_csv(file)
        # number_of_rows = dataframe.shape[0]
        # y = np.zeros((number_of_rows,))
        # x = np.zeros((number_of_rows,47))
        x = dataframe.iloc[:, 2:49].to_numpy()
        y = dataframe.iloc[:, 50].to_numpy()
        print("Input: ")
        print(x)
        print("Target: ")
        print(y)
        callbacks = []
        if stop_early:
            callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True))
        opt = optimizers.Adam(lr=lr)
        self.loaded_model.compile(loss=keras.losses.mean_absolute_error, optimizer=opt)
        print("Shape of input: ",tf.convert_to_tensor(x).get_shape())
        print("Shape of target: ",tf.convert_to_tensor(y).get_shape())
        self.loaded_model.summary()
        if not visualize_only:
            self.loaded_model.fit(
                x, y,
                batch_size=batch_size,
                epochs=epochs, shuffle=True, callbacks=[callbacks])

    def approximate_reward(self, data, action):
        return \
            self.loaded_model.call(tf.expand_dims(tf.convert_to_tensor((np.append(data, [action]))), axis=0)).numpy()[0][0]
        # return random.uniform(-1, 1)


# Takes AI data from the physics simulation as input and performs some normalization on it
# TODO: Normalize data, https://stackoverflow.com/questions/61710791/should-i-use-tf-keras-utils-normalize-to-normalize-my-targets
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

#TODO: Switch this from interactive to creating plot images in a new directory
class ScorePlot:
    scores = []
    figure, ax = plt.subplots(figsize=(10, 8))
    line1, = ax.plot([0, 1, 2])

    def __init__(self):

        plt.ylim(-1, 1)

    def draw_plot(self):
        plt.figure()
        x_data = [x for x in range(len(self.scores))]
        y_data = self.scores
        plt.title("AI average scores over time", fontsize=20)
        plt.xlabel('Round number')
        plt.ylabel('Score')
        plt.xlim(0, max(x_data) + 1)
        plt.ylim(-1, 1)
        plt.plot(x_data,y_data, marker='o')
        filename = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) + ".jpg"
        plt.savefig("Plots/"+filename)
    def draw_plot_old(self):
        x_data = [x for x in range(len(self.scores))]
        self.line1.set_xdata(x_data)
        self.line1.set_ydata(self.scores)
        plt.xlim(0, max(x_data) + 1)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        plt.show()


my_plot = ScorePlot()


def aver(lst):
    return sum(lst) / len(lst)


# CSV data structure:
# index, AI_data (from simulation), action, round, result, lambda = 51 columns
# TODO: Measure inference time over a single simulation to check if a
#  bigger network might actually be cheap and/or helpful
# TODO: With this multiplier and doing stuff every 10 frames the AI doesn't have time to do anything after changing the throttle,
#  #in the next move it's already out of fuel; therefore changed fuel flow change to +/- 1
# TODO: Maybe I shouldn't treat waiting like all other actions? Maybe I should train it mostly on the other ones? But then it'll just rotate around....
# TODO: Make it have to make a move, not wait. It can rotate but has a limited number of rotations that are available. Give it how many rotations it has left.
# TODO: Print max predicted Q at each step and verify that the formula for updating the dataframe at the end works

def simulate(rounds, epsilon, discount_factor, thread_filename="-1"):
    print("Rounds: ",rounds," thread_filename: ",thread_filename)
    dataframe_created = False
    dataframe = None
    int_column_names = [x for x in range(51)]
    string_column_names = [str(x) for x in int_column_names]
    scores = []
    model = MyModel()
    model.load_model()
    previous_dataframe_index=-1
    for current_round in range(rounds):
        first_choice = True
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
        right_side = np.array([0, 0, 0, 0], dtype=float)  # action,round,result, discount factor
        data = Simulation.get_AI_data()
        min_distance_from_mars_to_spaceship = Simulation.get_distance_from_mars_to_spaceship()
        choice_dictionary = {}
        time_appending = 0.0
        time_infering = 0.0
        while Simulation.current_time <= 60423640:
            Simulation.update()
            distance_to_mars = Simulation.get_distance_from_mars_to_spaceship()
            if (distance_to_mars < min_distance_from_mars_to_spaceship):
                min_distance_from_mars_to_spaceship = distance_to_mars
            if (frame_number % SAMPLING_INTERVAL == 0 and Spaceship.fuel_mass > 0):
                # Getting the decision
                data = Simulation.get_AI_data()
                random_value = random.uniform(0, 1)
                for i in range(1, 5):
                    start_time = time.time()
                    a = model.approximate_reward(data, i)  # i is the action, a is the reward
                    end_time = time.time()
                    time_infering += (end_time - start_time)
                    choice_dictionary[i] = a
                choice = max(choice_dictionary,
                             key=choice_dictionary.get)  # choose the best action according to approx. reward
                predicted_max_Q = choice_dictionary[choice]
                print("Max predicted Q value for round ",previous_dataframe_index+1,": ",predicted_max_Q)
                if (random_value < epsilon):
                    print("Made a random move instead")
                    choice = random.randint(1, 4)
                rotations_left = Spaceship.rotations_left
                # If choice==0 then we wait
                if choice == 1 or rotations_left <= 0:
                    Spaceship.current_flow_rate = min(Spaceship.max_flow_rate, Spaceship.current_flow_rate + 1)
                elif choice == 2:
                    Spaceship.current_flow_rate = max(0, Spaceship.current_flow_rate - 1)
                    Spaceship.rotations_left -= 1
                elif choice == 3:
                    Spaceship.rotate_anticlockwise(ROTATION_ANGLE)
                    Spaceship.rotations_left -= 1
                elif choice == 4:
                    Spaceship.rotate_anticlockwise(360 - ROTATION_ANGLE)
                    Spaceship.rotations_left -= 1
                # Adding data to the dataset
                data = np.append(index, data)
                right_side[0] = choice
                right_side[1] = current_round
                data = np.append(data, right_side)
                if (not dataframe_created):
                    dataframe_created = True
                    dataframe = pd.DataFrame([data], columns=string_column_names)
                else:
                    start_time = time.time()
                    dataframe = pd.concat([dataframe, pd.DataFrame([data],
                                                                   columns=string_column_names)])  # TODO: Check performance implications
                    end_time = time.time()
                    time_appending += (end_time - start_time)
                if first_choice!=True:
                    dataframe.iloc[previous_dataframe_index]['49']=predicted_max_Q
                first_choice=False
                previous_dataframe_index+=1
                if (Simulation.get_distance_from_mars_to_spaceship() >= 500090000000.0):
                    Simulation.current_time = 70423640
                index += 1
            frame_number += 1
        final_index = index
        dataframe.iloc[-1]['49'] = 0
        query = "@dataframe['48']==" + str(current_round)
        final_score = score_from_distance(min_distance_from_mars_to_spaceship)
        scores.append(final_score)
        print(dataframe.loc[dataframe.eval(query), '50'].shape[0])
        dataframe.loc[dataframe.eval(query), '50'] = pow(discount_factor,
                                                         final_index - dataframe.loc[dataframe.eval(query), '0'] - 1)
        dataframe.loc[dataframe.eval(query), '49'] = final_score * dataframe.loc[dataframe.eval(query), '50'] #dataframe.loc[dataframe.eval(query), '49']*dataframe.loc[dataframe.eval(query), '50']*(2/3)+final_score*dataframe.loc[dataframe.eval(query), '50']*(2/3)   #final_score * dataframe.loc[dataframe.eval(query), '50']
        print("Finished simulation round: ", current_round + 1, " of ", rounds, ", final score: ", final_score,
              "; Epsilon was: ", epsilon, ", time spent appending DF (in  seconds): ", time_appending,
              "; total inference time this round (s): ", time_infering)
    if thread_filename=="-1":
        filename = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) + ".csv"
    else:
        filename=thread_filename
    print("Average score this training round was: ",aver(scores))
    my_plot.scores.append(aver(scores))
    # print("Dataframe memory usage in bytes: ",dataframe.memory_usage(index=True).sum())
    # print(dataframe.columns)
    dataframe.to_csv(filename)
    return filename

def multithread_simulation(threads, simulation_rounds_per_thread, epsilon, discount_factor):
    directory_name = "multithread_batches/"+str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(directory_name)
    thread_list = []
    start_time = time.time()
    for i in range(threads):
        filename = directory_name+"_"+str(i)
        # def simulate(rounds, epsilon, discount_factor, thread_filename="-1"):
        new_thread = threading.Thread(target=simulate,args=(simulation_rounds_per_thread,epsilon,discount_factor,filename))
        thread_list.append(new_thread)
        new_thread.start()
    for current_thread in thread_list:
        current_thread.join()
    end_time = time.time()
    print("Multithreaded simulation time in seconds: "+str(start_time-end_time))
    csv_files = [f for f in os.listdir(directory_name) if os.isfile(join(directory_name, f))]
    dataframe_read = False
    for csv_file in csv_files:

        if not dataframe_read:
            dataframe_read=True
            dataframe = pd.read_csv(csv_file)
        else:
            new_dataframe = pd.read_csv(csv_file)
            dataframe = pd.concat(dataframe,new_dataframe)
    return_filename = "multithreaded_batch"+str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    dataframe.to_csv(return_filename)
    return return_filename

def train(input_file, visualize_only=False):
    model = MyModel()
    model.load_model()
    model.train_model(input_file,visualize_only=visualize_only)
    model.save()


# TODO: save intermediate models, don't overwrite (or make it an option to save backups)
# TODO: Plot progress (maybe look online how to do it in a separate window in PyCharm) https://www.geeksforgeeks.org/how-to-update-a-plot-on-same-figure-during-the-loop/
def simulate_and_train(rounds, simulations_per_round, starting_epsilon, starting_learning_rate, draw_plot,
                       epsilon_decay, lr_decay, discount_factor, epochs_per_round, stop_early):
    model = MyModel()
    model.load_model()
    epsilon = starting_epsilon
    learning_rate = starting_learning_rate
    very_tiny_learning_rate = starting_learning_rate*0.001
    for i in range(rounds):
        if epsilon_decay:
            epsilon = starting_epsilon - (i / rounds) * starting_epsilon
        if lr_decay:
            learning_rate*=pow(0.5,i)
            #learning_rate = (starting_learning_rate - (
            #        i / rounds) * starting_learning_rate) + very_tiny_learning_rate  # 0 or less doesn't make any sense
        print("Starting training round ", i + 1, " of ", rounds, ", learning rate is: ", learning_rate)
        filename = simulate(simulations_per_round, epsilon, discount_factor)
        if (draw_plot == True):
            my_plot.draw_plot()
        model.train_model(filename, lr=learning_rate, epochs=epochs_per_round, stop_early=stop_early)
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

if __name__ == '__main__':
    multithread_simulation(threads=50, simulation_rounds_per_thread=2, epsilon=0.3, discount_factor=0.85)

#model = MyModel().create_model().save("my_model")
#train("training_file.csv", visualize_only=False)


#simulate_and_train(rounds=15, simulations_per_round=100, starting_epsilon=0.3, starting_learning_rate=0.001,
#                   draw_plot=True, epsilon_decay=True, lr_decay=True, discount_factor=0.85, epochs_per_round=1,
#                   stop_early=False)
#train("training_file.csv", visualize_only=False)
#simulate(6,0.3,0.9)
