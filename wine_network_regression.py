
import glob, os
import numpy as np
from random import shuffle
import pandas as pd
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, AveragePooling2D, LeakyReLU
from keras.optimizers import SGD, Adam
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf

def read_data(num_vectors):
    """
    Reads in images of dogs and cats from a fixed directory containing
    approximately 25,000 images.  Each image has been resized and padded
    so that it's shape is (128, 128, 3).  Each image filename indicates
    whether it is an image of a dog or a cat.
    Do not copy these images to your own directory! Just read them from
    the given directory.
    num_patterns: Allows you to limit the number of patterns to include
    in the data set.  It's recommended that you use a subset of the data for
    testing, particularly early on, so your runs complete faster.
    returns: a list of input vectors and a list of target vectors
    """
    data = pd.read_csv("Global_Wine_Points_Formatted.csv", engine='python')

    data.dropna()

    indices = data['Province']

    indices = pd.DataFrame(indices)

    price_pre = np.array(data['IntPrice'])
    price_pre = price_pre.astype(np.float)
    min_price = np.min(price_pre)
    mean_price = np.mean(price_pre)
    max_price = np.max(price_pre)
    prices_norm = [(price - min_price)/(max_price-min_price) for price in price_pre]

    vintage_pre = np.array(data['IntVintage'])
    vintage_pre = vintage_pre.astype(np.float)
    min_vintage = np.min(vintage_pre)
    mean_vintage = np.mean(vintage_pre)
    max_vintage = np.max(vintage_pre)
    vintages_norm = [(vintage - min_vintage)/(max_vintage-min_vintage) for vintage in vintage_pre]

    one_hot_encoder = OneHotEncoder(sparse=False)
    one_hot_encoder.fit(indices)

    provinces_encoded = one_hot_encoder.transform(indices)


    data_vec = np.zeros((len(prices_norm), 2 + len(provinces_encoded[0])))

    for i in range(len(prices_norm)):
        data_vec[i][0] = prices_norm[i]
        data_vec[i][1] = vintages_norm[i]
        for j in range(len(provinces_encoded[0])):
            data_vec[i][2+j] = provinces_encoded[i][j]

    input_vectors = data_vec

    target_vectors = np.array(data['Points'])
    target_vectors = target_vectors.astype('float')
    min_points = 50
    max_points = np.max(target_vectors)
    targets_norm = [(target - min_points)/(max_points-min_points) for target in target_vectors]
    targets_norm = np.array(targets_norm)
    targets_norm = targets_norm.astype('float')

    s = np.arange(input_vectors.shape[0])
    np.random.shuffle(s)

    input_vectors = input_vectors[s]
    targets_norm = targets_norm[s]

    # Verify the shapes of both the inputs and targets
    print("input_vectors shape", input_vectors.shape)
    print("target_vectors shape", targets_norm.shape)

    return input_vectors[:num_vectors], targets_norm[:num_vectors]

def test_results(num_vectors, x_test, y_test):
    """
    After training, use this function to look at how test images are being
    classified.
    num_patterns: allows you to limit the number of patterns you show
    x_test: validation set of inputs
    y_test: validation set of targets
    returns: Nothing
    """
    answers = neural_net.predict(x_test)
    targets = y_test

    counter = 0
    for i in range(len(answers)):
        print("Network predicted", answers[i], "Target is", targets[i])
        error = 100*abs(answers[i]-targets[i])/targets[i]
        print(error)

        counter += 1
        if counter > num_vectors:
            break

# Read in the data, start with a small amount of data while you debug
num_vectors = 24000
inp_vecs, tar_vecs = read_data(num_vectors)
index = num_vectors*0.8
# TODO: Divide the data into training and testing sets
inp_train = inp_vecs[:int(index)]
tar_train_vectors = tar_vecs[:int(index)]

inp_test = inp_vecs[int(index+1):]
tar_test_vectors = tar_vecs[int(index+1):]



# TODO: Construct the model
neural_net = Sequential()
neural_net.add(Dense(256, input_dim=(318),activation='relu'))
neural_net.add(Dropout(0.15))
neural_net.add(Dense(128,activation='relu'))
neural_net.add(Dropout(0.25))
neural_net.add(Dense(64,activation='relu'))
neural_net.add(Dense(1, activation='linear'))
neural_net.summary()

# Compile the model
opt = Adam(learning_rate=0.00001)
neural_net.compile(optimizer=opt, loss="mean_squared_error",
                   metrics=['mean_absolute_percentage_error'])

history = neural_net.fit(inp_train, tar_train_vectors,verbose=1,
                         validation_data=(inp_test, tar_test_vectors),
                         epochs=300)
test_results(30, inp_test, tar_test_vectors)

loss, accuracy = neural_net.evaluate(inp_test, tar_test_vectors, verbose=1)
