import numpy as np
from collections import Counter
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
import keras, math


# TODO: fill out the function below that transforms the input series
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    # We have a number of windows based on the input length and window size, so we index the input series based on how many windows we have and their size. We add each window's input and output to their respective lists.
    for window in range(len(series)-window_size):
        begin = window
        end = begin + window_size
        X.append(series[begin:end])
        y.append(series[end])

    # reshape each
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    # We're using the basic keras network.
    model = Sequential()
    # We add our long short term memory layer, with 5 nodes and input shape appropriate for the input window size.
    model.add(LSTM(5, input_shape=(window_size,1)))
    # Our final layer is a fully connected one.
    model.add(Dense(1))
    return model



### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    punctuation = ['!', ',', '.', ':', ';', '?']
    text = text.replace('_', ' ')
    # Enumerate all unique characters in the text
    chars = Counter(text)
    for char in chars:
        # If a character isn't an english letter or in the punctuation list, replace it with a space
        if char not in letters:
            if char not in punctuation:
                text = text.replace(char, ' ')
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    # We have a number of windows equal to the ceiling of the size of the input series minus the window size divided by the step size.
    for window in range(int(math.ceil((len(text)-window_size)/step_size))):
        # Each window should begin a number of steps away from the beginning of the previous window, equal to step_size.
        begin = window * step_size
        # The window should end a number of steps from where it began equal to window_size.
        end = begin + window_size
        inputs.append(text[begin:end])
        # We select a final character after the input window to serve as our target output.
        outputs.append(text[end])

    return inputs,outputs

# TODO build the required RNN model:
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss
def build_part2_RNN(window_size, num_chars):
    # Like for the other RNN, we use Keras' basic model.
    model = Sequential()
    # This time, our LSTM will have 200 nodes.
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    # We have a fully connected linear layer, with a final softmax layer to help us predict the output categories (letters).
    model.add(Dense(num_chars, activation='linear'))
    model.add(Activation('softmax'))
    return model
