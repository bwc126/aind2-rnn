import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
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
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size,1)))
    model.add(Dense(1))
    return model



### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):

    punctuation = ['!', ',', '.', ':', ';', '?']
    def alpha_pun(letter):
        return str.isalpha or letter in punctuation
    # words = text.split(" ")
    words = [str().join(filter(alpha_pun, word)) for word in text]
    words = [word for word in words if word]
    text = words.join()
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    return inputs,outputs

# TODO build the required RNN model:
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss
def build_part2_RNN(window_size, num_chars):
    pass
