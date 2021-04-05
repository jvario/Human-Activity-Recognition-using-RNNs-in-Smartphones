#   ------------------------------Human Activity Recognition using Deep Recurrent Neural Networks on Motion Capture Data------------------------------
#                    ----------har-smartphone/TrainTheModel----------
#                              Name: Giannis
#                              Surname: Variozidis
#                              Email: cs141065@uniwa.gr
#                              ID: cs141065
#   ---------------------------------------------------------------------------

# https://medium.com/datadriveninvestor/human-activity-recognition-using-cnn-lstm-104eea952daf
# https://medium.com/@curiousily/human-activity-recognition-using-lstms-on-android-tensorflow-

"""IMPORTS"""
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, TimeDistributed, Conv1D, MaxPooling1D, GRU, SimpleRNN
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential
from keras.utils import to_categorical
from pandas import read_csv
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import ModelCheckpoint

'''FUNCTION: load_file()
   DESCRIPTION: Reads values of csv
   RETURNS: Dataframe Values  '''


def load_file(filepath):
    dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values


'''FUNCTION: load_group()
   DESCRIPTION: Load a list of files
   RETURNS: 3D Numpy Array  '''


def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    loaded = np.dstack(loaded)

    return loaded


'''FUNCTION: load_dataset_group()
   DESCRIPTION: Load a dataset group, such as train or test
   RETURNS: X,y Dataset  '''


def load_dataset_group(group, prefix=''):
    filepath = prefix + group + '/Inertial Signals/'
    # load all 9 files as a single array
    filenames = list()
    # total acceleration
    filenames += ['total_acc_x_' + group + '.txt', 'total_acc_y_' + group + '.txt', 'total_acc_z_' + group + '.txt']
    # body acceleration
    filenames += ['body_acc_x_' + group + '.txt', 'body_acc_y_' + group + '.txt', 'body_acc_z_' + group + '.txt']
    # body gyroscope
    filenames += ['body_gyro_x_' + group + '.txt', 'body_gyro_y_' + group + '.txt', 'body_gyro_z_' + group + '.txt']
    # load input Data
    X = load_group(filenames, filepath)
    # load class output
    y = load_file(prefix + group + '/y_' + group + '.txt')
    return X, y


'''FUNCTION: load_TestDataset()
   RETURNS: Test x and y elements '''


def load_TrainDataset(prefix=''):
    # load all train
    trainX, trainy = load_dataset_group('train',
                                        prefix + 'E:/Documents/Thesis/har1/har-smartphone/Data/UCI HAR Dataset/')
    print(trainX.shape, trainy.shape)
    # zero-offset class values
    trainy = trainy - 1

    # one hot encode y
    trainy = to_categorical(trainy)

    print(trainX.shape, trainy.shape)

    return trainX, trainy


'''FUNCTION: evaluate_model()
   DESCRIPTION: Model architectures and train of them
   RETURNS: History of Models '''


def evaluate_model(trainX, trainy, ModelType):
    print(ModelType, " Training...")
    # define model
    verbose, epochs, batch_size = 1, 100, 64
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    # reshape Data into time steps of sub-sequences
    n_steps, n_length = 4, 32
    # trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_features))

    # Split train & val
    trainX, x_val, trainy, y_val = train_test_split(trainX, trainy, test_size=0.03)

    # define model architecture
    model = Sequential()
    if ModelType == 'LSTM':
        model.add(LSTM(200, input_shape=(n_timesteps, n_features)))
    elif ModelType == 'GRU':
        model.add(GRU(200, input_shape=(n_timesteps, n_features)))
    else:
        model.add(SimpleRNN(200, input_shape=(n_timesteps, n_features)))

    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Checkpoint
    filepath = "Saved Models/CheckPoints/" + str(ModelType) + "weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    history = model.fit(trainX, trainy, epochs=epochs, callbacks=callbacks_list, batch_size=batch_size, verbose=verbose,
                        validation_data=(x_val, y_val))
    # Saving our model
    model_name = 'Saved Models/' + str(ModelType) + '.h5'
    model.save(model_name)

    return history


'''FUNCTION: plot_accuracy()
   DESCRIPTION: Plotting training accuracy of model
   RETURNS: Plot of Training Accuracy  '''


def plot_accuracy(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'orange', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.show()
    plt.savefig('Results/Training/Train-Accuracy.png')


'''FUNCTION: plot_loss()
   DESCRIPTION: Plotting training loss of model
   RETURNS: Plot of Training loss  '''


def plot_loss(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'orange', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
    plt.savefig('Results/Training/Train-Loss.png')


if __name__ == "__main__":
    # load Data
    trainX, trainy = load_TrainDataset()
    # repeat experiment
    scores = list()

    ModelType = input("Choose a model: LSTM / GRU / RNN \n")
    for r in range(1):
        history = evaluate_model(trainX, trainy, ModelType)
        plot_accuracy(history)
        plot_loss(history)
