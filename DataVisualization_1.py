#   ------------------------------Human Activity Recognition using Deep Recurrent Neural Networks on Motion Capture Data------------------------------
#                    ----------har-smartphone/DataVisualization----------
#                              Name: Giannis
#                              Surname: Variozidis
#                              Email: cs141065@uniwa.gr
#                              ID: cs141065
#   ---------------------------------------------------------------------------

# https://machinelearningmastery.com/how-to-model-human-activity-from-smartphone-data/

"""Imports"""
import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
from numpy import unique

'''FUNCTION: load_file()
   DESCRIPTION: Reads values of csv
   RETURNS: Dataframe Values  '''


def load_file(filepath):
    dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values


'''FUNCTION: load_group()
   DESCRIPTION: Load a list of files
   RETURNS: 3D Numpy Array '''


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


'''FUNCTION: data_for_subject()
   DESCRIPTION: Get all data for one subject
   RETURNS: Test x and y elements '''


def data_for_subject(X, y, sub_map, sub_id):
    # get row indexes for the subject id
    ix = [i for i in range(len(sub_map)) if sub_map[i] == sub_id]
    # return the selected samples
    return X[ix, :, :], y[ix]


'''FUNCTION: to_series()
   DESCRIPTION: Convert a series of windows to a 1D list
   RETURNS: 1D series '''


def to_series(windows):
    series = list()
    for window in windows:
        # remove the overlap from the window
        half = int(len(window) / 2) - 1
        for value in window[-half:]:
            series.append(value)
    return series


'''FUNCTION: plot_activity_durations_by_subject()
   DESCRIPTION: plot activity durations by subject
   RETURNS: Plots for activity duration '''


def plot_activity_durations_by_subject(X, y, sub_map, ticks):
    # get unique subjects and activities
    subject_ids = unique(sub_map[:, 0])
    activity_ids = unique(y[:, 0])
    # enumerate subjects
    activity_windows = {a: list() for a in activity_ids}
    for sub_id in subject_ids:
        # get data for one subject
        _, subj_y = data_for_subject(X, y, sub_map, sub_id)
        # count windows by activity
        for a in activity_ids:
            activity_windows[a].append(len(subj_y[subj_y[:, 0] == a]))
    # organize durations into a list of lists
    durations = [activity_windows[a] for a in activity_ids]
    plt.boxplot(durations, labels=ticks)
    plt.savefig('Results/Visualisation/BoxPlot.png')
    plt.show()


'''FUNCTION: plot_subject()
   DESCRIPTION: Plot the data for one subject
   RETURNS: Plots for one subject'''


def plot_subject(X, y):
    plt.figure()
    # determine the total number of plots
    n, off = X.shape[2] + 1, 0
    # plot total acc
    for i in range(3):
        plt.subplot(n, 1, off + 1)
        plt.plot(to_series(X[:, :, off]))
        plt.title('total acc ' + str(i), y=0, loc='left')
        off += 1
    # plot body acc
    for i in range(3):
        plt.subplot(n, 1, off + 1)
        plt.plot(to_series(X[:, :, off]))
        plt.title('body acc ' + str(i), y=0, loc='left')
        off += 1
    # plot body gyro
    for i in range(3):
        plt.subplot(n, 1, off + 1)
        plt.plot(to_series(X[:, :, off]))
        plt.title('body gyro ' + str(i), y=0, loc='left')
        off += 1
    # plot activities
    plt.subplot(n, 1, n)
    plt.plot(y)
    plt.title('activity', y=0, loc='left')
    plt.savefig('Results/Visualisation/SensorValues-Subject.png')
    plt.show()


trainX, trainy = load_dataset_group('train',
                                    prefix='' + 'E:/Documents/Thesis/har1/har-smartphone/Data/UCI HAR Dataset/')
trainy = trainy - 1

print('Total Train Samples: ', trainX.shape[0])

# plotting graph bar for activities
ticks = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']
unique2, counts = np.unique(trainy, return_counts=True)
plt.ylabel('Frequency')
plt.title('Graph Bars - Samples/Class')
plt.xlabel('Class')
plt.xticks(unique2, ticks)
plt.bar(unique2, counts)
plt.savefig('Results/Visualisation/Samples-Class.png')
plt.show()
# load subjects duration
sub_map = load_file('E:/Documents/Thesis/har1/har-smartphone/Data/UCI HAR Dataset/train/subject_train.txt')
train_subjects = unique(sub_map)
print(train_subjects)
sub_id = train_subjects[1]
subX, suby = data_for_subject(trainX, trainy, sub_map, sub_id)
print(subX.shape, suby.shape)
plot_subject(subX, suby)
plot_activity_durations_by_subject(trainX, trainy, sub_map, ticks)
print(counts/(sum(counts))*100)
