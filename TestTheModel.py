

"""IMPORTS"""

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tensorflow.python.keras.utils.np_utils import to_categorical
import seaborn as sns;
sns.set()


'''FUNCTION: load_file()
   DESCRIPTION: reads values of csv
   RETURNS: dataframe values  '''

def load_file(filepath):
    dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values


'''FUNCTION: load_group()
   DESCRIPTION: load a list of files
   RETURNS: 3d Numpy Array  '''

# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    loaded = np.dstack(loaded)
    return loaded


'''FUNCTION: load_dataset_group()
   DESCRIPTION: load a dataset group, such as train or test
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
   RETURNS: test x and y elements '''

def load_TestDataset(prefix=''):
    # load all test
    testX, testy = load_dataset_group('test', prefix + 'E:/Documents/Thesis/har1/har-smartphone/Data/UCI HAR Dataset/')
    print(testX.shape, testy.shape)
    # zero-offset class values
    testy = testy - 1
    # one hot encode y
    testy = to_categorical(testy)

    print(testX.shape, testy.shape)
    return testX, testy


'''FUNCTION: test_model()
   DESCRIPTION: Evaluating models with metrics and confusion matrix
   RETURNS: Plots and results of testing '''


def test_model(model, testX, testY, n_outputs):
    score = loaded_model.evaluate(testX, testy, verbose=1)
    prediction_list = model.predict(testX)
    predictions_transformed = np.eye(n_outputs, dtype=int)[np.argmax(prediction_list, axis=1)]

    # ------------Metrics ------------
    acc_test = accuracy_score(testY, predictions_transformed)
    pre_test = precision_score(testY, predictions_transformed, average='macro')
    rec_test = recall_score(testY, predictions_transformed, average='macro')
    f1_test = f1_score(testY, predictions_transformed, average='macro')

    print('-----------------METRICS FOR OUR MODEL-----------------')
    print('--------------TEST-------------')
    print('Accuracy score:', '{:.2f}.'.format(acc_test))
    print('Precision score:', '{:.2f}.'.format(pre_test))
    print('Recall score:', '{:.2f}.'.format(rec_test))
    print('F1  score:', '{:.2f}.'.format(f1_test))

    # -----------Export Results to EXCEL-----------
    np.savetxt("E:/Documents/Thesis/har1/har-smartphone/Results/Testing/predictions.txt", prediction_list)
    np.savetxt("E:/Documents/Thesis/har1/har-smartphone/Results/Testing/trans_predictions.txt",
               (np.argmax(predictions_transformed, axis=1)))
    # Fields of Excel with init train_data
    cols = {
        'Technique': ['HAR1'],
        'Train Data ratio': ['60:40'],
        'Precision(te)': [pre_test],
        'Recall(te)': [rec_test],
        'F1 Score(te)': [f1_test],
        'Accuracy(te)': [acc_test],
    }
    df = pd.DataFrame(cols,
                      columns=['Technique', 'Train Data ratio', 'Precision(te)', 'Recall(te)', 'F1 Score(te)',
                               'Accuracy(te)'])
    export_excel = df.to_excel("E:/Documents/Thesis/har1/har-smartphone/Results/Testing/Metrics.xls")
    print("Write to Excel File Completed!")
    print(classification_report(testY, predictions_transformed))

    # Calculating confusion matrix
    print("Confusion Matrix:")
    confusion_matrix = metrics.confusion_matrix(np.argmax(testy, axis=1), np.argmax(prediction_list, axis=1))
    print(confusion_matrix)
    normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32) / np.sum(confusion_matrix) * 100

    LABELS = [
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING"
    ]
    sns.heatmap(confusion_matrix, square=True, annot=True, fmt='d', cbar=True, xticklabels=LABELS,
                yticklabels=LABELS)
    # Plot Results:
    width = 6
    height = 6

    plt.figure(figsize=(width, height))
    plt.imshow(
        normalised_confusion_matrix,
        interpolation='nearest',
        cmap=plt.cm.rainbow
    )
    plt.title("Confusion matrix \n(normalised to % of total test Data)")
    plt.savefig('Results/Testing/Confusion Matrix.png')

    print(normalised_confusion_matrix)

    plt.colorbar()
    tick_marks = np.arange(6)
    plt.xticks(tick_marks, LABELS, rotation=90)
    plt.yticks(tick_marks, LABELS)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('Results/Testing/PredictedActivity.png')
    plt.show()


if __name__ == "__main__":
    # Loading our Trained Model
    model_name = 'Saved Models/GRU.h5'
    loaded_model = keras.models.load_model(model_name)
    testX, testy = load_TestDataset()
    # testX = testX.reshape((testX.shape[0], 4, 32, 9))
    test_model(loaded_model, testX, testy, 6)
