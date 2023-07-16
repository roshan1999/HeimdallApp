import numpy as np
import pandas as pd
import h5py
import csv
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import optimizers
from keras import initializers
from keras import metrics

from sklearn.metrics import roc_auc_score


# Reading a CSV file into a dictionary
def csv2dict(filename):
    """
    Creates a dictionary from the CSV file.
    which means all the 'columns' of the CSV file will be our 'key'
    and its 'data' will be our 'values'
    dictionary = {'x':[1, 2, 3, ...],
                  'y':[4, 5, 7, ...],
                  'z':[2, 7, 0, ...],
                  and so on...}

    ARGS:
        filename -- name to the csv file

    RETURNS:
        -- a dictionary object
    """

    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        dictionary = {} # initialise our dictionary
        for row in reader:
            for column, data in row.items():
                # for the same key we will append the data to it in form of a list
                dictionary.setdefault(column, []).append(data)
    return dictionary


# Write a dictionary file to HDF file
def dict2HDF(filename, columns, data):
    """
    Writes a dictionary to a h5 file format

    ARGS:
        filename -- name of the .h5 file in which we want to write the data
        columns{list} -- list of names
        data{dict} -- a dictionary

    RETURNS:
        -- 0
        """

    with h5py.File(filename, 'w') as file:
        for column in columns:
            file.create_dataset(column, data=data[column])
    return 0


# Read filenames from HDF file
def readHDF(filename, column):
    """
    Reads a particular column from a HDF file

    ARGS:
        filename{string} -- name of the HDF file
        column{string} -- name of the column that we want to read

    RETURNS:
        -- a numpy array containing the value of the column
    """
    with h5py.File(filename, 'r') as file:
        requiredCol = np.squeeze(np.array(file.get(column)))
    return requiredCol


def loadDataFromHDF(filename, features_in, features_out):
    """
    load data from the HDF and create X and y,
    X -> consist of feature on which are model will be trained
    y -> label

    ARGS:
        filename -- name of the HDF file from which we want to create the X, y pair
        features_in -- a list of features
        features_out -- output feature, or label

    RETURNS:
        X -- a numpy array containing data to be trained on
        y -- output ground truth label
    """

    with h5py.File(filename, 'r') as file:
        # let's read a feature, to get the idea of the size
        # temp_in is just a column vector
        tmp_in = readHDF(filename, features_in[0])
        X = np.zeros([np.size(tmp_in), len(features_in) * 2])
        # say we have 5 features and 100 rows
        # then IN will have dimension -> (100, 10)
        for i, column in enumerate(features_in):
            X[:, i] = np.abs(readHDF(filename, column))
        for i, column in enumerate(features_in):
            X[:, len(features_in) + i] = -1.0 * np.abs(readHDF(filename, column))
        y = readHDF(filename, features_out)
        X = X / 1.0e6
        return X, y


def loadDataFromDict(data, features_in):
    """
    loads data from a dictionary file

    ARGS:
        data -- dictionary file
        features_in -- the features that were used for training

    RETURNS:
        X -- the data affiliated to the features
    """
    tmp = np.double(data[features_in[0]])
    X = np.zeros([np.size(tmp), len(features_in) * 2])
    for i, column in enumerate(features_in):
        X[:, i] = np.abs(np.double(data[column]))
    for i, column in enumerate(features_in):
        X[:, len(features_in) + i] = -1.0 * np.abs(np.double(data[column]))
    X = X / 1.0e6
    return X


def findBestSignConventionCFS(data, aftershock_count):
    """
    We want to check the best sign convention for CFS out of four CFS columns

    ARGS:
        data -- dictionary containing the data
        aftershock_count -- a list containing the total number of aftershocks

    RETURNS:
        a number between 1 - 4 specifying the column
    """

    auc = np.zero(4)
    # as we have four column related to AUC in which
    #we have two distinct columns and the other two contain the same data but sign is different

    auc[0] = roc_auc_score(aftershock_count.transpose(), np.double(data['stresses_full_cfs_1']))
    auc[1] = roc_auc_score(aftershock_count.transpose(), np.double(data['stresses_full_cfs_2']))
    auc[2] = roc_auc_score(aftershock_count.transpose(), np.double(data['stresses_full_cfs_3']))
    auc[3] = roc_auc_score(aftershock_count.transpose(), np.double(data['stresses_full_cfs_4']))

    whichIsMax = np.argmax(auc)
    return whichIsMax + 1


def createModel():
    """
    Return an architecture of a Sequential model
    """
    model = Sequential()
    model.add(Dense(units=50, input_dim=12, kernel_initializer='lecun_uniform', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, kernel_initializer='lecun_uniform', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, kernel_initializer='lecun_uniform', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, kernel_initializer='lecun_uniform', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, kernel_initializer='lecun_uniform', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, kernel_initializer='lecun_uniform', activation= 'tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(1, kernel_initializer='lecun_uniform', activation='sigmoid'))

    model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=[metrics.binary_accuracy])
    return model


def generateData(posData, negData, batch_size, posStart, negStart):
    """
    It generates random selected data from posData and negData
    depending upon the batch_size, posStart and negStart
    say shapeOfNeg -> 100
    and shapeOfPos -> 50
    and batch_size -> 5

    """
    shapeOfPos = np.shape(posData)
    shapeOfNeg = np.shape(negData)
    while True:
        if posStart + round(batch_size / 2.0) >= shapeOfPos[0]:
            posStart = 0
            np.random.shuffle(posData)
        posEnd = posStart + int(round(batch_size / 2.0))
        if negStart + round(batch_size / 2.0) >= shapeOfNeg[0]:
            negStart = 0
            np.random.shuffle(negData)
        negEnd = negStart + int(round(batch_size / 2.0))

        data = np.row_stack((posData[posStart:posEnd, :], negData[negStart:negEnd, :]))
        np.random.shuffle(data)
        posStart = posStart + int(round(batch_size / 2.))
        negStart = negStart + int(round(batch_size / 2.))
        yield (data[:, :12], data[:, 12]) # 12 is number of features_in


# writeHDF is used in the eval model
def writeHDF(filename, X, y):
    """
    write X, y to a HDF file

    ARGS:
        filename -- specify the name of the file where you want to save the X, y
        X -- independent variables on which we train our model
        y -- dependent variable or label

    RETURNS:
        -- 0
    """
    with h5py.File(filename, 'w') as file:
        file.create_dataset('X', data=X)
        file.create_dataset('y', data=y)
    return 0
