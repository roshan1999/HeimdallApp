import numpy as np
from collections import defaultdict
import helper_functions

"""
The idea behind this script is to create a training and testing data
and save that it to a .h5 format, so that it can be easily load in the memory
unlike their csv files which often takes a lot of time to read if they are
bigger in size.
"""

# All the csv are placed in a folder called Data/csvfiles
# we will read certain csv files to create training set and other to create test set
trainingFilenames = helper_functions.readHDF('Data/training_filenames.h5', 'file_names_training')
testingFilenames = helper_functions.readHDF('Data/testing_filenames.h5', 'file_names_testing')

# initialise the dictionary
# we will use our helper_functions csv2dict() functions to populate this dictionary sets
trainingSet = defaultdict(list)
testingSet = defaultdict(list)

# columns present in the CSV file are
# notation in the column names stresses_full_xx -> partial diferentiation of some function 'f' with 'x' and again with 'x'
columns = ['stresses_full_xx', 'stresses_full_yy', 'stresses_full_xy', 'stresses_full_xz','stresses_full_yz','stresses_full_zz']
def makeDataDict(filenames):
    """
    Returns a dictionary having data combined of all the CSV files

    ARGS:
        filename{list} -- a list containg names of all the files

    RETURNS:
        -- a dictionary having the data
    """
    dictionary = defaultdict(list)
    for i, filename in enumerate(filenames):
        print('{}. working with {}, please wait...'.format(i, filename.decode('utf-8')))
        # calling csv2dict(), to convert the csv file to dictionary
        data = helper_functions.csv2dict('Data/csvfiles/' + str(filename.decode('utf-8')))
        # accessing the key aftershocksyn to check for unique values, similar like (set(list[1, 1, 0, 2, 3])) -> outputs [1, 1, 0, 2, 3]
        grid_aftershock_count = np.double(data['aftershocksyn'])
        # no use of if
        #if len(np.unique(grid_aftershock_count)) < 2:
        #   continue
        temp = grid_aftershock_count.tolist()
        # adding a (key, value) to the testingSet
        dictionary['aftershocksyn'].extend(temp)
        # now adding remaining columns
        for column in columns:
            dictionary[column].extend(np.double(data[column]))

    return dictionary

##########################CREATING TESTING DATASET#############################
testingSet = makeDataDict(testingFilenames)

##########################CREATING TRAINING DATASET#############################
trainingSet = makeDataDict(trainingFilenames)

#############################SAVING THE DATASET################################
columns.append('aftershocksyn')
helper_functions.dict2HDF('training_tmp.h5', columns, trainingSet)
helper_functions.dict2HDF('testing_tmp.h5', columns, testingSet)
