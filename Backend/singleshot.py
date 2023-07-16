import numpy as np
import helper_functions
from collections import defaultdict
import sklearn
import csv


##input a file name of csv format from the singleCsv folder in the Data directory.

filename = 'rt-data-io/incoming.csv'
weightFile = 'Data/weights.h5'
predFile = 'Data/singleCSV/singlePred.h5'
columns = ['stresses_full_xx', 'stresses_full_yy', 
           'stresses_full_xy', 'stresses_full_xz',
           'stresses_full_yz','stresses_full_zz']
testFile = 'single.h5'

dictionary = defaultdict(list)
print('working with {},...'.format(filename.split('/')[-1]))
data = helper_functions.csv2dict(filename)
grid_aftershock_count = np.double(data['aftershocksyn'])
temp = grid_aftershock_count.tolist()
dictionary['aftershocksyn'].extend(temp)
for column in columns:
    dictionary[column].extend(np.double(data[column]))


columns.append('aftershocksyn')
helper_functions.dict2HDF('single.h5', columns, dictionary)
features_in = ['stresses_full_xx',
               'stresses_full_yy',
               'stresses_full_xy',
               'stresses_full_xz',
               'stresses_full_yz',
               'stresses_full_zz']

features_out = 'aftershocksyn'
model = helper_functions.createModel()
model.load_weights(weightFile)
X, y = helper_functions.loadDataFromHDF(testFile, features_in, features_out)
y_pred = model.predict(X)
helper_functions.writeHDF(predFile, X, y)
auc = sklearn.metrics.roc_auc_score(y, y_pred)

writer = csv.writer(open('rt-data-io/outgoing.csv','w'))
writer.writerow(auc);
