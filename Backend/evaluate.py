import numpy as np
import sklearn as sklearn
import helper_functions

# name of weights
weightFile = 'Data/weights.h5'
predFile = 'predicted.h5'
testFile = 'Data/testing.h5'

# name of features in our dataset
features_in = ['stresses_full_xx',
               'stresses_full_yy',
               'stresses_full_xy',
               'stresses_full_xz',
               'stresses_full_yz',
               'stresses_full_zz']

# name of label
features_out = 'aftershocksyn'

# load the model
model = helper_functions.createModel()

# load the weights and evaluate on test file
model.load_weights(weightFile)
X, y = helper_functions.loadDataFromHDF(testFile, features_in, features_out)
y_pred = model.predict(X)
helper_functions.writeHDF(predFile, X, y)
auc = sklearn.metrics.roc_auc_score(y, y_pred)
print('merged AUC on testing data set: ' + str(auc))
