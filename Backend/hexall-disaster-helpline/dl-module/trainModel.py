import helper_functions
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
import copy
import numpy as np

weightFile = 'Data/weights.h5'
trainFile = 'Data/training.h5'
testFile = 'Data/testing.h5'

features_in = ['stresses_full_xx', 'stresses_full_yy', 'stresses_full_xy', 'stresses_full_xz','stresses_full_yz','stresses_full_zz']#train 
features_out = 'aftershocksyn'#predict


X, y = helper_functions.loadDataFromHDF(trainFile, features_in, features_out)

posIndex = np.where(y == 1) 
posSize = np.size(posIndex) # total number of the elements having y==1
negIndex = np.where(y == 0)
negSize = np.size(negIndex)
posData = np.column_stack((np.squeeze(X[posIndex, :]), y[posIndex].T))
negData = np.column_stack((np.squeeze(X[negIndex, :]), y[negIndex].T))
np.random.seed(5)
np.random.shuffle(posData)
np.random.shuffle(negData)
np.random.seed()
# validation dataset-0% of positive samples and negative samples
cutoff = int(round(posSize / 10)) # 10% of the total number
Xp_val = copy.copy(posData[:cutoff, :len(features_in) * 2])
yp_val = copy.copy(posData[:cutoff, len(features_in) * 2])
Xn_val = copy.copy(negData[:cutoff, :len(features_in) * 2])
yn_val = copy.copy(negData[:cutoff, len(features_in) * 2])
X_val = np.row_stack((Xp_val, Xn_val))
y_val = np.append(yp_val, yn_val)
# the remaining dataset is our training set
posDataFinal = copy.copy(posData[cutoff:, :])
negDataFinal = copy.copy(negData[cutoff:, :])

shapeOfTrainingDataset = np.shape(posDataFinal)
batch_size = 4000
steps_per_epoch = int(round((shapeOfTrainingDataset[0]) / batch_size))
epochs = 300
posStart = 0
negStart = 0
model = helper_functions.createModel()
checkpointer = ModelCheckpoint(filepath=weightFile, monitor='val_loss', verbose=2, save_best_only=True)
history = model.fit_generator(helper_functions.generateData(posDataFinal, negDataFinal, batch_size, posStart, negStart), steps_per_epoch, validation_data=(X_val, y_val), callbacks=[checkpointer], verbose=2, epochs=epochs)
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
