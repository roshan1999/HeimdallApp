from flask import Flask, request, render_template
import csv
from twilio.rest import Client
import numpy as np
import helper_functions
from collections import defaultdict
import sklearn
import json
import random

app = Flask(__name__)

#compute function
def singleshot():
	fn = str(random.randint(1,25))
	filename = 'rt-data-io/'+fn+'.csv'
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
	return auc;

def send_sms():

	f = open("strings.json")
	data = json.load(f)
	account_sid = data["ids"]["acc_sid"]
	auth_token = data["ids"]["auth_token"]

	client = Client(account_sid, auth_token)

	message = client.messages \
				.create(
					body="Aftershock alert. Initiate Evacuation procedures immediately.",
					from_='+13153874357',
					to='+918531018296' #this can be sent as a broadcast in a regin but the team is constrained by twilio's trial SMS plan for now.  
					#Further versions of the DL module and SMS-Modules will have broadcast feature included.
                 )


@app.route('/')
def hexalldm():
	return render_template('index.html')

@app.route('/', methods=['GET','POST'])
def predict():
	#fetch input data from textfields
	#sx = request.form['sx']
	#sy = request.form['sy']
	#sz = request.form['sz']
	#sxx = request.form['sxx']
	#syy = request.form['syy']
	#sxy = request.form['sxy']
	#sxz = request.form['sxz']
	#syz = request.form['syz']
	#szz = request.form['szz']
	#asyn = request.form['asyn']

			#random.randint(1,5)

	#call computing function		
	auc = singleshot()
	
	#call caution function
	if auc>0.700:
		send_sms()

	#return output on webpage
	stng = "Probability of Aftershock: "+str(auc)
	return stng

if __name__ == '__main__':
	app.debug = True
	app.run()
