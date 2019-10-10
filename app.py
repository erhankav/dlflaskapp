"""
This script runs the application using a development server.
It contains the definition of routes and views for the application.
"""
# Load libraries
from flask import Flask,request,jsonify
import sqlite3 as db
import numpy as np
from numpy import mean
from numpy import std
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import itertools
from keras import metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import ConvLSTM2D
import tensorflow as tf
import ast
from ModelLoad import *
import json

app = Flask(__name__)

# Make the WSGI interface available at the top level so wfastcgi can get it.
wsgi_app = app.wsgi_app
global model
model,scalers=init()

@app.route('/')
def hello():
    """Renders a sample page."""
    return "Hello World!"


@app.route('/predict' , methods=['GET'])
def predict():
    
     
     X = np.empty((0,30,9))
     
     dizi=request.args.get('dnm');
     
     dizi=dizi.replace(" ",",")
     dizi=dizi.replace("[,","[")
     dnm=ast.literal_eval(dizi);
     dnm=list(dnm)
     npdizi=np.array(dnm)
     #npdizi=pd.DataFrame(npdizi)
     X = np.append(X, [npdizi], axis=0)
     for i in range(X.shape[1]):
        X[:, i, :] = scalers[i].transform(X[:, i, :]) 
   

     n_timesteps, n_features = X.shape[1], X.shape[2]
     n_steps, n_length = 3, 10
     X = X.reshape((X.shape[0], n_steps, n_length, n_features))
     result=model.predict_classes(X).tolist()
     return json.dumps({'prediction': result})
     #return jsonify({'prediction': result.tolist()})
     #return str(npdizi)
     #return str(result)
if __name__ == '__main__':
    import os
    HOST = os.environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(os.environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    app.run(HOST, PORT,threaded=False)
    #app.run(HOST, PORT)
