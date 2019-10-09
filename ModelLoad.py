from keras.models import load_model
from keras.models import model_from_json
import keras.models
from sklearn.externals.joblib import dump
from sklearn.externals.joblib import load
import tensorflow as tf

def init():
    model = load_model('model.h5')
    filename = 'Scalar.pkl'
    scalers = load(filename)
    return model,scalers