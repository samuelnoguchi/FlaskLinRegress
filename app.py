import numpy as numpy
from flask import Flask, request, jsonify, render_template
import pickle

#initialize flask app and load model
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')