
import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)


with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [float(x) for x in request.form.values()] #Convert string inputs to float.
    features = np.array([int_features])  #Convert to numpy format for input to the model
    prediction = model.predict(features)  # predict based on features
    pred = float(prediction[0])
    output = round(pred, 2)

    return render_template('index.html', prediction_text='Predicted Volume is {}'.format(output))

if __name__ == "__main__":
    app.run()

