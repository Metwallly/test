'''from flask import Flask,request, jsonify
from flask_restful import Resource, Api
import joblib
import pandas as pd
from flask_cors import CORS


app = Flask(__name__)
#
CORS(app)
# creating an API object
api = Api(app)

#prediction api call 'F:/v2 machine learning/model 1/deploy api machine learning/
model = joblib.load(open('autism_model.pkl','rb'))


@app.route('/')
def home():
    return 'Autism prediction'

@app.route("/predict",methods=["post"])
def predict():
    rates = request.json
    
    quary_df = pd.DataFrame(rates)
    predection = model.predict(quary_df)
    return jsonify(list(predection))
    

if __name__ == '__main__':
    app.run(debug=True)'''





from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Assuming you have a pre-trained model loaded here
model = LogisticRegression()

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the POST request
    data = request.json

    # Convert the data to a pandas DataFrame or the appropriate format
    input_data = pd.DataFrame(data)

    # Make a prediction using your model
    prediction = model.predict(input_data)

    # Convert prediction to a list to ensure it's JSON serializable
    prediction_list = prediction.tolist()

    # Return the prediction as a JSON response
    return jsonify(prediction_list)

if __name__ == '__main__':
    app.run(debug=True)

