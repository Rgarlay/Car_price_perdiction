import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from flask import Flask,request
from flask.templating import render_template
from src.prediction.predict_pipeline import feature_insertion,prediction

app = Flask(__name__)


@app.route('/')
def welcome():
    return "This will be the front page."


@app.route('/predict_price',methods = ['GET','POST'])
def predict_pr():
    if request.method == "GET":
        return render_template("home.html")
    else:

        drivewheel = request.form.get('drivewheel_selector')  

        drivewheel_fwd = 1 if drivewheel == 'fwd' else 0
        drivewheel_rwd = 1 if drivewheel == 'rwd' else 0

        data = feature_insertion(
            wheelbase = float(request.form.get('wheelbase')),
            carlength = float(request.form.get('carlength')),
            carwidth = float(request.form.get('carwidth')),
            enginesize = float(request.form.get('enginesize')),
            boreratio = float(request.form.get('boreratio')),
            stroke = float(request.form.get('stroke')),
            efficency = float(request.form.get('efficency')),
            aspiration = float(request.form.get('aspiration')),
            cylindernumber = int(request.form.get('cylindernumber')),
            drivewheel_fwd = drivewheel_fwd,
            drivewheel_rwd = drivewheel_rwd)
            
        pred = data.get_data_into_df()
        predict = prediction()
        result = predict.model_prediction(pred)
        return render_template("home.html",results = result[0])
    
if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)
    