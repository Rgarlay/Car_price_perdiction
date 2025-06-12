import pandas as pd
import os
import sys
import numpy as np
from ..exception import CustomException
from ..logger import logging
from ..utils import load_object

### Let's create a class to get data

class feature_insertion():
    def __init__(self,wheelbase, carlength,carwidth, enginesize,boreratio,
                         stroke,efficency, aspiration,cylindernumber,drivewheel_fwd,drivewheel_rwd):
        
        self.wheelbase = wheelbase
        
        self.carlength = carlength

        self.carwidth = carwidth

        self.enginesize = enginesize

        self.boreratio = boreratio

        self.stroke = stroke

        self.efficency = efficency

        self.aspiration = aspiration

        self.cylindernumber = cylindernumber

        self.drivewheel_fwd = drivewheel_fwd
        
        self.drivewheel_rwd = drivewheel_rwd


    def get_data_into_df(self):
        try:

            car_spec_data = {
                'wheelbase' :[self.wheelbase],
                'carlength' :[self.carlength],
                'carwidth' :[self.carwidth],
                'enginesize' :[self.enginesize],
                'boreratio' :[self.boreratio],
                'stroke' :[self.stroke],
                'efficency' :[self.efficency],
                'aspiration' :[self.aspiration],
                'cylindernumber' :[self.cylindernumber],
                'drivewheel_fwd' :[self.drivewheel_fwd],
                'drivewheel_rwd' :[self.drivewheel_rwd]
            }

            df = pd.DataFrame(car_spec_data)
            return df
        
        except Exception as e:
            raise CustomException(e,sys)
        
class prediction():
    def __init__(self):
        pass
    def model_prediction(self,df):
        try:
            model_path = r'archieve\model.pkl'
            preprocessor_path = r'archieve\preprocessor.pkl'

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            numerical_cols = ['wheelbase', 'carlength', 'carwidth',
            'enginesize', 'boreratio', 'stroke',
            'efficency']

            df_num = df[numerical_cols]

            df_num = preprocessor.transform(df_num)
            df_num  = pd.DataFrame(df_num, columns = numerical_cols)
            df_num[['aspiration', 'cylindernumber', 'drivewheel_fwd', 'drivewheel_rwd']] = df[['aspiration', 'cylindernumber', 'drivewheel_fwd', 'drivewheel_rwd']].values

            preds = model.predict(df_num)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
        