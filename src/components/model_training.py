import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from xgboost import XGBRegressor
import pandas as pd
from ..utils import model_prediction,save_object
from ..exception import CustomException
from ..logger import logging

class model_save():
    def __init__(self):
        self.model = os.path.join(r'C:\Users\rgarlay\Desktop\DA\Projects\PWskills project\ML project\crypto_liquidation\archieve','model.pickle')

class initiate_model_training():
    def __init__(self):
        self.model_savning = model_save()

    def training_applying(self,train,test):
        
        try:

            logging.info("Reading transformed training and testing datasets from CSV.")

            train = pd.read_csv(r'C:\Users\rgarlay\Desktop\DA\Projects\PWskills project\ML project\crypto_liquidation\archieve\train_scaled.csv')
            test = pd.read_csv(r'C:\Users\rgarlay\Desktop\DA\Projects\PWskills project\ML project\crypto_liquidation\archieve\test_scaled.csv')
            
            logging.info("Splitting features and target variable for both train and test sets.")

            x_train = train.drop(columns = ['price'])
            y_train = train['price']
            x_test = test.drop(columns = ['price'])
            y_test = test['price']

            logging.info("Initializing XGBoost Regressor with specified hyperparameters.")

            xgb = XGBRegressor(gamma = 1,
                               learning_rate = 0.1, 
                               max_depth = 3,
                        n_estimator = 100,
                        min_child_weight = 8,
                        subsample = 0.5,
                        reg_alpha = 1,
                        reg_lambda = 1)

            logging.info("Training the model and evaluating on test data.")

            model,model_metrics = model_prediction(xgb, x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test)

            logging.info(f"The performace of the model is {model_metrics}")
            logging.info("Saving trained model to pickle file.")

            save_object(self.model_savning.model,model)

            return model

        except Exception as e:
            raise CustomException(e,sys)


        
    
