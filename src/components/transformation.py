import pandas as pd
import numpy as np
import os
import sys
import pickle
from ..utils import encoding,outlier_capping,save_object,feature_eng,oversampling
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from ..exception import CustomException
from ..logger import logging

class transformation_files_saving():
    def __init__(self):
        self.preprocessor_file = os.path.join(r'C:\Users\rgarlay\Desktop\DA\Projects\PWskills project\ML project\crypto_liquidation\archieve','preprocessor.pkl')
        self.train_file = os.path.join(r'C:\Users\rgarlay\Desktop\DA\Projects\PWskills project\ML project\crypto_liquidation\archieve','train_scaled.csv')
        self.test_file = os.path.join(r'C:\Users\rgarlay\Desktop\DA\Projects\PWskills project\ML project\crypto_liquidation\archieve','test_scaled.csv')

class transformations():
    def __init__(self):

        self.transformation = transformation_files_saving()

    def preprocessor_pipeline(self):

        numerical_cols = ['wheelbase', 'carlength', 'carwidth',
            'enginesize', 'boreratio', 'stroke',
            'efficency']

        numeric_pipeline = Pipeline(steps=[('stdscaler', StandardScaler())])

        preprocessing_pipeline = ColumnTransformer(
        transformers=[('numeric_pipe', numeric_pipeline, numerical_cols)])

        return preprocessing_pipeline


    def applying_transformation(self,train,test):

        try:

            train = pd.read_csv(r'C:\Users\rgarlay\Desktop\DA\Projects\PWskills project\ML project\crypto_liquidation\archieve\train.csv')    
            test = pd.read_csv(r'C:\Users\rgarlay\Desktop\DA\Projects\PWskills project\ML project\crypto_liquidation\archieve\test.csv')    
            
            logging.info("Reading training and testing datasets from CSV is done.")

            logging.info("Filtering out rows with 'three' and 'twelve' cylinder numbers.")

            train = train[train['cylindernumber'] != 'three']
            test = test[test['cylindernumber'] != 'three']
            train = train[train['cylindernumber'] != 'twelve']
            test = test[test['cylindernumber'] != 'twelve']

            logging.info("Applying encoding and feature engineering on training and test data.")

            train = encoding(train)
            train = feature_eng(train)
            test = encoding(test)
            test = feature_eng(test)
            
            

            col_list = ['wheelbase', 'carlength', 'carwidth', 'cylindernumber',
            'enginesize', 'boreratio', 'stroke',
            'efficency']

            logging.info("Applying outlier capping on specified columns.")

            for i in col_list:
                train = outlier_capping(train,i)
                test = outlier_capping(test,i)

            numerical_cols = ['wheelbase', 'carlength', 'carwidth',
            'enginesize', 'boreratio', 'stroke',
            'efficency']
            cat_cols = ['aspiration','cylindernumber','drivewheel_fwd','drivewheel_rwd']

            logging.info("Generating preprocessing pipeline.")

            preprocessor = self.preprocessor_pipeline()

            logging.info("Fitting and transforming training data with preprocessing pipeline.")


            train_copy = train[numerical_cols]
            train_copy = preprocessor.fit_transform(train_copy)
            train_scaled_data  = pd.DataFrame(train_copy, columns = numerical_cols)
            train_scaled_data[['aspiration', 'cylindernumber', 'drivewheel_fwd', 'drivewheel_rwd','price']] = train[['aspiration', 'cylindernumber', 'drivewheel_fwd', 'drivewheel_rwd','price']].values
            
            test_copy = test[numerical_cols]
            test_copy = preprocessor.transform(test_copy)
            test_scaled_data  = pd.DataFrame(test_copy, columns = numerical_cols)
            test_scaled_data[['aspiration', 'cylindernumber', 'drivewheel_fwd', 'drivewheel_rwd','price']] = test[['aspiration', 'cylindernumber', 'drivewheel_fwd', 'drivewheel_rwd','price']].values

            logging.info("Saving transformed training data to CSV and preprocessor into pickle.")


            train_scaled_data.to_csv(self.transformation.train_file,index=False,header=True)
            test_scaled_data.to_csv(self.transformation.test_file,index=False,header=True)

            save_object(file_path=self.transformation.preprocessor_file,
                    obj=preprocessor)
            
            return train_scaled_data,test_scaled_data,self.transformation.preprocessor_file
    
        except Exception as e:
            raise CustomException(e,sys)

