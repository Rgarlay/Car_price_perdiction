import os
import sys
from ..exception import CustomException
from ..logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from ..utils import oversampling
from src.components.transformation import transformations
from src.components.model_training import initiate_model_training



class ingestion_files_saving():
    def __init__(self):
        self.train_data_path = os.path.join(r'C:\Users\rgarlay\Desktop\DA\Projects\PWskills project\ML project\crypto_liquidation\archieve','train.csv')
        self.test_data_path = os.path.join(r'C:\Users\rgarlay\Desktop\DA\Projects\PWskills project\ML project\crypto_liquidation\archieve','test.csv')
        self.raw_data_path = os.path.join(r'C:\Users\rgarlay\Desktop\DA\Projects\PWskills project\ML project\crypto_liquidation\archieve','raw.csv')


class data_ingestion():
    def __init__(self):
        self.ingest_save = ingestion_files_saving()

    def code_initiating(self):
            
        try:

            logging.info('The data ingestion has started.')

            df = pd.read_csv(r'C:\Users\rgarlay\Desktop\DA\Projects\PWskills project\ML project\crypto_liquidation\Crypto Project\CarPrice_Assignment.csv')

            cols_to_drop = ['car_ID','CarName','fuelsystem','enginetype','fueltype',
                            'carbody','citympg','enginelocation',
                            'carheight','symboling',
                            'doornumber','peakrpm'
                            ,'compressionratio','highwaympg']
            
            df = df.drop(columns = cols_to_drop,axis = 1)

            logging.info('Columns that are not useful have been dropped.')

            train,test = train_test_split(df,test_size = 0.2, random_state=25)

            logging.info('This data has been split into training and testing portion.')

            os.makedirs(os.path.dirname(self.ingest_save.train_data_path), exist_ok=True)

            df = oversampling(df)

            df.to_csv(self.ingest_save.raw_data_path, index = False, header=True)

            train.to_csv(self.ingest_save.train_data_path, index = False, header = True)
            
            test.to_csv(self.ingest_save.test_data_path, index = False, header = True)

            logging.info('The data has successfully been saved.')

            return self.ingest_save.train_data_path,self.ingest_save.test_data_path
        
        except Exception as e:
            raise CustomException(e,sys)

                
if __name__=='__main__':
    obj = data_ingestion()
    train_data, test_data = obj.code_initiating()

    DataTransformation = transformations()
    train_data_scaled,test_data_scaled,preprocessor = DataTransformation.applying_transformation(train_data,test_data)

    model_trainer = initiate_model_training()
    model = model_trainer.training_applying(train_data_scaled,test_data_scaled)