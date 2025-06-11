import pandas as pd
import numpy as np
import os
import sys
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.utils import resample


def oversampling(df):
  least_cat = df[df['drivewheel']=='4wd']
  smote = resample(least_cat,replace=True, n_samples=30, random_state = 42)
  df = pd.concat([df,smote], ignore_index=True)
  return df



def encoding(df):
  df['cylindernumber'] = df['cylindernumber'].replace({'eight':5,'six':4,'five':3,'four':1,'two':2})
  df['aspiration'] = df['aspiration'].replace({'std':1,'turbo':0})
  return df

def feature_eng(df_oversampled):
  df_oversampled = pd.get_dummies(data = df_oversampled,columns = ['drivewheel'],drop_first=True,dtype='int')
  df_oversampled['efficency'] = df_oversampled['horsepower']/df_oversampled['curbweight']
  df_oversampled = df_oversampled.drop(columns=['horsepower','curbweight'])
  return df_oversampled


def outlier_capping(df,col):
  upper = df[col].quantile(0.75)
  lower = df[col].quantile(0.25)
  IQR = upper - lower
  df.loc[(df[col]>upper),col] = upper
  df.loc[(df[col]<lower),col] = lower
  return  df

def save_object(file_path, obj):
  try:
    dir_path = os.path.dirname(file_path)
        
    os.makedirs(dir_path, exist_ok=True)

    with open(file_path, "wb") as file_obj:
      dill.dump(obj, file_obj)
    
  except Exception as e:
    raise CustomException(e,sys)
  
def model_prediction(model,x_train,y_train,x_test,y_test):
  model.fit(x_train,y_train)
  train_pred = model.predict(x_train)
  test_pred = model.predict(x_test)

  train_r2_score= r2_score(y_train,train_pred)
  test_r2_score = r2_score(y_test,test_pred)

  train_rmse_score = mean_squared_error(y_train,train_pred)
  test_rmse_score = mean_squared_error(y_test,test_pred)

  model_metrics = {'Name': 'XGBoose',
                   'Train r2 score':train_r2_score,
                   'Test_r2_score':test_r2_score,
                   'Train_rmse_score':np.sqrt(train_rmse_score),
                   'Test_rmse_score':np.sqrt(test_rmse_score)}
  return model,model_metrics