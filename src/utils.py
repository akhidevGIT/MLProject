import os
import sys
import pandas as pd
import numpy as np
import dill
import pickle

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.logger import logging
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, X_test, y_train, y_test, models, params):
    try:
        report={}
        for i in range(len(models)):
            model = list(models.values())[i]
            param = params[list(params.keys())[i]]
            
            GridSearch = GridSearchCV(model, param, cv=3)
            GridSearch.fit(X_train, y_train) 

            logging.info(f"Best GridSearch params: {model}: {GridSearch.best_params_} ")
            
            model.set_params(**GridSearch.best_params_)
            model.fit(X_train, y_train) #Train model with best_params from GridSearchCV

            #y_pred_train = model.predict(X_train)

            y_pred_test = model.predict(X_test)

            #train_model_score = r2_score(y_train,y_pred_train)

            test_model_score = r2_score(y_test, y_pred_test)
            
            report[list(models.keys())[i]] = test_model_score                 
        
        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)
