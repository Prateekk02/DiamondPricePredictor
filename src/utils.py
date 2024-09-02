import os, sys
import pickle
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report={}
        for i in range(len(list(models))):
        models = list(models.values())[i]
        models.fit(X_train,y_train)
        # Predicting result.
        y_test_pred = model.predict(X_test)
        
        # Getting r2 score for train and test data
        # train_model_score = r2_score(y_test, y_train_pred)
        test_model_score = r2_score(y_test, )
        
    
  