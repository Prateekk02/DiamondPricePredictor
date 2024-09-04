import numpy as np   
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from src.utils import evaluate_model
from dataclasses import dataclass
import sys, os

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')
    
class ModelTrainer:
    
    def __init__(self):
        self.model_training_config = ModelTrainerConfig()
        
    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting Dependent and Independent variables from train_array and test_array")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:, -1],
                test_array[:,:-1] ,
                test_array[:,-1]               
            )      
            
            models = {
                'LinearRigression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'ElasticNet': ElasticNet(),
                'DecisionTreeRegression': DecisionTreeRegressor()
            }   
            
            model_report:dict = evaluate_model(X_train,y_train, X_test,y_test, models)
            print(model_report)
            print("\n==========================================================")
            logging.info(f'Model Report:  {model_report}')
            
            
            
            # Best model score
            
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]
            
            print(f'Best model found, model name: {best_model_name}, R2_Score : {best_model_score}')
            print('\n=================================')
            logging.info(f'Best model found, model name: {best_model_name}, R2_Score : {best_model_score}')
            
            save_object(
                file_path= self.model_training_config.trained_model_file_path, 
                obj = best_model
            )
            
        except Exception as e:
            logging.info("Exception occured during model training")
            raise CustomException(e, sys)     
           