from sklearn.impute import SimpleImputer  # Handling missing values
from sklearn.preprocessing import StandardScaler # Handling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder # Ordinal Encoding

# Pipelining
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import os, sys
from dataclasses import dataclass
import pandas as pd
import numpy as np 

from src.logger import logging
from src.exception import CustomException

from src.utils import save_object

# Data Transformation Config

@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path = os.path.join('artifacts', 'preprocessor.pkl')


# Data Transformation Class  

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation Initiated')
            
            # Segregating numerical and categorical columns
            categorical_cols = ['cut', 'color','clarity']
            numerical_cols = ['carat', 'depth','table', 'x', 'y', 'z']
            
            # Defining custom ranking for each ordinal variable.
            cut_categories = ['Fair','Good','Very Good', 'Premium','Ideal']
            color_categories = ['D','E','F','G','H','I','J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
            
            logging.info("Pipeline initiated")
            
            # Numerical Pipeline
            numerical_pipeline = Pipeline(
                                steps=[
                                    ('imputer',SimpleImputer(strategy='median')),
                                    ('scaler',StandardScaler())
                                ]) 
            # Categorical Pipeline
            categorical_pipeline = Pipeline(
                                steps=[
                                    ('imputer', SimpleImputer(strategy='most_frequent')),
                                    ('ordinalencoder', OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                                    ('scaler',StandardScaler())
                                ])
            preprocessor = ColumnTransformer([
                                                ('numerical_pipeline', numerical_pipeline,numerical_cols),
                                                ('categorical_pipeline',categorical_pipeline, categorical_cols)
                                            ])
            
            return preprocessor
            logging.info("Pipeline Completed")
            
        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e,sys)
            
            
    
    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            
            logging.info("Read Train and Test data completed")
            logging.info(f'Train DataFrame Head: \n{train_df.head().to_string()}')
            logging.info(f'Test DataFrame Head: \n{test_df.head().to_string()}')
            logging.info("Obtaining Preprocessed objects")
            
            preprocessing_obj = self.get_data_transformation_object()
            
            target_col_name = 'price'
            drop_cols_train = [target_col_name] + ['id'] if 'id' in train_df.columns else [target_col_name]
            
            # Dependent and indipendent features
            input_feature_train_df = train_df.drop(columns=drop_cols_train, axis=1)
            target_feature_train_df = train_df[target_col_name]
            
            drop_cols_test = [target_col_name] + ['id'] if 'id' in test_df.columns else [target_col_name]
            input_feature_test_df = test_df.drop(columns=drop_cols_test, axis=1)
            target_feature_test_df = test_df[target_col_name]
            
            # Applying transformation
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)           
            
            logging.info("Applying preprocessing object on training and test datasets.")
            
            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            save_object(
                
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessing_obj  
            )       
            
            logging.info("Preprocessor pickle is created and saved") 
            
            return (
                train_arr, 
                test_arr, 
                self.data_transformation_config.preprocessor_ob_file_path
            )
        except Exception as e:
            logging.info("Exception occured in initiate_data_transformation")
            raise CustomException(e,sys)
    