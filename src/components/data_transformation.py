import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object 

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    
        def __init__(self):
            self.preprocessor_path = DataTransformationConfig()
    
        def TransformerObject(self):
            try:
                 numeric_columns = ['reading_score', 'writing_score']
                 cat_columns = [
                      'gender',
                      'race_ethnicity',
                      'parental_level_of_education',
                      'lunch',
                      'test_preparation_course'
                      ]
                 num_pipleline = Pipeline(
                      steps=[
                           ("imputer", SimpleImputer(strategy='median')),
                           ("scaler", StandardScaler())                           
                      ]
                )
                 cat_pipeline = Pipeline(
                      steps=[
                           ("impute", SimpleImputer(strategy="most_frequent")),
                           ("onehotencoder", OneHotEncoder())
                           
                      ]
                 )
                 logging.info(f"categorical columns: {cat_columns}")
                 logging.info(f"numeric columns: {numeric_columns}")

                 preprocessor = ColumnTransformer(
                      [
                           ("numeric_pipleline", num_pipleline, numeric_columns),
                           ("categoric_pipeline", cat_pipeline, cat_columns)
                      ]
                 )

                 return preprocessor

            except Exception as e:
                raise CustomException(e,sys)
            
        def InitiateDataTransformation(self, train_path, test_path):
             try:
                  train_df = pd.read_csv(train_path)
                  test_df = pd.read_csv(test_path)

                  logging.info("Read train and test data completed")
                  logging.info("instantiating preprocessing object")

                  PreprocessorObj = self.TransformerObject()

                  target_column = 'math_score'
                  
                  input_feature_train = train_df.drop(columns=target_column, axis=1)
                  target_feature_train = train_df[target_column]

                  input_feature_test = test_df.drop(columns=target_column, axis=1)
                  target_feature_test = test_df[target_column]

                  logging.info("Applying Preprocessing object on train and test data frames")

                  input_feature_preprocess_train = PreprocessorObj.fit_transform(input_feature_train)
                  input_feature_preprocess_test = PreprocessorObj.transform(input_feature_test)

                  train_preprocess_arr = np.c_[input_feature_preprocess_train, target_feature_train]
                  test_preprocess_arr = np.c_[input_feature_preprocess_test, target_feature_test]

                  logging.info("Saving preprocessing object")

                  save_object(
                       file_path=self.preprocessor_path.preprocessor_obj_path,
                       obj= PreprocessorObj
                  )

                  logging.info("Saved preprocessing object")

                  return (
                    train_preprocess_arr,
                    test_preprocess_arr,
                    self.preprocessor_path.preprocessor_obj_path
                  )

             except Exception as e:
                  raise CustomException(e,sys)
             
             





     