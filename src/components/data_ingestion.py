import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# from src.components.data_transformation import DataTransformationConfig, DataTransformation
# from src.components.model_trainer import ModelTrainerConfig, ModelTrainer



@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join("artifacts", 'train.csv')
    test_data_path:str = os.path.join("artifacts", 'test.csv')
    raw_data_path:str = os.path.join("artifacts", 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion method or component")
        try:
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info("Read dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("train test split initiated")
            train_data, test_data = train_test_split(df, test_size=0.2,random_state=42)

            train_data.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            test_data.to_csv(self.ingestion_config.test_data_path, index=False, header = True)

            logging.info('data ingestion is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )  
       
        except Exception as e:
            raise CustomException(e, sys)
        


# if __name__ =="__main__":
#     ing_obj = DataIngestion()
#     train_path, test_path = ing_obj.initiate_data_ingestion()

#     trns_obj = DataTransformation()
#     train_arr, test_arr, _ = trns_obj.InitiateDataTransformation(train_path, test_path)

#     model_train_obj = ModelTrainer()
#     model_train_obj.initiate_model_training(train_arr, test_arr)

