import pandas as pd
from pathlib import Path

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path = Path('artifacts/train.csv')
    test_data_path = Path('artifacts/test.csv')
    raw_data_path = Path('artifacts/data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info('Initiated data ingestion component.')
        try:
            df = pd.read_csv('notebook/data/The_Cancer_data_1500_V2.csv')
            logging.info('Read dataset to DataFrame.')

            self.ingestion_config.train_data_path.parent.mkdir(exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info('Initiated train test split.')

            target = 'Diagnosis'
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target], shuffle=True)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False)

            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info('Data ingestion completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e)
        
if __name__ == '__main__':
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_path, test_path)

    model_trainer = ModelTrainer()
    f2score = model_trainer.initiate_model_trainer(train_arr, test_arr)
    print('F2 score of best model: ', f2score)