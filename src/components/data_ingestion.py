import os
import sys
from pathlib import Path
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: Path = Path('artifacts/train.csv')
    test_data_path: Path = Path('artifacts/test.csv')
    raw_data_path: Path = Path('artifacts/data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info('Started data ingestion component')
        try:
            df = pd.read_csv('notebook/data/The_Cancer_data_1500_V2.csv')
            logging.info('Read the dataset as DataFrame')

            self.ingestion_config.train_data_path.parent.mkdir(exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info('Train test split initiated')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

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
    obj.initiate_data_ingestion()