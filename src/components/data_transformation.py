from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_path = Path('artifacts/preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_preprocessor(self):
        cat_features= ['Gender', 'Smoking', 'GeneticRisk', 'CancerHistory']
        num_features = ['Age', 'BMI', 'PhysicalActivity', 'AlcoholIntake']

        try:
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            logging.info('Numerical features pipeline created.')

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent'))
                ]
            )
            logging.info('Categorical features pipeline created.')

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, num_features),
                    ('cat_pipeline', cat_pipeline, cat_features)
                ],
                remainder='passthrough'
            )
        except Exception as e:
            raise CustomException(e)

        return preprocessor
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info('Initiated data transformation component.')
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Train and test data loaded.')

            logging.info('Creating preprocessor.')

            preprocessor = self.get_preprocessor()

            logging.info('Created preprocessor.')

            target = 'Diagnosis'

            X_train = train_df.drop(target, axis=1)
            y_train = train_df[target]

            X_test = test_df.drop(target, axis=1)
            y_test = test_df[target]

            logging.info('Applying preprocessor on train and test data.')

            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)

            train_arr = np.column_stack((X_train, y_train))
            test_arr = np.column_stack((X_test, y_test))

            logging.info('Preprocessed train and test data.')

            logging.info('Saving the preprocessor object.')

            save_object(preprocessor, self.data_transformation_config.preprocessor_obj_path)

            logging.info(f'Saved preprocessor object to {self.data_transformation_config.preprocessor_obj_path}')

            logging.info('Data transformation completed.')

        except Exception as e:
            raise CustomException(e)
            
        return(
            train_arr,
            test_arr,
            self.data_transformation_config.preprocessor_obj_path
        )