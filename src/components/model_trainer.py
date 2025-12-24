from dataclasses import dataclass

import logging
import pandas as pd
from pathlib import Path

from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import fbeta_score

from src.exception import CustomException
from src.utils import evaluate_models, save_object

@dataclass
class ModelTrainerConfig:
    trained_model_path = Path('artifacts/model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Initiated model trainer component.')

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            scale_pos_weight = sum(y_train == 0)/sum(y_train == 1)
            seed = 0
            models = {
                'CatBoostClassifier': CatBoostClassifier(verbose=False, random_seed=seed),
                'CatBoostClassifier (scale_pos_weight)': CatBoostClassifier(verbose=False, scale_pos_weight=scale_pos_weight, random_state=seed),
                'RandomForestClassifier': RandomForestClassifier(verbose=False, random_state=seed),
                'RandomForestClassifier (Balanced class_weight)': RandomForestClassifier(verbose=False, class_weight='balanced', random_state=seed),
                'AdaBoostClassifier': AdaBoostClassifier(random_state=seed),
            }
            
            model_report = evaluate_models(X_train, y_train, X_test, y_test, models)
            model_report = pd.DataFrame(model_report).sort_values(by='f2score', axis=1, ascending=False).T

            best_model_name = model_report.index[0]
            best_model = models[best_model_name]
            best_model_score = model_report.loc[best_model_name, 'f2score']

            if best_model_score < 0.6:
                raise CustomException('No best model found.')
            logging.info('Best model found.')

            save_object(best_model, self.model_trainer_config.trained_model_path)
            logging.info(f'Saved best model to {self.model_trainer_config.trained_model_path}')

            predicted = best_model.predict(X_test)
            f2score = fbeta_score(y_test, predicted, beta=2.0)

        except Exception as e:
            raise CustomException(e)
        
        return f2score