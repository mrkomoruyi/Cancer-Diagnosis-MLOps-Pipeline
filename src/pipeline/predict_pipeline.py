import pandas as pd

from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor = load_object('artifacts/preprocessor.pkl')
            model = load_object('artifacts/model.pkl')
            
            data_transformed = preprocessor.transform(features)
            y_pred = model.predict(data_transformed)
            
            return y_pred

        except Exception as e:
            raise CustomException(e)

class CustomData:
    def __init__(self, age: float, gender: int, bmi: float, smoking: int, 
                 genetic_risk: int, physical_activity: float, 
                 alcohol_intake: float, cancer_history: int):
        self.age = age
        self.gender = gender
        self.bmi = bmi
        self.smoking = smoking
        self.genetic_risk = genetic_risk
        self.physical_activity = physical_activity
        self.alcohol_intake = alcohol_intake
        self.cancer_history = cancer_history

    def get_data_as_dataframe(self):
        try:
            data_input_dict = {
                'Age': [self.age],
                'Gender': [self.gender],
                'BMI': [self.bmi],
                'Smoking': [self.smoking],
                'GeneticRisk': [self.genetic_risk],
                'PhysicalActivity': [self.physical_activity],
                'AlcoholIntake': [self.alcohol_intake],
                'CancerHistory': [self.cancer_history]
            }

            return pd.DataFrame(data_input_dict)

        except Exception as e:
            raise CustomException(e)