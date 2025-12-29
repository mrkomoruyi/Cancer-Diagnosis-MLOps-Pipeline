import sys
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        """Run the training pipeline.
        
        Returns:
            float: F2 score of the best model.
        """
        try:
            # Step 1: Data Ingestion
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

            # Step 2: Data Transformation
            data_transformation = DataTransformation()
            train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

            # Step 3: Model Training
            model_trainer = ModelTrainer()
            f2score = model_trainer.initiate_model_trainer(train_arr, test_arr)

            return f2score

        except Exception as e:
            raise CustomException(e, sys.exc_info())

if __name__ == "__main__":
    try:
        train_pipeline = TrainPipeline()
        f2_score = train_pipeline.run_pipeline()
        print(f"Training completed successfully. Best Model F2 Score: {f2_score:.4f}")
    except Exception as e:
        print(f"Training failed: {e}")
