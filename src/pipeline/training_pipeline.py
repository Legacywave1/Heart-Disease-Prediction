import os
import sys
from src.logger import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.components.data_ingestion import IngestData
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTraining
from src.components.model_promotion import ModelPromotion
from src.exception import CustomException

class TrainPipeline:
    def __init__(self):
        self.ingestor = IngestData()
        self.transformer = DataTransformation()
        self.trainer = ModelTraining()
        self.promoter = ModelPromotion()

    def run_pipeline(self):
        try:
            logging.info("Starting Full Training Pipeline")
            db_path = self.ingestor.initiate_data_ingestion()

            train_path, test_path = self.transformer.commence_transformation(db_path)

            run_id, test_accuracy = self.trainer.initiate_train(train_path, test_path)

            success = self.promoter.promote(model_name = 'Heart_Disease_XGBoost', run_id= run_id,  score = test_accuracy)

            if success:
                logging.info('Model promoted to Production')
            else:
                logging.error('Promotion Failed')
            logging.info("FULL PIPELINE COMPLETED SUCCESSFULLY!")
            print("\n" + "="*60)
            print("   HEART DISEASE PREDICTION PIPELINE FINISHED")
            print("="*60)
            print(f"   Test Accuracy : {test_accuracy:.1%}")
            print("="*60)

        except Exception as e:
            logging.error("Pipeline failed!")
            raise CustomException(e, sys)

if __name__ == '__main__':
    pipeline = TrainPipeline()
    pipeline.run_pipeline()
