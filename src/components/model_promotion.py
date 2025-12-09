# src/components/model_promotion.py
import mlflow
from mlflow.tracking import MlflowClient
from src.logger import logging
from src.exception import CustomException
import sys

class ModelPromotion:
    def __init__(self):
        self.client = MlflowClient()
        self.staging = "Staging"
        self.production = "Production"
        self.threshold = 0.70
        self.min_improvement = 0.01

    def promote(self, model_name: str, run_id: str, score: float):
        versions = self.client.search_model_versions(f"run_id='{run_id}'")
        if not versions:
            logging.error("No model version found for run")
            return False

        version = versions[0].version

        # Always go to Staging
        self.client.set_registered_model_alias(model_name, self.staging, version)

        # Check if good enough for Production
        prod = None
        try:
            prod = self.client.get_model_version_by_alias(model_name, self.production)
            current_score = self.client.get_run(prod.run_id).data.metrics.get("test_accuracy", 0)
            if score - current_score < self.min_improvement:
                logging.info(f"Staging only: improvement {score-current_score:.4f} < {self.min_improvement}")
                return True
        except:
            pass  # No prod model yet

        if score >= self.threshold:
            self.client.set_registered_model_alias(model_name, self.production, version)
            logging.info(f"PROMOTED TO PRODUCTION: v{version} | Score: {score:.4f}")
        else:
            logging.info(f"Staging only: score {score:.4f} < threshold")

        return True
