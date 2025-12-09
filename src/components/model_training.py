import optuna
import os
import sys
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score
from dataclasses import dataclass
from src.utils import save_obj
from pathlib import Path
import shutil
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ModelTrainingConfig:
    model_path: str = os.path.join('artifacts', 'trained_model.pkl')
    mlflow_dir: str = 'mlrun'

class ModelTraining:
    def __init__(self):
        self.config = ModelTrainingConfig()
        mlflow.set_experiment('Heart Disease Predictor')

    def objective(self, trial, X, y):
        params = {
                    "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
                    "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
                    "random_state": 42,
                    "eval_metric": "logloss",
                    "tree_method": "hist",
                    "verbosity": 0
                }
        model = XGBClassifier(**params)
        model.fit(X,y)
        return accuracy_score(y, model.predict(X))

    def initiate_train(self, train_path, test_path):
        with mlflow.start_run(run_name = 'XGBoost Final Training') as run:
            self.run_id = run.info.run_id
            test = pd.read_parquet(test_path)
            train = pd.read_parquet(train_path)
            X_train = train.drop('DiagnosisResult',axis = 1)
            y_train = train['DiagnosisResult']
            X_test = test.drop('DiagnosisResult', axis = 1)
            y_test = test['DiagnosisResult']


            study = optuna.create_study(direction = 'maximize')
            study.optimize(lambda trial: self.objective(trial, X_train, y_train), n_trials = 20)
            best_model = XGBClassifier(**study.best_params, random_state = 42, eval_metric = 'logloss')
            best_model.fit(X_train,y_train)
            test_acc = accuracy_score(y_test, best_model.predict(X_test))
            mlflow.log_metric('test_accuracy', test_acc)
            mlflow.log_params(study.best_params)


            importances = best_model.feature_importances_
            feature_names = X_train.columns
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending = False)
            importance_df.to_csv('artifacts/feature_importance.csv', index = False)
            importance_df.head(20).to_json('artifacts/feature_importance.json', orient = 'records')
            save_obj(self.config.model_path, best_model)
            mlflow.sklearn.log_model(sk_model = best_model, artifact_path='model', registered_model_name = "Heart_Disease_XGBoost", input_example = X_train.head(1))



            return run.info.run_id, test_acc



if __name__ == '__main__':
    train_obj = ModelTraining()
    result = train_obj.initiate_train(
        r'C:\Users\Ankuc\Documents\Heart Disease Prediction\Data\transformed_data\train_heart_disease.parquet',
        r'C:\Users\Ankuc\Documents\Heart Disease Prediction\Data\transformed_data\test_heart_disease.parquet')
