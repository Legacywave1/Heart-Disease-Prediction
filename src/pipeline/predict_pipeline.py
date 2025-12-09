import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any
import logging
import shap

logging.basicConfig(level=logging.INFO)

class PredictPipeline:
    def __init__(self):
        self.model_name = "Heart_Disease_XGBoost"
        self.preprocessor_path = Path("artifacts/preprocessor.pkl")
        self.preprocessor = self._load_preprocessor()
        self.model = self._load_best_available_model()

    def _load_preprocessor(self):
        if not self.preprocessor_path.exists():
            raise FileNotFoundError("Preprocessor not found at artifacts/preprocessor.pkl. Run training pipeline first.")
        return joblib.load(self.preprocessor_path)

    def _load_best_available_model(self):
        client = MlflowClient()

        # Try Production → Staging → Latest version
        try:
            model = mlflow.sklearn.load_model(f"models:/{self.model_name}/Production")
            logging.info("Loaded model: Production")
            return model
        except:
            pass

        try:
            model = mlflow.sklearn.load_model(f"models:/{self.model_name}/Staging")
            logging.info("Loaded model: Staging")
            return model
        except:
            pass

        # Fallback: get latest version by version number
        try:
            versions = client.search_model_versions(f"name='{self.model_name}'")
            if not versions:
                raise ValueError(f"No versions found for model '{self.model_name}'")
            latest = max(versions, key=lambda x: int(x.version))
            model = mlflow.sklearn.load_model(f"models:/{self.model_name}/{latest.version}")
            logging.info(f"Loaded model: Latest version {latest.version}")
            return model
        except Exception as e:
            raise RuntimeError(f"Could not load any version of {self.model_name}. Error: {e}")

    def recreate_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Heart Rate Reserve'] = (220 - df['Age']) - df['thalch']
        conditions = [
            (df['RestingBP'] < 120),
            (df['RestingBP'].between(120, 129)),
            (df['RestingBP'].between(130, 139)),
            (df['RestingBP'] >= 140)
        ]
        choices = ['Normal', 'Elevated', 'Hypertension_Stage_1', 'Hypertension_Stage_2']
        df['Blood Pressure Category'] = np.select(conditions, choices, default='unknown')

        df['Age Category'] = pd.cut(df['Age'], bins=[0, 39, 60, np.inf],
                                   labels=['Young_Adult', 'Middle_Aged', 'Senior'], include_lowest=True)
        return df

    def prepare_input(self, raw_data: Dict) -> pd.DataFrame:
        df = pd.DataFrame([raw_data])
        required = ['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS',
                    'RestingECG','thalch','ExerciseAngina','OldPeak','ST_Slope']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing fields: {missing}")

        df = self.recreate_engineered_features(df)
        final_cols = ['Age','RestingBP','Cholesterol','thalch','OldPeak','Heart Rate Reserve',
                      'Sex','ChestPainType','RestingECG','Blood Pressure Category',
                      'ST_Slope','Age Category','FastingBS','ExerciseAngina']
        return df[final_cols]

    def predict(self, patient_data: Dict) -> Dict[str, Any]:
        X = self.prepare_input(patient_data)
        X_processed = self.preprocessor.transform(X)

        # 1. Standard Prediction
        proba = float(self.model.predict_proba(X_processed)[0][1])
        prediction = int(proba >= 0.5)

        # 2. SHAP Calculation (The "Why" behind the prediction)
        # TreeExplainer is fast for XGBoost
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_processed)

        # Handle shape: XGBoost binary usually returns shape (n_samples, n_features)
        # We take the first (and only) row
        instance_shap_values = shap_values[0]

        feat_names = self.preprocessor.get_feature_names_out()

        # 3. Sort by ABSOLUTE impact (magnitude), but keep the sign
        # We zip names, values, and absolute values together
        feature_contributions = zip(feat_names, instance_shap_values, np.abs(instance_shap_values))

        # Sort by absolute value (highest impact first) and take top 5
        top_5 = sorted(feature_contributions, key=lambda x: x[2], reverse=True)[:5]

        def clean_feature_name(name: str) -> str:
            # ... (Keep your existing cleaning logic here) ...
            name = name.replace("Categorical_one_hot_pipeline__", "")
            name = name.replace("Categorical_ordinal_encoding__", "")
            name = name.replace("numerical_pipeline__", "")
            name = name.replace("Chestpaintype", " Pain Type")
            name = name.replace("Thalach", "Max Heart Rate")
            name = name.replace("Oldpeak", "ST Depression")
            name = name.replace("Restingbp", "Resting BP")
            name = name.replace("Fastingbs", "Fasting Blood Sugar >120")
            name = name.replace("Exerciseangina", "Exercise Angina")
            name = name.replace("St Slope", "ST Slope")
            name = name.replace("_", " ").title()
            return name

        top_features = []
        for name, shap_val, abs_val in top_5:
            # Determine direction for UI
            impact_type = "increase" if shap_val > 0 else "decrease"

            top_features.append({
                "feature": clean_feature_name(name),
                "importance": round(float(abs_val), 4), # Magnitude for the bar width
                "raw_shap": round(float(shap_val), 4),  # Actual value for debugging
                "impact": impact_type                   # "increase" = Red (Bad), "decrease" = Green (Good)
            })

        return {
            "risk_level": "High Risk" if prediction else "Low Risk",
            "risk_probability": round(proba, 4),
            "top_contributors": top_features,
            "recommendation": "Kindly treat as urgent — consult cardiologist immediately" if prediction else "Low risk — continue healthy lifestyle",
            "model_status": "Production" if "Production" in str(self.model) else "Staging/Latest"
        }


if __name__ == "__main__":
    predictor = PredictPipeline()
    sample = {
        "Age": 55,
        "Sex": "Male",
        "ChestPainType": "NAP",
        "RestingBP": 140,
        "Cholesterol": 217,
        "FastingBS": 0,
        "RestingECG": "Normal",
        "thalch": 111,
        "ExerciseAngina": "Y",
        "OldPeak": 5.6,
        "ST_Slope": "Flat"
    }
    print(predictor.predict(sample))
