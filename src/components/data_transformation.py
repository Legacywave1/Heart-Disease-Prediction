import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj
import sys
import sqlite3
from dataclasses import dataclass
import os

@dataclass
class DataTransformationConfig:
    preprocessor_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    train_transformed_data: str = os.path.join('Data', 'transformed_data', 'train_heart_disease.parquet')
    test_transformed_data: str = os.path.join('Data', 'transformed_data', 'test_heart_disease.parquet')
class DataTransformation:
    def __init__(self):
        self.dtconfig = DataTransformationConfig()

    def commence_transformation(self, data_path):
        try:
            conn = sqlite3.connect(data_path)
            df = pd.read_sql('SELECT * FROM cardiac_records', conn)
            conn.close()

            df1 = df.copy()
            
            # Rename columns to match expected schema
            df1 = df1.rename(columns={
                'age': 'Age',
                'sex': 'Sex',
                'cp': 'ChestPainType',
                'trestbps': 'RestingBP',
                'chol': 'Cholesterol',
                'fbs': 'FastingBS',
                'restecg': 'RestingECG',
                'thalch': 'thalch',
                'exang': 'ExerciseAngina',
                'oldpeak': 'OldPeak',
                'slope': 'ST_Slope',
                'num': 'DiagnosisResult'
            })

            # Drop columns that may not exist
            cols_to_drop = [col for col in ['Patient_id', 'Data_source','Thalassemia','MajorVessels', 'id', 'dataset', 'ca', 'thal'] if col in df1.columns]
            df1 = df1.drop(cols_to_drop, axis=1)

            
            categorical_col = ['Sex', 'ChestPainType', 'RestingECG']
            df1[categorical_col] = df1[categorical_col].astype('category')
            bool_cols = ['ExerciseAngina', 'FastingBS']
            df1[bool_cols] = df1[bool_cols].apply(lambda x: pd.to_numeric(x, errors='coerce').notna().astype('boolean'))

            df1['Heart Rate Reserve'] = (220 - df1['Age']) - df1['thalch']

            conditions = [
                (df1['RestingBP'] < 120),
                (df1['RestingBP'].between(120, 129)),
                (df1['RestingBP'].between(130, 139)),
                (df1['RestingBP'] >= 140)
            ]

            choices = ['Normal', 'Elevated', 'Hypertension_Stage_1', 'Hypertension_Stage_2']

            df1['Blood Pressure Category'] = np.select(conditions, choices, default = 'unknown')


            age_bins = [0,39, 60, np.inf]
            age_labels = ['Young_Adult', 'Middle_Aged', 'Senior']

            df1['Age Category'] = pd.cut(df1['Age'], bins = age_bins, labels = age_labels, include_lowest = True)

            df1['DiagnosisResult'] = df1['DiagnosisResult'].apply(lambda x: 0 if x == 0 else 1)

            df1[['RestingBP', 'Cholesterol']] = df1[['RestingBP', 'Cholesterol']].replace({0: np.nan})

            df1['Cholesterol'] = np.log1p(df1['Cholesterol'])

            numerical = ['Age', 'RestingBP', 'Cholesterol', 'thalch', 'OldPeak', 'Heart Rate Reserve']
            cat_oh = [ 'RestingECG', 'ChestPainType', 'Blood Pressure Category']
            cat_oe = ['ST_Slope', 'Age Category', 'Sex', 'FastingBS', 'ExerciseAngina']

            slope_order = ['Up', 'Flat', 'Down']
            age_order = ['Young_Adult', 'Middle_Aged', 'Senior']
            sex_order = ['Female', 'Male']
            bool_order = [False, True]

            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(missing_values = np.nan, strategy ='median')),
                ('StandardScaler', StandardScaler())
                 ])

            cat_oh_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(missing_values = np.nan, strategy='most_frequent')),
                ('one hot', OneHotEncoder(handle_unknown = 'ignore', sparse_output = False))
                 ])

            cat_oe_pipeline = Pipeline(steps = [
                ('imputer', SimpleImputer(strategy = 'constant', fill_value = 'Unknown')),
                ('Ordinal encoding', OrdinalEncoder(categories=[slope_order, age_order, sex_order, bool_order, bool_order],handle_unknown = 'use_encoded_value', unknown_value = -1))
                ])

            preprocessor = ColumnTransformer([
                ('numerical_pipeline', numerical_pipeline, numerical),
                ('Categorical_one_hot_pipeline', cat_oh_pipeline, cat_oh),
                ('Categorical_ordinal_encoding', cat_oe_pipeline, cat_oe)

            ], remainder = 'drop')

            train_data, test_data = train_test_split(df1, test_size = 0.2, random_state= 42,stratify = df1['DiagnosisResult'])

            X_train = train_data.drop('DiagnosisResult', axis = 1)
            y_train = train_data['DiagnosisResult']
            X_test = test_data.drop('DiagnosisResult', axis = 1)
            y_test = test_data['DiagnosisResult']
            train_arr = preprocessor.fit_transform(X_train)
            test_arr = preprocessor.transform(X_test)
            cols = preprocessor.get_feature_names_out()
            os.makedirs('artifacts', exist_ok = True)
            save_obj(self.dtconfig.preprocessor_path,preprocessor)
            self._save_array(train_arr, y_train,cols, self.dtconfig.train_transformed_data)
            self._save_array(test_arr, y_test,cols, self.dtconfig.test_transformed_data)

            return(
                self.dtconfig.train_transformed_data,
                self.dtconfig.test_transformed_data
            )

        except Exception as e:
            raise CustomException(str(e), sys)

    def _save_array(self, X, y, cols, path):
        df = pd.DataFrame(X, columns=cols)
        df['DiagnosisResult'] = y.values
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path, index=False)

if __name__ == '__main__':
    preprocessing = DataTransformation()
    preprocessing.commence_transformation('Data/heart_disease.db')
