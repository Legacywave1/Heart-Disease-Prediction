import os
import sqlite3
import pandas as pd
import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

@dataclass
class IngestDataConfig:
    raw_data: str = os.path.join('Data', 'heart_disease.db')

class IngestData:
    def __init__(self):
        self.rawData = IngestDataConfig()
        self.path = r'Data\raw_data\heart_disease_uci.csv'

    def initiate_data_ingestion(self, path = None):
        try:
            if path is None:
                path = self.path
            df = pd.read_csv(path)
            df = df.rename(columns={
                'id': 'Patient_id',
                'age': 'Age',
                'sex': 'Sex',
                'dataset': 'Data_source',
                'cp': 'ChestPainType',        # 0-3 scale
                'trestbps': 'RestingBP',      # Blood Pressure
                'chol': 'Cholesterol',
                'fbs': 'FastingBS',           # Fasting Blood Sugar > 120
                'restecg': 'RestingECG',
                'thalach': 'MaxHeartRate',
                'exang': 'ExerciseAngina',    # 1 = Yes, 0 = No
                'oldpeak': 'OldPeak',         # ST depression
                'slope': 'ST_Slope',
                'ca': 'MajorVessels',         # Number of vessels colored by flourosopy
                'thal': 'Thalassemia',
                'num': 'DiagnosisResult'
            })
        except Exception as e:
           CustomException(str(e),sys)
        os.makedirs('Data', exist_ok = True)
        logging.info(f'Updating Knowledge Base at {self.rawData.raw_data}')
        try:
            TABLE_NAME = 'cardiac_records'
            conn = sqlite3.connect(self.rawData.raw_data)

            df.to_sql(TABLE_NAME, conn, if_exists='replace', index = False)
            conn.close()

            logging.info('Success! The Heart Disease Knowledge Base is ready.')
            logging.info(f'{self.rawData.raw_data} - Database File')

        except Exception as e:
            CustomException(str(e), sys)

        return self.rawData.raw_data

if __name__ == '__main__':
    ingest_data_obj = IngestData()
    ingest_data_obj.initiate_data_ingestion()
