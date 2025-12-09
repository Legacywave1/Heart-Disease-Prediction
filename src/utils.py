import os
import joblib

def save_obj(file_path, model):
   os.makedirs(os.path.dirname(file_path), exist_ok = True)
   joblib.dump(model, file_path)
