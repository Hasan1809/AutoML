import json 
import os
import sys
import joblib
import shutil
from src.exception import AutoMLException

def save_object(file_path:str , obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path , exist_ok = True)
        with open(file_path , 'w') as file_obj:
            json.dump(obj, file_obj, indent=4, sort_keys=True)
    except Exception as e:
        raise AutoMLException(e, sys)

def load_object(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_pickle(file_path: str, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(obj, file_path)
    except Exception as e:
        raise AutoMLException(e, sys)

def load_pickle(file_path: str):
    try:
        return joblib.load(file_path)
    except Exception as e:
        raise AutoMLException(e, sys)

def copy_file(src: str, dst: str):
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    except Exception as e:
        raise AutoMLException(e, sys)
