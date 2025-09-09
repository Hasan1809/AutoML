import json 
import os
from src.exception import AutoMLException
import sys

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