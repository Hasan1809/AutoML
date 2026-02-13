import os
import pandas as pd
from src.utils import load_pickle
from src.logger import logging
from src.exception import AutoMLException


class Predictor:
    def __init__(self, model_path: str, preprocessor_path: str, target: str):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.target = target
        self.model = None
        self.preprocessor = None

    def _load(self):
        if self.model is None:
            self.model = load_pickle(self.model_path)
        if self.preprocessor is None:
            self.preprocessor = load_pickle(self.preprocessor_path)

    def predict_dataframe(self, df: pd.DataFrame):
        try:
            self._load()
            if self.target in df.columns:
                df = df.drop(columns=[self.target])
            transformed = self.preprocessor.transform(df)
            preds = self.model.predict(transformed)
            return preds
        except Exception as e:
            raise AutoMLException(e, None)

    def predict_csv(self, csv_path: str):
        try:
            logging.info(f"Running prediction for {csv_path}")
            df = pd.read_csv(csv_path)
            return self.predict_dataframe(df)
        except Exception as e:
            raise AutoMLException(e, None)
