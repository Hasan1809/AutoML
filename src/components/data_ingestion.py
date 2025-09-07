from src.entity.config_entity import PipelineConfig, DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception import AutoMLException
from src.logger import logging
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import os


class DataIngestion:
    def __init__(self, data_ingestion_config:DataIngestionConfig , data_file_path:str):
        self.data_ingestion_config = data_ingestion_config
        self.data_file_path = data_file_path
    
    def initiate_data_ingestion(self)->DataIngestionArtifact:
        try:
            logging.info("Initiating data ingestion")
            #reading the dataset
            df = pd.read_csv(self.data_file_path)
            logging.info(f"Read the dataset as dataframe {df.shape}")

            #creating feature store folder
            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir, exist_ok=True)

            #saving the df to feature store folder
            df.to_csv(self.data_ingestion_config.feature_store_file_path, index=False)
            logging.info(f"Saved the df to feature store folder {self.data_ingestion_config.feature_store_file_path}")

            #splitting the dataset into train and test set
            train_set, test_set = train_test_split(df, test_size=self.data_ingestion_config.train_test_split_ratio, random_state=42)

            #creating dataset directory folder
            dataset_dir = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dataset_dir, exist_ok=True)

            #saving the train and test set to dataset directory
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)

            logging.info(f"Saved the train and test set to dataset directory {self.data_ingestion_config.training_file_path} and {self.data_ingestion_config.testing_file_path}")

            #prepare artifact
            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path,
                status=True
            )
            logging.info(f"Data Ingestion artifact: {data_ingestion_artifact}")
            
            return data_ingestion_artifact
        except Exception as e:
            raise AutoMLException(e, sys)