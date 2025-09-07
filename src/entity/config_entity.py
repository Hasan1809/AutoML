from datetime import datetime
import os 
from src.constant import *

class PipelineConfig:
    def __init__(self, timestamp=datetime.now()):
        # Format: DD_MM_YYYY_HH_MM
        timestamp = timestamp.strftime("%d_%m_%Y_%H_%M")
        self.artifact_name = ARTIFACT_DIR
        self.artifact_dir = os.path.join(self.artifact_name, timestamp)
        self.timestamp: str = timestamp
    

class DataIngestionConfig:
    def __init__(self,pipeline_config:PipelineConfig):
        self.data_ingestion_dir:str=os.path.join(
            pipeline_config.artifact_dir, DATA_INGESTION_DIR
        )
        self.feature_store_file_path: str = os.path.join(
                self.data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, DATA_INGESTION_FEATURE_STORE_FILE_NAME
            )
        self.training_file_path: str = os.path.join(
                self.data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME
            )
        self.testing_file_path: str = os.path.join(
                self.data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME
            )
        self.train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO

class DataValidationConfig:
    def __init__(self, pipeline_config:PipelineConfig):
        self.data_validation_dir = os.path.join(pipeline_config.artifact_dir, DATA_VALIDATION_DIR)
        self.report_file_path = os.path.join(self.data_validation_dir, "report.yaml")
        self.missing_threshold:float = MISSING_THRESHOLD