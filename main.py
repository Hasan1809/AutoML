from src.logger import logging
from src.exception import AutoMLException
from src.entity.config_entity import PipelineConfig, DataIngestionConfig
from src.components.data_ingestion import DataIngestion
import sys


try:
    pipeline_config = PipelineConfig()
    data_ingestion_config = DataIngestionConfig(pipeline_config=pipeline_config)
    data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config, data_file_path=r"C:\Users\hasan\End to end\AutoML\test_data\Titanic-Dataset.csv")
    data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
except Exception as e:
    raise AutoMLException(e, sys)