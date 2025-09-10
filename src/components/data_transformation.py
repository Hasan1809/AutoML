from src.entity.artifact_entity import DataValidationArtifact , DataTransformationArtifact
from src.entity.config_entity import DataTransformationConfig
from src.logger import logging
from src.exception import AutoMLException
from src.utils import load_object , save_object
import sys



class DataTransformation:
    def __init__(self, data_validation_artifact:DataValidationArtifact, data_transformation_config:DataTransformationConfig):
        self.data_validation_artifact = data_validation_artifact
        self.data_transformation_config = data_transformation_config
    
    def initiate_data_transformation(self):
        try:
            logging.info("Starting data transformation")
            validation_report = load_object(self.data_validation_artifact.report_file_path)
        except Exception as e:
            raise AutoMLException(e, sys)