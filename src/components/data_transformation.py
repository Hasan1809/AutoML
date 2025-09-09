from src.entity.artifact_entity import DataValidationArtifact , DataTransformationArtifact
from src.entity.config_entity import DataTransformationConfig
from src.logger import logging
from src.exception import AutoMLException
import sys



class DataTransformation:
    def __init__(self, data_validation_artifact:DataValidationArtifact, data_transformation_config:DataTransformationConfig):
        self.data_validation_artifact = data_validation_artifact
        self.data_transformation_config = data_transformation_config
    
    def initiate_data_transformation(self):
        logging.info("Starting data transformation")
        
        