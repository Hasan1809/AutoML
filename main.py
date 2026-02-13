import os
import sys
import yaml
from typing import Optional
from src.logger import logging
from src.exception import AutoMLException
from src.entity.config_entity import *
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def load_settings(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_pipeline(config_path: str = os.path.join("config", "config.yaml"),
                 override_data_path: Optional[str] = None,
                 override_target: Optional[str] = None,
                 override_problem_type: Optional[str] = None):
    settings = load_settings(config_path)
    target = override_target or settings.get("target")
    problem_type = override_problem_type or settings.get("problem_type")
    data_path = override_data_path or settings.get("data_path")

    pipeline_config = PipelineConfig()
    data_ingestion_config = DataIngestionConfig(pipeline_config=pipeline_config)
    data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config, data_file_path=data_path)
    data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

    data_validation_config = DataValidationConfig(pipeline_config)
    data_validation = DataValidation(
        data_ingestion_artifact=data_ingestion_artifact,
        data_validatoin_config=data_validation_config,
        target=target,
        problem_type=problem_type,
    )
    data_validation_artifact = data_validation.initiate_data_validation()

    data_transformation_config = DataTransformationConfig(pipeline_config)
    data_transformation = DataTransformation(
        data_validation_artifact=data_validation_artifact,
        data_transformation_config=data_transformation_config,
    )
    data_transformation_artifact = data_transformation.initiate_data_transformation()

    model_trainer_config = ModelTrainerConfig(pipeline_config)
    model_trainer = ModelTrainer(
        model_trainer_config=model_trainer_config,
        data_transformation_artifact=data_transformation_artifact,
        data_validation_artifact=data_validation_artifact,
    )
    model_trainer_artifact = model_trainer.initiate_model_trainer()
    return model_trainer_artifact


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        raise AutoMLException(e, sys)
