import argparse
import os
import sys
import yaml
from src.pipeline.predictor import Predictor
from src.entity.config_entity import PipelineConfig, ModelTrainerConfig
from src.exception import AutoMLException


def load_settings(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Run batch predictions using the latest registered model.")
    parser.add_argument("--config", default=os.path.join("config", "config.yaml"), help="Path to config YAML.")
    parser.add_argument("--data", required=True, help="Path to CSV with inference data.")
    args = parser.parse_args()

    settings = load_settings(args.config)
    target = settings.get("target")

    pipeline_config = PipelineConfig()
    model_trainer_config = ModelTrainerConfig(pipeline_config)

    predictor = Predictor(
        model_path=model_trainer_config.registry_model_path,
        preprocessor_path=model_trainer_config.registry_preprocessor_path,
        target=target,
    )

    preds = predictor.predict_csv(args.data)
    for i, p in enumerate(preds[:10]):
        print(f"row {i}: {p}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise AutoMLException(e, sys)
