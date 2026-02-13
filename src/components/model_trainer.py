import os
import sys
from itertools import product
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, r2_score, mean_squared_error
from src.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact, ModelTrainerArtifact
from src.entity.config_entity import ModelTrainerConfig
from src.exception import AutoMLException
from src.logger import logging
from src.utils import load_object, save_object, save_pickle, copy_file


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact, data_validation_artifact: DataValidationArtifact):
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifact = data_transformation_artifact
        self.data_validation_artifact = data_validation_artifact

    def _get_models_and_params_classification(self):
        models = {
            "logistic_regression": LogisticRegression(max_iter=2000),
            "random_forest": RandomForestClassifier(),
            "gradient_boosting": GradientBoostingClassifier(),
        }
        param_grid = {
            "logistic_regression": [
                {"C": [0.1, 1.0, 10.0], "solver": ["lbfgs"]},
            ],
            "random_forest": [
                {"n_estimators": [100, 200], "max_depth": [None, 5, 10]},
            ],
            "gradient_boosting": [
                {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [2, 3]},
            ],
        }
        return models, param_grid

    def _get_models_and_params_regression(self):
        models = {
            "linear_regression": LinearRegression(),
            "random_forest_regressor": RandomForestRegressor(),
            "gradient_boosting_regressor": GradientBoostingRegressor(),
        }
        param_grid = {
            "linear_regression": [{}],
            "random_forest_regressor": [
                {"n_estimators": [100, 200], "max_depth": [None, 5, 10]},
            ],
            "gradient_boosting_regressor": [
                {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [2, 3]},
            ],
        }
        return models, param_grid

    def _expand_param_grid(self, grid: List[Dict]) -> List[Dict]:
        expanded: List[Dict] = []
        for params in grid:
            if not params:
                expanded.append({})
                continue
            keys = list(params.keys())
            values = [v if isinstance(v, list) else [v] for v in params.values()]
            for combo in product(*values):
                expanded.append(dict(zip(keys, combo)))
        return expanded

    def _select_best_model(self, metrics: List[Dict], scoring_key: str) -> Dict:
        return max(metrics, key=lambda m: m["metrics"][scoring_key])

    def _evaluate_classification(self, model, X_test, y_test) -> Dict[str, float]:
        preds = model.predict(X_test)
        return {
            "accuracy": accuracy_score(y_test, preds),
            "f1_weighted": f1_score(y_test, preds, average="weighted"),
            "precision_weighted": precision_score(y_test, preds, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_test, preds, average="weighted"),
        }

    def _evaluate_regression(self, model, X_test, y_test) -> Dict[str, float]:
        preds = model.predict(X_test)
        rmse = mean_squared_error(y_test, preds, squared=False)
        return {
            "r2": r2_score(y_test, preds),
            "rmse": rmse,
        }

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Starting model training")
            report = load_object(self.data_validation_artifact.report_file_path)
            target_column = report.get("target_analysis", {}).get("name")
            problem_type = report.get("target_analysis", {}).get("type", "categorical")

            if not target_column:
                raise ValueError("Target column name not found in validation report.")

            train_df = pd.read_csv(self.data_transformation_artifact.transformed_train)
            test_df = pd.read_csv(self.data_transformation_artifact.transformed_test)

            if target_column not in train_df.columns or target_column not in test_df.columns:
                raise ValueError(f"Target column '{target_column}' not found in transformed data.")

            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            if problem_type == "categorical":
                models, param_grid = self._get_models_and_params_classification()
                scoring_key = "f1_weighted"
            else:
                models, param_grid = self._get_models_and_params_regression()
                scoring_key = "r2"

            model_results: List[Dict] = []

            for model_name, model in models.items():
                grid = self._expand_param_grid(param_grid.get(model_name, [{}]))
                best_model = None
                best_metrics = None
                best_params = None

                for params in grid:
                    candidate = clone(model)
                    candidate.set_params(**params)
                    candidate.fit(X_train, y_train)

                    if problem_type == "categorical":
                        metrics = self._evaluate_classification(candidate, X_test, y_test)
                    else:
                        metrics = self._evaluate_regression(candidate, X_test, y_test)

                    if best_metrics is None or metrics[scoring_key] > best_metrics[scoring_key]:
                        best_model = candidate
                        best_metrics = metrics
                        best_params = params

                model_results.append(
                    {
                        "name": model_name,
                        "best_params": best_params,
                        "metrics": best_metrics,
                    }
                )
                logging.info(f"Completed training for {model_name} with best {scoring_key}: {best_metrics[scoring_key]:.4f}")

            if not model_results:
                raise ValueError("No models were trained.")

            best_model_entry = self._select_best_model(model_results, scoring_key)
            best_model_name = best_model_entry["name"]
            best_model_params = best_model_entry["best_params"]

            # Refit best model on full training data
            final_model = clone(models[best_model_name])
            final_model.set_params(**best_model_params)
            final_model.fit(X_train, y_train)

            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.model_trainer_config.metrics_file_path), exist_ok=True)

            save_pickle(self.model_trainer_config.trained_model_file_path, final_model)

            report_payload = {
                "status": True,
                "problem_type": problem_type,
                "target": target_column,
                "scoring_key": scoring_key,
                "models": model_results,
                "best_model": {
                    "name": best_model_name,
                    "best_params": best_model_params,
                    "metrics": best_model_entry["metrics"],
                    "trained_model_path": self.model_trainer_config.trained_model_file_path,
                },
            }

            save_object(self.model_trainer_config.metrics_file_path, report_payload)

            logging.info(f"Saved trained model to {self.model_trainer_config.trained_model_file_path}")
            logging.info(f"Saved model report to {self.model_trainer_config.metrics_file_path}")

            # Promote to registry/latest for inference
            copy_file(self.model_trainer_config.trained_model_file_path, self.model_trainer_config.registry_model_path)
            copy_file(self.data_transformation_artifact.preprocessor_path, self.model_trainer_config.registry_preprocessor_path)
            copy_file(self.model_trainer_config.metrics_file_path, self.model_trainer_config.registry_report_path)

            logging.info(f"Promoted model to registry at {self.model_trainer_config.registry_model_path}")
            logging.info(f"Promoted preprocessor to registry at {self.model_trainer_config.registry_preprocessor_path}")
            logging.info(f"Promoted model report to registry at {self.model_trainer_config.registry_report_path}")

            return ModelTrainerArtifact(
                trained_model_path=self.model_trainer_config.trained_model_file_path,
                metrics_path=self.model_trainer_config.metrics_file_path,
                best_model_name=best_model_name,
                best_score=best_model_entry["metrics"][scoring_key],
            )
        except Exception as e:
            raise AutoMLException(e, sys)
