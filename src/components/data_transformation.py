import os
import sys
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from src.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from src.entity.config_entity import DataTransformationConfig
from src.logger import logging
from src.exception import AutoMLException
from src.utils import load_object, save_pickle


class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact, data_transformation_config: DataTransformationConfig):
        self.data_validation_artifact = data_validation_artifact
        self.data_transformation_config = data_transformation_config
    
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Starting data transformation")
            report = load_object(self.data_validation_artifact.report_file_path)

            if not report.get("status", False):
                raise ValueError("Data validation status is False; cannot proceed with transformation.")

            target_column = report.get("target_analysis", {}).get("name")
            if not target_column:
                raise ValueError("Target column name not found in validation report.")

            actions = report.get("actions", {})
            drop_columns = actions.get("drop_columns", [])
            encoding_actions = actions.get("encoding", {})
            one_hot_cols = encoding_actions.get("one_hot", [])
            label_encode_cols = encoding_actions.get("label_encode", [])
            remove_duplicates = actions.get("remove_duplicates", False)
            column_analysis = report.get("column_analysis", {})

            train_df = pd.read_csv(self.data_validation_artifact.train_file_path)
            test_df = pd.read_csv(self.data_validation_artifact.test_file_path)

            if remove_duplicates:
                train_df = train_df.drop_duplicates()
                test_df = test_df.drop_duplicates()

            drop_columns = [c for c in drop_columns if c != target_column]
            for col in drop_columns:
                if col in train_df.columns:
                    train_df = train_df.drop(columns=col)
                if col in test_df.columns:
                    test_df = test_df.drop(columns=col)

            if target_column not in train_df.columns or target_column not in test_df.columns:
                raise ValueError(f"Target column '{target_column}' not present in train/test data.")

            y_train = train_df[target_column].reset_index(drop=True)
            y_test = test_df[target_column].reset_index(drop=True)
            X_train = train_df.drop(columns=[target_column])
            X_test = test_df.drop(columns=[target_column])

            available_cols = [c for c in X_train.columns if c in column_analysis]
            categorical_like = ("object", "category", "string", "str")
            numeric_cols = [
                c for c in available_cols
                if not any(t in column_analysis[c]["dtype"] for t in categorical_like)
            ]
            categorical_cols = [c for c in available_cols if c not in numeric_cols]

            one_hot_cols = [c for c in one_hot_cols if c in X_train.columns]
            label_encode_cols = [c for c in label_encode_cols if c in X_train.columns]
            other_cat_cols = [c for c in categorical_cols if c not in one_hot_cols and c not in label_encode_cols]
            label_encode_cols = label_encode_cols + other_cat_cols

            transformers = []
            if numeric_cols:
                transformers.append(
                    ("numeric", SimpleImputer(strategy="median"), numeric_cols)
                )
            if one_hot_cols:
                transformers.append(
                    (
                        "one_hot",
                        Pipeline(
                            steps=[
                                ("imputer", SimpleImputer(strategy="most_frequent")),
                                ("encoder", OneHotEncoder(handle_unknown="ignore")),
                            ]
                        ),
                        one_hot_cols,
                    )
                )
            if label_encode_cols:
                transformers.append(
                    (
                        "label_encode",
                        Pipeline(
                            steps=[
                                ("imputer", SimpleImputer(strategy="most_frequent")),
                                (
                                    "encoder",
                                    OrdinalEncoder(
                                        handle_unknown="use_encoded_value",
                                        unknown_value=-1,
                                    ),
                                ),
                            ]
                        ),
                        label_encode_cols,
                    )
                )

            if not transformers:
                raise ValueError("No transformers configured; check validation report and data types.")

            preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
            preprocessor.fit(X_train)

            train_array = preprocessor.transform(X_train)
            test_array = preprocessor.transform(X_test)

            feature_names = preprocessor.get_feature_names_out()
            transformed_train_df = pd.DataFrame(train_array, columns=feature_names)
            transformed_test_df = pd.DataFrame(test_array, columns=feature_names)

            transformed_train_df[target_column] = y_train.values
            transformed_test_df[target_column] = y_test.values

            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessing_obj), exist_ok=True)

            transformed_train_df.to_csv(self.data_transformation_config.transformed_train, index=False)
            transformed_test_df.to_csv(self.data_transformation_config.transformed_test, index=False)
            save_pickle(self.data_transformation_config.preprocessing_obj, preprocessor)

            logging.info(f"Saved transformed train to {self.data_transformation_config.transformed_train}")
            logging.info(f"Saved transformed test to {self.data_transformation_config.transformed_test}")
            logging.info(f"Saved preprocessing object to {self.data_transformation_config.preprocessing_obj}")

            return DataTransformationArtifact(
                transformed_train=self.data_transformation_config.transformed_train,
                transformed_test=self.data_transformation_config.transformed_test,
                preprocessor_path=self.data_transformation_config.preprocessing_obj,
            )
        except Exception as e:
            raise AutoMLException(e, sys)
