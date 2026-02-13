from src.entity.config_entity import PipelineConfig , DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.exception import AutoMLException
from src.logger import logging
import sys
import pandas as pd
import os
from src.utils import save_object


class DataValidation:
    def __init__(self, data_validatoin_config: DataValidationConfig, data_ingestion_artifact: DataIngestionArtifact , target: str , problem_type: str):
        self.data_validation_config = data_validatoin_config
        self.data_ingestion_artifact = data_ingestion_artifact
        self.target = target
        self.problem_type = problem_type
        self.report = {
            "status": True,
            "dataset_summary": {},
            "target_analysis" : {},
            "column_analysis": {},
            "actions": {}
        }
    
    def check_size(self):
        
        size_mb = os.path.getsize(self.data_ingestion_artifact.feature_store_file_path)/(1024 * 1024)
        if size_mb > self.data_validation_config.max_file_size:
            self.report["status"] = False

    
    def check_empty(self, df: pd.DataFrame):
        if df.empty:
            self.report["status"] = False

    
    def get_data_summary(self, df: pd.DataFrame):
        self.report["dataset_summary"] =  {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "file_size_mb": round(os.path.getsize(self.data_ingestion_artifact.feature_store_file_path) / (1024 * 1024), 2),
        }
    
    def get_target_analysis(self, df:pd.DataFrame):
        
        if self.target not in df.columns:
            target_info = {
                "name": self.target,
                "type": None,
                "class_balance": None,
                "imbalance_flag": True,
                "error": f"Target column '{self.target}' not found"
            }

        else:
            target_series = df[self.target].dropna()

            if self.problem_type == "regression":
                target_info = {
                    "name": self.target,
                    "type": "numeric",
                    "class_balance": None,
                    "imbalance_flag": False
                }

            elif self.problem_type == "classification":
                class_distribution = target_series.value_counts(normalize=True).to_dict()
                
                target_info = {
                    "name": self.target,
                    "type": "categorical",
                    "class_balance": {str(k): round(v, 4) for k, v in class_distribution.items()},
                }

        self.report["target_analysis"] = target_info
    
    def get_column_analysis(self , df: pd.DataFrame):
        column_analysis = {}

        for col in df.columns:
            series = df[col]

            dtype = str(series.dtype)
            missing_pct = series.isna().mean()
            unique_values = series.nunique(dropna=True)

            column_analysis[col] = {
                    "dtype": dtype,
                    "missing_pct": float(missing_pct),
                    "unique_values": int(unique_values),
                }
            
        self.report["column_analysis"] = column_analysis
    
    def get_actions(self, df: pd.DataFrame):

        drop_columns = [col for col, stats in self.report["column_analysis"].items() if stats["missing_pct"] > self.data_validation_config.missing_threshold]
        impute_columns = [col for col, stats in self.report["column_analysis"].items() 
                            if 0 < stats["missing_pct"] <= self.data_validation_config.missing_threshold]
            
        remove_duplicates = df.duplicated().sum() > 0
            
        one_hot = []
        label_encode = []
        categorical_like = ("object", "category", "string", "str")
            
        for col, stats in self.report["column_analysis"].items():
            if any(t in stats["dtype"] for t in categorical_like):
                if stats["unique_values"] <= self.data_validation_config.one_hot_threshold:
                    one_hot.append(col)
                else:
                        label_encode.append(col)
            
        actions = {
                "drop_columns": drop_columns,
                "impute_columns": impute_columns,
                "remove_duplicates": bool(remove_duplicates),
                "encoding":
                    {
                        "one_hot" : one_hot,
                        "label_encode" : label_encode
                    }
            }
        self.report["actions"] = actions
        
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Initiating data validation")
            
            #checking file size and ensuring it is less than a certain threshold
            self.check_size()
            
            df = pd.read_csv(self.data_ingestion_artifact.feature_store_file_path)
            
            #making checks on the data
            self.check_empty(df)
            logging.info("Checked whether dataframe is empty or not")
            
            #build report
            self.get_data_summary(df)
            logging.info("Adding data summary to report")
            self.get_target_analysis(df)
            logging.info("Adding target summary to report")
            self.get_column_analysis(df)
            logging.info("Adding column summary to report")
            self.get_actions(df)
            logging.info("Adding actions to report")
                        
            save_object( self.data_validation_config.report_file_path , self.report)
            logging.info(f"Saved report to {self.data_validation_config.report_file_path}")
            
            return DataValidationArtifact(report_file_path=self.data_validation_config.report_file_path,
                                          status= self.report["status"],
                                          train_file_path=self.data_ingestion_artifact.train_file_path,
                                          test_file_path=self.data_ingestion_artifact.test_file_path)
            
        except Exception as e:
            raise AutoMLException(e,sys)
