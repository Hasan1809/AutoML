import os
import json
from datetime import datetime
from uuid import uuid4
from typing import List, Optional, Dict, Any

import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from pydantic import BaseModel

from main import run_pipeline, load_settings
from src.pipeline.predictor import Predictor
from src.entity.config_entity import PipelineConfig, ModelTrainerConfig


app = FastAPI(title="AutoML API", version="0.1.0")

jobs: Dict[str, Dict[str, Any]] = {}


class PredictRequest(BaseModel):
    records: List[dict]


class PredictResponse(BaseModel):
    predictions: List[Any]


class TrainRequest(BaseModel):
    data_path: Optional[str] = None
    target: Optional[str] = None
    problem_type: Optional[str] = None


class TrainResponse(BaseModel):
    job_id: str
    status: str


def _registry_paths():
    pipeline_config = PipelineConfig()
    model_trainer_config = ModelTrainerConfig(pipeline_config)
    return model_trainer_config.registry_model_path, model_trainer_config.registry_preprocessor_path, model_trainer_config.registry_report_path


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _latest_log_file():
    latest_path = None
    latest_mtime = 0
    for root, _, files in os.walk("logs"):
        for f in files:
            if f.endswith(".log"):
                path = os.path.join(root, f)
                mtime = os.path.getmtime(path)
                if mtime > latest_mtime:
                    latest_mtime = mtime
                    latest_path = path
    return latest_path


@app.get("/health")
def health():
    model_path, preproc_path, report_path = _registry_paths()
    return {
        "status": "ok",
        "model_available": os.path.exists(model_path),
        "preprocessor_available": os.path.exists(preproc_path),
        "model_report_available": os.path.exists(report_path),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    model_path, preproc_path, report_path = _registry_paths()
    if not (os.path.exists(model_path) and os.path.exists(preproc_path)):
        raise HTTPException(status_code=404, detail="No registered model/preprocessor found.")

    settings = load_settings(os.path.join("config", "config.yaml"))
    target = settings.get("target")

    df = pd.DataFrame(payload.records)
    predictor = Predictor(model_path=model_path, preprocessor_path=preproc_path, target=target)
    preds = predictor.predict_dataframe(df)
    return PredictResponse(predictions=[p.item() if hasattr(p, "item") else p for p in preds])


@app.post("/train", response_model=TrainResponse)
def trigger_train(request: TrainRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid4())
    jobs[job_id] = {"status": "running", "started_at": datetime.utcnow().isoformat() + "Z"}

    def _job():
        try:
            artifact = run_pipeline(
                override_data_path=request.data_path,
                override_target=request.target,
                override_problem_type=request.problem_type,
            )
            jobs[job_id] = {
                "status": "succeeded",
                "finished_at": datetime.utcnow().isoformat() + "Z",
                "artifact": {
                    "trained_model_path": artifact.trained_model_path,
                    "metrics_path": artifact.metrics_path,
                    "best_model_name": artifact.best_model_name,
                    "best_score": artifact.best_score,
                },
            }
        except Exception as e:
            jobs[job_id] = {
                "status": "failed",
                "finished_at": datetime.utcnow().isoformat() + "Z",
                "error": str(e),
            }

    background_tasks.add_task(_job)
    return TrainResponse(job_id=job_id, status="running")


@app.get("/train/{job_id}")
def get_train_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/reports/latest")
def get_latest_report():
    _, _, report_path = _registry_paths()
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="No model report found.")
    return _load_json(report_path)


@app.get("/models/latest")
def get_latest_model():
    model_path, preproc_path, report_path = _registry_paths()
    available = os.path.exists(model_path) and os.path.exists(preproc_path)
    payload = {
        "available": available,
        "model_path": model_path,
        "preprocessor_path": preproc_path,
        "report_path": report_path,
    }
    if os.path.exists(report_path):
        payload["report"] = _load_json(report_path)
    return payload


@app.get("/logs/latest")
def get_latest_logs(lines: int = 200):
    log_file = _latest_log_file()
    if not log_file:
        raise HTTPException(status_code=404, detail="No logs found.")
    with open(log_file, "r", encoding="utf-8") as f:
        content = f.readlines()[-lines:]
    return {"log_file": log_file, "lines": content}


@app.post("/train/upload")
async def train_with_upload(background_tasks: BackgroundTasks, file: UploadFile = File(...), target: Optional[str] = None, problem_type: Optional[str] = None):
    upload_dir = os.path.join("artifacts", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    save_path = os.path.join(upload_dir, f"{uuid4()}_{file.filename}")
    try:
        with open(save_path, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    # Run pipeline synchronously for simplicity; could move to background_tasks if desired
    try:
        artifact = run_pipeline(
            override_data_path=save_path,
            override_target=target,
            override_problem_type=problem_type,
        )
        _, _, report_path = _registry_paths()
        report = _load_json(report_path) if os.path.exists(report_path) else None
        return {
            "status": "succeeded",
            "uploaded_path": save_path,
            "trained_model_path": artifact.trained_model_path,
            "metrics_path": artifact.metrics_path,
            "best_model": report.get("best_model") if report else None,
            "report": report,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")
