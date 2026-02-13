import { useEffect, useMemo, useState } from "react";
import { getHealth, getModelReport, getValidationReport, predictJson, predictUpload, trainUpload } from "./api";
import type { Health, ModelReport, ValidationReport } from "./types";

const apiBase = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

type PredictMode = "json" | "form" | "csv";

const parseCsvColumns = async (file: File): Promise<string[]> => {
  const text = await file.text();
  const firstLine = text.split(/\r?\n/)[0] || "";
  return firstLine.split(",").map((c) => c.replace(/^"|"$/g, "").trim()).filter(Boolean);
};

function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [modelReport, setModelReport] = useState<ModelReport | null>(null);
  const [validationReport, setValidationReport] = useState<ValidationReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [trainFile, setTrainFile] = useState<File | null>(null);
  const [trainColumns, setTrainColumns] = useState<string[]>([]);
  const [targetColumn, setTargetColumn] = useState<string>("");
  const [trainResult, setTrainResult] = useState<any>(null);
  const [trainError, setTrainError] = useState<string | null>(null);
  const [problemType, setProblemType] = useState<string>("classification");
  const [predictMode, setPredictMode] = useState<PredictMode>("form");
  const [predictFile, setPredictFile] = useState<File | null>(null);
  const [predictJsonText, setPredictJsonText] = useState(
    '[{"PassengerId":1,"Pclass":3,"Name":"Smith, Mr. John","Sex":"male","Age":22,"SibSp":1,"Parch":0,"Ticket":"A/5 21171","Fare":7.25,"Cabin":"","Embarked":"S"}]'
  );
  const [predictFormValues, setPredictFormValues] = useState<Record<string, string>>({});
  const [predictions, setPredictions] = useState<any[]>([]);
  const [predictError, setPredictError] = useState<string | null>(null);

  useEffect(() => {
    const init = async () => {
      try {
        const [h, mr, vr] = await Promise.all([getHealth(), getModelReport(), getValidationReport()]);
        setHealth(h);
        setModelReport(mr);
        setValidationReport(vr);
      } catch (e) {
        console.error(e);
      }
    };
    init();
  }, []);

  useEffect(() => {
    if (validationReport?.column_analysis) {
      const initVals: Record<string, string> = {};
      Object.keys(validationReport.column_analysis).forEach((k) => {
        initVals[k] = "";
      });
      setPredictFormValues(initVals);
    }
  }, [validationReport]);

  const featureColumns = useMemo(() => {
    if (!validationReport?.column_analysis || !validationReport?.target_analysis?.name) return [];
    return Object.keys(validationReport.column_analysis).filter((c) => c !== validationReport.target_analysis.name);
  }, [validationReport]);

  const onTrainFileChange = async (file: File | null) => {
    setTrainFile(file);
    setTrainColumns([]);
    setTargetColumn("");
    if (file) {
      try {
        const cols = await parseCsvColumns(file);
        setTrainColumns(cols);
      } catch (e) {
        console.error(e);
      }
    }
  };

  const onTrain = async () => {
    if (!trainFile) {
      setTrainError("Please select a CSV file.");
      return;
    }
    if (!targetColumn) {
      setTrainError("Please select the target column.");
      return;
    }
    setTrainError(null);
    setTrainResult(null);
    setLoading(true);
    try {
      const res = await trainUpload(trainFile, targetColumn, problemType);
      setTrainResult(res);
      const [mr, vr] = await Promise.all([getModelReport(), getValidationReport()]);
      setModelReport(mr);
      setValidationReport(vr);
    } catch (e: any) {
      setTrainError(e?.message || "Training failed");
    } finally {
      setLoading(false);
    }
  };

  const onPredict = async () => {
    setPredictError(null);
    setPredictions([]);
    try {
      if (predictMode === "csv") {
        if (!predictFile) throw new Error("Please select a CSV file.");
        const res = await predictUpload(predictFile);
        setPredictions(res.predictions);
      } else if (predictMode === "json") {
        const parsed = JSON.parse(predictJsonText);
        const res = await predictJson(parsed);
        setPredictions(res.predictions);
      } else {
        const record: Record<string, any> = {};
        featureColumns.forEach((c) => {
          if (predictFormValues[c] !== "") record[c] = predictFormValues[c];
        });
        const res = await predictJson([record]);
        setPredictions(res.predictions);
      }
    } catch (e: any) {
      setPredictError(e?.message || "Prediction failed");
    }
  };

  return (
    <div className="max-w-5xl mx-auto p-6">
      <header className="mb-6">
        <h1 className="text-3xl font-semibold">AutoML UI</h1>
        <p className="text-sm text-gray-600">API base: {apiBase}</p>
      </header>

      <section className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm mb-4">
        <h2 className="text-xl font-semibold mb-2">Health & Model</h2>
        <div className="text-sm text-gray-800 space-y-1">
          <div>Status: {health?.status ?? "unknown"}</div>
          <div>Model available: {health?.model_available ? "yes" : "no"}</div>
          <div>Preprocessor available: {health?.preprocessor_available ? "yes" : "no"}</div>
          {modelReport && (
            <div className="pt-1">
              <span className="font-semibold">Best model:</span> {modelReport.best_model?.name} | f1/metric:{" "}
              {modelReport.best_model?.metrics?.f1_weighted ?? modelReport.best_model?.metrics?.r2}
            </div>
          )}
        </div>
      </section>

      <section className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm mb-4">
        <h2 className="text-xl font-semibold mb-3">Train on CSV Upload</h2>
        <div className="space-y-3">
          <div>
            <label className="block text-sm font-medium mb-1">Dataset (CSV)</label>
            <input
              type="file"
              accept=".csv"
              onChange={(e) => onTrainFileChange(e.target.files?.[0] || null)}
              className="block w-full text-sm text-gray-700"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Target column</label>
            <select
              value={targetColumn}
              onChange={(e) => setTargetColumn(e.target.value)}
              className="w-full border border-gray-300 rounded px-2 py-2 text-sm"
            >
              <option value="">Select target</option>
              {trainColumns.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Problem type</label>
            <select
              value={problemType}
              onChange={(e) => setProblemType(e.target.value)}
              className="w-full border border-gray-300 rounded px-2 py-2 text-sm"
            >
              <option value="classification">Classification</option>
              <option value="regression">Regression</option>
            </select>
          </div>
          <button
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-60"
            onClick={onTrain}
            disabled={loading}
          >
            {loading ? "Training..." : "Start Training"}
          </button>
          {trainError && <div className="text-sm text-red-600">{trainError}</div>}
          {trainResult && (
            <pre className="bg-gray-100 text-xs p-2 rounded border border-gray-200 overflow-auto max-h-64">
              {JSON.stringify(trainResult, null, 2)}
            </pre>
          )}
        </div>
      </section>

      <section className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm mb-4">
        <h2 className="text-xl font-semibold mb-3">Predict</h2>
        <div className="flex gap-2 mb-3">
          <button
            className={`px-3 py-2 rounded border ${predictMode === "form" ? "bg-blue-100 border-blue-400" : "border-gray-300"}`}
            onClick={() => setPredictMode("form")}
          >
            Form
          </button>
          <button
            className={`px-3 py-2 rounded border ${predictMode === "json" ? "bg-blue-100 border-blue-400" : "border-gray-300"}`}
            onClick={() => setPredictMode("json")}
          >
            JSON
          </button>
          <button
            className={`px-3 py-2 rounded border ${predictMode === "csv" ? "bg-blue-100 border-blue-400" : "border-gray-300"}`}
            onClick={() => setPredictMode("csv")}
          >
            CSV Upload
          </button>
        </div>

        {predictMode === "form" && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {featureColumns.length === 0 && <div className="text-sm text-gray-500">No schema available yet.</div>}
            {featureColumns.map((col) => (
              <div key={col} className="flex flex-col gap-1">
                <label className="text-sm font-medium">{col}</label>
                <input
                  className="border border-gray-300 rounded px-2 py-2 text-sm"
                  value={predictFormValues[col] || ""}
                  onChange={(e) => setPredictFormValues((prev) => ({ ...prev, [col]: e.target.value }))}
                  placeholder="Enter value"
                />
              </div>
            ))}
          </div>
        )}

        {predictMode === "json" && (
          <div>
            <label className="block text-sm font-medium mb-1">JSON records</label>
            <textarea
              className="w-full border border-gray-300 rounded px-2 py-2 text-sm font-mono min-h-40"
              value={predictJsonText}
              onChange={(e) => setPredictJsonText(e.target.value)}
            />
          </div>
        )}

        {predictMode === "csv" && (
          <div>
            <label className="block text-sm font-medium mb-1">Predict CSV</label>
            <input type="file" accept=".csv" onChange={(e) => setPredictFile(e.target.files?.[0] || null)} />
          </div>
        )}

        <button
          className="mt-3 px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
          onClick={onPredict}
        >
          Predict
        </button>
        {predictError && <div className="text-sm text-red-600 mt-2">{predictError}</div>}
        {predictions.length > 0 && (
          <pre className="bg-gray-100 text-xs p-2 rounded border border-gray-200 overflow-auto max-h-64 mt-3">
            {JSON.stringify(predictions, null, 2)}
          </pre>
        )}
      </section>

      <section className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm mb-4">
        <h2 className="text-xl font-semibold mb-3">Reports</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <h4 className="font-semibold mb-1 text-sm">Model Report</h4>
            {modelReport ? (
              <pre className="bg-gray-100 text-xs p-2 rounded border border-gray-200 overflow-auto max-h-64">
                {JSON.stringify(modelReport, null, 2)}
              </pre>
            ) : (
              <div className="text-sm text-gray-600">No report yet.</div>
            )}
          </div>
          <div>
            <h4 className="font-semibold mb-1 text-sm">Validation Report</h4>
            {validationReport ? (
              <pre className="bg-gray-100 text-xs p-2 rounded border border-gray-200 overflow-auto max-h-64">
                {JSON.stringify(validationReport, null, 2)}
              </pre>
            ) : (
              <div className="text-sm text-gray-600">No validation report yet.</div>
            )}
          </div>
        </div>
      </section>
    </div>
  );
}

export default App;
