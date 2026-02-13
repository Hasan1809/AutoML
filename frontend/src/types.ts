export interface Health {
  status: string;
  model_available: boolean;
  preprocessor_available: boolean;
  model_report_available: boolean;
  timestamp: string;
}

export interface ModelReport {
  status: boolean;
  problem_type: string;
  target: string;
  scoring_key: string;
  models: Array<{
    name: string;
    best_params: Record<string, any>;
    metrics: Record<string, any>;
  }>;
  best_model: {
    name: string;
    best_params: Record<string, any>;
    metrics: Record<string, any>;
    trained_model_path: string;
  };
}

export interface ValidationReport {
  status: boolean;
  dataset_summary: {
    rows: number;
    columns: number;
    file_size_mb: number;
  };
  target_analysis: {
    name: string;
    type: string;
    class_balance?: Record<string, number>;
  };
  column_analysis: Record<
    string,
    {
      dtype: string;
      missing_pct: number;
      unique_values: number;
    }
  >;
  actions: {
    drop_columns: string[];
    impute_columns: string[];
    remove_duplicates: boolean;
    encoding: {
      one_hot: string[];
      label_encode: string[];
    };
  };
}
