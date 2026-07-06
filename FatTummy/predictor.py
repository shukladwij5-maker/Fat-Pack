"""Prediction helpers for numeric time-series and tabular trend forecasting."""

from __future__ import annotations

import csv
import importlib.util
import os
import subprocess
import sys
from typing import Any, List, Optional, Sequence


def ensure_predictor_dependencies() -> bool:
    """Check whether common forecasting dependencies are available without installing them."""
    required = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("statsmodels", "statsmodels"),
        ("scikit-learn", "sklearn"),
    ]
    missing = [pkg for _, pkg in required if importlib.util.find_spec(pkg) is None]
    if not missing:
        return True

    return False


def _coerce_numeric(values: Sequence[float]) -> List[float]:
    numeric: List[float] = []
    for value in values:
        try:
            numeric.append(float(value))
        except (TypeError, ValueError) as exc:
            raise ValueError("predict() expects numeric values.") from exc
    return numeric


def _linear_trend(values: Sequence[float], steps: int = 1) -> List[float]:
    if steps <= 0:
        return []
    numeric = _coerce_numeric(values)
    if len(numeric) < 2:
        return [numeric[-1]] * steps if numeric else []

    x_values = list(range(len(numeric)))
    n = len(numeric)
    mean_x = sum(x_values) / n
    mean_y = sum(numeric) / n
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, numeric))
    denominator = sum((x - mean_x) ** 2 for x in x_values)
    if denominator == 0:
        last = numeric[-1]
        return [last] * steps
    slope = numerator / denominator
    intercept = mean_y - slope * mean_x
    return [float(intercept + slope * (n + offset)) for offset in range(steps)]


def _lag_regression(values: Sequence[float], steps: int = 1) -> List[float]:
    if steps <= 0:
        return []
    numeric = _coerce_numeric(values)
    if len(numeric) < 3:
        return _linear_trend(numeric, steps)

    try:
        import numpy as np
        from sklearn.linear_model import LinearRegression
    except Exception:
        return _linear_trend(numeric, steps)

    max_lag = min(6, len(numeric) - 1)
    if max_lag <= 0:
        return _linear_trend(numeric, steps)
    lags = list(range(1, max_lag + 1))
    rows: List[List[float]] = []
    targets: List[float] = []
    for index in range(max_lag, len(numeric)):
        rows.append([numeric[index - lag] for lag in lags])
        targets.append(numeric[index])

    if len(rows) < 2:
        return _linear_trend(numeric, steps)

    model = LinearRegression()
    model.fit(np.array(rows), np.array(targets))

    history = list(numeric)
    predictions: List[float] = []
    for _ in range(steps):
        feature_row = [history[-lag] for lag in lags]
        next_value = float(model.predict(np.array([feature_row]))[0])
        predictions.append(next_value)
        history.append(next_value)
    return predictions


def _arima_forecast(values: Sequence[float], steps: int = 1) -> List[float]:
    if steps <= 0:
        return []
    numeric = _coerce_numeric(values)
    if len(numeric) < 5:
        return _linear_trend(numeric, steps)

    try:
        from statsmodels.tsa.arima.model import ARIMA
    except Exception:
        return _linear_trend(numeric, steps)

    try:
        model = ARIMA(list(numeric), order=(1, 0, 1))
        result = model.fit()
        forecast = result.forecast(steps=steps)
        if hasattr(forecast, "tolist"):
            return [float(item) for item in forecast.tolist()]
        return [float(item) for item in list(forecast)]
    except Exception:
        return _linear_trend(numeric, steps)


def predict(values: Sequence[float], steps: int = 1, model: str = "auto") -> List[float]:
    """Predict future values from a numeric sequence using an adaptive forecasting strategy."""
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if steps == 0:
        return []

    numeric = _coerce_numeric(values)
    if len(numeric) < 2:
        return [numeric[-1]] * steps if numeric else []

    ensure_predictor_dependencies()

    if model.lower() in {"arima", "statsmodels"}:
        return _arima_forecast(numeric, steps)
    if model.lower() in {"regression", "sklearn", "ml"}:
        return _lag_regression(numeric, steps)

    if model.lower() == "linear":
        return _linear_trend(numeric, steps)

    arima_result = _arima_forecast(numeric, steps)
    if arima_result and len(arima_result) == steps:
        return arima_result
    return _lag_regression(numeric, steps)


def predict_csv(
    csv_path: str,
    target_column: Optional[str] = None,
    steps: int = 1,
    model: str = "auto",
    date_column: Optional[str] = None,
) -> List[float]:
    """Load a CSV file and predict the next values from a numeric target column."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    if target_column is None:
        try:
            import pandas as pd
        except Exception:
            pandas = None
        if pandas is not None:
            frame = pd.read_csv(csv_path)
            for column in frame.columns:
                if column.lower() == date_column.lower() if date_column else False:
                    continue
                try:
                    series = pd.to_numeric(frame[column], errors="coerce").dropna()
                except Exception:
                    continue
                if len(series) >= 2:
                    target_column = column
                    break
            if target_column is None:
                raise ValueError("No numeric target column found in CSV")
        else:
            with open(csv_path, newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                rows = list(reader)
            if len(rows) < 2:
                raise ValueError("CSV is empty")
            header = rows[0]
            numeric_candidates = []
            for column_index, column_name in enumerate(header):
                values = [row[column_index] for row in rows[1:] if column_index < len(row)]
                try:
                    numeric_values = [float(item) for item in values if item not in {"", None}]
                except ValueError:
                    continue
                if numeric_values:
                    numeric_candidates.append((column_name, numeric_values))
            if not numeric_candidates:
                raise ValueError("No numeric target column found in CSV")
            target_column = numeric_candidates[0][0]
            series = numeric_candidates[0][1]
            return predict(series, steps=steps, model=model)

    try:
        import pandas as pd
    except Exception:
        with open(csv_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            values = []
            for row in reader:
                raw_value = row.get(target_column, "")
                try:
                    values.append(float(raw_value))
                except (TypeError, ValueError):
                    continue
            return predict(values, steps=steps, model=model)

    frame = pd.read_csv(csv_path)
    if date_column and date_column in frame.columns:
        frame = frame.sort_values(date_column)
    if target_column not in frame.columns:
        raise KeyError(target_column)
    series = pd.to_numeric(frame[target_column], errors="coerce").dropna()
    return predict(series.tolist(), steps=steps, model=model)
