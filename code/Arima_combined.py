import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
DEFAULT_RESULTS_DIR = Path("results_international_2006")
DEFAULT_TARGET_SERIES_PATH = Path(
    "student_data/international_new_students_2006/international_students_nixtla.csv"
)
DEFAULT_COVARIATES_PATH = Path(
    "student_data/international_new_students_2006/covariates/"
    "international_students_nixtla_0_100_news_covariates_Good_News_engineered_1.csv"
)
FREQ = "D"
DEFAULT_START_POINT = 9


def load_target_series(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=None)
    df.columns = ["unique_id", "ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"])
    df["unique_id"] = df["unique_id"].astype(str)
    return df.sort_values("ds").reset_index(drop=True)


def load_covariates(path: Path) -> Tuple[pd.DataFrame, List[str]]:
    X = pd.read_csv(path, index_col=None)
    X["ds"] = pd.to_datetime(X["ds"])
    X["unique_id"] = X["unique_id"].astype(str)
    X = X.sort_values("ds").reset_index(drop=True)

    exog_cols = [col for col in X.columns if col not in {"unique_id", "ds"}]
    usable_cols: List[str] = []
    for col in exog_cols:
        if X[col].nunique(dropna=True) > 1:
            usable_cols.append(col)

    if not usable_cols:
        raise ValueError("No usable exogenous columns found in the covariates file.")

    X = X[["unique_id", "ds"] + usable_cols]
    return X, usable_cols


def compute_metrics(predictions: pd.Series, actual: pd.Series) -> Dict[str, float]:
    errors = predictions - actual
    mae = float(errors.abs().mean())
    mse = float((errors**2).mean())
    rmse = float(np.sqrt(mse))
    mape = float(
        (errors.abs() / actual.replace(0, np.nan)).dropna().mean() * 100
    )
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape}


def run_arima_without_covariates(
    df: pd.DataFrame,
    start_point: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    records = []
    for step in range(start_point, len(df)):
        train = df.iloc[:step][["unique_id", "ds", "y"]]
        test = df.iloc[step: step + 1][["unique_id", "ds", "y"]]

        sf = StatsForecast(models=[AutoARIMA()], freq=FREQ)
        sf.fit(train)
        forecast = sf.forecast(df=train, h=1)
        prediction = float(forecast["AutoARIMA"].iloc[0])

        records.append(
            {
                "unique_id": test["unique_id"].iloc[0],
                "ds": test["ds"].iloc[0],
                "y": test["y"].iloc[0],
                "AutoARIMA": prediction,
            }
        )

    result_df = pd.DataFrame(records)
    metrics = compute_metrics(result_df["AutoARIMA"], result_df["y"])
    return result_df, metrics


def run_arima_with_covariates(
    df: pd.DataFrame,
    covariates: pd.DataFrame,
    exog_cols: List[str],
    start_point: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    records = []

    for step in range(start_point, len(df)):
        y_train = df.iloc[:step][["unique_id", "ds", "y"]]
        y_test = df.iloc[step: step + 1][["unique_id", "ds", "y"]]

        train = y_train.merge(covariates, on=["unique_id", "ds"], how="left")
        train = train[["unique_id", "ds", "y"] + exog_cols]

        X_next = covariates[covariates["ds"].isin(y_test["ds"])]
        if X_next.empty:
            raise ValueError(
                f"Missing exogenous values for forecast date(s): {y_test['ds'].tolist()}"
            )

        sf = StatsForecast(models=[AutoARIMA()], freq=FREQ)
        sf.fit(train)
        forecast = sf.forecast(df=train, h=1, X_df=X_next)
        prediction = float(forecast["AutoARIMA"].iloc[0])

        records.append(
            {
                "unique_id": y_test["unique_id"].iloc[0],
                "ds": y_test["ds"].iloc[0],
                "y": y_test["y"].iloc[0],
                "AutoARIMA": prediction,
            }
        )

    result_df = pd.DataFrame(records)
    metrics = compute_metrics(result_df["AutoARIMA"], result_df["y"])
    return result_df, metrics


def save_outputs(
    df_forecasts: pd.DataFrame,
    metrics: Dict[str, float],
    filename_prefix: str,
    results_dir: Path,
    suffix: str
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    forecast_path = results_dir / f"AutoArima_forecasts_{suffix}.csv"
    df_forecasts[:-1].to_csv(forecast_path, index=False)

    report = pd.DataFrame(
        {
            "MAE": [f"{metrics['MAE']:.6f}"],
            "MSE": [f"{metrics['MSE']:.6f}"],
            "RMSE": [f"{metrics['RMSE']:.6f}"],
            "MAPE": [f"{metrics['MAPE']:.2f}%"],
        }
    )
    report_path = results_dir / f"{filename_prefix}_report_{suffix}.csv"


def plot_forecasts(
    df_actual: pd.DataFrame,
    df_forecasts: pd.DataFrame,
    title: str,
    axis: plt.Axes,
    start_point: int,
) -> None:
    actual_segment = df_actual.iloc[start_point: start_point + len(df_forecasts)]
    axis.plot(actual_segment["ds"], actual_segment["y"], label="Actual")
    axis.plot(df_forecasts["ds"], df_forecasts["AutoARIMA"], label="Forecast")
    axis.set_title(title)
    axis.set_xlabel("Date")
    axis.set_ylabel("Value")
    axis.legend()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AutoARIMA with or without covariates."
    )
    parser.add_argument(
        "--mode",
        choices=["both", "no_co", "co"],
        default="both",
        help=(
            "'both' (default) runs with and without covariates; "
            "'no_co' runs only without covariates; "
            "'co' runs only with covariates."
        ),
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory to write forecasts and metrics (default: results_international_2006).",
    )
    parser.add_argument(
        "--target-series-path",
        default=str(DEFAULT_TARGET_SERIES_PATH),
        help="CSV path for the target series (default: international_students_nixtla.csv).",
    )
    parser.add_argument(
        "--covariates-path",
        default=str(DEFAULT_COVARIATES_PATH),
        help="CSV path for exogenous covariates (used when mode includes 'co').",
    )
    parser.add_argument(
        "--start-point",
        type=int,
        default=DEFAULT_START_POINT,
        help="History length before the first forecast (default: 9).",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="no_co",
        help="suffix for saved filename.",
    )


    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    target_series_path = Path(args.target_series_path)
    covariates_path = Path(args.covariates_path)
    start_point = args.start_point

    if start_point < 1:
        raise ValueError("--start-point must be at least 1.")

    df_target = load_target_series(target_series_path)

    plot_payload: List[Tuple[str, pd.DataFrame]] = []

    if args.mode in {"both", "no_co"}:
        forecasts_no_co, metrics_no_co = run_arima_without_covariates(df_target, start_point)
        save_outputs(forecasts_no_co, metrics_no_co, "AutoArima", results_dir, args.suffix)
        print("AutoARIMA (no covariates) metrics:")
        for key, value in metrics_no_co.items():
            unit = "%" if key == "MAPE" else ""
            precision = 2 if key == "MAPE" else 6
            print(f"  {key}: {value:.{precision}f}{unit}")
        plot_payload.append(("AutoARIMA without covariates", forecasts_no_co))

    if args.mode in {"both", "co"}:
        covariates, exog_cols = load_covariates(covariates_path)
        forecasts_co, metrics_co = run_arima_with_covariates(
            df_target, covariates, exog_cols, start_point
        )
        save_outputs(forecasts_co, metrics_co, "AutoArima_co", results_dir, args.suffix)
        print("\nAutoARIMA (with covariates) metrics:")
        for key, value in metrics_co.items():
            unit = "%" if key == "MAPE" else ""
            precision = 2 if key == "MAPE" else 6
            print(f"  {key}: {value:.{precision}f}{unit}")
        plot_payload.append(("AutoARIMA with covariates", forecasts_co))

    if plot_payload:
        fig, axes = plt.subplots(len(plot_payload), 1, figsize=(10, 4 * len(plot_payload)), sharex=True)
        if len(plot_payload) == 1:
            axes = [axes]
        for axis, (title, forecast_df) in zip(axes, plot_payload):
            plot_forecasts(df_target, forecast_df, title, axis, start_point)
        plt.tight_layout()


if __name__ == "__main__":
    main()
