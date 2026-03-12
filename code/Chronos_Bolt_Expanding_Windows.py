import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor


DEFAULT_MODEL_SIZES = [
    "chronos-bolt-mini",
    "chronos-bolt-tiny",
    "chronos-bolt-small",
    "chronos-bolt-base",
]


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    """Return MAE, MSE, RMSE, MAPE (%), MSPE (%)."""
    y_true = pd.to_numeric(y_true, errors="coerce")
    y_pred = pd.to_numeric(y_pred, errors="coerce")
    mask = y_true.notna() & y_pred.notna()
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    e = y_true - y_pred
    mae = np.mean(np.abs(e)) if len(e) else np.nan
    mse = np.mean(np.square(e)) if len(e) else np.nan
    rmse = np.sqrt(mse) if np.isfinite(mse) else np.nan
    not_zero = y_true != 0
    mape = (np.mean(np.abs(e[not_zero] / y_true[not_zero])) * 100) if not_zero.any() else np.nan
    mspe = (np.mean(np.square(e[not_zero] / y_true[not_zero])) * 100) if not_zero.any() else np.nan
    return pd.DataFrame(
        {"MAE": [mae], "MSE": [mse], "RMSE": [rmse], "MAPE (%)": [mape], "MSPE (%)": [mspe]}
    )


def load_dataframes(
    target_path: Path,
    covariate_path: Optional[Path],
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], TimeSeriesDataFrame]:
    """Read input CSVs and build the TimeSeriesDataFrame."""
    if not target_path.exists():
        raise FileNotFoundError(f"Target CSV not found: {target_path}")

    raw_df = pd.read_csv(target_path)
    required = {"item_id", "timestamp", "target"}
    if not required.issubset(raw_df.columns):
        missing = ", ".join(sorted(required - set(raw_df.columns)))
        raise ValueError(f"Target CSV missing required columns: {missing}")

    cov_df: Optional[pd.DataFrame] = None
    if covariate_path:
        if not covariate_path.exists():
            raise FileNotFoundError(f"Covariate CSV not found: {covariate_path}")
        cov_df = pd.read_csv(covariate_path)

    tsdf = TimeSeriesDataFrame.from_data_frame(
        raw_df,
        id_column="item_id",
        timestamp_column="timestamp",
    )
    return raw_df, cov_df, tsdf


def parse_known_covariates(names: Optional[str]) -> List[str]:
    if not names:
        return []
    return [name.strip() for name in names.split(",") if name.strip()]


def build_predictor(
    prediction_length: int,
    model_size: str,
    model_name: str,
    per_iter_time: int,
    run_dir: Path,
    known_covariates: List[str],
) -> TimeSeriesPredictor:
    """Create and fit a TimeSeriesPredictor for a single cut."""
    hyperparameters: Dict[str, List[Dict]] = {"Chronos": []}
    # print('known_covariates:', known_covariates)
    if known_covariates:
        hyperparameters["Chronos"].append(
            {
                "model_path": f"amazon/{model_size}",
                "covariate_regressor": "CAT",
                "target_scaler": "standard",
                "ag_args": {"name_suffix": f"{model_name}"},
                "ag_args_fit": {"num_gpus": 1},
            }
        )
    else:
        hyperparameters["Chronos"].append(
            {
                "model_path": f"amazon/{model_size}",
                "ag_args": {"name_suffix": f"{model_name}"},
                "ag_args_fit": {"num_gpus": 1},
            }
        )
    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        target="target",
        path=str(run_dir),
        known_covariates_names=known_covariates,
    )
    return predictor, hyperparameters


def run_expanding_forecast(
    tsdf: TimeSeriesDataFrame,
    cov_df: Optional[pd.DataFrame],
    item_id: str,
    model_sizes: List[str],
    n_steps: int,
    prediction_length: int,
    per_iter_time: int,
    known_covariate_names: List[str],
    output_dir: Path,
    suffix: str,
    model_name: str
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Run expanding window forecasts for the selected item across model sizes."""
    history = tsdf.loc[item_id]
    if len(history) < (n_steps + 1):
        raise ValueError("Not enough history for the requested rolling steps.")

    cut_times = history.index[n_steps:]
    results: Dict[str, Dict[str, pd.DataFrame]] = {}

    known_covs: Optional[TimeSeriesDataFrame] = None
    if known_covariate_names:
        if cov_df is None:
            raise ValueError("Known covariate names provided but no covariate CSV supplied.")
        known_covs = TimeSeriesDataFrame.from_data_frame(
            cov_df, id_column="item_id", timestamp_column="timestamp"
        )

    for model_size in model_sizes:
        print(f"Running model size '{model_size}' for item '{item_id}'.")
        median_points: List[pd.Series] = []
        p10_points: List[pd.Series] = []
        p30_points: List[pd.Series] = []
        p50_points: List[pd.Series] = []
        p70_points: List[pd.Series] = []
        p90_points: List[pd.Series] = []
        true_points: List[pd.Series] = []
        # run_dir = Path(f"C:/temps_chronos/chronos_expand_chronos_expand_bolt")  # ---> Few shots (zero_with-fintuned)
        for i, current_cut in enumerate(cut_times, start=1):
            train_df = tsdf[tsdf.index.get_level_values("timestamp") < current_cut]
            # run_dir = f"C:/temps_chronos/chronos_expand_{model_size}_{int(time.time())}_{i}" # ----> Zero-shot
            
            str_time = pd.to_datetime(current_cut).strftime('%Y-%m-%d')
            run_dir = Path(f"C:/temps_chronos/chronos_expand_chronos_expand_bolt_{model_size}_{str_time}")
            # run_dir = f"C:/temps_chronos/chronos_expand_{model_size}"
            # run_dir.mkdir(parents=True, exist_ok=True)
            predictor, hyperparameters = build_predictor(
                prediction_length=prediction_length,
                model_size=model_size,
                model_name=model_name,
                per_iter_time=per_iter_time,
                run_dir=run_dir,
                known_covariates=known_covariate_names,
            )
            predictor = predictor.fit(
                train_df,
                hyperparameters=hyperparameters,
                enable_ensemble=False,
                time_limit=per_iter_time,
            )
            if known_covs is not None:
                preds = predictor.predict(train_df, known_covariates=known_covs)
            else:
                preds = predictor.predict(train_df)

            fc_item = preds.loc[item_id]
            if "0.5" in fc_item.columns:
                yhat = fc_item["0.5"].iloc[0]
            elif "mean" in fc_item.columns:
                yhat = fc_item["mean"].iloc[0]
            else:
                yhat = fc_item.iloc[0, 0]

            median_points.append(pd.Series([yhat], index=[current_cut], name="median"))
            if "0.1" in fc_item.columns:
                p10_points.append(pd.Series([fc_item["0.1"].iloc[0]], index=[current_cut], name="p10"))
            if "0.3" in fc_item.columns:
                p30_points.append(pd.Series([fc_item["0.3"].iloc[0]], index=[current_cut], name="p30"))
            if "0.5" in fc_item.columns:
                p50_points.append(pd.Series([fc_item["0.5"].iloc[0]], index=[current_cut], name="p50"))
            if "0.7" in fc_item.columns:
                p70_points.append(pd.Series([fc_item["0.7"].iloc[0]], index=[current_cut], name="p70"))
            if "0.9" in fc_item.columns:
                p90_points.append(pd.Series([fc_item["0.9"].iloc[0]], index=[current_cut], name="p90"))

            if current_cut in history.index:
                true_points.append(
                    pd.Series([history.loc[current_cut, "target"]], index=[current_cut], name="y_true")
                )

        median_series = pd.concat(median_points).sort_index() if median_points else pd.Series(dtype=float)
        p10_series = pd.concat(p10_points).sort_index() if p10_points else None
        p30_series = pd.concat(p30_points).sort_index() if p30_points else None
        p50_series = pd.concat(p50_points).sort_index() if p50_points else None
        p70_series = pd.concat(p70_points).sort_index() if p70_points else None
        p90_series = pd.concat(p90_points).sort_index() if p90_points else None
        true_series = pd.concat(true_points).sort_index() if true_points else pd.Series(dtype=float)

        out_df = pd.DataFrame({"yhat": median_series})
        if true_series.size:
            out_df["y_true"] = true_series
            out_df["error"] = out_df["y_true"] - out_df["yhat"]
            out_df["abs_error"] = out_df["error"].abs()
            out_df["q10"] = p10_series
            out_df["q30"] = p30_series
            out_df["q50"] = p50_series
            out_df["q70"] = p70_series
            out_df["q90"] = p90_series

        if not out_df.empty:
            out_df["yhat"] = out_df["yhat"].iloc[:-1]

        metrics_df = compute_metrics(out_df["y_true"], out_df["yhat"]) if "y_true" in out_df else None

        fig = plt.figure(figsize=(14, 6))
        plt.plot(history.index[:-1], history["target"].iloc[:-1], label="History")
        plt.plot(median_series.index, median_series.values, label="Forecast (median)")
        plt.title(f"{prediction_length}-step rolling forecast (last {n_steps}) - item_id={item_id}")
        plt.xlabel("Time")
        plt.ylabel("Target")
        plt.legend()
        plt.tight_layout()

        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        forecast_path = output_dir / f"actual_vs_forecast_{model_size}_{n_steps}_{suffix}.csv"
        plot_path = output_dir / f"chronos_forecast_{model_size}_{n_steps}_{suffix}.png"
        out_df.to_csv(forecast_path)
        # fig.savefig(plot_path, bbox_inches="tight")
        # plt.close(fig)

        if metrics_df is not None:
            metrics_path = output_dir / f"metrics_{model_size}_{n_steps}.csv"
            metrics_df.to_csv(metrics_path, index=False)
        else:
            metrics_path = None

        results[model_size] = {
            "forecast": out_df,
            "metrics": metrics_df if metrics_df is not None else pd.DataFrame(),
        }

        print(f"Saved forecast to {forecast_path}")
        print(f"Saved plot to {plot_path}")
        if metrics_path:
            print(f"Saved metrics to {metrics_path}")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chronos Bolt expanding window forecast without Streamlit.")
    parser.add_argument("--target", type=Path, required=True, help="Target CSV with columns item_id,timestamp,target.")
    parser.add_argument("--covariate", type=Path, help="Optional covariate CSV with known future covariates.")
    parser.add_argument("--item-id", dest="item_id", required=False, help="Item ID to model. Defaults to first in CSV.")
    parser.add_argument("--n-steps", type=int, default=9, help="Number of expanding steps.")
    parser.add_argument("--prediction-length", type=int, default=1, help="Forecast horizon.")
    parser.add_argument(
        "--model-sizes",
        nargs="+",
        choices=DEFAULT_MODEL_SIZES,
        default=DEFAULT_MODEL_SIZES,
        help="Chronos model sizes to run.",
    )
    parser.add_argument("--time-limit", type=int, default=30, help="Per-iteration time limit in seconds.")
    parser.add_argument(
        "--known-covariates",
        type=str,
        help="Comma separated list of known covariate column names (requires --covariate).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory to store CSVs and plots.",
    )
    parser.add_argument(
        "--suffix",
        default="no_co",
        help="suffix for saved filename .",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="FewShotFT",
        help="Model name suffix for saved filename.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df, cov_df, tsdf = load_dataframes(args.target, args.covariate)
    item_id = args.item_id or next(iter(tsdf.item_ids))

    known_covariate_names = parse_known_covariates(args.known_covariates)

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Running Chronos Bolt expanding window for item '{item_id}'.")

    run_expanding_forecast(
        tsdf=tsdf,
        cov_df=cov_df,
        item_id=item_id,
        model_sizes=list(dict.fromkeys(args.model_sizes)),
        model_name=str(args.model_name),
        n_steps=args.n_steps,
        prediction_length=args.prediction_length,
        per_iter_time=args.time_limit,
        known_covariate_names=known_covariate_names,
        output_dir=output_dir,
        suffix=args.suffix,
    )


if __name__ == "__main__":
    main()
