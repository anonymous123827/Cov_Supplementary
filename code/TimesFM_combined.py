import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import timesfm


@dataclass(frozen=True)
class Scenario:
    key: str
    description: str
    results_dir: Path
    checkpoint_repo: str
    hparams_kwargs: Dict[str, object]
    results_prefix: str
    uses_covariates: bool = False
    covariate_columns: Optional[Dict[str, str]] = None


SCENARIOS: Dict[str, Scenario] = {
    "main-200": Scenario(
        key="main-200",
        description="TimesFM 200M without covariates",
        results_dir=Path("results_grand_2006"),
        checkpoint_repo="google/timesfm-1.0-200m-pytorch",
        hparams_kwargs={"per_core_batch_size": 32, "horizon_len": 1},
        results_prefix="TimesFM200",
    ),
    "main-500": Scenario(
        key="main-500",
        description="TimesFM 500M without covariates",
        results_dir=Path("results_grand_2006"),
        checkpoint_repo="google/timesfm-2.0-500m-pytorch",
        hparams_kwargs={
            "per_core_batch_size": 32,
            "horizon_len": 1,
            "num_layers": 50,
            "use_positional_embedding": False,
            "context_len": 2048,
        },
        results_prefix="TimesFM500",
    ),
    "co-200": Scenario(
        key="co-200",
        description="TimesFM 200M with covariates",
        results_dir=Path("results_international_2006"),
        checkpoint_repo="google/timesfm-1.0-200m-pytorch",
        hparams_kwargs={"per_core_batch_size": 32, "horizon_len": 1},
        results_prefix="TimesFM200",
        uses_covariates=True,
        covariate_columns={"news100_good_ewma2y": "news100_good_ewma2y"},
    ),
    "co-500": Scenario(
        key="co-500",
        description="TimesFM 500M with covariates",
        results_dir=Path("results_international_2006"),
        checkpoint_repo="google/timesfm-2.0-500m-pytorch",
        hparams_kwargs={
            "per_core_batch_size": 32,
            "horizon_len": 1,
            "num_layers": 50,
            "use_positional_embedding": False,
            "context_len": 2048,
        },
        results_prefix="TimesFM500",
        uses_covariates=True,
        covariate_columns={"news100_bad_ewma2y": "news100_bad_ewma2y"},
    ),
}

DEFAULT_DATA_PATHS: Dict[str, Path] = {
    "main-200": Path("student_data/grand_new_students_2006/domestic_students_timesfm.csv"),
    "main-500": Path("student_data/grand_new_students_2006/domestic_students_timesfm.csv"),
    "co-200": Path(
        "student_data/international_new_students_2006/covariates/"
        "international_students_timesfm_0_100_news_covariates_Good_News_engineered_1.csv"
    ),
    "co-500": Path(
        "student_data/international_new_students_2006/covariates/"
        "international_students_timesfm_0_100_news_covariates_Bad_News_engineered_1.csv"
    ),
}

DEFAULT_RESULTS_DIRS: Dict[str, Path] = {key: scenario.results_dir for key, scenario in SCENARIOS.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple TimesFM scenarios (with and without covariates)."
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        choices=sorted(SCENARIOS.keys()),
        default=sorted(SCENARIOS.keys()),
        help="Scenario keys to execute (default: all).",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=9,
        help="Burn-in history length before the first forecast (default: 9).",
    )
    parser.add_argument(
        "--plot",
        dest="plot",
        action="store_true",
        help="Display matplotlib plots for each scenario.",
    )
    parser.add_argument(
        "--no-plot",
        dest="plot",
        action="store_false",
        help="Disable plotting (useful for headless runs).",
    )
    parser.set_defaults(plot=True)
    parser.add_argument(
        "--data-paths",
        nargs="+",
        help=(
            "Optional overrides in the form scenario_key=path. "
            "Example: --data-paths main-200=data.csv co-200=covariates.csv"
        ),
    )
    parser.add_argument(
        "--results-dirs",
        nargs="+",
        help=(
            "Optional overrides for output directories in the form scenario_key=dir. "
            "Example: --results-dirs main-200=results/custom_main"
        ),
    )
    parser.add_argument(
        "--covariate-columns",
        nargs="+",
        help=(
            "Optional overrides for covariate columns in the form "
            "scenario_key=timesfm_name:csv_column[,timesfm_name:csv_column]. "
            "Example: --covariate-columns co-200=news100_good_ewma2y:news100_good_ewma2y"
        ),
    )

    parser.add_argument(
        "--suffix",
        type=str,
        help=(
            "Optional suffixes for output files in the form scenario_key=suffix. "
            "Example: --suffix main-200=custom_suffix"
        ),
    )
    return parser.parse_args()


def build_model(scenario: Scenario, backend: str) -> timesfm.TimesFm:
    hparams = timesfm.TimesFmHparams(backend=backend, **scenario.hparams_kwargs)
    checkpoint = timesfm.TimesFmCheckpoint(huggingface_repo_id=scenario.checkpoint_repo)
    return timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)


def load_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=1, parse_dates=True)
    df = df.sort_index()
    return df


def ensure_minimum_history(df: pd.DataFrame, n_steps: int, scenario_key: str) -> None:
    if len(df) <= n_steps:
        raise ValueError(
            f"Scenario '{scenario_key}' has insufficient observations "
            f"({len(df)} rows) for n_steps={n_steps}."
        )


def compute_metrics(result_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    result_with_errors = result_df.copy()

    if result_with_errors.empty:
        empty_metrics = pd.DataFrame(
            [{"MAE": np.nan, "MSE": np.nan, "RMSE": np.nan, "MAPE (%)": np.nan, "MSPE (%^2)": np.nan}]
        )
        return empty_metrics, result_with_errors

    y_true = result_with_errors["actual"].to_numpy(dtype=float)
    y_pred = result_with_errors["forecast"].to_numpy(dtype=float)
    err = y_pred - y_true

    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))

    eps = 1e-8
    mask = np.abs(y_true) > eps
    if mask.any():
        mape = float(np.mean(np.abs(err[mask] / y_true[mask])) * 100.0)
        mspe = float(np.mean((err[mask] / y_true[mask]) ** 2) * 100.0)
    else:
        mape = np.nan
        mspe = np.nan

    result_with_errors["abs_error"] = np.abs(err)
    result_with_errors["squared_error"] = err**2
    result_with_errors["ape_%"] = np.where(mask, np.abs(err / y_true) * 100.0, np.nan)
    result_with_errors["spe_%^2"] = np.where(mask, (err / y_true) ** 2 * 100.0, np.nan)

    metrics = pd.DataFrame(
        [
            {
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "MAPE (%)": mape,
                "MSPE (%^2)": mspe,
            }
        ]
    )
    return metrics, result_with_errors


def run_main_forecast(
    model: timesfm.TimesFm,
    df_all: pd.DataFrame,
    n_steps: int,
) -> pd.DataFrame:
    col = "y"
    preds: List[float] = []

    for t in range(n_steps, len(df_all)):
        train_series = df_all[col].iloc[:t]
        pf, _ = model.forecast(
            [np.asarray(train_series, dtype=np.float32)],
            freq=[0] * len(train_series),
        )
        preds.append(float(pf[0]))

    forecast_index = df_all.index[n_steps:]
    forecast_series = pd.Series(preds, index=forecast_index, name="forecast")
    actual_series = df_all[col].iloc[n_steps:].rename("actual")
    combined = pd.concat([actual_series, forecast_series], axis=1)
    if len(combined) > 0:
        combined = combined.iloc[:-1]
    return combined


def build_dynamic_covariates(
    groups: Sequence[Tuple[object, pd.DataFrame]],
    covariate_columns: Dict[str, str],
    horizon: int,
) -> Dict[str, Sequence[np.ndarray]]:
    covariates: Dict[str, Sequence[np.ndarray]] = {}
    for cov_name, column in covariate_columns.items():
        hist_arrays = [g[column].to_numpy(dtype=float) for _, g in groups]
        fut_arrays = [np.repeat(h[-1], horizon) for h in hist_arrays]
        covariates[cov_name] = [np.concatenate([h, f]) for h, f in zip(hist_arrays, fut_arrays)]
    return covariates


def run_covariate_forecast(
    model: timesfm.TimesFm,
    df_all: pd.DataFrame,
    n_steps: int,
    covariate_columns: Dict[str, str],
) -> pd.DataFrame:
    preds: List[float] = []
    for t in range(n_steps, len(df_all)):
        train_df = df_all.iloc[:t].copy()
        groups = list(train_df.groupby("unique_id", sort=False))
        inputs = [g["y"].to_numpy(dtype=float) for _, g in groups]
        dynamic_covs = build_dynamic_covariates(groups, covariate_columns, horizon=1)

        cov_forecast, _ = model.forecast_with_covariates(
            inputs=inputs,
            dynamic_numerical_covariates=dynamic_covs,
            freq=[0] * len(train_df),
            xreg_mode="xreg + timesfm",
            ridge=0.0,
            force_on_cpu=False,
            normalize_xreg_target_per_input=True,
        )
        preds.append(float(cov_forecast[0]))

    forecast_index = df_all.index[n_steps:]
    forecast_series = pd.Series(preds, index=forecast_index, name="forecast")
    actual_series = df_all["y"].iloc[n_steps:].rename("actual")
    combined = pd.concat([actual_series, forecast_series], axis=1)
    if len(combined) > 0:
        combined = combined.iloc[:-1]
    return combined


def plot_results(df_all: pd.DataFrame, result_df: pd.DataFrame, title: str) -> None:
    if result_df.empty:
        return
    plt.figure(figsize=(11, 4))
    trimmed_actual = df_all.iloc[:-1]
    plt.plot(trimmed_actual.index, trimmed_actual["y"], label="Actual")
    plt.plot(result_df.index, result_df["forecast"], label="Forecast")
    fc_start = result_df.index[0]
    plt.axvline(fc_start, linestyle="--", label="Forecast start")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()


def save_outputs(
    scenario: Scenario,
    n_steps: int,
    result_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    results_dir: Path,
    suffix: str
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    forecast_path = results_dir / f"{scenario.results_prefix}_{n_steps}_{suffix}.csv"
    metrics_path = results_dir / f"{scenario.results_prefix}_{n_steps}_{suffix}_metrics.csv"
    result_df.to_csv(forecast_path, index=True)


def run_scenario(
    scenario: Scenario,
    backend: str,
    n_steps: int,
    plot: bool,
    data_path: Path,
    results_dir: Path,
    suffix: str
) -> None:
    print(f"\n=== Running {scenario.key}: {scenario.description} ===")
    df_all = load_dataframe(data_path)
    ensure_minimum_history(df_all, n_steps, scenario.key)

    model = build_model(scenario, backend)
    if scenario.uses_covariates:
        if not scenario.covariate_columns:
            raise ValueError(f"Scenario '{scenario.key}' requires covariate columns.")
        result_df = run_covariate_forecast(model, df_all, n_steps, scenario.covariate_columns)
    else:
        result_df = run_main_forecast(model, df_all, n_steps)

    metrics_df, result_with_errors = compute_metrics(result_df)
    print(metrics_df.round(4))

    save_outputs(scenario, n_steps, result_with_errors, metrics_df, results_dir, suffix)

    if plot:
        title = f"{scenario.description} (n_steps={n_steps})"
        plot_results(df_all, result_with_errors, title)


def main() -> None:
    args = parse_args()
    backend = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"Using backend: {backend}")

    data_path_overrides: Dict[str, Path] = {}
    if args.data_paths:
        for mapping in args.data_paths:
            if "=" not in mapping:
                raise ValueError(f"Invalid data-path mapping '{mapping}', expected key=path.")
            key, path_str = mapping.split("=", 1)
            if key not in SCENARIOS:
                raise ValueError(f"Unknown scenario key '{key}' in data-path override.")
            data_path_overrides[key] = Path(path_str)

    results_dir_overrides: Dict[str, Path] = {}
    if args.results_dirs:
        for mapping in args.results_dirs:
            if "=" not in mapping:
                raise ValueError(f"Invalid results-dir mapping '{mapping}', expected key=dir.")
            key, dir_str = mapping.split("=", 1)
            if key not in SCENARIOS:
                raise ValueError(f"Unknown scenario key '{key}' in results-dir override.")
            results_dir_overrides[key] = Path(dir_str)

    covariate_overrides: Dict[str, Dict[str, str]] = {}
    if args.covariate_columns:
        for mapping in args.covariate_columns:
            if "=" not in mapping:
                raise ValueError(f"Invalid covariate-columns mapping '{mapping}', expected key=mapping.")
            key, columns_str = mapping.split("=", 1)
            key = key.strip()
            if key not in SCENARIOS:
                raise ValueError(f"Unknown scenario key '{key}' in covariate-columns override.")
            if not columns_str.strip():
                raise ValueError(
                    f"Invalid covariate-columns mapping '{mapping}', expected at least one name:column pair."
                )
            column_pairs = [pair for pair in columns_str.split(",") if pair.strip()]
            if not column_pairs:
                raise ValueError(
                    f"Invalid covariate-columns mapping '{mapping}', expected at least one name:column pair."
                )
            override_for_key = covariate_overrides.setdefault(key, {})
            for pair in column_pairs:
                if ":" not in pair:
                    raise ValueError(
                        f"Invalid covariate pair '{pair}' in mapping '{mapping}', expected name:column."
                    )
                cov_name, column_name = pair.split(":", 1)
                cov_name = cov_name.strip()
                column_name = column_name.strip()
                if not cov_name or not column_name:
                    raise ValueError(
                        f"Invalid covariate pair '{pair}' in mapping '{mapping}', expected name:column."
                    )
                override_for_key[cov_name] = column_name

    for key in args.runs:
        scenario = SCENARIOS[key]
        if key in covariate_overrides:
            scenario = replace(
                scenario,
                covariate_columns=covariate_overrides[key],
                uses_covariates=True,
            )
        data_path = data_path_overrides.get(key, DEFAULT_DATA_PATHS[key])
        results_dir = results_dir_overrides.get(key, DEFAULT_RESULTS_DIRS[key])
        run_scenario(scenario, backend, args.n_steps, args.plot, data_path, results_dir, args.suffix)


if __name__ == "__main__":
    main()
