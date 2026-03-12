import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule


DEFAULT_MODEL_SIZES = ["small", "base", "large"]
CV_MODE_CHOICES = {
    "manual": "Expanding (manually looping, 1-step ahead)",
    "library": "Expanding windows via Uni2TS library",
}


def parse_patch_size(value: str) -> Union[str, int]:
    """Allow passing integers or the literal 'auto' for patch size."""
    lowered = value.lower()
    if lowered == "auto":
        return "auto"
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid patch size '{value}'. Use an integer or 'auto'."
        ) from exc


def name_or_list(cols: Sequence[str], list_only: bool = False) -> Union[str, List[str]]:
    cols = list(cols)
    if list_only or len(cols) != 1:
        return cols
    return cols[0]


def load_dataset(
    target_path: Path,
    covariate_path: Optional[Path],
    past_path: Optional[Path],
    convert_to_year: bool,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame], pd.DataFrame]:
    """Load CSV files and optionally convert indices to annual periods."""

    if not target_path.exists():
        raise FileNotFoundError(f"Target file not found: {target_path}")

    target_df = pd.read_csv(target_path, index_col=0, parse_dates=True).sort_index()
    target_display_df = target_df.copy()  # preserve original index for plotting
    if convert_to_year:
        target_df = target_df.to_period("A")

    cov_df: Optional[pd.DataFrame] = None
    if covariate_path:
        if not covariate_path.exists():
            raise FileNotFoundError(f"Covariate file not found: {covariate_path}")
        cov_df = pd.read_csv(covariate_path, index_col=0, parse_dates=True).sort_index()
        if convert_to_year:
            cov_df = cov_df.to_period("A")

    past_df: Optional[pd.DataFrame] = None
    if past_path:
        if not past_path.exists():
            raise FileNotFoundError(f"Past feature file not found: {past_path}")
        past_df = pd.read_csv(past_path, index_col=0, parse_dates=True).sort_index()

    return target_df, cov_df, past_df, target_display_df


def build_predictor(
    model_name: str,
    size: str,
    prediction_length: int,
    context_length: int,
    patch_size: Union[str, int],
    target_dim: int,
    co_dim: int,
    past_dim: int,
    num_samples: int,
    batch_size: int,
):
    """Instantiate a Moirai predictor."""
    if model_name != "moirai":
        print(
            f"Warning: model '{model_name}' is not supported in this script. "
            "Falling back to 'moirai'."
        )
    module = MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{size}")
    model = MoiraiForecast(
        module=module,
        prediction_length=prediction_length,
        context_length=context_length,
        patch_size=patch_size,
        num_samples=num_samples,
        target_dim=target_dim,
        feat_dynamic_real_dim=co_dim,
        past_feat_dynamic_real_dim=past_dim,
    )
    return model.create_predictor(batch_size=batch_size)


def concatenate_inputs(
    target_df: pd.DataFrame,
    cov_df: Optional[pd.DataFrame],
    past_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    parts = [target_df]
    if cov_df is not None:
        parts.append(cov_df)
    if past_df is not None:
        parts.append(past_df)
    if len(parts) == 1:
        return target_df
    return pd.concat(parts, axis=1)


def run_forecast(
    target_df: pd.DataFrame,
    cov_df: Optional[pd.DataFrame],
    past_df: Optional[pd.DataFrame],
    display_df: pd.DataFrame,
    model_name: str,
    size: str,
    start_point: int,
    step_horizon: int,
    patch_size: Union[str, int],
    num_samples: int,
    batch_size: int,
    cv_mode: str,
) -> Tuple[pd.DataFrame, plt.Figure, pd.DataFrame]:
    target_cols = list(target_df.columns)
    target_dim = len(target_cols)
    if start_point <= 0:
        raise ValueError("start_point must be positive.")
    if start_point >= len(target_df):
        raise ValueError("start_point must be smaller than the length of the target series.")

    co_cols = name_or_list(cov_df.columns, True) if cov_df is not None else None
    past_cols = name_or_list(past_df.columns, True) if past_df is not None else None

    combined_df = concatenate_inputs(target_df, cov_df, past_df)

    test_len = len(combined_df) - start_point
    if test_len <= 0:
        raise ValueError(
            "Not enough observations after the chosen start point to run a forecast."
        )

    context_len = start_point
    prediction_len = step_horizon

    if cv_mode == "manual":
        if target_dim != 1:
            raise NotImplementedError("Manual expanding mode currently supports univariate targets only.")

        preds_by_dim: List[List[float]] = [[] for _ in range(target_dim)]
        preds_by_dim_prob: List[List[dict]] = [[] for _ in range(target_dim)]

        for k in range(test_len):
            hist_df = combined_df.iloc[: start_point + k]
            predictor = build_predictor(
                model_name=model_name,
                size=size,
                prediction_length=1,
                context_length=start_point + k,
                patch_size=patch_size,
                target_dim=target_dim,
                co_dim=len(cov_df.columns) if cov_df is not None else 0,
                past_dim=len(past_df.columns) if past_df is not None else 0,
                num_samples=num_samples,
                batch_size=batch_size,
            )

            ds_hist = PandasDataset(
                hist_df,
                target=name_or_list(target_cols),
                feat_dynamic_real=co_cols,
                past_feat_dynamic_real=past_cols,
            )

            forecast = next(predictor.predict(ds_hist))
            samples = np.asarray(forecast.samples)
            step0 = samples[:, 0]
            preds_by_dim[0].append(float(np.median(step0)))

            percentiles = [90, 70, 50, 30, 10]
            perc_values = np.nanpercentile(step0, percentiles, axis=0)
            preds_by_dim_prob[0].append(
                {
                    "p90": perc_values[0],
                    "p70": perc_values[1],
                    "p50": perc_values[2],
                    "p30": perc_values[3],
                    "p10": perc_values[4],
                }
            )

        forecast_index = target_df.index[start_point : start_point + test_len]
        forecast_df = pd.DataFrame(preds_by_dim[0], index=forecast_index, columns=["forecast"])
        forecast_prob_df = pd.DataFrame(preds_by_dim_prob[0], index=forecast_index)

        fig, ax = plt.subplots(figsize=(15, 8))
        for col in display_df.columns:
            ax.plot(
                display_df.iloc[:-1].index,
                display_df.iloc[:-1][col],
                label=f"Actual {col}",
                color="#000c66",
                alpha=0.7,
            )
        ax.plot(
            forecast_df.index,
            forecast_df["forecast"],
            label="Forecast (median)",
            linewidth=3,
            color="#CE9DD9",
        )
        ax.set_title("Walk-forward expanding window (1-step ahead)")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Value")
        plt.xticks(rotation=45)
        ax.legend(loc="upper left", ncol=2)
        plt.tight_layout()

        return forecast_df, fig, forecast_prob_df

    # Library-based expanding windows
    multivar_ds = PandasDataset(
        combined_df,
        target=name_or_list(target_cols),
        feat_dynamic_real=co_cols,
        past_feat_dynamic_real=past_cols,
    )

    train, test_template = split(multivar_ds, offset=-(len(combined_df) - start_point))
    test_data = test_template.generate_instances(
        prediction_length=prediction_len,
        windows=(len(combined_df) - start_point) // prediction_len,
        distance=prediction_len,
    )

    predictor = build_predictor(
        model_name=model_name,
        size=size,
        prediction_length=prediction_len,
        context_length=context_len,
        patch_size=patch_size,
        target_dim=target_dim,
        co_dim=len(cov_df.columns) if cov_df is not None else 0,
        past_dim=len(past_df.columns) if past_df is not None else 0,
        num_samples=num_samples,
        batch_size=batch_size,
    )

    forecasts = predictor.predict(test_data.input)

    median_values: List[List[float]] = [[] for _ in range(target_dim)]
    sample_store: List[np.ndarray] = []

    if target_dim == 1:
        for fc in forecasts:
            samples = np.asarray(fc.samples)
            sample_store.append(samples)
            median_values[0].append(float(np.median(samples[:, 0])))
        forecast_index = target_df.index[start_point:]
        forecast_df = pd.DataFrame(median_values[0], index=forecast_index, columns=["forecast"])
    else:
        for fc in forecasts:
            samples = np.asarray(fc.samples)
            medians = np.median(samples[:, 0, :], axis=0)
            for idx, value in enumerate(medians):
                median_values[idx].append(float(value))
        forecast_index = target_df.index[start_point:]
        forecast_df = pd.DataFrame(
            {col: series for col, series in zip(target_cols, median_values)},
            index=forecast_index,
        )

    forecast_prob_df = pd.DataFrame(sample_store, index=forecast_index)

    fig, ax = plt.subplots(figsize=(15, 8))
    for col in display_df.columns:
        ax.plot(
            display_df.iloc[:-1].index,
            display_df.iloc[:-1][col],
            label=f"Actual {col}",
            color="#000c66",
            alpha=0.7,
        )
    for col in forecast_df.columns:
        ax.plot(forecast_df.index, forecast_df[col], label=f"Forecast {col}", linewidth=3)
    ax.set_title("Expanding windows via Uni2TS")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Value")
    plt.xticks(rotation=45)
    ax.legend(loc="upper left", ncol=2)
    plt.tight_layout()

    return forecast_df, fig, forecast_prob_df


def assemble_actual_vs_forecast(
    target_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    forecast_prob_df: pd.DataFrame,
    start_point: int,
) -> pd.DataFrame:
    """Align actuals and predictions for inspection and saving."""
    target_cols = list(target_df.columns)
    n_pred = len(forecast_df)
    actual_slice = target_df.iloc[start_point : start_point + n_pred].copy()
    actual_slice.index = forecast_df.index
    if not target_cols:
        raise ValueError("Target dataframe has no columns.")
    result = pd.concat(
        [
            actual_slice.iloc[:, 0].rename("Actual"),
            forecast_df.iloc[:, 0].rename("Forecast"),
            forecast_prob_df,
        ],
        axis=1,
        join="inner",
    )
    if len(result) > 1:
        result = result.iloc[:-1]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Salesforce Moirai forecasts without the Streamlit UI."
    )
    parser.add_argument("--target", required=True, type=Path, help="CSV with the target time series.")
    parser.add_argument(
        "--covariate",
        type=Path,
        help="Optional CSV with dynamic real-valued covariates.",
    )
    parser.add_argument(
        "--past",
        type=Path,
        help="Optional CSV with past dynamic real-valued features.",
    )
    parser.add_argument(
        "--convert-to-year",
        dest="convert_to_year",
        action="store_true",
        help="Convert indices to yearly PeriodIndex (default).",
    )
    parser.add_argument(
        "--keep-datetime-index",
        dest="convert_to_year",
        action="store_false",
        help="Keep the original datetime index.",
    )
    parser.set_defaults(convert_to_year=True)

    parser.add_argument("--model", default="moirai", help="Model name (currently only 'moirai' is supported).")
    parser.add_argument(
        "--sizes",
        nargs="+",
        choices=DEFAULT_MODEL_SIZES,
        default=DEFAULT_MODEL_SIZES,
        help="Model sizes to evaluate. Defaults to all sizes.",
    )
    parser.add_argument(
        "--start-point",
        type=int,
        default=9,
        help="Context length (number of historical steps to condition on).",
    )
    parser.add_argument(
        "--step-horizon",
        type=int,
        default=1,
        help="Prediction length (number of steps to forecast).",
    )
    parser.add_argument(
        "--patch-size",
        default="auto",
        type=parse_patch_size,
        help="Patch size used by Moirai (integer or 'auto').",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples drawn from the predictor.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--cv-mode",
        choices=list(CV_MODE_CHOICES.keys()),
        default="manual",
        help="Cross-validation / backtest mode.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory where CSVs and figures will be stored.",
    )
    parser.add_argument(
        "--save-plot",
        action="store_true",
        help="If set, save matplotlib figures to the output directory.",
    )
    parser.add_argument(
        "--plot-format",
        default="png",
        help="Image format for saved plots (used with --save-plot).",
    )

    parser.add_argument(
        "--suffix",
        default="no_co",
        help="suffix for saved filename.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    target_df, cov_df, past_df, display_df = load_dataset(
        target_path=args.target,
        covariate_path=args.covariate,
        past_path=args.past,
        convert_to_year=args.convert_to_year,
    )

    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for size in args.sizes:
        print(f"Running Moirai forecast for size='{size}' ({CV_MODE_CHOICES[args.cv_mode]}).")
        forecast_df, fig, forecast_prob_df = run_forecast(
            target_df=target_df,
            cov_df=cov_df,
            past_df=past_df,
            display_df=display_df,
            model_name=args.model,
            size=size,
            start_point=args.start_point,
            step_horizon=args.step_horizon,
            patch_size=args.patch_size,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            cv_mode=args.cv_mode,
        )

        actual_vs_forecast = assemble_actual_vs_forecast(
            target_df=target_df,
            forecast_df=forecast_df,
            forecast_prob_df=forecast_prob_df,
            start_point=args.start_point,
        )

        csv_path = output_dir / f"actual_vs_forecast_Moirai_{size}_{args.start_point}_{args.suffix}.csv"
        forecast_csv_path = output_dir / f"forecast_Moirai_{size}_{args.start_point}_{args.suffix}.csv"
        prob_csv_path = output_dir / f"forecast_prob_Moirai_{size}_{args.start_point}_{args.suffix}.csv"

        actual_vs_forecast.to_csv(csv_path)
        # forecast_df.to_csv(forecast_csv_path)
        # forecast_prob_df.to_csv(prob_csv_path)
        print(f"Saved actual vs forecast data to {csv_path}")

        if args.save_plot:
            plot_path = output_dir / f"forecast_plot_{size}.{args.plot_format}"
            fig.savefig(plot_path, bbox_inches="tight")
            print(f"Saved plot to {plot_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
