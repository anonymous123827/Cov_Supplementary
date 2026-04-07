#!/usr/bin/env python3
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

n_steps = 9  # forecast horizon

# Global aesthetics for consistent readability
plt.rcParams.update({
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

def apply_axis_style(ax):
    ax.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.35)
    ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
def load_mode(mode: str):
    match mode:
        case "no_co_domestic":
            FILES = {
                "ARIMA": "results_domestic_2006/no_co/AutoArima_forecasts_no_co.csv",
                "TimesFM_200": "results_domestic_2006/no_co/TimesFM200_9_no_co.csv",
                "TimesFM_500": "results_domestic_2006/no_co/TimesFM500_9_no_co.csv",
                "Moirai_large": "results_domestic_2006/no_co/actual_vs_forecast_9_moirai_large_auto_no_co.csv",
                "Moirai_base": "results_domestic_2006/no_co/Moirai_Base_None.csv",
                "Moirai_small": "results_domestic_2006/no_co/Moirai_Small_None.csv",
                "Chronos_bolt_base": "results_domestic_2006/no_co/actual_vs_forecast_chronos-bolt-base_9_no_co.csv",
                "Chronos_bolt_mini": "results_domestic_2006/no_co/actual_vs_forecast_chronos-bolt-mini_9_no_co.csv",
                "Chronos_bolt_small": "results_domestic_2006/no_co/actual_vs_forecast_chronos-bolt-small_9_no_co.csv",
                "Chronos_bolt_tiny": "results_domestic_2006/no_co/Chronos_Bolt_Tiny_None.csv",
                "Chronos_2": "results_domestic_2006/no_co/Chronos-2_None.csv",
                "Persistence": "results_domestic_2006/no_co/Persistence_merged.csv",
            }
        case "domestic_IOCI":
            FILES = {
                "ARIMAX": "results_domestic_2006/IOCI/AutoArima_forecasts_pressure.csv",
                "TimesFM_200": "results_domestic_2006/IOCI/TimesFM200_9_pressure.csv",
                "TimesFM_500": "results_domestic_2006/IOCI/TimesFM500_9_pressure.csv",
                "Moirai_large": "results_domestic_2006/IOCI/actual_vs_forecast_Moirai_large_9_pressure.csv",
                "Moirai_base": "results_domestic_2006/IOCI/actual_vs_forecast_Moirai_base_9_pressure.csv",
                "Moirai_small": "results_domestic_2006/IOCI/actual_vs_forecast_Moirai_small_9_pressure.csv",
                "Chronos_bolt_base": "results_domestic_2006/IOCI/Chronos_Bolt_Base_IOCI.csv",
                "Chronos_bolt_mini": "results_domestic_2006/IOCI/Chronos_Bolt_Mini_IOCI.csv",
                "Chronos_bolt_small": "results_domestic_2006/IOCI/Chronos_Bolt_Small_IOCI.csv",
                "Chronos_bolt_tiny": "results_domestic_2006/IOCI/Chronos_Bolt_Tiny_IOCI.csv",
                "Chronos_2": "results_domestic_2006/IOCI/actual_vs_forecast_chronos-2_9_pressure.csv",
                "Persistence": "results_domestic_2006/IOCI/Persistence_merged.csv",
            }
        case "no_co_international":
            FILES = {
                "ARIMA": "results_international_2006/no_co/AutoArima_forecasts_no_co.csv",
                "TimesFM_200": "results_international_2006/no_co/TimesFM200_9_no_co.csv",
                "TimesFM_500": "results_international_2006/no_co/TimesFM500_9_no_co.csv",
                "Moirai_large": "results_international_2006/no_co/actual_vs_forecast_9_moirai_large_auto_no_co.csv",
                "Moirai_base": "results_international_2006/no_co/Moirai_Base_None.csv",
                "Moirai_small": "results_international_2006/no_co/Moirai_Small_None.csv",
                "Chronos_bolt_base": "results_international_2006/no_co/Chronos_Bolt_Base_None.csv",
                "Chronos_bolt_mini": "results_international_2006/no_co/Chronos_Bolt_Mini_None.csv",
                "Chronos_bolt_small": "results_international_2006/no_co/Chronos_Bolt_Small_None.csv",
                "Chronos_bolt_tiny": "results_international_2006/no_co/Chronos_Bolt_Tiny_None.csv",
                "Chronos_2": "results_international_2006/no_co/Chronos-2_None.csv",
                "Persistence": "results_international_2006/no_co/Persistence_merged.csv",
            }
        
        case "international_google_3":
            FILES = {
                "ARIMAX": "results_international_2006/google_3/AutoArima_forecasts_google_3.csv",
                "TimesFM_200": "results_international_2006/google_3/TimesFM200_9_google_3.csv",
                "TimesFM_500": "results_international_2006/google_3/TimesFM500_9_google_3.csv",
                "Moirai_large": "results_international_2006/google_3/actual_vs_forecast_Moirai_large_9_google_3.csv",
                "Moirai_base": "results_international_2006/google_3/Moirai_Base_Google_Trends.csv",
                "Moirai_small": "results_international_2006/google_3/Moirai_Small_Google_Trends.csv",
                "Chronos_bolt_base": "results_international_2006/google_3/Chronos_Bolt_Base_Google_Trends.csv",
                "Chronos_bolt_mini": "results_international_2006/google_3/Chronos_Bolt_Mini_Google_Trends.csv",
                "Chronos_bolt_small": "results_international_2006/google_3/Chronos_Bolt_Small_Google_Trends.csv",
                "Chronos_bolt_tiny": "results_international_2006/google_3/actual_vs_forecast_chronos-bolt-tiny_9_google_3.csv",
                "Persistence": "results_international_2006/google_3/Persistence_merged.csv",
                "Chronos_2": "results_international_2006/google_3/actual_vs_forecast_chronos-2_9_google_3.csv",
            }
        case _:
            raise ValueError(f"Unknown mode: {mode}")

    return FILES

def build_aligned(file: str):
    FILES = load_mode(file)
    baseline_model_name = "ARIMAX" if "ARIMAX" in FILES else "ARIMA"
    autoArima = pd.read_csv(FILES[baseline_model_name])
    autoArima["ds"] = pd.to_datetime(autoArima["ds"])
    base = autoArima[["ds","AutoARIMA"]].rename(columns={"AutoARIMA": baseline_model_name}).copy()
    
    # base = auto[["ds","y","AutoARIMA"]].rename(columns={"y":"actual"}).copy()
    # GRAND
    g200 = pd.read_csv(FILES["TimesFM_200"]); g200["ds"] = pd.to_datetime(g200["ds"])
    g500 = pd.read_csv(FILES["TimesFM_500"]); g500["ds"] = pd.to_datetime(g500["ds"])
    base = base.merge(g200[["ds","forecast"]].rename(columns={"forecast":"TimesFM_200"}), on="ds", how="left")
    base = base.merge(g500[["ds","forecast"]].rename(columns={"forecast":"TimesFM_500"}), on="ds", how="left")

    persistence = pd.read_csv(FILES["Persistence"]); persistence["ds"] = pd.to_datetime(persistence["ds"])
    base = base.merge(persistence[["ds","y_forecast"]].rename(columns={"y_forecast":"Persistence"}), on="ds", how="left")

    # Moirai (by index order)
    def attach_moirai(df_base, path, name):
        df = pd.read_csv(path)
        if "Forecast" not in df.columns:
            return df_base
        n = min(len(df_base), len(df))
        out = df_base.copy()
        out.loc[out.index[:n], name] = df["Forecast"].values[:n]
        return out

    base = attach_moirai(base, FILES["Moirai_large"], "Moirai_large")
    base = attach_moirai(base, FILES["Moirai_base"], "Moirai_base")
    base = attach_moirai(base, FILES["Moirai_small"], "Moirai_small")

    # Chronos (by index order, 'yhat')
    def attach_chronos(df_base, path, name):
        df = pd.read_csv(path)
        col = "yhat" if "yhat" in df.columns else None
        if col is None:
            return df_base
        n = min(len(df_base), len(df))
        out = df_base.copy()
        out.loc[out.index[:n], name] = df[col].values[:n]
        return out

    base = attach_chronos(base, FILES["Chronos_bolt_base"], "Chronos_bolt_base")
    base = attach_chronos(base, FILES["Chronos_bolt_mini"], "Chronos_bolt_mini")
    base = attach_chronos(base, FILES["Chronos_bolt_tiny"], "Chronos_bolt_tiny")
    base = attach_chronos(base, FILES["Chronos_bolt_small"], "Chronos_bolt_small")
    base = attach_chronos(base, FILES["Chronos_2"], "Chronos_2")

    return base

def smape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred))
    mask = denom == 0
    out = np.zeros_like(denom, dtype=float)
    out[~mask] = 200.0 * np.abs(y_pred[~mask] - y_true[~mask]) / denom[~mask]
    return np.mean(out[~np.isnan(out)]) if np.any(~np.isnan(out)) else np.nan

def mape(y, yhat, eps=1e-8):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    # ignore positions where |y| <= eps to avoid division by ~0
    mask = np.abs(y) > eps
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.abs((yhat[mask] - y[mask]) / y[mask])) * 100.0)

def main():
    actual_inter_df = pd.read_csv("actual_new_international_students_2006.csv")
    actual_domestic_df = pd.read_csv("actual_new_domestic_students_2006.csv")

    files_arr = [
        'no_co_domestic',
        'domestic_IOCI',
        'no_co_international',
        'international_google_3',
    ]
    
    for file in files_arr:
        df = build_aligned(file)
        # df.to_csv("aligned_actual_vs_forecasts.csv", index=False)
        if "inter" in file:
            actual_df = actual_inter_df.copy()
        elif "domestic" in file:
            actual_df = actual_domestic_df.copy()
        else:
            raise ValueError(f"Unknown file type: {file}")
        # Plot Actual vs Models
        fig, ax = plt.subplots(figsize=(10, 5))
        actual_df["ds"] = pd.to_datetime(actual_df["ds"])
        ax.plot(
            actual_df["ds"],
            actual_df["actual"],
            label="Actual",
            color="black",
            linewidth=2.2,
            zorder=3,
        )
        ax.axvline(
            pd.Timestamp("2016"),
            color="#d62728",
            linestyle="--",
            linewidth=1.4,
            label="Train/Test split (2016)",
            zorder=2,
        )
        models = ["Chronos_2","Persistence", "ARIMA", "ARIMAX", "TimesFM_200","TimesFM_500","Moirai_large","Moirai_base","Moirai_small","Chronos_bolt_base","Chronos_bolt_mini","Chronos_bolt_tiny", "Chronos_bolt_small"]
        cmap = plt.get_cmap("tab20")
        for m in models:
            if m in df.columns:
                print(m)
                ax.plot(
                    df["ds"],
                    df[m],
                    label=m,
                    color=cmap(models.index(m) % cmap.N),
                    linewidth=1.2,
                    alpha=0.9,
                    zorder=1,
                )
        title_prefix = "International" if "inter" in file else "Domestic"
        ax.set_title(f"{title_prefix} Student Enrolment Forecasts")
        ax.set_xlabel("Time")
        ax.set_ylabel("Student Enrolments")
        apply_axis_style(ax)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.0, frameon=False)
        fig.tight_layout()
        plt.savefig(f"results_all/actual_vs_models_{file}.pdf", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # Metrics
        rows = []
        for m in models:
            if m not in df.columns:
                continue
            tmp = df[[m]].dropna()
            if len(tmp)==0:
                continue
            y = actual_df["actual"][n_steps:].values
            yhat = tmp[m].values
            print(m)
            print(y)
            print(yhat)
            rows.append({
                "model": m,
                "MAE": round(float(np.mean(np.abs(yhat - y))), 2),
                "RMSE": round(float(np.sqrt(np.mean((yhat - y) ** 2))), 2),
                # "MSE": float(np.mean((yhat - y) ** 2)),
                "SMAPE": round(float(smape(y, yhat)), 2),
                "MAPE": round(float(mape(y, yhat)), 2)
            })
        metrics = pd.DataFrame(rows).sort_values("MAE")
        metrics.to_csv(f"results_all/model_metrics_{file}.csv", index=False)
        print(metrics)

if __name__ == "__main__":
    main()
