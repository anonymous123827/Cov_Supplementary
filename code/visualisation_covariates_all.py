import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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

name = "international_students_0_100_pressure_moirai_3_google_trends"
google_df = pd.read_csv(f"student_data\international_new_students_2006\covariates\{name}.csv", index_col=0, parse_dates=True)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(google_df.index[:-1], google_df["google_trends_ewma2y"][:-1], label="google_trends_ewma2y", linewidth=1.4)
ax.plot(google_df.index[:-1], google_df["google_trends_ewma3y"][:-1], label="google_trends_ewma3y", linewidth=1.4)
ax.plot(google_df.index[1:-1], google_df["google_trends_lag1y"][1:-1], label="google_trends_lag1y", linewidth=1.4)

ax.axvline(pd.Timestamp("2016"), color="#d62728", linestyle="--", linewidth=1.4, label="Train/Test split (2016)")
ax.set_title("Google Trends Covariate")
ax.set_xlabel("Time")
ax.set_ylabel("Search Trend Value")
apply_axis_style(ax)
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.0, frameon=False)
fig.tight_layout()
fig.savefig(f"results_all\{name}.pdf", dpi=200, bbox_inches="tight")
plt.close(fig)

name = "domestic_students_0_100_pressure_moirai"
pressure_df = pd.read_csv(f"student_data\domestic_new_students_2006\covariates\{name}.csv", index_col=0, parse_dates=True)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(pressure_df.index[:-1], pressure_df["pressure"][:-1], label="IOCI", linewidth=1.4)

ax.axvline(pd.Timestamp("2016"), color="#d62728", linestyle="--", linewidth=1.4, label="Train/Test split (2016)")
ax.set_title("IOCI Covariate")
ax.set_xlabel("Time")
ax.set_ylabel("IOCI Value")
apply_axis_style(ax)
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.0, frameon=False)
fig.tight_layout()
fig.savefig(f"results_all\{name}.pdf", dpi=200, bbox_inches="tight")
plt.close(fig)