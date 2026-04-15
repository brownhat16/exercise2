"""Run the Statistical Inference project analysis for UPI and digital payments."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
MONTHLY_DATA_FILE = ROOT / "upi_monthly_dataset.csv"
PERIOD_TESTS_FILE = ROOT / "upi_period_tests.csv"
FESTIVAL_TESTS_FILE = ROOT / "upi_festival_tests.csv"
REGRESSION_FILE = ROOT / "upi_regression_results.csv"
AUTOCORRELATION_FILE = ROOT / "upi_autocorrelation_tests.csv"
ANNUAL_ROBUSTNESS_FILE = ROOT / "upi_annual_growth_robustness.csv"
FIGURE_FILE = ROOT / "upi_project_overview.png"

PRE_CUTOFF = pd.Timestamp("2022-01-01")
HAC_LAGS = 3


def load_monthly_data() -> pd.DataFrame:
    """Load the monthly UPI panel and derive helper variables."""
    df = pd.read_csv(MONTHLY_DATA_FILE, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df["post_2022"] = (df["date"] >= PRE_CUTOFF).astype(int)
    df["t"] = np.arange(len(df), dtype=float)
    return df


def mean_ci(series: pd.Series, confidence: float = 0.95) -> tuple[float, float, float]:
    """Return mean and t-based confidence interval."""
    clean = series.dropna().astype(float)
    n = len(clean)
    mean = clean.mean()
    se = clean.std(ddof=1) / np.sqrt(n)
    ci_low, ci_high = stats.t.interval(confidence, n - 1, loc=mean, scale=se)
    return mean, ci_low, ci_high


def ljung_box_test(series: pd.Series, lags: list[int]) -> pd.DataFrame:
    """Compute Ljung-Box statistics without relying on statsmodels."""
    clean = series.dropna().astype(float).to_numpy()
    n = clean.size
    centered = clean - clean.mean()
    denom = np.sum(centered**2)
    records = []

    for lag in lags:
        autocorr_sq_terms = []
        for k in range(1, lag + 1):
            numerator = np.sum(centered[k:] * centered[:-k])
            rho_k = numerator / denom
            autocorr_sq_terms.append((rho_k**2) / (n - k))
        q_stat = n * (n + 2) * np.sum(autocorr_sq_terms)
        p_value = 1 - stats.chi2.cdf(q_stat, df=lag)
        records.append({"lag": lag, "lb_stat": q_stat, "p_value": p_value})

    return pd.DataFrame(records)


def ols_hac(y: pd.Series, x: pd.DataFrame, max_lag: int = HAC_LAGS) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate OLS with Newey-West HAC covariance."""
    y_array = np.asarray(y, dtype=float)
    x_array = np.asarray(x, dtype=float)
    n_obs, n_coef = x_array.shape

    xtx_inv = np.linalg.inv(x_array.T @ x_array)
    beta = xtx_inv @ (x_array.T @ y_array)
    resid = y_array - x_array @ beta

    s_matrix = np.zeros((n_coef, n_coef))
    for t in range(n_obs):
        x_t = x_array[t : t + 1].T
        s_matrix += resid[t] ** 2 * (x_t @ x_t.T)

    for lag in range(1, max_lag + 1):
        weight = 1 - lag / (max_lag + 1)
        gamma = np.zeros((n_coef, n_coef))
        for t in range(lag, n_obs):
            x_t = x_array[t : t + 1].T
            x_lag = x_array[t - lag : t - lag + 1].T
            gamma += resid[t] * resid[t - lag] * (x_t @ x_lag.T)
        s_matrix += weight * (gamma + gamma.T)

    covariance = xtx_inv @ s_matrix @ xtx_inv
    standard_errors = np.sqrt(np.diag(covariance))
    t_stats = beta / standard_errors
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n_obs - n_coef))
    return beta, covariance, standard_errors, p_values


def wald_test(beta: np.ndarray, covariance: np.ndarray, restriction: np.ndarray) -> tuple[float, float]:
    """Run a Wald chi-square test for linear restrictions R * beta = 0."""
    diff = restriction @ beta
    middle = np.linalg.inv(restriction @ covariance @ restriction.T)
    statistic = float(diff.T @ middle @ diff)
    p_value = 1 - stats.chi2.cdf(statistic, df=restriction.shape[0])
    return statistic, p_value


def compare_pre_post_growth(df: pd.DataFrame) -> pd.DataFrame:
    """Compare growth rates before 2022 and from 2022 onward."""
    growth_df = df.dropna(subset=["growth_volume", "growth_value"]).copy()
    records = []

    for variable in ["growth_volume", "growth_value"]:
        pre = growth_df.loc[growth_df["post_2022"] == 0, variable]
        post = growth_df.loc[growth_df["post_2022"] == 1, variable]

        pre_mean, pre_low, pre_high = mean_ci(pre)
        post_mean, post_low, post_high = mean_ci(post)
        welch = stats.ttest_ind(post, pre, equal_var=False, nan_policy="omit")
        mann_whitney = stats.mannwhitneyu(post, pre, alternative="two-sided")

        records.append(
            {
                "variable": variable,
                "n_pre": len(pre),
                "n_post": len(post),
                "mean_pre": pre_mean,
                "ci_pre_low": pre_low,
                "ci_pre_high": pre_high,
                "mean_post": post_mean,
                "ci_post_low": post_low,
                "ci_post_high": post_high,
                "difference_post_minus_pre": post_mean - pre_mean,
                "welch_t_stat": welch.statistic,
                "welch_p_value": welch.pvalue,
                "mann_whitney_u": mann_whitney.statistic,
                "mann_whitney_p_value": mann_whitney.pvalue,
            }
        )

    return pd.DataFrame(records)


def compare_festival_months(df: pd.DataFrame) -> pd.DataFrame:
    """Test whether festival months differ from non-festival months."""
    records = []

    for variable in ["upi_volume", "upi_value", "log_volume", "log_value"]:
        festival = df.loc[df["festival_month"] == 1, variable].dropna()
        non_festival = df.loc[df["festival_month"] == 0, variable].dropna()
        welch = stats.ttest_ind(festival, non_festival, equal_var=False, nan_policy="omit")
        mann_whitney = stats.mannwhitneyu(festival, non_festival, alternative="two-sided")

        records.append(
            {
                "variable": variable,
                "n_festival": len(festival),
                "n_non_festival": len(non_festival),
                "mean_festival": festival.mean(),
                "mean_non_festival": non_festival.mean(),
                "difference_festival_minus_non_festival": festival.mean() - non_festival.mean(),
                "welch_t_stat": welch.statistic,
                "welch_p_value": welch.pvalue,
                "mann_whitney_u": mann_whitney.statistic,
                "mann_whitney_p_value": mann_whitney.pvalue,
            }
        )

    return pd.DataFrame(records)


def run_growth_regressions(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate seasonality-adjusted growth regressions with HAC standard errors."""
    growth_df = df.dropna(subset=["growth_volume", "growth_value"]).copy()
    month_dummies = pd.get_dummies(growth_df["month"].astype(int), prefix="m", drop_first=True, dtype=float)
    design = pd.concat(
        [
            pd.Series(1.0, index=growth_df.index, name="const"),
            growth_df[["post_2022"]].astype(float),
            month_dummies,
        ],
        axis=1,
    )

    records = []
    parameter_names = list(design.columns)

    for dependent_variable in ["growth_volume", "growth_value"]:
        beta, covariance, standard_errors, p_values = ols_hac(growth_df[dependent_variable], design)

        for variable_name in ["post_2022", "m_10", "m_11", "m_12"]:
            idx = parameter_names.index(variable_name)
            records.append(
                {
                    "model": dependent_variable,
                    "term": variable_name,
                    "coefficient": beta[idx],
                    "std_error_hac": standard_errors[idx],
                    "p_value": p_values[idx],
                    "test_type": "coefficient",
                }
            )

        restriction = np.zeros((3, len(parameter_names)))
        for row_idx, variable_name in enumerate(["m_10", "m_11", "m_12"]):
            restriction[row_idx, parameter_names.index(variable_name)] = 1
        statistic, p_value = wald_test(beta, covariance, restriction)
        records.append(
            {
                "model": dependent_variable,
                "term": "joint_oct_nov_dec",
                "coefficient": statistic,
                "std_error_hac": np.nan,
                "p_value": p_value,
                "test_type": "wald_joint",
            }
        )

    return pd.DataFrame(records)


def run_festival_trend_regressions(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate log-level regressions with a linear trend and festival dummy."""
    records = []
    design = pd.concat(
        [
            pd.Series(1.0, index=df.index, name="const"),
            df[["t", "festival_month"]].astype(float),
        ],
        axis=1,
    )
    parameter_names = list(design.columns)

    for dependent_variable in ["log_volume", "log_value"]:
        beta, covariance, standard_errors, p_values = ols_hac(df[dependent_variable], design)
        for variable_name in ["t", "festival_month"]:
            idx = parameter_names.index(variable_name)
            records.append(
                {
                    "model": dependent_variable,
                    "term": variable_name,
                    "coefficient": beta[idx],
                    "std_error_hac": standard_errors[idx],
                    "p_value": p_values[idx],
                    "test_type": "coefficient",
                }
            )

    return pd.DataFrame(records)


def build_annual_robustness(df: pd.DataFrame) -> pd.DataFrame:
    """Construct annual average growth rates and flag complete years."""
    annual = (
        df.dropna(subset=["growth_volume", "growth_value"])
        .groupby("year", as_index=False)
        .agg(
            avg_monthly_growth_volume=("growth_volume", "mean"),
            avg_monthly_growth_value=("growth_value", "mean"),
            observations=("date", "size"),
        )
        .sort_values("year")
        .reset_index(drop=True)
    )
    annual["complete_year"] = (annual["observations"] == 12).astype(int)
    return annual


def make_overview_figure(df: pd.DataFrame) -> None:
    """Save a compact overview figure for the report."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    axes[0, 0].plot(df["date"], df["upi_volume"] / 1e9, color="#0b6e4f", linewidth=2)
    axes[0, 0].set_title("Monthly UPI Volume (billions)")
    axes[0, 0].set_ylabel("Billions")

    axes[0, 1].plot(df["date"], df["upi_value"] / 1e12, color="#aa3a00", linewidth=2)
    axes[0, 1].set_title("Monthly UPI Value (trillion rupees)")
    axes[0, 1].set_ylabel("Rs trillion")

    axes[1, 0].plot(df["date"], df["growth_volume"], color="#1f4b99", linewidth=1.8)
    axes[1, 0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1, 0].set_title("Monthly Growth in UPI Volume (%)")
    axes[1, 0].set_ylabel("Percent")

    axes[1, 1].plot(df["date"], df["growth_value"], color="#7a1ea1", linewidth=1.8)
    axes[1, 1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1, 1].set_title("Monthly Growth in UPI Value (%)")
    axes[1, 1].set_ylabel("Percent")

    for axis in axes.ravel():
        axis.set_xlabel("Date")
        axis.grid(alpha=0.25)

    fig.savefig(FIGURE_FILE, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_autocorrelation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Run Ljung-Box tests on growth series."""
    records = []
    growth_df = df.dropna(subset=["growth_volume", "growth_value"])

    for variable in ["growth_volume", "growth_value"]:
        result = ljung_box_test(growth_df[variable], lags=[1, 3, 6])
        for row in result.itertuples(index=False):
            records.append(
                {
                    "variable": variable,
                    "lag": row.lag,
                    "lb_stat": row.lb_stat,
                    "p_value": row.p_value,
                }
            )

    return pd.DataFrame(records)


def main() -> None:
    df = load_monthly_data()

    period_tests = compare_pre_post_growth(df)
    festival_tests = compare_festival_months(df)
    growth_regressions = run_growth_regressions(df)
    festival_trend_regressions = run_festival_trend_regressions(df)
    regression_results = pd.concat([growth_regressions, festival_trend_regressions], ignore_index=True)
    autocorrelation_tests = build_autocorrelation_table(df)
    annual_robustness = build_annual_robustness(df)

    period_tests.to_csv(PERIOD_TESTS_FILE, index=False)
    festival_tests.to_csv(FESTIVAL_TESTS_FILE, index=False)
    regression_results.to_csv(REGRESSION_FILE, index=False)
    autocorrelation_tests.to_csv(AUTOCORRELATION_FILE, index=False)
    annual_robustness.to_csv(ANNUAL_ROBUSTNESS_FILE, index=False)
    make_overview_figure(df)

    print("Saved:")
    print(f"  {PERIOD_TESTS_FILE.name}")
    print(f"  {FESTIVAL_TESTS_FILE.name}")
    print(f"  {REGRESSION_FILE.name}")
    print(f"  {AUTOCORRELATION_FILE.name}")
    print(f"  {ANNUAL_ROBUSTNESS_FILE.name}")
    print(f"  {FIGURE_FILE.name}")
    print()

    for row in period_tests.itertuples(index=False):
        print(
            f"{row.variable}: pre={row.mean_pre:.2f}, post={row.mean_post:.2f}, "
            f"diff={row.difference_post_minus_pre:.2f}, Welch p={row.welch_p_value:.4f}, "
            f"Mann-Whitney p={row.mann_whitney_p_value:.4f}"
        )


if __name__ == "__main__":
    main()
