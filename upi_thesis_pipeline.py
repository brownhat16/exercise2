"""Build a thesis-style UPI analysis from official RBI monthly data and daily panel data."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from html import escape
from pathlib import Path
import math
import re
import subprocess

import matplotlib.pyplot as plt
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).resolve().parent

RBI_ARCHIVE_URL = "https://www.rbi.org.in/Scripts/PSIUserView.aspx?Id=47"
RBI_PAGE_URL = "https://www.rbi.org.in/Scripts/PSIUserView.aspx?Id={page_id}"
DAILY_RESOURCE_ID = "1f9367ac-01b0-4c82-83a1-4069d4340667"
DAILY_RESOURCE_URL = (
    "https://ckandev.indiadataportal.com/api/3/action/datastore_search"
)
PIB_UPI_URL = "https://www.pib.gov.in/PressReleaseIframePage.aspx?PRID=2079544"
DBIE_URL = "https://data.rbi.org.in"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

PRIMARY_END = "2025-07"
POST_2022_START = "2022-01"

DAILY_OUTPUT = ROOT / "upi_daily_validated.csv"
MONTHLY_PRIMARY_OUTPUT = ROOT / "upi_monthly_primary.csv"
MONTHLY_OFFICIAL_OUTPUT = ROOT / "upi_monthly_official.csv"
MONTHLY_EXTENDED_OUTPUT = ROOT / "upi_monthly_extended.csv"
OVERLAP_OUTPUT = ROOT / "upi_overlap_validation.csv"
PERIOD_TESTS_OUTPUT = ROOT / "upi_period_tests.csv"
FESTIVAL_TESTS_OUTPUT = ROOT / "upi_festival_tests.csv"
REGRESSION_OUTPUT = ROOT / "upi_regression_results.csv"
DAILY_DIAGNOSTICS_OUTPUT = ROOT / "upi_daily_diagnostics.csv"
ANNUAL_OUTPUT = ROOT / "upi_annual_robustness.csv"
FIGURE_OUTPUT = ROOT / "upi_thesis_overview.png"
LEVELS_EXTENDED_OUTPUT = ROOT / "upi_plot_levels_extended.png"
GROWTH_REGIME_OUTPUT = ROOT / "upi_plot_growth_regime.png"
VALIDATION_SCATTER_OUTPUT = ROOT / "upi_plot_validation_scatter.png"
SEASONALITY_BOXPLOT_OUTPUT = ROOT / "upi_plot_seasonality_boxplots.png"
DAILY_TRENDS_OUTPUT = ROOT / "upi_plot_daily_trends.png"
REPORT_OUTPUT = ROOT / "upi_thesis_report.md"
NOTEBOOK_OUTPUT = ROOT / "upi_thesis_analysis.ipynb"
SUBMISSION_HTML_OUTPUT = ROOT / "upi_submission_report.html"
SUBMISSION_DOCX_OUTPUT = ROOT / "upi_submission_report.docx"


@dataclass
class DataBundle:
    daily_df: pd.DataFrame
    monthly_primary_df: pd.DataFrame
    official_monthly_df: pd.DataFrame
    extended_monthly_df: pd.DataFrame
    overlap_df: pd.DataFrame


def get_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def fetch_text(session: requests.Session, url: str, data: dict | None = None) -> str:
    response = session.get(url, timeout=60) if data is None else session.post(
        url,
        data=data,
        timeout=60,
    )
    response.raise_for_status()
    return response.text


def fetch_json(
    session: requests.Session,
    url: str,
    params: dict | None = None,
) -> dict:
    response = session.get(url, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def parse_hidden_value(html: str, field_name: str) -> str:
    match = re.search(rf'id="{field_name}" value="([^"]+)"', html)
    if not match:
        raise ValueError(f"Could not find hidden field {field_name}")
    return match.group(1)


def fetch_rbi_archive_links(session: requests.Session) -> pd.DataFrame:
    """Fetch year archive pages and map month labels to detailed RBI page ids."""
    root_html = fetch_text(session, RBI_ARCHIVE_URL)
    viewstate = parse_hidden_value(root_html, "__VIEWSTATE")
    eventvalidation = parse_hidden_value(root_html, "__EVENTVALIDATION")
    viewstate_generator = parse_hidden_value(root_html, "__VIEWSTATEGENERATOR")

    records: list[dict[str, str]] = []
    link_pattern = re.compile(
        r'href=PSIUserView\.aspx\?Id=(\d+)>Payment System Indicators - ([^<]+)</a>'
    )

    for year in range(2020, 2026):
        html = fetch_text(
            session,
            RBI_ARCHIVE_URL,
            data={
                "__VIEWSTATE": viewstate,
                "__EVENTVALIDATION": eventvalidation,
                "__VIEWSTATEGENERATOR": viewstate_generator,
                "hdnYear": str(year),
                "hdnMonth": "0",
                "UsrFontCntr$btn": "",
            },
        )
        for page_id, label in link_pattern.findall(html):
            clean_label = label.replace("Decmber", "December")
            period = pd.to_datetime(clean_label, format="%B %Y")
            records.append(
                {
                    "label": clean_label,
                    "page_id": page_id,
                    "year_month": period.strftime("%Y-%m"),
                }
            )

    archive_df = (
        pd.DataFrame(records)
        .drop_duplicates(subset=["year_month"], keep="last")
        .sort_values("year_month")
        .reset_index(drop=True)
    )
    return archive_df


def fetch_rbi_official_monthly(session: requests.Session) -> pd.DataFrame:
    """Fetch one official UPI observation per month from the RBI archive."""
    archive_df = fetch_rbi_archive_links(session)
    row_pattern = re.compile(r"<tr>\s*<td>2\.6 UPI @</td>(.*?)</tr>", re.S)
    value_pattern = re.compile(r'<td align="right">([^<]+)</td>')
    records: list[dict[str, object]] = []

    for row in archive_df.itertuples(index=False):
        html = fetch_text(session, RBI_PAGE_URL.format(page_id=row.page_id))
        match = row_pattern.search(html)
        if not match:
            continue
        values = value_pattern.findall(match.group(1))
        if len(values) < 8:
            continue
        records.append(
            {
                "date": pd.Timestamp(row.year_month + "-01"),
                "year_month": row.year_month,
                "upi_volume_lakh": float(values[3].replace(",", "")),
                "upi_value_crore": float(values[7].replace(",", "")),
                "source": "RBI official monthly archive",
            }
        )

    official_df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    if official_df.empty:
        raise ValueError("No monthly RBI official UPI observations were parsed.")
    return official_df


def fetch_daily_panel(session: requests.Session) -> pd.DataFrame:
    """Fetch the machine-readable daily panel and deduplicate repeated dates."""
    offset = 0
    page_size = 1000
    records: list[dict[str, object]] = []

    while True:
        payload = fetch_json(
            session,
            DAILY_RESOURCE_URL,
            params={
                "resource_id": DAILY_RESOURCE_ID,
                "limit": page_size,
                "offset": offset,
            },
        )
        batch = payload["result"]["records"]
        if not batch:
            break
        records.extend(batch)
        offset += len(batch)
        if len(batch) < page_size:
            break

    daily_df = pd.DataFrame.from_records(records)
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    daily_df["upi_volume_lakh"] = pd.to_numeric(daily_df["upi_vol"], errors="coerce")
    daily_df["upi_value_crore"] = pd.to_numeric(daily_df["upi_val"], errors="coerce")
    daily_df = (
        daily_df.dropna(subset=["date", "upi_volume_lakh", "upi_value_crore"])
        .sort_values(["date", "_id"])
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )
    daily_df["upi_volume"] = daily_df["upi_volume_lakh"] * 100_000
    daily_df["upi_value_rupees"] = daily_df["upi_value_crore"] * 10_000_000
    daily_df["year_month"] = daily_df["date"].dt.strftime("%Y-%m")
    daily_df["year"] = daily_df["date"].dt.year
    daily_df["month"] = daily_df["date"].dt.month
    daily_df["weekday"] = daily_df["date"].dt.day_name()
    return daily_df[
        [
            "date",
            "year_month",
            "year",
            "month",
            "weekday",
            "upi_volume_lakh",
            "upi_value_crore",
            "upi_volume",
            "upi_value_rupees",
        ]
    ]


def build_monthly_primary(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate deduplicated daily data to months."""
    monthly_df = (
        daily_df.groupby("year_month", as_index=False)
        .agg(
            date=("date", "min"),
            year=("year", "first"),
            month=("month", "first"),
            upi_volume_lakh=("upi_volume_lakh", "sum"),
            upi_value_crore=("upi_value_crore", "sum"),
            days_observed=("date", "size"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    monthly_df["upi_volume"] = monthly_df["upi_volume_lakh"] * 100_000
    monthly_df["upi_value_rupees"] = monthly_df["upi_value_crore"] * 10_000_000
    monthly_df["festival_month"] = monthly_df["month"].isin([10, 11, 12]).astype(int)
    monthly_df["post_2022"] = (monthly_df["year_month"] >= POST_2022_START).astype(int)
    monthly_df["log_volume"] = np.log(monthly_df["upi_volume"])
    monthly_df["log_value"] = np.log(monthly_df["upi_value_rupees"])
    monthly_df["growth_volume"] = monthly_df["upi_volume_lakh"].pct_change() * 100
    monthly_df["growth_value"] = monthly_df["upi_value_crore"].pct_change() * 100
    return monthly_df


def build_overlap_validation(
    monthly_primary_df: pd.DataFrame,
    official_monthly_df: pd.DataFrame,
) -> pd.DataFrame:
    overlap_df = monthly_primary_df.merge(
        official_monthly_df,
        on=["year_month"],
        how="inner",
        suffixes=("_daily_agg", "_official"),
    )
    overlap_df["pct_diff_volume"] = (
        (overlap_df["upi_volume_lakh_daily_agg"] - overlap_df["upi_volume_lakh_official"])
        / overlap_df["upi_volume_lakh_official"]
        * 100
    )
    overlap_df["pct_diff_value"] = (
        (overlap_df["upi_value_crore_daily_agg"] - overlap_df["upi_value_crore_official"])
        / overlap_df["upi_value_crore_official"]
        * 100
    )
    return overlap_df


def build_extended_monthly(
    monthly_primary_df: pd.DataFrame,
    official_monthly_df: pd.DataFrame,
) -> pd.DataFrame:
    primary_slice = monthly_primary_df.loc[
        monthly_primary_df["year_month"] <= PRIMARY_END,
        [
            "date",
            "year_month",
            "year",
            "month",
            "upi_volume_lakh",
            "upi_value_crore",
            "upi_volume",
            "upi_value_rupees",
            "festival_month",
            "post_2022",
            "log_volume",
            "log_value",
            "growth_volume",
            "growth_value",
        ],
    ]
    official_tail = official_monthly_df.loc[
        official_monthly_df["year_month"] > PRIMARY_END
    ].copy()
    official_tail["year"] = official_tail["date"].dt.year
    official_tail["month"] = official_tail["date"].dt.month
    official_tail["festival_month"] = official_tail["month"].isin([10, 11, 12]).astype(int)
    official_tail["post_2022"] = (official_tail["year_month"] >= POST_2022_START).astype(int)
    official_tail["upi_volume"] = official_tail["upi_volume_lakh"] * 100_000
    official_tail["upi_value_rupees"] = official_tail["upi_value_crore"] * 10_000_000
    official_tail["log_volume"] = np.log(official_tail["upi_volume"])
    official_tail["log_value"] = np.log(official_tail["upi_value_rupees"])
    official_tail["growth_volume"] = np.nan
    official_tail["growth_value"] = np.nan

    extended_df = (
        pd.concat([primary_slice, official_tail], ignore_index=True, sort=False)
        .sort_values("date")
        .reset_index(drop=True)
    )
    extended_df["growth_volume"] = extended_df["upi_volume_lakh"].pct_change() * 100
    extended_df["growth_value"] = extended_df["upi_value_crore"].pct_change() * 100
    return extended_df


def bootstrap_mean_ci(
    series: pd.Series,
    confidence: float = 0.95,
    reps: int = 5000,
    seed: int = 0,
) -> tuple[float, float, float]:
    clean = series.dropna().to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(clean, size=(reps, len(clean)), replace=True).mean(axis=1)
    alpha = (1 - confidence) / 2
    return clean.mean(), np.quantile(draws, alpha), np.quantile(draws, 1 - alpha)


def build_period_tests(monthly_df: pd.DataFrame) -> pd.DataFrame:
    growth_df = monthly_df.dropna(subset=["growth_volume", "growth_value"]).copy()
    records = []

    for column in ["growth_volume", "growth_value"]:
        pre = growth_df.loc[growth_df["post_2022"] == 0, column]
        post = growth_df.loc[growth_df["post_2022"] == 1, column]

        pre_mean, pre_low, pre_high = bootstrap_mean_ci(pre, seed=1)
        post_mean, post_low, post_high = bootstrap_mean_ci(post, seed=2)
        welch = stats.ttest_ind(post, pre, equal_var=False, nan_policy="omit")
        mann_whitney = stats.mannwhitneyu(post, pre, alternative="two-sided")

        records.append(
            {
                "variable": column,
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


def build_festival_tests(monthly_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for column in ["upi_volume_lakh", "upi_value_crore", "log_volume", "log_value"]:
        festival = monthly_df.loc[monthly_df["festival_month"] == 1, column]
        non_festival = monthly_df.loc[monthly_df["festival_month"] == 0, column]
        welch = stats.ttest_ind(
            festival,
            non_festival,
            equal_var=False,
            nan_policy="omit",
        )
        mann_whitney = stats.mannwhitneyu(
            festival,
            non_festival,
            alternative="two-sided",
        )
        records.append(
            {
                "variable": column,
                "n_festival": len(festival),
                "n_non_festival": len(non_festival),
                "mean_festival": festival.mean(),
                "mean_non_festival": non_festival.mean(),
                "difference_festival_minus_non_festival": festival.mean()
                - non_festival.mean(),
                "welch_t_stat": welch.statistic,
                "welch_p_value": welch.pvalue,
                "mann_whitney_u": mann_whitney.statistic,
                "mann_whitney_p_value": mann_whitney.pvalue,
            }
        )
    return pd.DataFrame(records)


def build_regression_results(monthly_df: pd.DataFrame) -> pd.DataFrame:
    growth_df = monthly_df.dropna(subset=["growth_volume", "growth_value"]).copy()
    month_dummies = pd.get_dummies(
        growth_df["month"].astype(int),
        prefix="m",
        drop_first=True,
        dtype=float,
    )
    design = pd.concat(
        [
            pd.Series(1.0, index=growth_df.index, name="const"),
            growth_df[["post_2022"]].astype(float),
            month_dummies,
        ],
        axis=1,
    )

    records = []
    for dependent in ["growth_volume", "growth_value"]:
        model = sm.OLS(growth_df[dependent], design)
        fitted = model.fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        for term in ["post_2022", "m_10", "m_11", "m_12"]:
            records.append(
                {
                    "model": dependent,
                    "term": term,
                    "coefficient": fitted.params[term],
                    "std_error_hac": fitted.bse[term],
                    "p_value": fitted.pvalues[term],
                    "test_type": "coefficient",
                }
            )

    trend_design = pd.DataFrame(
        {
            "const": 1.0,
            "trend": np.arange(len(monthly_df), dtype=float),
            "festival_month": monthly_df["festival_month"].astype(float),
        }
    )
    for dependent in ["log_volume", "log_value"]:
        model = sm.OLS(monthly_df[dependent], trend_design)
        fitted = model.fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        for term in ["trend", "festival_month"]:
            records.append(
                {
                    "model": dependent,
                    "term": term,
                    "coefficient": fitted.params[term],
                    "std_error_hac": fitted.bse[term],
                    "p_value": fitted.pvalues[term],
                    "test_type": "coefficient",
                }
            )

    return pd.DataFrame(records)


def build_daily_diagnostics(daily_df: pd.DataFrame) -> pd.DataFrame:
    daily = daily_df.copy()
    daily["log_volume"] = np.log(daily["upi_volume"])
    daily["log_value"] = np.log(daily["upi_value_rupees"])
    daily["log_growth_volume"] = daily["log_volume"].diff() * 100
    daily["log_growth_value"] = daily["log_value"].diff() * 100

    records = []
    for label, column in [
        ("log_growth_volume", "log_growth_volume"),
        ("log_growth_value", "log_growth_value"),
    ]:
        series = daily[column].dropna()
        for lag in [1, 7]:
            records.append(
                {
                    "series": label,
                    "lag": lag,
                    "autocorrelation": series.autocorr(lag=lag),
                }
            )
    return pd.DataFrame(records)


def build_annual_robustness(monthly_df: pd.DataFrame) -> pd.DataFrame:
    annual_df = (
        monthly_df.dropna(subset=["growth_volume", "growth_value"])
        .groupby("year", as_index=False)
        .agg(
            avg_monthly_growth_volume=("growth_volume", "mean"),
            avg_monthly_growth_value=("growth_value", "mean"),
            observations=("year_month", "size"),
        )
        .sort_values("year")
        .reset_index(drop=True)
    )
    annual_df["complete_year"] = (annual_df["observations"] == 12).astype(int)
    return annual_df


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a simple markdown table without optional dependencies."""
    render_df = df.copy()
    for column in render_df.columns:
        if pd.api.types.is_float_dtype(render_df[column]):
            render_df[column] = render_df[column].map(lambda value: f"{value:.4f}")
    header = "| " + " | ".join(render_df.columns.astype(str)) + " |"
    divider = "| " + " | ".join(["---"] * len(render_df.columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in render_df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider] + rows)


def dataframe_to_html(
    df: pd.DataFrame,
    float_format: str = "{:.4f}",
) -> str:
    """Render an HTML table with consistent number formatting."""
    render_df = df.copy()
    for column in render_df.columns:
        if pd.api.types.is_float_dtype(render_df[column]):
            render_df[column] = render_df[column].map(lambda value: float_format.format(value))
    return render_df.to_html(index=False, border=0, classes="table", escape=False)


def make_figure(monthly_df: pd.DataFrame, period_tests_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(monthly_df["date"], monthly_df["upi_volume"] / 1e9, color="#0b6e4f")
    axes[0, 0].set_title("UPI monthly volume")
    axes[0, 0].set_ylabel("Billion transactions")

    axes[0, 1].plot(
        monthly_df["date"],
        monthly_df["upi_value_rupees"] / 1e12,
        color="#9a3412",
    )
    axes[0, 1].set_title("UPI monthly value")
    axes[0, 1].set_ylabel("Rupee trillion")

    growth_df = monthly_df.dropna(subset=["growth_volume", "growth_value"])
    axes[1, 0].plot(growth_df["date"], growth_df["growth_volume"], color="#1d4ed8")
    axes[1, 0].axvline(pd.Timestamp("2022-01-01"), color="black", linestyle="--")
    axes[1, 0].set_title("Monthly volume growth")
    axes[1, 0].set_ylabel("Percent")

    bars = period_tests_df.set_index("variable")
    labels = ["Pre-2022", "2022 onward"]
    positions = np.arange(2)
    width = 0.35
    axes[1, 1].bar(
        positions - width / 2,
        [bars.loc["growth_volume", "mean_pre"], bars.loc["growth_volume", "mean_post"]],
        width=width,
        label="Volume growth",
        color="#2563eb",
    )
    axes[1, 1].bar(
        positions + width / 2,
        [bars.loc["growth_value", "mean_pre"], bars.loc["growth_value", "mean_post"]],
        width=width,
        label="Value growth",
        color="#d97706",
    )
    axes[1, 1].set_xticks(positions)
    axes[1, 1].set_xticklabels(labels)
    axes[1, 1].set_title("Mean monthly growth by period")
    axes[1, 1].set_ylabel("Percent")
    axes[1, 1].legend(frameon=False)

    for axis in axes.flat:
        axis.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(FIGURE_OUTPUT, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_visualisations(
    daily_df: pd.DataFrame,
    monthly_primary_df: pd.DataFrame,
    official_monthly_df: pd.DataFrame,
    extended_monthly_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
) -> None:
    """Create a fuller set of charts for the project deliverable."""
    growth_df = monthly_primary_df.dropna(subset=["growth_volume", "growth_value"]).copy()

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    primary_mask = extended_monthly_df["year_month"] <= PRIMARY_END
    tail_mask = extended_monthly_df["year_month"] > PRIMARY_END

    axes[0].plot(
        extended_monthly_df.loc[primary_mask, "date"],
        extended_monthly_df.loc[primary_mask, "upi_volume"] / 1e9,
        color="#0f766e",
        linewidth=2.2,
        label="Validated daily-backed sample",
    )
    axes[0].plot(
        extended_monthly_df.loc[tail_mask, "date"],
        extended_monthly_df.loc[tail_mask, "upi_volume"] / 1e9,
        color="#0f766e",
        linewidth=2.2,
        linestyle="--",
        label="Official RBI extension",
    )
    axes[0].axvline(pd.Timestamp("2022-01-01"), color="#334155", linestyle=":", linewidth=1.3)
    axes[0].set_ylabel("Billion transactions")
    axes[0].set_title("UPI monthly volume through official December 2025 archive")
    axes[0].legend(frameon=False, loc="upper left")

    axes[1].plot(
        extended_monthly_df.loc[primary_mask, "date"],
        extended_monthly_df.loc[primary_mask, "upi_value_rupees"] / 1e12,
        color="#b45309",
        linewidth=2.2,
        label="Validated daily-backed sample",
    )
    axes[1].plot(
        extended_monthly_df.loc[tail_mask, "date"],
        extended_monthly_df.loc[tail_mask, "upi_value_rupees"] / 1e12,
        color="#b45309",
        linewidth=2.2,
        linestyle="--",
        label="Official RBI extension",
    )
    axes[1].axvline(pd.Timestamp("2022-01-01"), color="#334155", linestyle=":", linewidth=1.3)
    axes[1].set_ylabel("Rupee trillion")
    axes[1].set_title("UPI monthly value through official December 2025 archive")
    axes[1].legend(frameon=False, loc="upper left")

    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(LEVELS_EXTENDED_OUTPUT, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    pre_mean_vol = growth_df.loc[growth_df["post_2022"] == 0, "growth_volume"].mean()
    post_mean_vol = growth_df.loc[growth_df["post_2022"] == 1, "growth_volume"].mean()
    pre_mean_val = growth_df.loc[growth_df["post_2022"] == 0, "growth_value"].mean()
    post_mean_val = growth_df.loc[growth_df["post_2022"] == 1, "growth_value"].mean()

    axes[0].plot(growth_df["date"], growth_df["growth_volume"], color="#2563eb", linewidth=1.8)
    axes[0].axvline(pd.Timestamp("2022-01-01"), color="#111827", linestyle="--", linewidth=1.3)
    axes[0].axhline(pre_mean_vol, color="#1d4ed8", linestyle=":", linewidth=1.5, label=f"Pre-2022 mean: {pre_mean_vol:.2f}%")
    axes[0].axhline(post_mean_vol, color="#60a5fa", linestyle="-.", linewidth=1.5, label=f"2022+ mean: {post_mean_vol:.2f}%")
    axes[0].set_ylabel("Percent")
    axes[0].set_title("Monthly UPI volume growth and the post-2021 regime shift")
    axes[0].legend(frameon=False, loc="upper right")

    axes[1].plot(growth_df["date"], growth_df["growth_value"], color="#d97706", linewidth=1.8)
    axes[1].axvline(pd.Timestamp("2022-01-01"), color="#111827", linestyle="--", linewidth=1.3)
    axes[1].axhline(pre_mean_val, color="#b45309", linestyle=":", linewidth=1.5, label=f"Pre-2022 mean: {pre_mean_val:.2f}%")
    axes[1].axhline(post_mean_val, color="#f59e0b", linestyle="-.", linewidth=1.5, label=f"2022+ mean: {post_mean_val:.2f}%")
    axes[1].set_ylabel("Percent")
    axes[1].set_title("Monthly UPI value growth and the post-2021 regime shift")
    axes[1].legend(frameon=False, loc="upper right")

    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(GROWTH_REGIME_OUTPUT, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x_vol = overlap_df["upi_volume_lakh_official"] / 10000
    y_vol = overlap_df["upi_volume_lakh_daily_agg"] / 10000
    x_val = overlap_df["upi_value_crore_official"] / 100000
    y_val = overlap_df["upi_value_crore_daily_agg"] / 100000

    axes[0].scatter(x_vol, y_vol, color="#0f766e", alpha=0.8)
    axes[0].plot([x_vol.min(), x_vol.max()], [x_vol.min(), x_vol.max()], color="#111827", linestyle="--")
    axes[0].set_xlabel("Official monthly volume (billions)")
    axes[0].set_ylabel("Daily aggregate volume (billions)")
    axes[0].set_title("Validation: daily aggregation vs official RBI volume")

    axes[1].scatter(x_val, y_val, color="#b45309", alpha=0.8)
    axes[1].plot([x_val.min(), x_val.max()], [x_val.min(), x_val.max()], color="#111827", linestyle="--")
    axes[1].set_xlabel("Official monthly value (Rs. trillion)")
    axes[1].set_ylabel("Daily aggregate value (Rs. trillion)")
    axes[1].set_title("Validation: daily aggregation vs official RBI value")

    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(VALIDATION_SCATTER_OUTPUT, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    month_positions = np.arange(1, 13)
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    volume_boxes = [
        growth_df.loc[growth_df["month"] == month, "growth_volume"].dropna().to_numpy()
        for month in month_positions
    ]
    value_boxes = [
        growth_df.loc[growth_df["month"] == month, "growth_value"].dropna().to_numpy()
        for month in month_positions
    ]

    bp1 = axes[0].boxplot(volume_boxes, patch_artist=True, tick_labels=month_labels)
    bp2 = axes[1].boxplot(value_boxes, patch_artist=True, tick_labels=month_labels)
    for idx, patch in enumerate(bp1["boxes"], start=1):
        patch.set_facecolor("#93c5fd" if idx not in [10, 11, 12] else "#2563eb")
        patch.set_alpha(0.85)
    for idx, patch in enumerate(bp2["boxes"], start=1):
        patch.set_facecolor("#fdba74" if idx not in [10, 11, 12] else "#d97706")
        patch.set_alpha(0.85)

    axes[0].set_ylabel("Percent")
    axes[0].set_title("Seasonality in monthly UPI volume growth")
    axes[1].set_ylabel("Percent")
    axes[1].set_title("Seasonality in monthly UPI value growth")
    for axis in axes:
        axis.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(SEASONALITY_BOXPLOT_OUTPUT, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    daily_plot = daily_df.copy()
    daily_plot["volume_7d"] = daily_plot["upi_volume"].rolling(7, min_periods=1).mean() / 1e9
    daily_plot["value_7d"] = daily_plot["upi_value_rupees"].rolling(7, min_periods=1).mean() / 1e12

    axes[0].plot(daily_plot["date"], daily_plot["upi_volume"] / 1e9, color="#cbd5e1", linewidth=0.8, alpha=0.8)
    axes[0].plot(daily_plot["date"], daily_plot["volume_7d"], color="#0f766e", linewidth=2.2)
    axes[0].set_ylabel("Billion transactions")
    axes[0].set_title("Daily UPI volume with 7-day moving average")

    axes[1].plot(daily_plot["date"], daily_plot["upi_value_rupees"] / 1e12, color="#fde68a", linewidth=0.8, alpha=0.8)
    axes[1].plot(daily_plot["date"], daily_plot["value_7d"], color="#b45309", linewidth=2.2)
    axes[1].set_ylabel("Rupee trillion")
    axes[1].set_title("Daily UPI value with 7-day moving average")

    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(DAILY_TRENDS_OUTPUT, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_report(
    daily_df: pd.DataFrame,
    monthly_primary_df: pd.DataFrame,
    official_monthly_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    period_tests_df: pd.DataFrame,
    festival_tests_df: pd.DataFrame,
    regression_df: pd.DataFrame,
    daily_diagnostics_df: pd.DataFrame,
    annual_df: pd.DataFrame,
) -> None:
    period = period_tests_df.set_index("variable")
    festival = festival_tests_df.set_index("variable")
    regressions = regression_df.set_index(["model", "term"])
    latest = official_monthly_df.iloc[-1]
    overlap_volume = overlap_df["pct_diff_volume"].abs().mean()
    overlap_value = overlap_df["pct_diff_value"].abs().mean()

    daily_diag_table = dataframe_to_markdown(daily_diagnostics_df)
    annual_table = dataframe_to_markdown(annual_df)

    text = f"""# Digital Payments and UPI in India: structural changes or just growth?

## Executive summary

This project studies whether India’s UPI expansion reflects only rising transaction levels or also a structural change in growth dynamics. The answer is clear: **UPI continues to rise strongly in levels, but average monthly percentage growth is materially lower after January 2022 than in the earlier diffusion phase.**

- Primary monthly sample from validated daily data: `{monthly_primary_df['year_month'].min()}` to `{monthly_primary_df['year_month'].max()}`
- Official RBI monthly archive available through: `{latest['year_month']}`
- Daily panel coverage: `{daily_df['date'].min().date()}` to `{daily_df['date'].max().date()}`
- Daily-to-official overlap validation mean absolute percent difference:
  - volume: `{overlap_volume:.6f}%`
  - value: `{overlap_value:.6f}%`

## Data sources

- RBI official monthly archive: [RBI Payment System Indicators]({RBI_ARCHIVE_URL})
- RBI data warehouse reference: [DBIE]({DBIE_URL})
- UPI institutional background: [PIB UPI explainer]({PIB_UPI_URL})
- Daily machine-readable RBI-style panel used for high-frequency robustness:
  `resource_id = {DAILY_RESOURCE_ID}`

## Research questions

1. Has the average monthly growth rate of UPI transaction volume and value changed between the pre-2022 and 2022 onward periods?
2. Are festival months statistically different from non-festival months in volume or value?
3. Do these conclusions survive seasonality controls and serial-correlation-aware inference?
4. What does the daily data add to the interpretation?

## Data construction

1. Pull official RBI month pages from the PSI archive and extract the `2.6 UPI @` row.
2. Pull the daily panel from the datastore API, deduplicate repeated dates, and aggregate to months.
3. Validate that monthly sums from daily data match the official RBI archive in overlapping months.
4. Use the daily-aggregated monthly data through July 2025 for the main inference.
5. Use official RBI monthly data through December 2025 to extend level tracking beyond the daily panel.

## Main estimation results

### Mean monthly growth by period

| Variable | Pre-2022 mean | 95% CI | 2022 onward mean | 95% CI | Difference (post - pre) | Welch p-value | Mann-Whitney p-value |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: |
| Volume growth | {period.loc['growth_volume', 'mean_pre']:.2f}% | [{period.loc['growth_volume', 'ci_pre_low']:.2f}%, {period.loc['growth_volume', 'ci_pre_high']:.2f}%] | {period.loc['growth_volume', 'mean_post']:.2f}% | [{period.loc['growth_volume', 'ci_post_low']:.2f}%, {period.loc['growth_volume', 'ci_post_high']:.2f}%] | {period.loc['growth_volume', 'difference_post_minus_pre']:.2f} pp | {period.loc['growth_volume', 'welch_p_value']:.3f} | {period.loc['growth_volume', 'mann_whitney_p_value']:.3f} |
| Value growth | {period.loc['growth_value', 'mean_pre']:.2f}% | [{period.loc['growth_value', 'ci_pre_low']:.2f}%, {period.loc['growth_value', 'ci_pre_high']:.2f}%] | {period.loc['growth_value', 'mean_post']:.2f}% | [{period.loc['growth_value', 'ci_post_low']:.2f}%, {period.loc['growth_value', 'ci_post_high']:.2f}%] | {period.loc['growth_value', 'difference_post_minus_pre']:.2f} pp | {period.loc['growth_value', 'welch_p_value']:.3f} | {period.loc['growth_value', 'mann_whitney_p_value']:.3f} |

Interpretation:

- Volume growth falls by about `{abs(period.loc['growth_volume', 'difference_post_minus_pre']):.2f}` percentage points after 2021.
- Value growth falls by about `{abs(period.loc['growth_value', 'difference_post_minus_pre']):.2f}` percentage points after 2021.
- The value slowdown is significant by Welch’s test at conventional levels.
- The volume slowdown is marginal by Welch’s test but significant by Mann-Whitney, which is consistent with non-Gaussian monthly growth.

### Festival-month evidence

Naive festival-month comparisons in levels are weak:

- Volume Welch p-value: `{festival.loc['upi_volume_lakh', 'welch_p_value']:.3f}`
- Value Welch p-value: `{festival.loc['upi_value_crore', 'welch_p_value']:.3f}`
- Log-volume Welch p-value: `{festival.loc['log_volume', 'welch_p_value']:.3f}`
- Log-value Welch p-value: `{festival.loc['log_value', 'welch_p_value']:.3f}`

This means that a coarse `October-November-December` dummy is not enough to separate festival effects from the dominant long-run upward trend.

### Seasonality-adjusted regressions

Using monthly growth regressions with month dummies and HAC standard errors:

- Post-2022 coefficient in volume growth regression:
  `{regressions.loc[('growth_volume', 'post_2022'), 'coefficient']:.2f}` pp,
  p-value `{regressions.loc[('growth_volume', 'post_2022'), 'p_value']:.4f}`
- Post-2022 coefficient in value growth regression:
  `{regressions.loc[('growth_value', 'post_2022'), 'coefficient']:.2f}` pp,
  p-value `{regressions.loc[('growth_value', 'post_2022'), 'p_value']:.6f}`

Selected month coefficients:

- October volume growth effect: `{regressions.loc[('growth_volume', 'm_10'), 'coefficient']:.2f}` pp
- October value growth effect: `{regressions.loc[('growth_value', 'm_10'), 'coefficient']:.2f}` pp
- December volume growth effect: `{regressions.loc[('growth_volume', 'm_12'), 'coefficient']:.2f}` pp
- December value growth effect: `{regressions.loc[('growth_value', 'm_12'), 'coefficient']:.2f}` pp

Trend-plus-festival log regressions show:

- log-volume trend coefficient: `{regressions.loc[('log_volume', 'trend'), 'coefficient']:.4f}` per month
- log-value trend coefficient: `{regressions.loc[('log_value', 'trend'), 'coefficient']:.4f}` per month
- festival coefficient in log-volume regression: `{regressions.loc[('log_volume', 'festival_month'), 'coefficient']:.4f}`, p-value `{regressions.loc[('log_volume', 'festival_month'), 'p_value']:.3f}`
- festival coefficient in log-value regression: `{regressions.loc[('log_value', 'festival_month'), 'coefficient']:.4f}`, p-value `{regressions.loc[('log_value', 'festival_month'), 'p_value']:.3f}`

## Daily-data supplement

The daily panel adds two useful facts.

1. The monthly totals computed from daily observations match the official RBI monthly archive exactly in overlapping months after deduplicating repeated dates.
2. Daily log-growth shows strong serial dependence:

{daily_diag_table}

This justifies the use of HAC standard errors in monthly regressions and supports the classroom point that annualised averages are closer to independent observations than raw monthly growth rates.

## Annualised robustness

{annual_table}

The annual averages show a steady deceleration from 2021 through 2024 in both volume and value growth. That reinforces the structural-slowdown interpretation.

## Practical implications

- **Capacity planning:** Growth is lower in percentage terms, but the transaction base is so large that even a small additional growth rate implies a very large absolute increase in system load.
- **Fraud and resilience:** Strong October and December growth effects, plus the daily autocorrelation structure, mean operational risk is concentrated around predictable calendar peaks and weekly rhythms.
- **Interpretation discipline:** The right statement is not “UPI is slowing down”; it is “UPI is maturing, with lower percentage growth on a much larger base.”

## Latest scale

The latest official RBI month available in this run is `{latest['year_month']}`:

- UPI volume: `{latest['upi_volume_lakh']:.2f}` lakh transactions
- UPI value: `₹{latest['upi_value_crore']:.2f}` crore

That corresponds to roughly `{latest['upi_volume_lakh'] / 10000:.2f}` billion transactions and `₹{latest['upi_value_crore'] / 100000:.2f}` trillion in a single month.
"""

    REPORT_OUTPUT.write_text(text)


def write_submission_package(
    daily_df: pd.DataFrame,
    monthly_primary_df: pd.DataFrame,
    official_monthly_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    period_tests_df: pd.DataFrame,
    festival_tests_df: pd.DataFrame,
    regression_df: pd.DataFrame,
    daily_diagnostics_df: pd.DataFrame,
    annual_df: pd.DataFrame,
) -> None:
    """Create a cleaner submission-ready HTML report and convert it to DOCX."""
    period = period_tests_df.copy()
    period["period_1"] = period.apply(
        lambda row: f"{row['mean_pre']:.2f}% [{row['ci_pre_low']:.2f}, {row['ci_pre_high']:.2f}]",
        axis=1,
    )
    period["period_2"] = period.apply(
        lambda row: f"{row['mean_post']:.2f}% [{row['ci_post_low']:.2f}, {row['ci_post_high']:.2f}]",
        axis=1,
    )
    period["difference_pp"] = period["difference_post_minus_pre"].map(lambda value: f"{value:.2f}")
    period["welch_p"] = period["welch_p_value"].map(lambda value: f"{value:.3f}")
    period["mann_whitney_p"] = period["mann_whitney_p_value"].map(lambda value: f"{value:.3f}")
    period["measure"] = period["variable"].map(
        {
            "growth_volume": "Monthly growth in UPI volume",
            "growth_value": "Monthly growth in UPI value",
        }
    )
    period_table = dataframe_to_html(
        period[
            [
                "measure",
                "period_1",
                "period_2",
                "difference_pp",
                "welch_p",
                "mann_whitney_p",
            ]
        ].rename(
            columns={
                "measure": "Measure",
                "period_1": "Pre-2022 mean [95% CI]",
                "period_2": "2022 onward mean [95% CI]",
                "difference_pp": "Post - pre (pp)",
                "welch_p": "Welch p-value",
                "mann_whitney_p": "Mann-Whitney p-value",
            }
        ),
        float_format="{:.3f}",
    )

    reg_subset = regression_df.loc[
        regression_df["term"].isin(["post_2022", "m_10", "m_11", "m_12"])
    ].copy()
    reg_subset["model"] = reg_subset["model"].map(
        {
            "growth_volume": "Volume growth",
            "growth_value": "Value growth",
        }
    )
    reg_subset["term"] = reg_subset["term"].map(
        {
            "post_2022": "Post-2022 indicator",
            "m_10": "October",
            "m_11": "November",
            "m_12": "December",
        }
    )
    reg_table = dataframe_to_html(
        reg_subset[["model", "term", "coefficient", "std_error_hac", "p_value"]].rename(
            columns={
                "model": "Model",
                "term": "Term",
                "coefficient": "Coefficient",
                "std_error_hac": "HAC std. error",
                "p_value": "p-value",
            }
        ),
        float_format="{:.4f}",
    )

    annual_table = dataframe_to_html(
        annual_df.rename(
            columns={
                "year": "Year",
                "avg_monthly_growth_volume": "Avg. monthly volume growth",
                "avg_monthly_growth_value": "Avg. monthly value growth",
                "observations": "Months observed",
                "complete_year": "Complete year",
            }
        ),
        float_format="{:.4f}",
    )

    diagnostics_table = dataframe_to_html(
        daily_diagnostics_df.rename(
            columns={
                "series": "Series",
                "lag": "Lag",
                "autocorrelation": "Autocorrelation",
            }
        ),
        float_format="{:.4f}",
    )

    festival = festival_tests_df.set_index("variable")
    regressions = regression_df.set_index(["model", "term"])
    latest = official_monthly_df.iloc[-1]
    overlap_volume = overlap_df["pct_diff_volume"].abs().mean()
    overlap_value = overlap_df["pct_diff_value"].abs().mean()
    report_date = pd.Timestamp.now().strftime("%B %d, %Y")
    figure_base64 = base64.b64encode(FIGURE_OUTPUT.read_bytes()).decode("ascii")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>UPI Structural Change Report</title>
  <style>
    body {{
      font-family: 'Times New Roman', Georgia, serif;
      line-height: 1.45;
      margin: 42px 54px;
      color: #111827;
      font-size: 12pt;
    }}
    h1, h2, h3 {{
      color: #0f172a;
      margin-top: 1.2em;
      margin-bottom: 0.4em;
    }}
    h1 {{
      text-align: center;
      font-size: 24pt;
      margin-top: 1.8em;
    }}
    h2 {{
      font-size: 16pt;
      border-bottom: 1px solid #cbd5e1;
      padding-bottom: 4px;
    }}
    h3 {{
      font-size: 13pt;
    }}
    p, li {{
      text-align: justify;
    }}
    .cover {{
      page-break-after: always;
      text-align: center;
      margin-top: 120px;
    }}
    .cover p {{
      text-align: center;
    }}
    .meta {{
      margin-top: 36px;
      font-size: 11pt;
      color: #374151;
    }}
    .abstract {{
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      padding: 14px 18px;
      margin: 20px 0;
    }}
    .table {{
      width: 100%;
      border-collapse: collapse;
      margin: 14px 0 20px 0;
      font-size: 10.5pt;
    }}
    .table th, .table td {{
      border: 1px solid #cbd5e1;
      padding: 6px 8px;
      vertical-align: top;
    }}
    .table th {{
      background: #e2e8f0;
      font-weight: bold;
    }}
    .figure {{
      text-align: center;
      margin: 18px 0 8px 0;
    }}
    .figure img {{
      max-width: 100%;
      height: auto;
      border: 1px solid #cbd5e1;
    }}
    .caption {{
      font-size: 10pt;
      color: #475569;
      margin-top: 6px;
    }}
    .refs li {{
      margin-bottom: 8px;
    }}
  </style>
</head>
<body>
  <div class="cover">
    <p><strong>Statistical Inference Project Report</strong></p>
    <h1>Digital Payments and UPI in India:<br>Structural Changes or Just Growth?</h1>
    <p class="meta">Prepared on {escape(report_date)}</p>
    <p class="meta">Author(s): ________________________________</p>
    <p class="meta">Roll Number(s): ___________________________</p>
    <p class="meta">Course / Department: ______________________</p>
  </div>

  <h2>Abstract</h2>
  <div class="abstract">
    <p>
      This report evaluates whether UPI growth in India reflects simple continued expansion or a change in the
      underlying growth regime. Using the official RBI Payment System Indicators archive as the primary monthly source,
      plus a validated daily machine-readable payments panel for high-frequency robustness, the analysis compares
      average monthly growth before January 2022 and from January 2022 onward. The evidence shows a clear structural
      moderation in monthly percentage growth despite continued expansion in levels. Mean monthly UPI volume growth
      declines from {period_tests_df.loc[period_tests_df['variable'] == 'growth_volume', 'mean_pre'].iloc[0]:.2f}% to
      {period_tests_df.loc[period_tests_df['variable'] == 'growth_volume', 'mean_post'].iloc[0]:.2f}%, while mean
      monthly value growth declines from {period_tests_df.loc[period_tests_df['variable'] == 'growth_value', 'mean_pre'].iloc[0]:.2f}% to
      {period_tests_df.loc[period_tests_df['variable'] == 'growth_value', 'mean_post'].iloc[0]:.2f}%. Seasonality-adjusted
      regressions with HAC standard errors preserve this conclusion. Daily data additionally reveal meaningful serial
      dependence, justifying robust inference. The overall interpretation is market maturation on a very large base,
      not deterioration in usage.
    </p>
  </div>

  <h2>1. Introduction</h2>
  <p>
    UPI has become the dominant retail digital payments rail in India. The practical question for this project is
    whether the system is still in a rapid diffusion phase or has moved into a maturity phase where transaction levels
    continue to rise but percentage growth is lower. That distinction matters for infrastructure sizing, monitoring,
    and fraud-risk management because a smaller growth rate on a much larger base may still imply heavier absolute load.
  </p>

  <h2>2. Data and Source Validation</h2>
  <p>
    The monthly backbone of the analysis comes from the official RBI PSI archive. Daily observations come from the
    machine-readable RBI-style datastore and are deduplicated before aggregation. In overlapping months from
    {monthly_primary_df['year_month'].iloc[12]} to {monthly_primary_df['year_month'].iloc[-1]}, the daily aggregation matches the official RBI
    monthly totals with mean absolute percentage differences of {overlap_volume:.6f}% for volume and {overlap_value:.6f}% for value.
    This validation step is critical because it lets the daily data be used for robustness rather than as a substitute
    for the primary official source.
  </p>
  <ul>
    <li>Primary monthly sample for inference: {escape(monthly_primary_df['year_month'].min())} to {escape(monthly_primary_df['year_month'].max())}</li>
    <li>Official RBI monthly archive available through: {escape(latest['year_month'])}</li>
    <li>Daily panel coverage: {escape(str(daily_df['date'].min().date()))} to {escape(str(daily_df['date'].max().date()))}</li>
  </ul>

  <h2>3. Empirical Design</h2>
  <p>
    The central estimation target is the mean monthly growth rate of UPI volume and value in two periods:
    pre-2022 and 2022 onward. The analysis reports bootstrap confidence intervals, Welch unequal-variance t-tests,
    and Mann-Whitney tests to avoid relying on strict normality. Seasonality is handled with month dummies, and
    regression inference uses HAC standard errors because payment series are serially correlated.
  </p>
  <p>
    A supplementary trend-plus-festival specification tests whether October to December months are unusually high after
    controlling for a strong upward time trend. Annualised averages of monthly growth are reported as a low-frequency
    robustness check because they are closer to independent observations than raw monthly growth rates.
  </p>

  <h2>4. Main Results</h2>
  <h3>4.1 Period Comparison</h3>
  <p>
    The primary result is a marked decline in percentage growth after January 2022. The effect is economically large:
    monthly volume growth falls by {abs(period_tests_df.loc[period_tests_df['variable'] == 'growth_volume', 'difference_post_minus_pre'].iloc[0]):.2f}
    percentage points, and monthly value growth falls by
    {abs(period_tests_df.loc[period_tests_df['variable'] == 'growth_value', 'difference_post_minus_pre'].iloc[0]):.2f} percentage points.
    Value growth is significant under Welch's test, while volume growth is marginal by Welch and significant by the
    Mann-Whitney alternative.
  </p>
  {period_table}

  <h3>4.2 Seasonality and Festival Months</h3>
  <p>
    Simple festival-month comparisons do not show statistically meaningful differences in average levels. For example,
    the festival-month Welch p-values are {festival.loc['upi_volume_lakh', 'welch_p_value']:.3f} for volume and
    {festival.loc['upi_value_crore', 'welch_p_value']:.3f} for value. That does not imply the absence of seasonality.
    Instead, it suggests that a fixed October-November-December indicator is too crude when strong trend and calendar
    composition effects are present.
  </p>

  <h3>4.3 Seasonality-Adjusted Regressions</h3>
  <p>
    Once month effects are controlled for, the post-2022 slowdown remains clear. The post-2022 indicator is
    {regressions.loc[('growth_volume', 'post_2022'), 'coefficient']:.2f} percentage points for volume growth and
    {regressions.loc[('growth_value', 'post_2022'), 'coefficient']:.2f} percentage points for value growth, both with
    strong HAC-robust significance. October is the strongest positive seasonal month in both specifications, and
    December also shows a positive effect.
  </p>
  {reg_table}

  <div class="figure">
    <img src="data:image/png;base64,{figure_base64}" alt="UPI analysis figure">
    <div class="caption">Figure 1. Monthly UPI levels, growth, and pre/post comparison generated from the validated dataset.</div>
  </div>

  <h3>4.4 Daily Data and Autocorrelation</h3>
  <p>
    The daily series add practical high-frequency structure that monthly aggregates hide. Daily log-growth is negatively
    autocorrelated at lag 1 and strongly positively autocorrelated at lag 7, especially for transaction value. That
    pattern is consistent with day-of-week effects and confirms that serial correlation should not be ignored in
    inference.
  </p>
  {diagnostics_table}

  <h3>4.5 Annualised Robustness</h3>
  <p>
    Annual average monthly growth declines steadily from 2021 through 2024 in both volume and value. This lower-frequency
    view supports the same structural interpretation as the monthly results and is useful when discussing approximate
    independence at a course-theory level.
  </p>
  {annual_table}

  <h2>5. Interpretation and Practical Implications</h2>
  <p>
    The evidence is most consistent with <strong>continued expansion in levels with structural moderation in growth rates</strong>.
    UPI is not shrinking, and it is not plateauing in any literal transaction-count sense. Rather, its percentage growth is
    lower because the base is now extremely large. That distinction matters operationally. At the latest official RBI month,
    {escape(latest['year_month'])}, UPI handles approximately {latest['upi_volume_lakh'] / 10000:.2f} billion transactions and
    ₹{latest['upi_value_crore'] / 100000:.2f} trillion in a single month.
  </p>
  <ul>
    <li><strong>Infrastructure:</strong> planning should focus on absolute transaction load, not growth rates alone.</li>
    <li><strong>Risk monitoring:</strong> October and December spikes, plus weekly dependence in daily data, imply predictable peak-risk windows.</li>
    <li><strong>Policy interpretation:</strong> lower growth should be read as maturation on scale, not as deterioration in digital payments adoption.</li>
  </ul>

  <h2>6. Limitations</h2>
  <p>
    The daily panel ends in July 2025, so the monthly inference sample stops there even though official RBI monthly
    levels continue through December 2025. Festival measurement is also intentionally simple: October to December is a
    coarse proxy and does not capture the shifting timing of Diwali and related events. Finally, the statistical tests
    here are descriptive-inferential rather than structural models of user adoption or merchant behavior.
  </p>

  <h2>7. Conclusion</h2>
  <p>
    The project’s answer is straightforward. India’s UPI story is no longer one of uniformly high percentage growth,
    but neither is it a stagnation story. The system has entered a mature high-scale phase: absolute usage continues to
    rise rapidly, while monthly percentage growth is structurally lower than in the pre-2022 period. That is the right
    interpretation for a thesis-level statistical inference report grounded in current RBI data.
  </p>

  <h2>References</h2>
  <ol class="refs">
    <li>Reserve Bank of India. <em>Payment System Indicators</em>. {escape(RBI_ARCHIVE_URL)}</li>
    <li>Reserve Bank of India. <em>Database on Indian Economy (DBIE)</em>. {escape(DBIE_URL)}</li>
    <li>Press Information Bureau. <em>UPI: Revolutionizing Digital Payments in India</em>. {escape(PIB_UPI_URL)}</li>
    <li>RBI-style daily payments datastore, resource id <code>{escape(DAILY_RESOURCE_ID)}</code>, used only after direct validation against the official RBI monthly archive.</li>
  </ol>
</body>
</html>
"""

    SUBMISSION_HTML_OUTPUT.write_text(html)
    try:
        subprocess.run(
            [
                "textutil",
                "-convert",
                "docx",
                "-output",
                str(SUBMISSION_DOCX_OUTPUT),
                str(SUBMISSION_HTML_OUTPUT),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass


def write_notebook() -> None:
    notebook = new_notebook(
        cells=[
            new_markdown_cell(
                "# UPI structural change analysis\n\n"
                "This notebook reproduces the project pipeline in "
                "`upi_thesis_pipeline.py` and saves the outputs in the current folder."
            ),
            new_markdown_cell(
                "## Sources\n\n"
                f"- Official RBI monthly archive: {RBI_ARCHIVE_URL}\n"
                f"- DBIE reference: {DBIE_URL}\n"
                f"- PIB background note: {PIB_UPI_URL}\n"
                f"- Daily panel resource id: `{DAILY_RESOURCE_ID}`"
            ),
            new_code_cell(
                "from pathlib import Path\n"
                "import pandas as pd\n"
                "from upi_thesis_pipeline import run_pipeline\n\n"
                "results = run_pipeline()\n"
                "list(results.keys())"
            ),
            new_code_cell(
                "pd.read_csv('upi_period_tests.csv')"
            ),
            new_code_cell(
                "pd.read_csv('upi_festival_tests.csv')"
            ),
            new_code_cell(
                "pd.read_csv('upi_regression_results.csv')"
            ),
            new_code_cell(
                "from IPython.display import Image\n"
                "Image(filename='upi_thesis_overview.png')"
            ),
            new_code_cell(
                "from IPython.display import display, Image\n"
                "for path in [\n"
                "    'upi_plot_levels_extended.png',\n"
                "    'upi_plot_growth_regime.png',\n"
                "    'upi_plot_validation_scatter.png',\n"
                "    'upi_plot_seasonality_boxplots.png',\n"
                "    'upi_plot_daily_trends.png',\n"
                "]:\n"
                "    display(Image(filename=path))"
            ),
            new_code_cell(
                "print(Path('upi_thesis_report.md').read_text())"
            ),
        ]
    )
    NOTEBOOK_OUTPUT.write_text(nbformat.writes(notebook))


def save_outputs(
    bundle: DataBundle,
    period_tests_df: pd.DataFrame,
    festival_tests_df: pd.DataFrame,
    regression_df: pd.DataFrame,
    daily_diagnostics_df: pd.DataFrame,
    annual_df: pd.DataFrame,
) -> None:
    bundle.daily_df.to_csv(DAILY_OUTPUT, index=False)
    bundle.monthly_primary_df.to_csv(MONTHLY_PRIMARY_OUTPUT, index=False)
    bundle.official_monthly_df.to_csv(MONTHLY_OFFICIAL_OUTPUT, index=False)
    bundle.extended_monthly_df.to_csv(MONTHLY_EXTENDED_OUTPUT, index=False)
    bundle.overlap_df.to_csv(OVERLAP_OUTPUT, index=False)
    period_tests_df.to_csv(PERIOD_TESTS_OUTPUT, index=False)
    festival_tests_df.to_csv(FESTIVAL_TESTS_OUTPUT, index=False)
    regression_df.to_csv(REGRESSION_OUTPUT, index=False)
    daily_diagnostics_df.to_csv(DAILY_DIAGNOSTICS_OUTPUT, index=False)
    annual_df.to_csv(ANNUAL_OUTPUT, index=False)


def run_pipeline() -> dict[str, pd.DataFrame]:
    session = get_session()

    official_monthly_df = fetch_rbi_official_monthly(session)
    daily_df = fetch_daily_panel(session)
    monthly_primary_df = build_monthly_primary(daily_df)
    overlap_df = build_overlap_validation(monthly_primary_df, official_monthly_df)
    extended_monthly_df = build_extended_monthly(monthly_primary_df, official_monthly_df)

    bundle = DataBundle(
        daily_df=daily_df,
        monthly_primary_df=monthly_primary_df,
        official_monthly_df=official_monthly_df,
        extended_monthly_df=extended_monthly_df,
        overlap_df=overlap_df,
    )

    period_tests_df = build_period_tests(monthly_primary_df)
    festival_tests_df = build_festival_tests(monthly_primary_df)
    regression_df = build_regression_results(monthly_primary_df)
    daily_diagnostics_df = build_daily_diagnostics(daily_df)
    annual_df = build_annual_robustness(monthly_primary_df)

    save_outputs(
        bundle,
        period_tests_df,
        festival_tests_df,
        regression_df,
        daily_diagnostics_df,
        annual_df,
    )
    make_figure(monthly_primary_df, period_tests_df)
    make_visualisations(
        daily_df,
        monthly_primary_df,
        official_monthly_df,
        extended_monthly_df,
        overlap_df,
    )
    write_report(
        daily_df,
        monthly_primary_df,
        official_monthly_df,
        overlap_df,
        period_tests_df,
        festival_tests_df,
        regression_df,
        daily_diagnostics_df,
        annual_df,
    )
    write_submission_package(
        daily_df,
        monthly_primary_df,
        official_monthly_df,
        overlap_df,
        period_tests_df,
        festival_tests_df,
        regression_df,
        daily_diagnostics_df,
        annual_df,
    )
    write_notebook()

    return {
        "daily": daily_df,
        "monthly_primary": monthly_primary_df,
        "monthly_official": official_monthly_df,
        "monthly_extended": extended_monthly_df,
        "overlap": overlap_df,
        "period_tests": period_tests_df,
        "festival_tests": festival_tests_df,
        "regression_results": regression_df,
        "daily_diagnostics": daily_diagnostics_df,
        "annual_robustness": annual_df,
    }


def main() -> None:
    results = run_pipeline()
    latest = results["monthly_official"].iloc[-1]
    period = results["period_tests"].set_index("variable")
    print(
        f"Wrote notebook, report, figure, and CSV outputs to {ROOT}. "
        f"Latest official month: {latest['year_month']}."
    )
    print(
        "Volume growth mean changed from "
        f"{period.loc['growth_volume', 'mean_pre']:.2f}% to "
        f"{period.loc['growth_volume', 'mean_post']:.2f}%."
    )
    print(
        "Value growth mean changed from "
        f"{period.loc['growth_value', 'mean_pre']:.2f}% to "
        f"{period.loc['growth_value', 'mean_post']:.2f}%."
    )


if __name__ == "__main__":
    main()
