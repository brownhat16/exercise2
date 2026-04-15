"""Build monthly India digital payments datasets with a focus on UPI."""

import io
import re
import sys

import numpy as np
import pandas as pd
import requests

RBI_PAGE_URL = "https://www.rbi.org.in/Scripts/PSIUserView.aspx?Id=46"
CKAN_API_BASE = "https://ckandev.indiadataportal.com/api/3/action"
RESOURCE_ID = "1f9367ac-01b0-4c82-83a1-4069d4340667"
START_DATE = pd.Timestamp("2019-01-01")
PRE_2022_CUTOFF = pd.Timestamp("2022-01-01")
DAILY_OUTPUT = "upi_daily_dataset.csv"
WEEKLY_OUTPUT = "upi_weekly_dataset.csv"
MONTHLY_OUTPUT = "upi_monthly_dataset.csv"
ANNUAL_OUTPUT = "upi_annual_dataset.csv"
LAKH_TO_ABSOLUTE = 100_000
CRORE_TO_RUPEES = 10_000_000

SERIES_RENAME = {
    "upi_vol": "upi_volume",
    "upi_val": "upi_value",
    "imps_vol": "imps_volume",
    "imps_val": "imps_value",
    "neft_vol": "neft_volume",
    "neft_val": "neft_value",
    "rtgs_vol": "rtgs_volume",
    "rtgs_val": "rtgs_value",
    "credit_card_at_pos_and_e_commerce_vol": "credit_card_volume",
    "credit_card_at_pos_and_e_commerce_val": "credit_card_value",
    "debit_card_at_pos_and_e_commerce_vol": "debit_card_volume",
    "debit_card_at_pos_and_e_commerce_val": "debit_card_value",
}


def get_json(url: str, params: dict | None = None) -> dict:
    """Send a GET request and return the decoded JSON payload."""
    response = requests.get(
        url,
        params=params,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def fetch_latest_rbi_workbook_url() -> str | None:
    """Read the RBI payment indicators page and return the latest XLSX link."""
    response = requests.get(
        RBI_PAGE_URL,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=60,
    )
    response.raise_for_status()

    match = re.search(
        r"https://rbidocs\.rbi\.org\.in/rdocs/PSI/DOCs/[^\"']+\.XLSX",
        response.text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(0)
    return None


def fetch_resource_metadata(resource_id: str) -> dict:
    """Return CKAN metadata for the RBI-style payments resource."""
    payload = get_json(f"{CKAN_API_BASE}/resource_show", params={"id": resource_id})
    if not payload.get("success"):
        raise ValueError("Failed to fetch resource metadata from CKAN.")
    return payload["result"]


def try_direct_csv_download(resource_url: str) -> pd.DataFrame | None:
    """Try the direct CSV download endpoint before falling back to the API."""
    response = requests.get(
        resource_url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/csv,*/*;q=0.8",
        },
        timeout=60,
    )
    content_type = response.headers.get("content-type", "").lower()
    text = response.text.lstrip()

    if not response.ok:
        return None
    if "text/csv" not in content_type:
        return None
    if text.startswith("<!DOCTYPE html") or text.startswith("<html"):
        return None

    return pd.read_csv(io.StringIO(response.text))


def download_datastore_rows(resource_id: str, page_size: int = 1000) -> pd.DataFrame:
    """Download all rows from the CKAN datastore API with pagination."""
    rows = []
    offset = 0
    total = None

    while total is None or offset < total:
        payload = get_json(
            f"{CKAN_API_BASE}/datastore_search",
            params={
                "resource_id": resource_id,
                "limit": page_size,
                "offset": offset,
            },
        )
        if not payload.get("success"):
            raise ValueError("CKAN datastore request failed.")

        result = payload["result"]
        batch = result.get("records", [])
        if total is None:
            total = result.get("total", 0)
        if not batch:
            break

        rows.extend(batch)
        offset += len(batch)

    if not rows:
        raise ValueError("No records were returned by the CKAN datastore API.")

    return pd.DataFrame.from_records(rows)


def download_payments_data() -> tuple[pd.DataFrame, str, str | None]:
    """
    Download the full RBI-style payments history.

    The RBI official page is checked first for traceability, but the historical
    panel is built from the CKAN datastore because the RBI workbook is only a
    monthly snapshot, not a full back-history.
    """
    latest_rbi_workbook = fetch_latest_rbi_workbook_url()
    metadata = fetch_resource_metadata(RESOURCE_ID)

    direct_df = try_direct_csv_download(metadata["url"])
    if direct_df is not None and "date" in direct_df.columns:
        return direct_df, "direct_csv", latest_rbi_workbook

    api_df = download_datastore_rows(RESOURCE_ID)
    return api_df, "ckan_datastore_api", latest_rbi_workbook


def clean_daily_data(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates, coerce numeric columns, and keep the analysis window."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[df["date"] >= START_DATE].copy()
    df = df.sort_values("date").drop_duplicates(subset="date", keep="last")

    usable_columns = [column for column in SERIES_RENAME if column in df.columns]
    df[usable_columns] = df[usable_columns].apply(pd.to_numeric, errors="coerce")

    return df[["date"] + usable_columns]


def standardize_payment_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert units, rename columns, and add combined cards series."""
    value_columns = [column for column in SERIES_RENAME if column.endswith("_val")]
    volume_columns = [column for column in SERIES_RENAME if column.endswith("_vol")]
    df = df.copy()
    available_volume_columns = [column for column in volume_columns if column in df.columns]
    available_value_columns = [column for column in value_columns if column in df.columns]

    df[available_volume_columns] = df[available_volume_columns] * LAKH_TO_ABSOLUTE
    df[available_value_columns] = df[available_value_columns] * CRORE_TO_RUPEES
    df = df.rename(columns=SERIES_RENAME)

    # Create a simple combined cards measure for comparison if both sub-series exist.
    card_volume_columns = [column for column in ["credit_card_volume", "debit_card_volume"] if column in df.columns]
    card_value_columns = [column for column in ["credit_card_value", "debit_card_value"] if column in df.columns]
    if card_volume_columns:
        df["cards_volume"] = df[card_volume_columns].sum(axis=1, min_count=1)
    if card_value_columns:
        df["cards_value"] = df[card_value_columns].sum(axis=1, min_count=1)

    return df


def add_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add common growth, log, and indicator variables."""
    df = df.dropna(subset=["upi_volume", "upi_value"]).copy()
    df = df.sort_values("date").reset_index(drop=True)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    iso_calendar = df["date"].dt.isocalendar()
    df["iso_year"] = iso_calendar.year.astype(int)
    df["iso_week"] = iso_calendar.week.astype(int)
    df["growth_volume"] = df["upi_volume"].pct_change(fill_method=None) * 100
    df["growth_value"] = df["upi_value"].pct_change(fill_method=None) * 100
    df["log_volume"] = np.where(df["upi_volume"] > 0, np.log(df["upi_volume"]), np.nan)
    df["log_value"] = np.where(df["upi_value"] > 0, np.log(df["upi_value"]), np.nan)
    df["pre_2022"] = (df["date"] < PRE_2022_CUTOFF).astype(int)
    df["festival_month"] = df["month"].isin([10, 11, 12]).astype(int)
    return df


def order_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep a stable column order across daily, weekly, and monthly datasets."""
    ordered_columns = [
        "date",
        "year",
        "month",
        "iso_year",
        "iso_week",
        "upi_volume",
        "upi_value",
        "growth_volume",
        "growth_value",
        "log_volume",
        "log_value",
        "pre_2022",
        "festival_month",
    ]

    optional_columns = [
        "imps_volume",
        "imps_value",
        "neft_volume",
        "neft_value",
        "rtgs_volume",
        "rtgs_value",
        "cards_volume",
        "cards_value",
        "credit_card_volume",
        "credit_card_value",
        "debit_card_volume",
        "debit_card_value",
    ]
    available_optional_columns = [column for column in optional_columns if column in df.columns]

    return df[ordered_columns + available_optional_columns]


def build_daily_dataset(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the daily dataset in analysis-ready units."""
    daily = standardize_payment_columns(daily_df)
    daily = add_time_series_features(daily)
    return order_output_columns(daily)


def build_aggregated_dataset(daily_df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Aggregate daily data to weekly or monthly frequency and add features."""
    available_columns = [column for column in SERIES_RENAME if column in daily_df.columns]

    if freq == "W":
        grouped = daily_df.assign(date=daily_df["date"].dt.to_period("W-SUN").dt.start_time)
    elif freq == "M":
        grouped = daily_df.assign(date=daily_df["date"].dt.to_period("M").dt.to_timestamp())
    else:
        raise ValueError(f"Unsupported frequency: {freq}")

    aggregated = (
        grouped.groupby("date", as_index=False)[available_columns]
        .sum(min_count=1)
        .sort_values("date")
        .reset_index(drop=True)
    )

    aggregated = standardize_payment_columns(aggregated)
    aggregated = add_time_series_features(aggregated)
    return order_output_columns(aggregated)


def aggregate_to_monthly(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily payment data to monthly sums and build analysis features."""
    return build_aggregated_dataset(daily_df, freq="M")


def build_annual_dataset(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """Create an annual summary with average monthly growth rates."""
    annual = (
        monthly_df.groupby("year", as_index=False)
        .agg(
            avg_monthly_growth_volume=("growth_volume", "mean"),
            avg_monthly_growth_value=("growth_value", "mean"),
            avg_upi_volume=("upi_volume", "mean"),
            avg_upi_value=("upi_value", "mean"),
            observations=("date", "size"),
        )
        .sort_values("year")
        .reset_index(drop=True)
    )
    return annual


def print_summary(monthly_df: pd.DataFrame, source_name: str, latest_rbi_workbook: str | None) -> None:
    """Print summary statistics required for the project."""
    pre = monthly_df.loc[monthly_df["pre_2022"] == 1, ["growth_volume", "growth_value"]]
    post = monthly_df.loc[monthly_df["pre_2022"] == 0, ["growth_volume", "growth_value"]]

    print(f"Source used: {source_name}")
    if latest_rbi_workbook:
        print(f"Latest RBI payment indicators workbook detected: {latest_rbi_workbook}")

    print(
        "Monthly data coverage:",
        monthly_df["date"].min().date(),
        "to",
        monthly_df["date"].max().date(),
    )
    print(f"Number of observations: {len(monthly_df)}")

    print("\nMean monthly growth (pre-2022):")
    print(pre.mean().round(2).to_string())

    print("\nMean monthly growth (post-2022):")
    print(post.mean().round(2).to_string())

    print("\nStandard deviation of monthly growth:")
    print(monthly_df[["growth_volume", "growth_value"]].std().round(2).to_string())


def main() -> None:
    raw_df, source_name, latest_rbi_workbook = download_payments_data()
    daily_df = clean_daily_data(raw_df)
    daily_output_df = build_daily_dataset(daily_df)
    weekly_df = build_aggregated_dataset(daily_df, freq="W")
    monthly_df = aggregate_to_monthly(daily_df)
    annual_df = build_annual_dataset(monthly_df)

    daily_output_df.to_csv(DAILY_OUTPUT, index=False)
    weekly_df.to_csv(WEEKLY_OUTPUT, index=False)
    monthly_df.to_csv(MONTHLY_OUTPUT, index=False)
    annual_df.to_csv(ANNUAL_OUTPUT, index=False)

    if monthly_df["date"].min() > START_DATE:
        print(
            "Note: the structured RBI-style source starts on",
            daily_df["date"].min().date(),
            "so 2019 months are not available in the downloaded history.",
        )

    print(
        "Daily data coverage:",
        daily_output_df["date"].min().date(),
        "to",
        daily_output_df["date"].max().date(),
        f"({len(daily_output_df)} observations)",
    )
    print(
        "Weekly data coverage:",
        weekly_df["date"].min().date(),
        "to",
        weekly_df["date"].max().date(),
        f"({len(weekly_df)} observations)",
    )
    print_summary(monthly_df, source_name, latest_rbi_workbook)
    print("\nDaily dataset preview:")
    print(daily_output_df.head().to_string(index=False))
    print("\nWeekly dataset preview:")
    print(weekly_df.head().to_string(index=False))
    print("\nMonthly dataset preview:")
    print(monthly_df.head().to_string(index=False))
    print("\nAnnual dataset preview:")
    print(annual_df.to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
