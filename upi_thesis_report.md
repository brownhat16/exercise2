# Digital Payments and UPI in India: structural changes or just growth?

## Executive summary

This project studies whether India’s UPI expansion reflects only rising transaction levels or also a structural change in growth dynamics. The answer is clear: **UPI continues to rise strongly in levels, but average monthly percentage growth is materially lower after January 2022 than in the earlier diffusion phase.**

- Primary monthly sample from validated daily data: `2020-06` to `2025-07`
- Official RBI monthly archive available through: `2025-12`
- Daily panel coverage: `2020-06-01` to `2025-07-31`
- Daily-to-official overlap validation mean absolute percent difference:
  - volume: `0.000001%`
  - value: `0.000019%`

## Data sources

- RBI official monthly archive: [RBI Payment System Indicators](https://www.rbi.org.in/Scripts/PSIUserView.aspx?Id=47)
- RBI data warehouse reference: [DBIE](https://data.rbi.org.in)
- UPI institutional background: [PIB UPI explainer](https://www.pib.gov.in/PressReleaseIframePage.aspx?PRID=2079544)
- Full project repository with code, notebook, cleaned data, plots, and report files: [brownhat16/exercise2](https://github.com/brownhat16/exercise2)
- Daily machine-readable RBI-style panel used for high-frequency robustness:
  `resource_id = 1f9367ac-01b0-4c82-83a1-4069d4340667`

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
| Volume growth | 7.28% | [4.05%, 10.54%] | 3.65% | [2.03%, 5.36%] | -3.63 pp | 0.062 | 0.046 |
| Value growth | 6.80% | [3.76%, 9.98%] | 2.81% | [1.23%, 4.42%] | -3.99 pp | 0.036 | 0.056 |

Interpretation:

- Volume growth falls by about `3.63` percentage points after 2021.
- Value growth falls by about `3.99` percentage points after 2021.
- The value slowdown is significant by Welch’s test at conventional levels.
- The volume slowdown is marginal by Welch’s test but significant by Mann-Whitney, which is consistent with non-Gaussian monthly growth.

### Festival-month evidence

Naive festival-month comparisons in levels are weak:

- Volume Welch p-value: `0.861`
- Value Welch p-value: `0.931`
- Log-volume Welch p-value: `0.981`
- Log-value Welch p-value: `0.994`

This means that a coarse `October-November-December` dummy is not enough to separate festival effects from the dominant long-run upward trend.

### Seasonality-adjusted regressions

Using monthly growth regressions with month dummies and HAC standard errors:

- Post-2022 coefficient in volume growth regression:
  `-3.30` pp,
  p-value `0.0007`
- Post-2022 coefficient in value growth regression:
  `-3.74` pp,
  p-value `0.000000`

Selected month coefficients:

- October volume growth effect: `8.65` pp
- October value growth effect: `10.98` pp
- December volume growth effect: `3.81` pp
- December value growth effect: `4.61` pp

Trend-plus-festival log regressions show:

- log-volume trend coefficient: `0.0425` per month
- log-value trend coefficient: `0.0359` per month
- festival coefficient in log-volume regression: `0.0509`, p-value `0.247`
- festival coefficient in log-value regression: `0.0492`, p-value `0.320`

## Daily-data supplement

The daily panel adds two useful facts.

1. The monthly totals computed from daily observations match the official RBI monthly archive exactly in overlapping months after deduplicating repeated dates.
2. Daily log-growth shows strong serial dependence:

| series | lag | autocorrelation |
| --- | --- | --- |
| log_growth_volume | 1 | -0.3865 |
| log_growth_volume | 7 | 0.2170 |
| log_growth_value | 1 | -0.3783 |
| log_growth_value | 7 | 0.7033 |

This justifies the use of HAC standard errors in monthly regressions and supports the classroom point that annualised averages are closer to independent observations than raw monthly growth rates.

## Annualised robustness

| year | avg_monthly_growth_volume | avg_monthly_growth_value | observations | complete_year |
| --- | --- | --- | --- | --- |
| 2020 | 9.0277 | 8.1647 | 6 | 0 |
| 2021 | 6.4091 | 6.1157 | 12 | 1 |
| 2022 | 4.7332 | 3.8411 | 12 | 1 |
| 2023 | 3.7748 | 3.0769 | 12 | 1 |
| 2024 | 2.9148 | 2.1933 | 12 | 1 |
| 2025 | 2.7282 | 1.4604 | 6 | 0 |

The annual averages show a steady deceleration from 2021 through 2024 in both volume and value growth. That reinforces the structural-slowdown interpretation.

## Practical implications

- **Capacity planning:** Growth is lower in percentage terms, but the transaction base is so large that even a small additional growth rate implies a very large absolute increase in system load.
- **Fraud and resilience:** Strong October and December growth effects, plus the daily autocorrelation structure, mean operational risk is concentrated around predictable calendar peaks and weekly rhythms.
- **Interpretation discipline:** The right statement is not “UPI is slowing down”; it is “UPI is maturing, with lower percentage growth on a much larger base.”

## Latest scale

The latest official RBI month available in this run is `2025-12`:

- UPI volume: `216346.68` lakh transactions
- UPI value: `₹2796713.00` crore

That corresponds to roughly `21.63` billion transactions and `₹27.97` trillion in a single month.
