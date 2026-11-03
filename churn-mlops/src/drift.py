"""A simple, beginner-friendly data-drift check.

Drift = the data your model sees in production slowly stops looking like the
data it was trained on, so its predictions get worse.

We compare:
  reference = models/reference_data.csv   (a sample of the training data)
  current   = recent data (logs/predictions.csv written by the API,
              or a simulated drifted dataset for practice)

Tests used (p-value < 0.05 => "drift detected"):
  - numeric columns     -> Kolmogorov-Smirnov test
  - categorical columns -> chi-square test

Run:
    python -m src.drift --simulate                     # practice with fake drift
    python -m src.drift --current logs/predictions.csv # real API traffic

The HTML report is saved to reports/drift_report.html
(For a fancier report, look at the Evidently library once you understand this.)
"""
import argparse
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.data import (CATEGORICAL_FEATURES, FEATURES, MODELS_DIR, NUMERIC_FEATURES, ROOT,
                      clean_data, load_raw_data)

REPORT_PATH = ROOT / "reports" / "drift_report.html"
ALPHA = 0.05
# SeniorCitizen is 0/1, so it is treated as a category, not as a continuous number
CONTINUOUS = [c for c in NUMERIC_FEATURES if c != "SeniorCitizen"]
CATEGORICAL = CATEGORICAL_FEATURES + ["SeniorCitizen"]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only feature columns and make sure the types are right."""
    df = df[FEATURES].copy()
    for col in CONTINUOUS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL:
        df[col] = df[col].astype(str).str.strip()
    return df


def simulate_drift(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Fake 'new' data: customers are newer, pay more and prefer monthly contracts."""
    rng = np.random.default_rng(seed)
    df = clean_data(load_raw_data()).drop(columns=["Churn"]).sample(n, random_state=seed).copy()
    df["tenure"] = (df["tenure"] * 0.5).round().astype(int)
    df["MonthlyCharges"] = (df["MonthlyCharges"] * 1.15).round(2)
    df["TotalCharges"] = (df["tenure"] * df["MonthlyCharges"]).round(2)
    flip = rng.random(len(df)) < 0.5
    df.loc[flip, "Contract"] = "Month-to-month"
    return df


def check_numeric(ref: pd.Series, cur: pd.Series) -> dict:
    ref, cur = ref.dropna(), cur.dropna()
    stat, p = stats.ks_2samp(ref, cur)
    return {"test": "KS test", "stat": stat, "p": p,
            "ref": f"mean {ref.mean():.2f}", "cur": f"mean {cur.mean():.2f}"}


def check_categorical(ref: pd.Series, cur: pd.Series) -> dict:
    ref_counts, cur_counts = ref.value_counts(), cur.value_counts()
    cats = sorted(set(ref_counts.index) | set(cur_counts.index))
    table = np.array([[ref_counts.get(c, 0) for c in cats],
                      [cur_counts.get(c, 0) for c in cats]])
    if len(cats) < 2:
        return {"test": "chi-square", "stat": 0.0, "p": 1.0,
                "ref": str(cats[0]) if cats else "-", "cur": str(cats[0]) if cats else "-"}
    stat, p, _, _ = stats.chi2_contingency(table)
    top = lambda counts: f"top: {counts.index[0]} ({counts.iloc[0] / counts.sum():.0%})"
    return {"test": "chi-square", "stat": stat, "p": p, "ref": top(ref_counts), "cur": top(cur_counts)}


def build_report(rows: list, n_ref: int, n_cur: int) -> str:
    n_drift = sum(r["drift"] for r in rows)
    body = ""
    for r in rows:
        color = "#fde2e2" if r["drift"] else "#e6f4ea"
        status = "DRIFT" if r["drift"] else "ok"
        body += (f"<tr style='background:{color}'><td>{escape(r['feature'])}</td><td>{r['test']}</td>"
                 f"<td>{r['stat']:.3f}</td><td>{r['p']:.4f}</td><td>{escape(r['ref'])}</td>"
                 f"<td>{escape(r['cur'])}</td><td><b>{status}</b></td></tr>")
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Drift report</title>
<style>body{{font-family:Arial,sans-serif;margin:30px}}table{{border-collapse:collapse}}
td,th{{border:1px solid #bbb;padding:6px 10px;font-size:14px}}th{{background:#dce6f2}}</style></head>
<body><h1>Data drift report</h1>
<p>Reference rows: {n_ref} | Current rows: {n_cur} | Significance level: {ALPHA}</p>
<h2>{n_drift} of {len(rows)} features show drift</h2>
<table><tr><th>Feature</th><th>Test</th><th>Statistic</th><th>p-value</th><th>Reference</th>
<th>Current</th><th>Status</th></tr>{body}</table>
<p>Note: with very large samples even tiny, harmless changes can look significant.
Use this report as a signal to investigate, not as an automatic alarm.</p></body></html>"""


def run_drift(current: pd.DataFrame, reference_path: Path = MODELS_DIR / "reference_data.csv",
              out_path: Path = REPORT_PATH) -> int:
    if not Path(reference_path).exists():
        raise SystemExit("Reference data not found. Run `python -m src.train` first.")
    ref, cur = prepare(pd.read_csv(reference_path)), prepare(current)
    if len(cur) < 30:
        raise SystemExit(f"Only {len(cur)} current rows - need at least 30 for a meaningful check.")

    rows = []
    for col in CONTINUOUS:
        rows.append({"feature": col, **check_numeric(ref[col], cur[col])})
    for col in CATEGORICAL:
        rows.append({"feature": col, **check_categorical(ref[col], cur[col])})
    for r in rows:
        r["drift"] = bool(r["p"] < ALPHA)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_report(rows, len(ref), len(cur)), encoding="utf-8")

    drifted = [r["feature"] for r in rows if r["drift"]]
    print(f"[drift] {len(drifted)} of {len(rows)} features drifted: {drifted}")
    print(f"[drift] report saved to {out_path}")
    return len(drifted)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple data drift check")
    parser.add_argument("--current", help="CSV with recent data (e.g. logs/predictions.csv)")
    parser.add_argument("--simulate", action="store_true", help="use a simulated drifted dataset")
    args = parser.parse_args()

    if args.simulate:
        current_df = simulate_drift()
    elif args.current:
        current_df = pd.read_csv(args.current)
    else:
        raise SystemExit("Use --simulate or --current <file.csv>")
    run_drift(current_df)
