"""Create a synthetic dataset with the SAME columns as the Kaggle
"Telco Customer Churn" file (WA_Fn-UseC_-Telco-Customer-Churn.csv).

It exists so the project runs end-to-end straight away.
For your portfolio, download the real file from Kaggle, rename it to
data/telco_churn.csv and re-train (see README).

Run:  python -m src.generate_sample_data
"""
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "telco_churn.csv"


def generate(path: Path = DEFAULT_PATH, n: int = 7043, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    def choice(options, probs):
        return rng.choice(options, size=n, p=probs)

    gender = choice(["Male", "Female"], [0.5, 0.5])
    senior = rng.binomial(1, 0.16, n)
    partner = choice(["Yes", "No"], [0.48, 0.52])
    dependents = np.where(senior == 1, choice(["Yes", "No"], [0.05, 0.95]),
                          choice(["Yes", "No"], [0.35, 0.65]))
    tenure = np.clip(rng.gamma(2.2, 15, n).astype(int), 0, 72)

    phone = choice(["Yes", "No"], [0.90, 0.10])
    multiple = np.where(phone == "No", "No phone service", choice(["Yes", "No"], [0.47, 0.53]))

    internet = choice(["DSL", "Fiber optic", "No"], [0.34, 0.44, 0.22])
    no_net = internet == "No"

    def addon(p_yes):
        return np.where(no_net, "No internet service", choice(["Yes", "No"], [p_yes, 1 - p_yes]))

    security = addon(0.29)
    backup = addon(0.34)
    protection = addon(0.34)
    support = addon(0.29)
    tv = addon(0.38)
    movies = addon(0.39)

    contract = choice(["Month-to-month", "One year", "Two year"], [0.55, 0.21, 0.24])
    paperless = choice(["Yes", "No"], [0.59, 0.41])
    payment = choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        [0.34, 0.23, 0.22, 0.21],
    )

    monthly = (
        18
        + 25 * (internet == "DSL")
        + 55 * (internet == "Fiber optic")
        + 8 * (phone == "Yes")
        + 5 * (security == "Yes") + 5 * (backup == "Yes")
        + 5 * (protection == "Yes") + 5 * (support == "Yes")
        + 8 * (tv == "Yes") + 8 * (movies == "Yes")
        + rng.normal(0, 3, n)
    ).round(2)
    monthly = np.clip(monthly, 18.25, 118.75)

    total = (tenure * monthly * rng.uniform(0.95, 1.05, n)).round(2)

    # The hidden "rule" that decides who churns (this is what the model must learn)
    logit = (
        -1.35
        + 1.3 * (contract == "Month-to-month")
        - 1.0 * (contract == "Two year")
        + 0.8 * (internet == "Fiber optic")
        - 0.035 * tenure
        + 0.012 * (monthly - 65)
        + 0.4 * (paperless == "Yes")
        + 0.5 * (payment == "Electronic check")
        + 0.35 * senior
        - 0.5 * (security == "Yes")
        - 0.5 * (support == "Yes")
        + rng.normal(0, 0.6, n)
    )
    churn = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")

    letters = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    ids = [f"{i:04d}-" + "".join(rng.choice(letters, 5)) for i in range(n)]

    df = pd.DataFrame({
        "customerID": ids,
        "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone, "MultipleLines": multiple,
        "InternetService": internet, "OnlineSecurity": security, "OnlineBackup": backup,
        "DeviceProtection": protection, "TechSupport": support, "StreamingTV": tv,
        "StreamingMovies": movies, "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly,
        "TotalCharges": total.astype(object), "Churn": churn,
    })

    # The real dataset has blank TotalCharges for customers with tenure 0
    df.loc[df["tenure"] == 0, "TotalCharges"] = " "

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[data] wrote {len(df)} rows to {path}  (churn rate: {(df['Churn'] == 'Yes').mean():.1%})")
    return df


if __name__ == "__main__":
    generate()
