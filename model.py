from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_FILE_NAME = "insurance_data.csv"
MODEL_FILE_NAME = "model.pkl"
FEATURES = ["age", "gender", "bmi", "smoker", "alcohol", "dependents"]
TARGET = "premium"
VALID_GENDERS = {"Male", "Female", "Other"}
VALID_BINARY = {"Yes", "No"}


def _normalize_gender(gender: str) -> str:
    cleaned = str(gender).strip().title()
    if cleaned not in VALID_GENDERS:
        raise ValueError("Gender must be Male, Female, or Other.")
    return cleaned


def _normalize_flag(value: str, field_name: str) -> str:
    cleaned = str(value).strip().title()
    if cleaned not in VALID_BINARY:
        raise ValueError(f"{field_name} must be Yes or No.")
    return cleaned


def validate_inputs(
    *,
    age: int,
    gender: str,
    bmi: float,
    smoker: str,
    alcohol: str,
    dependents: int,
) -> dict[str, Any]:
    if not 18 <= int(age) <= 70:
        raise ValueError("Age must be between 18 and 70.")
    if not 16 <= float(bmi) <= 40:
        raise ValueError("BMI must be between 16 and 40.")
    if not 0 <= int(dependents) <= 6:
        raise ValueError("Dependents must be between 0 and 6.")

    return {
        "age": int(age),
        "gender": _normalize_gender(gender),
        "bmi": round(float(bmi), 1),
        "smoker": _normalize_flag(smoker, "Smoker"),
        "alcohol": _normalize_flag(alcohol, "Alcohol"),
        "dependents": int(dependents),
    }


def pricing_breakdown(
    *,
    age: int,
    gender: str,
    bmi: float,
    smoker: str,
    alcohol: str,
    dependents: int,
    include_variation: bool,
) -> dict[str, float]:
    age = int(age)
    bmi = float(bmi)
    dependents = int(dependents)

    age_component = 900 + (age * 32) + (max(age - 30, 0) * 18) + (max(age - 45, 0) * 28)

    if bmi < 18.5:
        bmi_component = 250
    elif bmi < 25:
        bmi_component = 0
    elif bmi < 30:
        bmi_component = 450 + ((bmi - 25) * 55)
    elif bmi < 35:
        bmi_component = 1100 + ((bmi - 30) * 100)
    else:
        bmi_component = 1650 + ((bmi - 35) * 120)

    smoker_component = 0
    if smoker == "Yes":
        smoker_component = 1800 if age < 30 else 2200 if age < 45 else 2800

    alcohol_component = 0
    if alcohol == "Yes":
        alcohol_component = 550 if age < 30 else 800 if age < 45 else 1100

    dependents_component = dependents * 300

    gender_component = 0
    if gender == "Female":
        gender_component = -100
    elif gender == "Other":
        gender_component = 50

    interaction_component = 0
    if bmi >= 30:
        interaction_component += 250
    if smoker == "Yes" and bmi >= 30:
        interaction_component += 450
    if smoker == "Yes" and age >= 45:
        interaction_component += 500
    if alcohol == "Yes" and bmi >= 30:
        interaction_component += 200

    variation_component = 0
    if include_variation:
        risk_seed = (
            (age * 13)
            + (int(round(bmi * 10)) * 7)
            + (dependents * 17)
            + (110 if smoker == "Yes" else 25)
            + (60 if alcohol == "Yes" else 10)
            + {"Male": 20, "Female": -15, "Other": 30}[gender]
        )
        variation_component = ((risk_seed % 9) - 4) * 55

    premium = (
        age_component
        + bmi_component
        + smoker_component
        + alcohol_component
        + dependents_component
        + gender_component
        + interaction_component
        + variation_component
    )

    return {
        "age_component": round(age_component, 2),
        "bmi_component": round(bmi_component, 2),
        "smoker_component": round(smoker_component, 2),
        "alcohol_component": round(alcohol_component, 2),
        "dependents_component": round(dependents_component, 2),
        "gender_component": round(gender_component, 2),
        "interaction_component": round(interaction_component, 2),
        "variation_component": round(variation_component, 2),
        "premium": round(max(premium, 1200), 2),
    }


def calculate_rule_based_premium(
    *,
    age: int,
    gender: str,
    bmi: float,
    smoker: str,
    alcohol: str,
    dependents: int,
    include_variation: bool = False,
) -> float:
    validated = validate_inputs(
        age=age,
        gender=gender,
        bmi=bmi,
        smoker=smoker,
        alcohol=alcohol,
        dependents=dependents,
    )
    return pricing_breakdown(**validated, include_variation=include_variation)["premium"]


def generate_synthetic_data(rows: int = 360) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    index = 0
    genders = ["Male", "Female", "Other"]

    while len(records) < rows:
        age = 18 + ((index * 7) % 53)
        gender = genders[index % len(genders)]
        bmi = round(16 + (((index * 11) + (index // 7)) % 241) / 10, 1)
        smoker = "Yes" if (index % 5 == 0 or (age >= 48 and index % 4 == 0)) else "No"
        alcohol = "Yes" if index % 7 in {1, 4, 6} else "No"
        dependents = min(6, max(0, ((age - 22) // 8) + ((index % 3) - 1)))

        key = (age, gender, bmi, smoker, alcohol, dependents)
        index += 1
        if key in seen:
            continue
        seen.add(key)

        premium = calculate_rule_based_premium(
            age=age,
            gender=gender,
            bmi=bmi,
            smoker=smoker,
            alcohol=alcohol,
            dependents=dependents,
            include_variation=True,
        )
        records.append(
            {
                "age": age,
                "gender": gender,
                "bmi": bmi,
                "smoker": smoker,
                "alcohol": alcohol,
                "dependents": dependents,
                "premium": premium,
            }
        )

    return pd.DataFrame(records).sort_values(["age", "bmi", "dependents", "gender"]).reset_index(drop=True)


def train_and_save_model(base_dir: Path) -> dict[str, Any]:
    data_path = base_dir / DATA_FILE_NAME
    model_path = base_dir / MODEL_FILE_NAME

    df = pd.read_csv(data_path)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                ["gender", "smoker", "alcohol"],
            )
        ],
        remainder="passthrough",
    )

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    x_train, x_test, y_train, y_test = train_test_split(
        df[FEATURES],
        df[TARGET],
        test_size=0.2,
        random_state=42,
    )
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    payload = {
        "model": pipeline,
        "features": FEATURES,
        "metrics": {
            "mae": round(float(mean_absolute_error(y_test, predictions)), 2),
            "r2": round(float(r2_score(y_test, predictions)), 4),
        },
    }
    joblib.dump(payload, model_path)
    return payload


def ensure_artifacts(base_dir: Path) -> None:
    data_path = base_dir / DATA_FILE_NAME
    model_path = base_dir / MODEL_FILE_NAME
    if data_path.exists() and model_path.exists():
        return

    df = generate_synthetic_data()
    df.to_csv(data_path, index=False)
    train_and_save_model(base_dir)


def load_predictor(base_dir: Path) -> dict[str, Any]:
    ensure_artifacts(base_dir)
    return joblib.load(base_dir / MODEL_FILE_NAME)


def predict_premium(
    predictor: dict[str, Any],
    *,
    age: int,
    gender: str,
    bmi: float,
    smoker: str,
    alcohol: str,
    dependents: int,
) -> float:
    validated = validate_inputs(
        age=age,
        gender=gender,
        bmi=bmi,
        smoker=smoker,
        alcohol=alcohol,
        dependents=dependents,
    )

    features = pd.DataFrame([validated])
    model_prediction = float(predictor["model"].predict(features)[0])
    rule_prediction = calculate_rule_based_premium(**validated, include_variation=False)

    blended_prediction = (rule_prediction * 0.65) + (model_prediction * 0.35)
    floor = rule_prediction * 0.92
    return round(max(blended_prediction, floor), 2)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    dataset = generate_synthetic_data()
    dataset.to_csv(base_dir / DATA_FILE_NAME, index=False)
    train_and_save_model(base_dir)
