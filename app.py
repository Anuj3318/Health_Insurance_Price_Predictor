from __future__ import annotations

import random
import re
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any

import pandas as pd
from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for

from model import ensure_artifacts, load_predictor, predict_premium, validate_inputs


BASE_DIR = Path(__file__).resolve().parent
USERS_FILE = BASE_DIR / "users.csv"
PREDICTIONS_FILE = BASE_DIR / "predictions.csv"
PURCHASES_FILE = BASE_DIR / "purchases.csv"
EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")

app = Flask(__name__)
app.secret_key = "smart-insurance-advisor-secret-key"


def format_inr(value: float) -> str:
    amount = f"{value:.2f}"
    integer_part, decimal_part = amount.split(".")
    sign = ""
    if integer_part.startswith("-"):
        sign = "-"
        integer_part = integer_part[1:]

    if len(integer_part) > 3:
        last_three = integer_part[-3:]
        remaining = integer_part[:-3]
        groups: list[str] = []
        while len(remaining) > 2:
            groups.insert(0, remaining[-2:])
            remaining = remaining[:-2]
        if remaining:
            groups.insert(0, remaining)
        integer_part = ",".join(groups + [last_three])

    return f"{sign}{integer_part}.{decimal_part}"


app.jinja_env.filters["inr"] = format_inr


def format_display_date(value: Any) -> str:
    """Return a readable date for stored CSV timestamp strings."""
    if value is None or (isinstance(value, float) and value != value):
        return ""
    raw_value = str(value).strip()
    if not raw_value:
        return ""
    for date_format in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw_value, date_format).strftime("%d %b %Y")
        except ValueError:
            continue
    return raw_value.split(" ")[0]


app.jinja_env.filters["display_date"] = format_display_date


def ensure_user_store() -> None:
    if USERS_FILE.exists():
        return
    pd.DataFrame(columns=["name", "email", "password"]).to_csv(USERS_FILE, index=False)


def load_users() -> pd.DataFrame:
    ensure_user_store()
    return pd.read_csv(USERS_FILE)


def save_users(users_df: pd.DataFrame) -> None:
    users_df.to_csv(USERS_FILE, index=False)


def find_user(email: str) -> dict[str, Any] | None:
    users = load_users()
    matches = users[users["email"].astype(str).str.strip().str.lower() == email.strip().lower()]
    if matches.empty:
        return None
    return matches.iloc[0].to_dict()


def get_user_profile(email: str) -> dict[str, Any]:
    """
    Safely retrieve user profile from users.csv.
    
    Args:
        email: User email to lookup
        
    Returns:
        Dictionary with user data if found, empty dictionary otherwise
    """
    try:
        if not email:
            return {}
        
        users = load_users()
        matches = users[users["email"].astype(str).str.strip().str.lower() == email.strip().lower()]
        
        if matches.empty:
            return {}
        
        return matches.iloc[0].to_dict()
    except Exception:
        # Return empty dict if any error occurs (file not found, corrupted data, etc.)
        return {}


def ensure_predictions_store() -> None:
    """Create predictions CSV if it doesn't exist."""
    if PREDICTIONS_FILE.exists():
        return
    pd.DataFrame(columns=[
        "email", "date", "age", "gender", "bmi", "smoker", "alcohol",
        "dependents", "diseases", "premium", "risk_score", "risk_label"
    ]).to_csv(PREDICTIONS_FILE, index=False)


def load_predictions() -> pd.DataFrame:
    """Load all predictions."""
    ensure_predictions_store()
    return pd.read_csv(PREDICTIONS_FILE)


def save_prediction(email: str, age: int, gender: str, bmi: float, smoker: bool,
                   alcohol: bool, dependents: int, diseases: list[str], 
                   premium: float, risk_score: int, risk_label: str) -> None:
    """Save a new prediction."""
    ensure_predictions_store()
    predictions = load_predictions()
    new_prediction = pd.DataFrame([{
        "email": email,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "age": age,
        "gender": gender,
        "bmi": bmi,
        "smoker": "Yes" if smoker else "No",
        "alcohol": "Yes" if alcohol else "No",
        "dependents": dependents,
        "diseases": "|".join(diseases) if diseases else "None",
        "premium": round(premium, 2),
        "risk_score": risk_score,
        "risk_label": risk_label,
    }])
    predictions = pd.concat([predictions, new_prediction], ignore_index=True)
    predictions.to_csv(PREDICTIONS_FILE, index=False)


def get_user_predictions(email: str) -> list[dict[str, Any]]:
    """Get all predictions for a user."""
    ensure_predictions_store()
    predictions = load_predictions()
    user_predictions = predictions[predictions["email"].astype(str).str.lower() == email.strip().lower()]
    if user_predictions.empty:
        return []
    # Sort by date in descending order (newest first)
    user_predictions = user_predictions.sort_values("date", ascending=False)
    
    # Normalize diseases field to ensure it's always a string
    records = user_predictions.to_dict('records')
    for record in records:
        # Handle None, NaN, float, or other non-string types
        diseases = record.get("diseases")
        if diseases is None or (isinstance(diseases, float) and diseases != diseases):  # NaN check
            record["diseases"] = ""
        elif not isinstance(diseases, str):
            record["diseases"] = str(diseases)
        elif diseases.lower() == "none" or not diseases.strip():
            record["diseases"] = ""
    
    return records


def ensure_purchases_store() -> None:
    """Create purchases CSV if it doesn't exist."""
    if PURCHASES_FILE.exists():
        return
    pd.DataFrame(columns=[
        "email", "date", "plan_name", "price", "age", "gender", "bmi",
        "smoker", "alcohol", "dependents", "diseases"
    ]).to_csv(PURCHASES_FILE, index=False)


def load_purchases() -> pd.DataFrame:
    """Load all purchases."""
    ensure_purchases_store()
    return pd.read_csv(PURCHASES_FILE)


def save_purchase(email: str, plan_name: str, price: float, age: int, gender: str,
                 bmi: float, smoker: bool, alcohol: bool, dependents: int,
                 diseases: list[str]) -> None:
    """Save a new purchase."""
    ensure_purchases_store()
    purchases = load_purchases()
    new_purchase = pd.DataFrame([{
        "email": email,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "plan_name": plan_name,
        "price": round(price, 2),
        "age": age,
        "gender": gender,
        "bmi": bmi,
        "smoker": "Yes" if smoker else "No",
        "alcohol": "Yes" if alcohol else "No",
        "dependents": dependents,
        "diseases": "|".join(diseases) if diseases else "None",
    }])
    purchases = pd.concat([purchases, new_purchase], ignore_index=True)
    purchases.to_csv(PURCHASES_FILE, index=False)


def get_plan_name_for_age(age: int | float | str) -> str:
    """Return the user-facing plan name based on age."""
    try:
        age_value = int(float(age))
    except (TypeError, ValueError):
        age_value = 0
    return "GENZ Plan" if 18 <= age_value <= 28 else "Millennial Plan"


def user_has_active_plan(email: str) -> bool:
    """A stored purchase represents the user's active plan."""
    return get_user_latest_purchase(email) is not None


def cancel_user_plan(email: str) -> bool:
    """Remove the logged-in user's active plan records."""
    ensure_purchases_store()
    purchases = load_purchases()
    user_mask = purchases["email"].astype(str).str.strip().str.lower() == email.strip().lower()
    if not user_mask.any():
        return False
    purchases = purchases[~user_mask].reset_index(drop=True)
    purchases.to_csv(PURCHASES_FILE, index=False)
    return True


def get_user_latest_purchase(email: str) -> dict[str, Any] | None:
    """Get the latest purchase for a user."""
    ensure_purchases_store()
    purchases = load_purchases()
    user_purchases = purchases[purchases["email"].astype(str).str.lower() == email.strip().lower()]
    if user_purchases.empty:
        return None
    # Get the latest purchase
    user_purchases = user_purchases.sort_values("date", ascending=False)
    purchase = user_purchases.iloc[0].to_dict()
    purchase["plan_name"] = get_plan_name_for_age(purchase.get("age", 0))
    
    # Normalize diseases field to ensure it's always a string
    diseases = purchase.get("diseases")
    if diseases is None or (isinstance(diseases, float) and diseases != diseases):  # NaN check
        purchase["diseases"] = ""
    elif not isinstance(diseases, str):
        purchase["diseases"] = str(diseases)
    elif diseases.lower() == "none" or not diseases.strip():
        purchase["diseases"] = ""
    
    return purchase


def delete_user_prediction_by_index(email: str, index: int) -> bool:
    """Delete one prediction from the logged-in user's sorted prediction list."""
    ensure_predictions_store()
    predictions = load_predictions()
    user_predictions = predictions[predictions["email"].astype(str).str.lower() == email.strip().lower()]
    if user_predictions.empty:
        return False

    user_predictions = user_predictions.sort_values("date", ascending=False)
    if index < 0 or index >= len(user_predictions):
        return False

    row_index = user_predictions.index[index]
    predictions = predictions.drop(index=row_index).reset_index(drop=True)
    predictions.to_csv(PREDICTIONS_FILE, index=False)
    return True


def get_latest_prediction_details(email: str) -> dict[str, Any] | None:
    """Return latest prediction data usable as purchase details."""
    predictions = get_user_predictions(email)
    return predictions[0] if predictions else None


def parse_bool_choice(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"yes", "true", "1", "on"}


def normalize_disease_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and value != value):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    value_text = str(value).strip()
    if not value_text or value_text.lower() == "none":
        return []
    separator = "|" if "|" in value_text else ","
    return [item.strip() for item in value_text.split(separator) if item.strip()]


def login_required(view_func):
    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if "user_email" not in session:
            flash("Please log in to access the predictor.", "error")
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)

    return wrapped_view


@app.before_request
def bootstrap() -> None:
    ensure_user_store()
    ensure_artifacts(BASE_DIR)


@app.route("/")
def landing() -> str:
    if session.get("user_email"):
        return redirect(url_for("dashboard"))
    return render_template("landing.html")


@app.route("/login", methods=["GET", "POST"])
def login() -> str:
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not email or not password:
            flash("Please enter both email and password.", "error")
        else:
            user = find_user(email)
            if not user or str(user["password"]) != password:
                flash("Invalid credentials.", "error")
                return redirect(url_for("login"))
            else:
                session.clear()
                session["user_email"] = str(user["email"])
                session["user_name"] = str(user["name"])
                flash("Login successful.", "success")
                return redirect(url_for("dashboard"))

        return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup() -> str:
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        otp_value = "".join(request.form.get(f"otp_{index}", "").strip() for index in range(1, 7))

        errors: list[str] = []
        if not name:
            errors.append("Name is required.")
        if not email:
            errors.append("Email is required.")
        elif not EMAIL_PATTERN.fullmatch(email):
            errors.append("Please enter a valid email address")
        if len(password) < 6:
            errors.append("Password must be at least 6 characters.")
        if password != confirm_password:
            errors.append("Passwords do not match.")
        if find_user(email):
            errors.append("User already exists")

        stored_otp = session.get("signup_otp")
        stored_otp_email = session.get("signup_otp_email")
        if not stored_otp or not stored_otp_email:
            errors.append("Please send OTP first.")
        elif stored_otp_email != email:
            errors.append("OTP was sent for a different email address.")
        elif otp_value != stored_otp:
            errors.append("Invalid OTP.")

        if errors:
            for error in errors:
                flash(error, "error")
        else:
            users = load_users()
            users.loc[len(users)] = [name, email, password]
            save_users(users)
            session.pop("signup_otp", None)
            session.pop("signup_otp_email", None)
            session["signup_success"] = True
            return redirect(url_for("signup"))

        return redirect(url_for("signup"))

    signup_success = bool(session.pop("signup_success", False))
    return render_template("signup.html", signup_success=signup_success)


@app.route("/send-otp", methods=["POST"])
def send_otp() -> Any:
    email = request.form.get("email", "").strip().lower()

    if not email:
        return jsonify({"success": False, "message": "Email is required."}), 400
    if not EMAIL_PATTERN.fullmatch(email):
        return jsonify({"success": False, "message": "Please enter a valid email address"}), 400
    if find_user(email):
        return jsonify({"success": False, "message": "User already exists"}), 400

    otp = f"{random.randint(100000, 999999):06d}"
    session["signup_otp"] = otp
    session["signup_otp_email"] = email
    return jsonify(
        {
            "success": True,
            "message": "OTP sent successfully",
            "otp": otp,
        }
    )


@app.route("/logout")
def logout() -> Any:
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("landing"))


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict() -> str:
    result: dict[str, Any] | None = None
    form_data = {
        "age": "32",
        "gender": "Male",
        "bmi": "24.5",
        "smoker": False,
        "alcohol": True,
        "dependents": "2",
    }

    if request.method == "POST":
        # Get diseases from form (multiple values possible)
        diseases_list = request.form.getlist("diseases")
        
        form_data = {
            "age": request.form.get("age", "").strip(),
            "gender": request.form.get("gender", "Male"),
            "bmi": request.form.get("bmi", "").strip(),
            "smoker": request.form.get("smoker") == "Yes",
            "alcohol": request.form.get("alcohol") == "Yes",
            "dependents": request.form.get("dependents", "0").strip(),
            "diseases": diseases_list,
        }

        errors: list[str] = []
        try:
            age = int(form_data["age"])
        except ValueError:
            age = -1
            errors.append("Age must be a valid number.")

        try:
            bmi = float(form_data["bmi"])
        except ValueError:
            bmi = -1.0
            errors.append("BMI must be a valid number.")

        try:
            dependents = int(form_data["dependents"])
        except ValueError:
            dependents = -1
            errors.append("Dependents must be a valid number.")

        if not errors:
            try:
                validate_inputs(
                    age=age,
                    gender=form_data["gender"],
                    bmi=bmi,
                    smoker="Yes" if form_data["smoker"] else "No",
                    alcohol="Yes" if form_data["alcohol"] else "No",
                    dependents=dependents,
                )
            except ValueError as exc:
                errors.append(str(exc))

        if errors:
            for error in errors:
                flash(error, "error")
        else:
            predictor = load_predictor(BASE_DIR)
            premium = predict_premium(
                predictor,
                age=age,
                gender=form_data["gender"],
                bmi=bmi,
                smoker="Yes" if form_data["smoker"] else "No",
                alcohol="Yes" if form_data["alcohol"] else "No",
                dependents=dependents,
            )
            
            # Calculate disease impact multiplier
            disease_multipliers = {
                "Diabetes": 0.15,
                "High Blood Pressure": 0.10,
                "Asthma": 0.08,
                "Heart Disease": 0.20,
            }
            disease_impact = 0.0
            for disease in diseases_list:
                if disease in disease_multipliers:
                    disease_impact += disease_multipliers[disease]
            
            # Apply disease adjustment to premium
            adjusted_premium = premium * (1 + disease_impact)
            
            risk_score = min(
                100,
                int(
                    ((age - 18) / 52) * 32
                    + ((bmi - 16) / 24) * 26
                    + (24 if form_data["smoker"] else 0)
                    + (10 if form_data["alcohol"] else 0)
                    + (dependents * 2)
                    + (disease_impact * 20)  # Disease impact on risk score
                ),
            )
            risk_label = "Low" if risk_score < 35 else "Medium" if risk_score < 65 else "High"
            result = {
                "premium": adjusted_premium,
                "risk_score": risk_score,
                "risk_label": risk_label,
                "risk_bar": max(10, min(risk_score, 100)),
                "diseases": diseases_list,
            }
            
            # Save prediction to database
            save_prediction(
                email=session.get("user_email"),
                age=age,
                gender=form_data["gender"],
                bmi=bmi,
                smoker=form_data["smoker"],
                alcohol=form_data["alcohol"],
                dependents=dependents,
                diseases=diseases_list,
                premium=adjusted_premium,
                risk_score=risk_score,
                risk_label=risk_label,
            )
            
            flash("Prediction generated successfully.", "success")

    return render_template("predictor.html", result=result, form_data=form_data, user_name=session.get("user_name"))


@app.route("/compare")
@login_required
def compare() -> Any:
    premium_raw = request.args.get("premium", "").strip()
    try:
        user_premium = float(premium_raw)
    except ValueError:
        flash("Generate a premium prediction before comparing plans.", "error")
        return redirect(url_for("predict"))

    if user_premium <= 0:
        flash("Premium value must be greater than zero to compare plans.", "error")
        return redirect(url_for("predict"))

    email = session.get("user_email")
    latest_prediction = get_latest_prediction_details(email)
    active_plan = get_user_latest_purchase(email)
    plan_age = latest_prediction.get("age", 0) if latest_prediction else 0
    dynamic_plan_name = get_plan_name_for_age(plan_age)

    lic_price = user_premium * 1.05
    hdfc_price = user_premium * 1.08
    icici_price = user_premium * 1.06
    our_price = user_premium * 0.98

    comparison_plans = [
        {
            "id": "lic",
            "company": "LIC",
            "price": lic_price,
            "features": [
                "Wide hospital network across India",
                "Stable claim servicing with traditional coverage",
                "Suitable for conservative long-term planning",
            ],
        },
        {
            "id": "hdfc",
            "company": "HDFC Ergo",
            "price": hdfc_price,
            "features": [
                "Fast digital onboarding and policy issuance",
                "Cashless claim support in major metros",
                "Well-suited for premium urban protection plans",
            ],
        },
        {
            "id": "icici",
            "company": "ICICI Lombard",
            "price": icici_price,
            "features": [
                "Balanced premium-to-coverage positioning",
                "Broad add-on ecosystem for family policies",
                "Strong digital self-service experience",
            ],
        },
        {
            "id": "our-plan",
            "company": dynamic_plan_name,
            "price": our_price,
            "features": [
                "AI-personalized premium using your ML prediction",
                "2% below your predicted market benchmark by default",
                "Extra 10% discount available with MYFIRSTTIME",
            ],
            "highlighted": True,
        },
    ]

    purchase_details = {}
    if latest_prediction:
        purchase_details = {
            "age": int(float(latest_prediction.get("age", 0) or 0)),
            "bmi": float(latest_prediction.get("bmi", 0) or 0),
            "gender": str(latest_prediction.get("gender") or ""),
            "diseases": str(latest_prediction.get("diseases") or ""),
            "tobacco": str(latest_prediction.get("smoker") or "No"),
            "alcohol": str(latest_prediction.get("alcohol") or "No"),
            "dependents": int(float(latest_prediction.get("dependents", 0) or 0)),
        }

    return render_template(
        "compare.html",
        user_premium=user_premium,
        lic_price=lic_price,
        hdfc_price=hdfc_price,
        icici_price=icici_price,
        our_price=our_price,
        comparison_plans=comparison_plans,
        purchase_details=purchase_details,
        dynamic_plan_name=dynamic_plan_name,
        active_plan=active_plan,
        user_name=session.get("user_name"),
    )


@app.route("/save_purchase", methods=["POST"])
@login_required
def save_purchase_route() -> Any:
    """Save the selected plan for the logged-in user after payment succeeds."""
    email = session.get("user_email")
    payload = request.get_json(silent=True) or {}
    latest_prediction = get_latest_prediction_details(email)
    details = payload.get("details") or latest_prediction or {}

    try:
        if user_has_active_plan(email):
            return jsonify({
                "success": False,
                "message": "You already have an active plan. Please cancel it before purchasing a new one.",
            }), 409

        price = float(payload.get("price") or 0)
        if price <= 0:
            return jsonify({"success": False, "message": "Invalid plan price."}), 400
        age = int(float(details.get("age", 0)))
        plan_name = get_plan_name_for_age(age)

        save_purchase(
            email=email,
            plan_name=plan_name,
            price=price,
            age=age,
            gender=str(details.get("gender") or "Not specified"),
            bmi=float(details.get("bmi", 0)),
            smoker=parse_bool_choice(details.get("tobacco", details.get("smoker", False))),
            alcohol=parse_bool_choice(details.get("alcohol", False)),
            dependents=int(float(details.get("dependents", 0) or 0)),
            diseases=normalize_disease_list(details.get("diseases")),
        )
    except (TypeError, ValueError):
        return jsonify({"success": False, "message": "Could not save purchase details."}), 400

    return jsonify({"success": True, "redirect_url": url_for("dashboard")})


@app.route("/cancel_plan", methods=["POST"])
@login_required
def cancel_plan() -> Any:
    """Cancel the logged-in user's active plan."""
    email = session.get("user_email")
    if cancel_user_plan(email):
        flash("Plan cancelled. You can purchase a new plan now.", "success")
    else:
        flash("No active plan found to cancel.", "error")
    return redirect(url_for("dashboard"))


@app.route("/delete_prediction/<int:index>", methods=["POST"])
@login_required
def delete_prediction(index: int) -> Any:
    """Delete one prediction from the current user's dashboard history."""
    email = session.get("user_email")
    if delete_user_prediction_by_index(email, index):
        flash("Prediction deleted.", "success")
    else:
        flash("Prediction could not be found.", "error")
    return redirect(url_for("dashboard"))


@app.route("/claims")
def claims() -> str:
    return render_template("claims.html", user_name=session.get("user_name"))


@app.route("/privacy-policy")
def privacy_policy() -> str:
    return render_template("privacy.html", user_name=session.get("user_name"))


@app.route("/terms")
def terms() -> str:
    return render_template("terms.html", user_name=session.get("user_name"))


@app.route("/contact", methods=["GET", "POST"])
def contact() -> Any:
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        message = request.form.get("message", "").strip()
        
        if not name or not email or not message:
            flash("Please fill in all fields.", "error")
        else:
            flash("Thank you for contacting us! We'll get back to you soon.", "success")
            return redirect(url_for("contact"))
    
    return render_template("contact.html", user_name=session.get("user_name"))


@app.route("/support")
def support() -> str:
    return render_template("support.html", user_name=session.get("user_name"))


@app.route("/knowledge-base")
def knowledge_base() -> str:
    return render_template("knowledge.html", user_name=session.get("user_name"))


@app.route("/dashboard")
@login_required
def dashboard() -> str:
    """User dashboard with prediction history and purchased plans."""
    email = session.get("user_email")
    user = get_user_profile(email)
    predictions = get_user_predictions(email)
    latest_purchase = get_user_latest_purchase(email)
    
    return render_template(
        "dashboard.html",
        user_name=session.get("user_name"),
        user_email=email,
        predictions=predictions,
        latest_purchase=latest_purchase,
        user_info=user,
    )


if __name__ == "__main__":
    app.run(debug=True)
