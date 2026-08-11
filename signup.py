from __future__ import annotations

import random
import re
from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
USERS_FILE = BASE_DIR / "users.csv"
EMAIL_PATTERN = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"


def ensure_user_store() -> None:
    if USERS_FILE.exists():
        return
    pd.DataFrame(columns=["name", "email", "password"]).to_csv(USERS_FILE, index=False)


def load_users() -> pd.DataFrame:
    ensure_user_store()
    return pd.read_csv(USERS_FILE)


def save_users(users_df: pd.DataFrame) -> None:
    users_df.to_csv(USERS_FILE, index=False)


def signup_page() -> None:
    st.markdown('<div class="panel-wrap"><div class="panel">', unsafe_allow_html=True)
    st.title("Signup")

    name = st.text_input("Name", key="signup_name")
    email = st.text_input("Email", key="signup_email")
    password = st.text_input("Password", type="password", key="signup_password")
    confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm_password")

    if st.button("Generate OTP", key="signup_generate_otp"):
        validation_errors: list[str] = []
        cleaned_name = name.strip()
        cleaned_email = email.strip().lower()

        if not cleaned_name:
            validation_errors.append("Name is required.")
        if not cleaned_email:
            validation_errors.append("Email is required.")
        elif not re.fullmatch(EMAIL_PATTERN, cleaned_email):
            validation_errors.append("Enter a valid email address.")
        if len(password) < 6:
            validation_errors.append("Password must be at least 6 characters.")
        if password != confirm_password:
            validation_errors.append("Passwords do not match.")

        users = load_users()
        if not users.empty and users["email"].astype(str).str.strip().str.lower().eq(cleaned_email).any():
            validation_errors.append("An account with this email already exists.")

        if validation_errors:
            for error in validation_errors:
                st.error(error)
        else:
            st.session_state["otp"] = f"{random.randint(100000, 999999):06d}"
            st.session_state["signup_pending_user"] = {
                "name": cleaned_name,
                "email": cleaned_email,
                "password": password,
            }
            st.success("OTP generated successfully.")
            st.info(f"Your OTP is: {st.session_state['otp']}")

    otp_input = st.text_input("Enter OTP", max_chars=6, key="signup_otp_input")

    if st.button("Verify OTP and Signup", key="signup_verify"):
        pending_user = st.session_state.get("signup_pending_user")
        generated_otp = st.session_state.get("otp")

        if not pending_user or not generated_otp:
            st.error("Generate an OTP before verifying.")
        elif otp_input.strip() != generated_otp:
            st.error("Invalid OTP")
        else:
            users = load_users()
            users.loc[len(users)] = [
                pending_user["name"],
                pending_user["email"],
                pending_user["password"],
            ]
            save_users(users)
            st.session_state["otp"] = None
            st.session_state["signup_pending_user"] = None
            st.success("Signup successful")

    if st.button("Already have account? Login", key="signup_to_login"):
        st.session_state.page = "login"
        st.rerun()

    st.markdown("</div></div>", unsafe_allow_html=True)
