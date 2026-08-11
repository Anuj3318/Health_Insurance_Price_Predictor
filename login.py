from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
USERS_FILE = BASE_DIR / "users.csv"


def ensure_user_store() -> None:
    if USERS_FILE.exists():
        return
    pd.DataFrame(columns=["name", "email", "password"]).to_csv(USERS_FILE, index=False)


def load_users() -> pd.DataFrame:
    ensure_user_store()
    return pd.read_csv(USERS_FILE)


def login_page() -> None:
    st.markdown('<div class="panel-wrap"><div class="panel">', unsafe_allow_html=True)
    st.title("Login")

    email = st.text_input("Email", key="login_email")
    password = st.text_input("Password", type="password", key="login_password")

    if st.button("Login", key="login_submit"):
        if not email.strip() or not password.strip():
            st.error("Please enter both email and password.")
        else:
            users = load_users()
            matched_user = users[
                (users["email"].astype(str).str.strip().str.lower() == email.strip().lower())
                & (users["password"].astype(str) == password)
            ]

            if matched_user.empty:
                st.error("Invalid credentials")
            else:
                st.success("Login successful")
                st.session_state.is_logged_in = True
                st.session_state.current_user = matched_user.iloc[0].to_dict()
                st.session_state.page = "predictor"
                st.rerun()

    if st.button("Don't have account? Signup", key="login_to_signup"):
        st.session_state.page = "signup"
        st.rerun()

    st.markdown("</div></div>", unsafe_allow_html=True)
