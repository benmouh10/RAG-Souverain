"""Login page with role-based authentication and dynamic redirection."""

from __future__ import annotations

import streamlit as st

from app.app_ui import _load_database
from app.css import login_logo_html
from app.session import sign_in


def render_login_page() -> None:
    """Login page with account creation and role detection."""
    Database = _load_database()
    db = Database()

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown(login_logo_html(), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    tab_login, tab_register = st.tabs(["Login", "Create Account"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="admin")
            password = st.text_input("Password", type="password", placeholder="enter your password")
            submitted = st.form_submit_button("Sign In", use_container_width=True)
            if submitted:
                if not username or not password:
                    st.error("Please fill in all fields.")
                else:
                    user = db.authenticate(username, password)
                    if user is not None:
                        sign_in(user.user_id, user.username, user.role)
                        st.success(f"Welcome back, {user.username}!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials.")

    with tab_register:
        with st.form("register_form"):
            new_user = st.text_input("Username", placeholder="alice")
            new_pass = st.text_input("Password", type="password", placeholder="****")
            new_pass2 = st.text_input("Confirm Password", type="password", placeholder="****")
            reg_submitted = st.form_submit_button("Create Account", use_container_width=True)
            if reg_submitted:
                if not new_user or not new_pass:
                    st.error("Please fill in all fields.")
                elif new_pass != new_pass2:
                    st.error("Passwords do not match.")
                elif len(new_pass) < 4:
                    st.error("Password must be at least 4 characters.")
                else:
                    created = db.create_user(new_user, new_pass, role="user")
                    if created is None:
                        st.error("Username already exists.")
                    else:
                        st.success("Account created. Please sign in.")
