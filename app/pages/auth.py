"""Login page with role-based authentication and dynamic redirection."""

from __future__ import annotations

import streamlit as st

from app.app_ui import _load_database
from app.css import login_logo_html
from app.session import sign_in


def render_login_page() -> None:
    """Login page with role detection. Account creation is reserved to the
    admin via the User Management page."""
    Database = _load_database()
    db = Database()

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown(login_logo_html(), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

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
