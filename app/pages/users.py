"""User Management page — admin-only panel to manage system accounts."""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from app.app_ui import _load_database


def render_users_page() -> None:
    Database = _load_database()
    db = Database()

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.header("User Management")
    st.caption("Create, view, and delete system user accounts.")
    st.markdown("</div>", unsafe_allow_html=True)

    current_user_id = st.session_state.get("user_id")

    users = db.list_users()
    if users:
        rows = [
            {
                "User ID": u.user_id,
                "Username": u.username,
                "Role": u.role,
                "Created": time.strftime("%Y-%m-%d %H:%M", time.localtime(u.created_at)) if u.created_at else "—",
            }
            for u in users
        ]
        st.subheader("Current users")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No users registered yet.")

    st.markdown("---")
    st.subheader("Create a new user")
    with st.form("create_user_form"):
        new_user = st.text_input("Username", placeholder="alice")
        new_pass = st.text_input("Password", type="password", placeholder="****")
        new_pass2 = st.text_input("Confirm password", type="password", placeholder="****")
        new_role = st.selectbox("Role", ["user", "admin"])
        create_btn = st.form_submit_button("Create user", use_container_width=True)
        if create_btn:
            if not new_user or not new_pass:
                st.error("Please fill in all fields.")
            elif new_pass != new_pass2:
                st.error("Passwords do not match.")
            elif len(new_pass) < 4:
                st.error("Password must be at least 4 characters.")
            else:
                created = db.create_user(new_user.strip(), new_pass, role=new_role)
                if created is None:
                    st.error("Username already exists.")
                else:
                    st.success(f"User '{created.username}' created with role '{created.role}'.")

    st.markdown("---")
    st.subheader("Delete a user")
    deletable = [u for u in users if u.user_id != current_user_id]
    if not deletable:
        st.info("No deletable users (you cannot remove your own account).")
    else:
        target = st.selectbox(
            "Select user to delete",
            options=[u.username for u in deletable],
        )
        if st.button("Delete user", use_container_width=True):
            if target:
                user_obj = next((u for u in deletable if u.username == target), None)
                if user_obj and db.delete_user(user_obj.user_id):
                    st.success(f"User '{target}' deleted.")
                else:
                    st.error(f"Failed to delete user '{target}'.")
