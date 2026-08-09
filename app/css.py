"""Centralized styling for the Streamlit UI.

Light theme only — soft blues on white, rounded cards, green status highlights.
"""

from __future__ import annotations

import streamlit as st

CSS = """
<style>
/* ==========================================================================
   Light theme defaults
   ========================================================================== */
html, body, [class*="css"] {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
}
.stAppHeader { padding-top: 6px; padding-bottom: 6px; }
section.main > div[role="main"] > div:first-child { padding-top: 6px; }
.block-container { padding-top: 1rem; }

/* ==========================================================================
   Cards
   ========================================================================== */
.metric-card, .status-summary-card {
  background: #FFFFFF;
  border: 1px solid #E2E8F0;
  border-radius: 14px;
  padding: 12px;
  box-shadow: 0 4px 16px rgba(15, 23, 42, 0.06);
}
.status-summary-card {
  background: #FFFFFF;
  border: 1px solid #E2E8F0;
  color: #0F172A;
  box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
}
.status-summary-card .label { font-size: 12px; color: #0F172A; opacity: 0.72; }
.status-summary-card .value { font-size: 20px; font-weight: 700; color: #0F172A; }
.metric-card .label { font-size: 12px; color: #0F172A; opacity: 0.72; }
.metric-card .value { font-size: 20px; font-weight: 700; color: #0F172A; }
.hero-panel {
  background: linear-gradient(135deg, rgba(37, 99, 235, 0.10), rgba(16, 185, 129, 0.10), rgba(248, 250, 252, 0.6));
  border: 1px solid rgba(37, 99, 235, 0.20);
  border-radius: 20px;
  padding: 18px 20px;
  margin-bottom: 16px;
  box-shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
}
.hero-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 10px;
}
.hero-badges .status-badge {
  background: rgba(255, 255, 255, 0.85);
  backdrop-filter: blur(6px);
}
.section-card {
  background: #FFFFFF;
  border: 1px solid #E2E8F0;
  border-radius: 14px;
  padding: 12px 14px;
  margin-bottom: 12px;
  box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
}
.status-value-green {
  display: inline-block;
  font-size: 20px;
  font-weight: 700;
  color: #059669;
  background-color: #ECFDF5;
  border: 1px solid #A7F3D0;
  border-radius: 999px;
  padding: 2px 12px;
}

/* ==========================================================================
   Buttons
   ========================================================================== */
.stButton > button {
  border-radius: 999px;
  border: 1px solid rgba(37, 99, 235, 0.20);
  background: linear-gradient(135deg, #eff6ff, #ecfeff);
  color: #0f172a;
  font-weight: 600;
}
.stButton > button:hover {
  border-color: rgba(16, 185, 129, 0.35);
  transform: translateY(-1px);
}

/* ==========================================================================
   Status badges
   ========================================================================== */
.status-badge {
  display: inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  font-weight: 700;
  font-size: 12px;
  background: rgba(37, 99, 235, 0.10);
  color: #0284C7;
  border: 1px solid rgba(37, 99, 235, 0.18);
}
.status-online { color: #0f766e; }
.status-offline { color: #92400e; }
.status-green {
  color: #059669;
  background-color: #ECFDF5;
  border-color: #A7F3D0;
}
.status-blue {
  color: #0284C7;
  background-color: #E0F2FE;
  border-color: #BAE6FD;
}
.status-amber {
  color: #B45309;
  background-color: #FEF3C7;
  border-color: #FDE68A;
}

/* ==========================================================================
   Engine status header cards — green values
   ========================================================================== */
.engine-status-card {
  background: #FFFFFF;
  border: 1px solid #E2E8F0;
  border-radius: 14px;
  padding: 12px;
  box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
}
.engine-status-card .label {
  font-size: 12px;
  color: #0F172A;
  opacity: 0.72;
  margin-bottom: 4px;
}

/* ==========================================================================
   Chat bubbles
   ========================================================================== */
.chat-user, .chat-assistant {
  padding: 12px 14px;
  border-radius: 16px;
  margin: 8px 0;
  max-width: 80%;
  background: #FFFFFF;
  color: #0F172A;
  border: 1px solid #E2E8F0;
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.04);
  line-height: 1.45;
}
.chat-user {
  background: linear-gradient(135deg, rgba(37, 99, 235, 0.09), rgba(191, 219, 254, 0.12));
  border-left: 4px solid rgba(37, 99, 235, 0.55);
}
.chat-assistant {
  background: linear-gradient(135deg, rgba(16, 185, 129, 0.10), rgba(14, 116, 144, 0.08));
  border-color: rgba(16, 185, 129, 0.28);
  border-left: 4px solid rgba(16, 185, 129, 0.55);
}

/* ==========================================================================
   Source cards
   ========================================================================== */
.source-card {
  background: #F8FAFC;
  padding: 10px 12px;
  border-radius: 12px;
  margin-bottom: 8px;
  border: 1px solid #E2E8F0;
  color: #0F172A;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.7);
}
.source-pill {
  display: inline-block;
  padding: 4px 8px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 700;
  background: rgba(37, 99, 235, 0.10);
  color: #1d4ed8;
  margin-top: 6px;
}
.source-score { font-family: monospace; color: #0F172A; opacity: 0.72; }
.source-snippet {
  white-space: pre-wrap;
  margin-top: 8px;
  font-size: 13px;
  line-height: 1.5;
  max-height: 180px;
  overflow-y: auto;
}
.source-readmore {
  margin-top: 6px;
  font-size: 12px;
  font-weight: 600;
  color: #0284C7;
  cursor: pointer;
}

/* ==========================================================================
   Tabs
   ========================================================================== */
.stTabs [data-testid="stBaseButton-header"] {
  border-radius: 999px;
  padding: 8px 12px;
}
.stTabs [data-testid="stBaseButton-header"][aria-selected="true"] {
  background: linear-gradient(135deg, rgba(37, 99, 235, 0.16), rgba(16, 185, 129, 0.14));
  border: 1px solid rgba(37, 99, 235, 0.18);
}
.stChatInput > div {
  border-radius: 16px;
  border: 1px solid #E2E8F0;
  box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
}
.stDataFrame > div { background: transparent; }

/* ==========================================================================
   Password input — keep the visibility toggle clear of placeholder text
   ========================================================================== */
div[data-baseweb="input"] input[type="password"],
div[data-baseweb="input"] input[type="text"] {
  padding-right: 3.5rem !important;
}

/* Properly position and vertically center the eye toggle icon container */
div[data-baseweb="input"] > div:last-child {
  position: absolute !important;
  right: 0.75rem !important;
  top: 50% !important;
  transform: translateY(-50%) !important;
  margin: 0 !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  z-index: 5 !important;
}

/* Remove the distracting 'Press Enter to submit form' hint if it collides */
[data-testid="InputInstructions"] {
  display: none !important;
}

/* ==========================================================================
   Sidebar — soft blue accents, clean dividers, centered logo
   ========================================================================== */
[data-testid="stSidebar"] {
  background: #F8FAFC;
  border-right: 1px solid #E2E8F0;
}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
  background: #F8FAFC;
}
/* Hide the native Streamlit multi-page nav — navigation lives in the top bar */
[data-testid="stSidebarNav"] {
  display: none !important;
}
[data-testid="stSidebar"] hr {
  border-color: #E2E8F0;
  margin: 12px 0;
}
.sidebar-logo {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
  padding: 8px 0 14px;
  text-align: center;
}
.sidebar-logo .logo-badge {
  width: 56px;
  height: 56px;
  border-radius: 16px;
  background: linear-gradient(135deg, #2563eb, #0d9488);
  color: #FFFFFF;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 26px;
  font-weight: 800;
  box-shadow: 0 8px 20px rgba(37, 99, 235, 0.25);
}
.sidebar-logo .logo-title {
  font-size: 17px;
  font-weight: 800;
  color: #0F172A;
  letter-spacing: 0.2px;
}
.sidebar-logo .logo-sub {
  font-size: 11px;
  color: #0284C7;
  font-weight: 600;
}
.sidebar-section-title {
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.6px;
  color: #64748B;
  margin: 10px 0 6px;
}
.sidebar-card {
  background: #FFFFFF;
  border: 1px solid #E2E8F0;
  border-radius: 14px;
  padding: 12px;
  margin-bottom: 10px;
  box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
}
.sidebar-card .label { font-size: 12px; color: #0F172A; opacity: 0.72; }
.sidebar-card .value { font-size: 16px; font-weight: 700; color: #0F172A; }

/* System Health box — compact 0.85rem typography */
.sidebar-health {
  font-size: 0.85rem;
  line-height: 1.5;
}
.sidebar-health .row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
}
.sidebar-health .row:last-child { margin-bottom: 0; }
.sidebar-health .row span:first-child { color: #0F172A; opacity: 0.8; }
.sidebar-health .status-badge {
  font-size: 11px;
  padding: 2px 8px;
}
.sidebar-health .chunk-count {
  font-weight: 700;
  color: #0F172A;
  font-size: 0.85rem;
}
.profile-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: #FFFFFF;
  border: 1px solid #E2E8F0;
  border-radius: 999px;
  padding: 6px 12px;
  font-weight: 700;
  color: #0F172A;
  font-size: 13px;
}
.role-pill {
  display: inline-block;
  border-radius: 999px;
  padding: 2px 10px;
  font-size: 11px;
  font-weight: 700;
  color: #FFFFFF;
}
.role-pill-admin { background: #2563eb; }
.role-pill-user { background: #0d9488; }

/* ==========================================================================
   Top navigation bar — horizontal pill/tab bar in the main content area
   ========================================================================== */
.top-nav-bar {
  display: flex;
  justify-content: center;
  margin: 0 auto 16px;
  max-width: 900px;
}
.top-nav-bar [data-testid="stSegmentedControl"] {
  background: #FFFFFF;
  border: 1px solid #E2E8F0;
  border-radius: 999px;
  padding: 6px;
  box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
}
.top-nav-bar [data-testid="stSegmentedControl"] > div {
  gap: 4px;
}
.top-nav-bar [data-testid="stSegmentedControl"] button {
  border-radius: 999px;
  padding: 8px 20px;
  font-size: 14px;
  font-weight: 600;
  color: #475569;
  background: transparent;
  border: none;
  transition: all 0.15s ease;
  white-space: nowrap;
}
.top-nav-bar [data-testid="stSegmentedControl"] button:hover {
  color: #0284C7;
  background: #F0F9FF;
}
.top-nav-bar [data-testid="stSegmentedControl"] button[aria-checked="true"] {
  background: linear-gradient(135deg, #2563eb, #0d9488);
  color: #FFFFFF;
  box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
}
.top-nav-bar [data-testid="stSegmentedControl"] button[aria-checked="true"]:hover {
  color: #FFFFFF;
  background: linear-gradient(135deg, #2563eb, #0d9488);
}
.top-nav-bar [data-testid="stElementToolbar"] {
  display: none;
}
.top-nav-bar [data-testid="stCaptionContainer"] {
  display: none;
}

/* ==========================================================================
   Sidebar action buttons — Reset Chat & Sign Out
   ========================================================================== */
.sidebar-actions {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.sidebar-actions [data-testid="stButton"] > button {
  width: 100%;
  height: 38px;
  border-radius: 999px;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.15s ease;
}
/* Reset Chat — compact soft blue, warm amber on hover */
.reset-chat-btn [data-testid="stButton"] > button {
  background: linear-gradient(135deg, #eff6ff, #ecfeff);
  border: 1px solid rgba(37, 99, 235, 0.25);
  color: #1d4ed8;
}
.reset-chat-btn [data-testid="stButton"] > button:hover {
  background: linear-gradient(135deg, #fef3c7, #fff7ed);
  border-color: rgba(245, 158, 11, 0.45);
  color: #b45309;
}
/* Sign Out — compact outline, muted red */
.signout-btn [data-testid="stButton"] > button {
  background: #FFFFFF;
  border: 1px solid rgba(239, 68, 68, 0.45);
  color: #EF4444;
}
.signout-btn [data-testid="stButton"] > button:hover {
  background: #FEF2F2;
  border-color: #EF4444;
  color: #B91C1C;
}

/* ==========================================================================
   Floating scroll buttons — small & discreet, chat page only
   ========================================================================== */
#scroll-top-btn, #scroll-bottom-btn {
  position: fixed;
  right: 20px;
  width: 32px;
  height: 32px;
  border-radius: 50%;
  background: rgba(37, 99, 235, 0.92);
  color: #FFFFFF;
  border: none;
  box-shadow: 0 2px 10px rgba(37, 99, 235, 0.28);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  cursor: pointer;
  z-index: 9999;
  text-decoration: none;
  opacity: 0.85;
  transition: opacity 0.2s, transform 0.2s;
}
#scroll-bottom-btn { bottom: 20px; }
#scroll-top-btn { bottom: 64px; }
#scroll-top-btn:hover, #scroll-bottom-btn:hover {
  opacity: 1;
  transform: translateY(-2px);
}
#scroll-top-btn:visited, #scroll-bottom-btn:visited { color: #FFFFFF; }
#scroll-top-btn:active, #scroll-bottom-btn:active { color: #FFFFFF; }

/* ==========================================================================
   Login page
   ========================================================================== */
.login-hero {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
  padding: 24px 0 10px;
  text-align: center;
}
.login-logo-badge {
  width: 72px;
  height: 72px;
  border-radius: 20px;
  background: linear-gradient(135deg, #2563eb, #0d9488);
  color: #FFFFFF;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 32px;
  font-weight: 800;
  box-shadow: 0 12px 28px rgba(37, 99, 235, 0.28);
}
.login-title {
  font-size: 26px;
  font-weight: 800;
  color: #0F172A;
}
.login-sub {
  font-size: 13px;
  color: #64748B;
  max-width: 360px;
}

/* ==========================================================================
   System Parameters — compact cards, constrained slider width
   ========================================================================== */
.page-title {
  font-size: 20px;
  font-weight: 800;
  color: #0F172A;
  margin: 0 0 4px;
}
.param-card {
  background: #FFFFFF;
  border: 1px solid #E2E8F0;
  border-radius: 14px;
  padding: 16px;
  box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
  margin-bottom: 12px;
}
.param-card .param-title {
  font-size: 15px;
  font-weight: 700;
  color: #0F172A;
  margin-bottom: 12px;
}
.params-layout {
  max-width: 860px;
}
.params-layout [data-testid="stSlider"] {
  max-width: 700px;
}

/* ==========================================================================
   Indexed Files — structured catalog table with green status badges
   ========================================================================== */
.file-table {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  background: #FFFFFF;
  border: 1px solid #E2E8F0;
  border-radius: 14px;
  overflow: hidden;
  box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
}
.file-table th {
  text-align: left;
  padding: 10px 14px;
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: #64748B;
  background: #F8FAFC;
  border-bottom: 1px solid #E2E8F0;
}
.file-table td {
  padding: 10px 14px;
  font-size: 13px;
  color: #0F172A;
  border-bottom: 1px solid #E2E8F0;
  vertical-align: middle;
}
.file-table tr:last-child td { border-bottom: none; }
.file-table .file-type {
  color: #64748B;
  font-weight: 600;
}
.file-table .chunk-count {
  font-weight: 700;
  color: #0F172A;
}
.file-table-th {
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: #64748B;
  padding: 4px 0;
  border-bottom: 1px solid #E2E8F0;
  margin-bottom: 6px;
}
</style>
"""


def inject_styles() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hide_sidebar_style() -> str:
    """CSS to fully hide the sidebar — used on the login/auth view."""
    return (
        "<style>"
        "[data-testid='stSidebar'] { display: none !important; }"
        "</style>"
    )


def sidebar_logo_html() -> str:
    """Sidebar logo block — clean and centered at the top."""
    return (
        "<div class='sidebar-logo'>"
        "<div class='logo-badge'>R</div>"
        "<div class='logo-title'>RAG Souverain</div>"
        "<div class='logo-sub'>Local & Private</div>"
        "</div>"
    )


def login_logo_html() -> str:
    """Login page logo block — centered above the login card."""
    return (
        "<div class='login-hero'>"
        "<div class='login-logo-badge'>R</div>"
        "<div class='login-title'>RAG Souverain</div>"
        "<div class='login-sub'>Private document intelligence, fully on-premise.</div>"
        "</div>"
    )
