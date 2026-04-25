# test change

import streamlit as st
from pathlib import Path

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="StudyTrack AI | Intelligent Education Suite",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS INJECTION ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

css_path = Path(__file__).parent / "style.css"
if css_path.exists():
    local_css(css_path)

# --- AUTHENTICATION ---
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

VALID_USERNAME = "admin"
VALID_PASSWORD = "password123"

def login_ui():
    _, col, _ = st.columns([1, 2, 1])
    with col:
        # st.markdown('<div class="card-container animate-fade-in">', unsafe_allow_html=True)
        # st.markdown('<h1 class="gradient-text" style="text-align: center;">StudyTrack AI</h1>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card-container animate-fade-in" style="text-align:center; padding:10px;">
            <h2 class="gradient-text">StudyTrack AI</h2>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Please enter username")
            password = st.text_input("Password", type="password", placeholder="Please enter password")
            login_btn = st.form_submit_button("Sign In")

            if login_btn:
                if username == VALID_USERNAME and password == VALID_PASSWORD:
                    st.session_state["logged_in"] = True
                    st.success("Welcome back, Admin!")
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please try again.")
        st.markdown('</div>', unsafe_allow_html=True)

if not st.session_state["logged_in"]:
    login_ui()
    st.stop()

# --- SIDEBAR LOGOUT ---
with st.sidebar:
    st.markdown('<h2 class="gradient-text">StudyTrack AI</h2>', unsafe_allow_html=True)
    st.markdown("---")
    if st.button("Logout"):
        st.session_state["logged_in"] = False
        st.rerun()

# --- MAIN DASHBOARD ---
st.markdown('<h1 class="animate-fade-in">Welcome to <span class="gradient-text">StudyTrack AI</span></h1>', unsafe_allow_html=True)
st.markdown('<p class="animate-fade-in" style="font-size: 1.1rem; color: #94a3b8; margin-bottom: 2rem;">Your unified command center for academic excellence and behavioral insights.</p>', unsafe_allow_html=True)

# Grid Layout for Modules
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div class="card-container animate-fade-in">
            <h3 style="margin-top: 0;">📊 Performance Analytics</h3>
            <p style="color: #94a3b8;">Deep dive into student metrics, quiz scores, and department-wide performance trends.</p>
            <p style="font-size: 0.9rem; color: #6366f1;"><b>Modules:</b> Regression, Heatmaps, Summary Stats</p>
        </div>
        <div class="card-container animate-fade-in">
            <h3 style="margin-top: 0;">🧠 Clustering Engine</h3>
            <p style="color: #94a3b8;">Uncover hidden patterns in student behavior using advanced K-Means clustering.</p>
            <p style="font-size: 0.9rem; color: #6366f1;"><b>Modules:</b> Pattern Detection, Radar Charts, Profiles</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="card-container animate-fade-in">
            <h3 style="margin-top: 0;">✨ AI Recommendations</h3>
            <p style="color: #94a3b8;">Personalized study routines and habit improvements based on behavioral clusters.</p>
            <p style="font-size: 0.9rem; color: #6366f1;"><b>Modules:</b> Habit Optimization, Smart Scheduling</p>
        </div>
        <div class="card-container animate-fade-in">
            <h3 style="margin-top: 0;">⚙️ Admin & Tracker</h3>
            <p style="color: #94a3b8;">Manage datasets, monitor system health, and the log of the real-time study sessions.</p>
            <p style="font-size: 0.9rem; color: #6366f1;"><b>Modules:</b> Data Management, Log Integration</p>
        </div>
    """, unsafe_allow_html=True)

st.info("System operational. Select a module from the sidebar to begin.")
