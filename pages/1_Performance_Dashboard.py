import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Performance Analytics | StudyTrack AI",
    page_icon="📊",
    layout="wide"
)

# --- CSS INJECTION ---
def local_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

css_path = Path(__file__).parent.parent / "style.css"
if css_path.exists():
    local_css(css_path)

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if not st.session_state["logged_in"]:
    st.warning("🔒 Please login on the main page to access dashboards.")
    st.stop()

# --- HEADER ---
st.markdown('<h1 class="animate-fade-in"><span class="gradient-text">📊 Student Performance Dashboard</span></h1>', unsafe_allow_html=True)
st.markdown('<p class="animate-fade-in" style="color: #94a3b8;">Analyze academic metrics, correlations, and predictive insights.</p>', unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Students Performance Dataset.csv")
    df.columns = df.columns.str.strip()
    if "Parent_Education_Level" in df.columns:
        df.drop(columns=["Parent_Education_Level"], inplace=True)
    return df

df = load_data()
# --- MAIN CONTENT ---
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown('<div class="card-container animate-fade-in"><h3>🔥 Correlation Heatmap</h3>', unsafe_allow_html=True)

    numeric_df = df.select_dtypes(include=["int64", "float64"])
    corr = numeric_df.corr()

    sns.set_style("dark")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1e293b')
    ax.set_facecolor('#1e293b')

    sns.heatmap(corr, annot=True, cmap='viridis', linewidths=0.5, fmt=".2f", ax=ax,
                annot_kws={"color": "white"})

    plt.xticks(color='white', rotation=45)
    plt.yticks(color='white', rotation=0)

    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)


with col_right:
    st.markdown('<div class="card-container animate-fade-in"><h3>📈 Study Time vs Quiz Scores</h3>', unsafe_allow_html=True)

    study = np.arange(1, 9)
    quiz = [65, 70, 75, 82, 85, 88, 90, 92]

    fig, ax = plt.subplots()
    fig.patch.set_facecolor('#1e293b')
    ax.set_facecolor('#1e293b')

    ax.plot(study, quiz, marker='o', color='#6366f1', linewidth=2)
    ax.set_xlabel("Study Time (Hours)", color='white')
    ax.set_ylabel("Quiz Score (%)", color='white')
    ax.tick_params(colors='white')

    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)


col3, col4 = st.columns([1, 1])

with col3:
    st.markdown('<div class="card-container animate-fade-in"><h3>🚫 Distraction Impact</h3>', unsafe_allow_html=True)

    distraction = ["Low", "Medium", "High"]
    performance = [88, 75, 65]

    fig, ax = plt.subplots()
    fig.patch.set_facecolor('#1e293b')
    ax.set_facecolor('#1e293b')

    ax.bar(distraction, performance, color=["#10b981", "#f59e0b", "#ef4444"])
    ax.set_ylabel("Average Quiz Score", color='white')
    ax.tick_params(colors='white')

    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)


with col4:
    st.markdown('<div class="card-container animate-fade-in"><h3>🕒 Study Pattern Distribution</h3>', unsafe_allow_html=True)

    labels = ["Morning", "Afternoon", "Evening", "Night"]
    sizes = [25, 30, 30, 15]

    fig, ax = plt.subplots()
    fig.patch.set_facecolor('#1e293b')
    ax.set_facecolor('#1e293b')

    colors = ['#6366f1', '#818cf8', '#a5b4fc', '#c7d2fe']
    ax.pie(sizes, labels=labels, autopct='%1.1f%%',
           wedgeprops=dict(width=0.4),
           colors=colors, textprops={'color': "white"})

    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

