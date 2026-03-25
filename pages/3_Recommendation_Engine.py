import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AI Recommendations | StudyTrack AI",
    page_icon="✨",
    layout="wide"
)

# --- CSS INJECTION ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

css_path = Path(__file__).parent.parent / "style.css"
if css_path.exists():
    local_css(css_path)

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if not st.session_state["logged_in"]:
    st.warning("🔒 Please login on the main page to access dashboards.")
    st.stop()

# --- LOAD DATASET ---
@st.cache_data
def load_data():
    df = pd.read_csv("Students Performance Dataset.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# --- CLUSTERING ---
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if "Student_ID" in numeric_cols:
    numeric_cols.remove("Student_ID")

scaler = StandardScaler()
# ROBUST DATA CLEANING
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(df[numeric_cols])

if len(X_imputed) == 0:
    st.error("❌ The dataset is empty. Please upload a structured CSV.")
    st.stop()

X_scaled = scaler.fit_transform(X_imputed)

# DYNAMIC CLUSTER COUNT
n_samples = len(X_scaled)
current_n_clusters = min(4, n_samples)

if n_samples < 2:
    st.warning("⚠️ Need at least 2 students to generate recommendations.")
    st.stop()

kmeans = KMeans(n_clusters=current_n_clusters, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

behavior_map = {
    0: "Distracted learners",
    1: "Visual learners",
    2: "Low focus",
    3: "High focus"
}
df["Behavior_Type"] = df["Cluster"].map(behavior_map)
# --- HEADER ---
st.markdown('<h1 class="animate-fade-in"><span class="gradient-text">✨ AI Recommendation Engine</span></h1>', unsafe_allow_html=True)
st.markdown('<p class="animate-fade-in" style="color: #94a3b8;">Personalized study routines, habit optimization, and performance trajectories.</p>', unsafe_allow_html=True)

# --- STUDENT SELECTION ---
st.markdown('<div class="card-container animate-fade-in"><h3>👤 Student Profile</h3>', unsafe_allow_html=True)
student_id = st.selectbox("Select Student Profile", df["Student_ID"].unique())
student = df[df["Student_ID"] == student_id].iloc[0]
cluster = student["Cluster"]
behavior = student["Behavior_Type"]

c_info1, c_info2 = st.columns(2)
c_info1.markdown(f"**Selected Student belongs to :** <span style='color: #c084fc; font-weight: bold;'>Cluster {cluster}</span>", unsafe_allow_html=True)
c_info2.markdown(f"**Behavior Type :** <span style='color: #6366f1; font-weight: bold;'>{behavior}</span>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- MAIN LAYOUT ---
l_col, r_col = st.columns([1.5, 1])

# LEFT SIDE
with l_col:
    # 📆 STRATEGY
    st.markdown('<div class="card-container animate-fade-in"><h3>📆 Your Weekly Study Strategy</h3>', unsafe_allow_html=True)
    
    cluster_profile = df.groupby("Cluster")[numeric_cols].mean()
    cluster_mean = cluster_profile.loc[cluster].mean()
    overall_mean = df[numeric_cols].mean().mean()

    if cluster_mean >= overall_mean:
        optimal_times = ["08:00 AM - 10:00 AM", "02:00 PM - 03:30 PM", "07:00 PM - 08:30 PM"]
        duration = "90-minute Deep Work sessions"
        break_schedule = "Pomodoro: 25m work / 5m break"
    else:
        optimal_times = ["04:00 PM - 05:30 PM", "06:00 PM - 08:00 PM", "09:00 PM - 10:00 PM"]
        duration = "60-minute Focus sessions"
        break_schedule = "Extended: 30m work / 10m break"

    effectiveness = int((cluster_mean / overall_mean) * 80)
    effectiveness = min(max(effectiveness, 60), 95)

    t1, t2, t3 = st.columns(3)
    t1.info(optimal_times[0])
    t2.info(optimal_times[1])
    t3.info(optimal_times[2])

    st.markdown(f"""
    **Session Duration:** {duration}  
    **Break Pattern:** {break_schedule}
    """)

    st.progress(effectiveness / 100)
    st.write(f"Confidence Level: **{effectiveness}%**")

    st.markdown('</div>', unsafe_allow_html=True)

    # 🎯 SUBJECTS (UNCHANGED LOGIC)
    st.markdown('<div class="card-container animate-fade-in"><h3>🎯 Priority Focus Areas</h3>', unsafe_allow_html=True)
    sub_cols = [col for col in df.columns if "Score" in col and col != "Quiz_Score"]
    low_subs = student[sub_cols][student[sub_cols] < 70]

    if len(low_subs) > 0:
        for sub in low_subs.index:
            st.warning(f"⚠️ Improve performance in {sub}")
    else:
        st.success("✅ Good performance across all subjects!")

    st.markdown('</div>', unsafe_allow_html=True)

# RIGHT SIDE
with r_col:
    # 🛠 TOOLS
    st.markdown('<div class="card-container animate-fade-in"><h3>🛠️ Recommended Tools</h3>', unsafe_allow_html=True)
    tool_map = {
        "Distracted learners": "Site Blocker Pro",
        "Visual learners": "Digital Canvas",
        "Low focus": "Pomodoro Timer",
        "High focus": "Neuro-Music Player"
    }

    st.markdown(f"""
        <div style="background: rgba(99, 102, 241, 0.1); border: 1px solid #6366f1; padding: 1rem; border-radius: 0.5rem; text-align: center;">
            <h4 style="margin:0; color: #6366f1;">{tool_map[behavior]}</h4>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # 📈 WEEKLY GRAPH (NOT MODIFIED)
    st.markdown('<div class="card-container animate-fade-in"><h3>📈 Weekly Study Schedule</h3>', unsafe_allow_html=True)
    weekly_cols = numeric_cols[:7]
    if len(weekly_cols) >= 7:
        weekly_data = student[weekly_cols].values[:7]
        fig_bar = px.bar(
            x=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            y=weekly_data,
            template="plotly_dark",
            color_discrete_sequence=["#c084fc"]
        )
        fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=10,b=10,l=10,r=10), height=250)
        st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# 📈 EXPECTED IMPROVEMENT (FIXED)
st.markdown('<div class="card-container animate-fade-in"><h3>📈 Expected Performance Improvement</h3>', unsafe_allow_html=True)

current_mock = student.get("Midterm_Score", 70)
predicted_score = df[df["Cluster"] == cluster]["Final_Score"].mean()

# Progress graph (unchanged)
prog_labels = ["Current (Midterm)", "Week 1", "Week 2", "Week 3", "Predicted Final"]
prog_scores = np.linspace(current_mock, predicted_score, 5)

fig_line = go.Figure(go.Scatter(
    x=prog_labels,
    y=prog_scores,
    mode="lines+markers",
    line=dict(color="#6366f1", width=3)
))
fig_line.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    yaxis=dict(range=[0, 100])
)

st.plotly_chart(fig_line, use_container_width=True)

# =========================
# 🔥 FIXED IMPROVEMENT LOGIC
# =========================

raw_improvement = ((predicted_score - current_mock) / current_mock) * 100

# 🚨 Prevent negative values
improvement = max(0, round(raw_improvement, 1))

# Clean delta display
delta_text = f"+{improvement}%" if improvement > 0 else "0%"

st.metric(
    "Estimated Improvement",
    f"{improvement}%",
    delta=delta_text
)

st.markdown('</div>', unsafe_allow_html=True)
import streamlit as st

# =========================
# 🤖 AI STUDY CHATBOT
# =========================

# Header with animation (HTML card)
st.markdown(
    '<div class="card-container animate-fade-in"><h3>🤖 AI Study Assistant</h3></div>', 
    unsafe_allow_html=True
)

# Display example questions
st.markdown(
    "**Example questions:** study plan, schedule, focus, concentrate, improve score, "
    "time management, motivation, lazy, remember, memory, exam preparation, stress, "
    "break, productive, night study, morning study, weak subject, revision, procrastination, health"
)

# Input box for user
prompt = st.text_input("Ask study-related questions...")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show last 5 chat messages
for msg in st.session_state.messages[-5:]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chatbot response function
def chatbot_response(question):
    q = question.lower()
    if "study plan" in q or "schedule" in q:
        return (
            "📌 **Create a structured study plan:**\n"
            "• Study 60–90 minutes\n"
            "• Take short breaks\n"
            "• Revise daily"
        )
    elif "focus" in q or "concentrate" in q:
        return (
            "📌 **Improve focus:**\n"
            "• Pomodoro technique\n"
            "• Remove distractions\n"
            "• Study in a quiet place"
        )
    elif "memory" in q or "remember" in q:
        return (
            "📌 **Improve memory:**\n"
            "• Flashcards\n"
            "• Active recall\n"
            "• Spaced repetition"
        )
    elif "exam" in q or "preparation" in q:
        return (
            "📌 **Exam preparation tips:**\n"
            "• Revise concepts\n"
            "• Solve previous papers\n"
            "• Take mock tests"
        )
    else:
        return (
            "You can ask about:\n"
            "• Study plan / Schedule\n"
            "• Focus / Concentration\n"
            "• Memory / Remembering\n"
            "• Exam preparation"
        )

# Handle user query
if prompt:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Get chatbot response
    response = chatbot_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)

# Feedback & Countdown
c_foot1, c_foot2 = st.columns(2)

with c_foot1:
    st.markdown('<div class="card-container animate-fade-in"><h3>🎯 Exam Countdown</h3>', unsafe_allow_html=True)

    exam_date = st.date_input("Exam Date")
    days = (exam_date - datetime.today().date()).days

    if days > 0:
        st.info(f"{days} days remaining")
    else:
        st.warning("Date passed")

    st.markdown('</div>', unsafe_allow_html=True)

# with c_foot2:
#     st.markdown('<div class="card-container animate-fade-in">', unsafe_allow_html=True)
#     st.subheader("💬 Routine Feedback")
#     st.radio("Is this working?", ["Yes", "Needs adjustment"], horizontal=True)
#     st.markdown('</div>', unsafe_allow_html=True)