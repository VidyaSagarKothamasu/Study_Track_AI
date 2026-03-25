import streamlit as st
import pandas as pd
import numpy as np
import os
import traceback
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Clustering & Pattern Detection | StudyTrack AI",
    page_icon="🧠",
    layout="wide"
)

# --- CSS INJECTION ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

css_path = Path(__file__).parent.parent / "style.css"
if css_path.exists():
    local_css(css_path)

# --- LOGIN CHECK ---
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if not st.session_state["logged_in"]:
    st.warning("🔒 Please login on the main page to access dashboards.")
    st.stop()

# --- LOAD DATASET ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Study Track AI Project.csv")
        if len(df) < 5 and os.path.exists("Students Performance Dataset.csv"):
            st.warning("⚠️ Using 'Students Performance Dataset.csv' for better pattern detection.")
            df = pd.read_csv("Students Performance Dataset.csv")
        df.columns = df.columns.str.strip()
        numeric_cols = df.select_dtypes(include=np.number).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        return df
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        st.code(traceback.format_exc())
        return pd.DataFrame()

df = load_data()
if df.empty:
    st.stop()

try:
    # --- HEADER ---
    st.markdown('<h1 class="animate-fade-in"><span class="gradient-text">🧠 Clustering & Pattern Detection</span></h1>', unsafe_allow_html=True)
    st.markdown('<p class="animate-fade-in" style="color: #94a3b8;">Identify student behavior types and derive actionable insights using machine learning.</p>', unsafe_allow_html=True)

    # --- PARAMETERS ---
    st.markdown('<div class="card-container animate-fade-in"><h3>⚙️ Clustering Controls</h3>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        n_clusters = st.slider("Number of Clusters", 2, 8, 4)
    with col2:
        retrain = st.button("🔄 Reload & Retrain", use_container_width=True)
    with col3:
        export = st.button("⬇ Export Clustered Data", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- PREPARE DATA ---
    features = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in ["Student_ID", "Cluster"]:
        if col in features: features.remove(col)

    if not features:
        st.error("❌ No numeric features found for clustering!")
        st.stop()

    scaler = StandardScaler()
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(df[features])
    X_scaled = scaler.fit_transform(X_imputed)

    kmeans = KMeans(n_clusters=min(n_clusters, len(X_imputed)), random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    # --- HANDLE RELOAD & RETRAIN (Older Streamlit Compatible) ---
    if retrain:
        st.cache_data.clear()
        st.session_state["reload_trigger"] = True
        st.info("✅ Data cleared. Please refresh the page to retrain.")

    # --- CLUSTER SELECTION ---
    st.markdown('<div class="card-container animate-fade-in" style="background: rgba(99, 102, 241, 0.05); border: 1px solid rgba(99, 102, 241, 0.2);"><h3>🎯 Cluster Selection</h3>', unsafe_allow_html=True)
    selected_cluster = st.selectbox("Select Cluster to Highlight & Inspect", sorted(df["Cluster"].unique()))
    st.markdown('</div>', unsafe_allow_html=True)

    # --- MAIN DASHBOARD ---
    left, right = st.columns([2,1])

    # 📊 SCATTER PLOT
    with left:
        st.markdown('<div class="card-container animate-fade-in"><h3>📊 Student Behavior Clusters</h3>', unsafe_allow_html=True)
        c_x, c_y = st.columns(2)
        x_axis = c_x.selectbox("X-Axis Feature", features, index=0)
        y_axis = c_y.selectbox("Y-Axis Feature", features, index=1 if len(features) > 1 else 0)

        clusters = sorted(df["Cluster"].unique())
        palette = px.colors.qualitative.Pastel
        color_map = {str(c): ("rgba(150,150,150,0.2)" if c != selected_cluster else palette[i % len(palette)])
                     for i, c in enumerate(clusters)}

        fig = px.scatter(
            df, x=x_axis, y=y_axis,
            color=df["Cluster"].astype(str),
            color_discrete_map=color_map,
            hover_data=df.columns,
            height=450,
            template="plotly_dark"
        )
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 📌 POPULATION DISTRIBUTION
    with right:
        st.markdown('<div class="card-container animate-fade-in"><h3>📌 Population Distribution</h3>', unsafe_allow_html=True)
        cluster_counts = df["Cluster"].value_counts().sort_index()
        total = len(df)
        for cluster_id, count in cluster_counts.items():
            percent = round((count/total)*100, 1)
            st.markdown(f"**Cluster {cluster_id}** ({count} Students)")
            st.progress(percent/100)
        st.markdown('</div>', unsafe_allow_html=True)

    # 📈 RADAR CHART
    st.markdown('<div class="card-container animate-fade-in"><h3>📈 Cluster Characteristics</h3>', unsafe_allow_html=True)
    cluster_means = df.groupby("Cluster")[features].mean()
    radar = go.Figure()
    for cluster_id in cluster_means.index:
        radar.add_trace(go.Scatterpolar(
            r=cluster_means.loc[cluster_id].values,
            theta=features,
            fill='toself',
            name=f"Cluster {cluster_id}",
            line=dict(width=4 if cluster_id == selected_cluster else 1),
            opacity=1.0 if cluster_id == selected_cluster else 0.4
        ))
    radar.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        height=550
    )
    st.plotly_chart(radar, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- LOWER SECTION ---
    low_left, low_right = st.columns([1, 1.2])

    # 🧾 CLUSTER METRICS PROFILE
    with low_left:
        st.markdown('<div class="card-container animate-fade-in"><h3>🧾 Cluster Metrics Profile</h3>', unsafe_allow_html=True)
        cluster_df = df[df["Cluster"] == selected_cluster]
        metrics = cluster_df[features].mean()
        cols = st.columns(2)
        for i, feature in enumerate(features):
            (cols[0] if i%2==0 else cols[1]).metric(feature, round(metrics[feature], 2))
        st.markdown('</div>', unsafe_allow_html=True)

    # ➕ ASSIGN NEW STUDENT
    with low_right:
        st.markdown('<div class="card-container animate-fade-in"><h3>➕ Assign New Student</h3>', unsafe_allow_html=True)
        with st.form("new_student_form"):
            new_values = []
            c1, c2 = st.columns(2)
            for i, feature in enumerate(features[:4]):
                val = (c1 if i%2==0 else c2).number_input(
                    feature,
                    float(df[feature].min()),
                    float(df[feature].max()),
                    float(df[feature].mean())
                )
                new_values.append(val)
            for feature in features[4:]:
                new_values.append(df[feature].mean())
            submitted = st.form_submit_button("Predict Cluster Group", use_container_width=True)
            if submitted:
                new_data_full = np.array([new_values])
                new_data_scaled = scaler.transform(imputer.transform(new_data_full))
                prediction = kmeans.predict(new_data_scaled)[0]
                st.success(f"Student assigned to: **Cluster {prediction}**")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- EXPORT ---
    if export:
        df.to_csv("clustered_students.csv", index=False)
        st.success("✅ Dataset exported to clustered_students.csv")

except Exception as e:
    st.error(f"❌ An error occurred: {e}")
    st.code(traceback.format_exc())