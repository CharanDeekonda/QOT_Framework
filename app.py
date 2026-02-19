import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import time
import networkx as nx
import pydeck as pdk
import pyrebase
import extra_streamlit_components as stx

# Page configuration
st.set_page_config(page_title="QOT Framework", page_icon="📡", layout="wide", initial_sidebar_state="expanded")

# Firebase configuration
firebaseConfig = st.secrets["firebase"]

# Initialize Firebase
firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()

# ==========================================
# 🔒 COOKIE + SESSION AUTH (COMPATIBLE FIX)
# ==========================================
cookie_manager = stx.CookieManager(key="qot_cookie_manager")

# 1. Start session state initialization
if "user" not in st.session_state:
    st.session_state.user = None
if "login_error" not in st.session_state:
    st.session_state.login_error = None
if "logged_out" not in st.session_state:
    st.session_state.logged_out = False

# 2. SYNC DELAY: Give the browser a moment to return the cookie
time.sleep(0.1) 
stored_token = cookie_manager.get("qot_auth_token")

# 3. RELOAD LOGIC: Catch the cookie and force a rerun if needed
if stored_token and st.session_state.user is None and not st.session_state.logged_out:
    st.session_state.user = {"idToken": stored_token}
    st.rerun() # Forces the dashboard to show immediately on refresh
# ==========================================

# ==========================================


# Custom CSS
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem !important; padding-bottom: 1rem !important; margin-top: 0rem !important; }
    header { visibility: hidden; }
    div[data-testid="stDeckGlJsonChart"] { border: 1px solid rgba(128, 128, 128, 0.5); border-radius: 8px; padding: 5px; background-color: #0e1117; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; height: 3em; background-color: #ff4b4b; color: white; border: none; }
    .login-card { background-color: var(--secondary-background-color); padding: 2.5rem; border-radius: 1rem; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); text-align: center; margin-bottom: 1.5rem; }
    .login-title { color: var(--text-color); font-family: 'Source Sans Pro', sans-serif; font-weight: 700; margin-bottom: 0.5rem; font-size: 1.8rem; }
    .login-subtitle { color: rgba(var(--text-color-rgb), 0.7); font-size: 1rem; margin-bottom: 2rem; }
    div[data-baseweb="input"] { background-color: var(--primary-background-color) !important; border: 1px solid rgba(var(--text-color-rgb), 0.2) !important; }
</style>
""", unsafe_allow_html=True)

# Load AI Model
MODEL_PATH = "models/saved_models/qot_model.keras"
MAX_PATH_LENGTH = 42

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Datasets
US_CITIES = {
    "New York": [-74.0060, 40.7128], "Washington DC": [-77.0369, 38.9072], "Chicago": [-87.6298, 41.8781],
    "Atlanta": [-84.3880, 33.7490], "Miami": [-80.1918, 25.7617], "Dallas": [-96.7970, 32.7767],
    "Denver": [-104.9903, 39.7392], "Seattle": [-122.3321, 47.6062], "San Francisco": [-122.4194, 37.7749],
    "Los Angeles": [-118.2437, 34.0522], "Boston": [-71.0589, 42.3601], "Houston": [-95.3698, 29.7604],
    "Phoenix": [-112.0740, 33.4484], "Philadelphia": [-75.1652, 39.9526], "Detroit": [-83.0458, 42.3314],
    "Minneapolis": [-93.2650, 44.9778], "St. Louis": [-90.1994, 38.6270], "Las Vegas": [-115.1398, 36.1699],
    "Montreal": [-73.5673, 45.5017]
}

US_CONNECTIONS = [
    ("New York", "Washington DC", 0), ("Washington DC", "Atlanta", 1), ("Atlanta", "Miami", 2),
    ("Atlanta", "Dallas", 3), ("Dallas", "Los Angeles", 4), ("Los Angeles", "San Francisco", 5),
    ("San Francisco", "Seattle", 6), ("Seattle", "Chicago", 7), ("Chicago", "New York", 8),
    ("Chicago", "Denver", 9), ("Denver", "San Francisco", 10), ("New York", "Boston", 14),
    ("Houston", "Dallas", 13), ("Miami", "Houston", 16), ("Detroit", "Chicago", 17),
    ("Minneapolis", "Chicago", 18), ("St. Louis", "Chicago", 19), ("St. Louis", "Dallas", 20),
    ("Phoenix", "Los Angeles", 21), ("Phoenix", "Dallas", 22), ("Las Vegas", "Los Angeles", 23),
    ("Las Vegas", "Denver", 24), ("Philadelphia", "New York", 25), ("Philadelphia", "Washington DC", 26),
    ("Seattle", "Denver", 27), ("Boston", "Montreal", 28), ("Detroit", "New York", 29),
    ("Atlanta", "Houston", 30), ("Denver", "Dallas", 31)
]

TG_CITIES = {
    "Hyderabad": [78.4867, 17.3850], "Warangal": [79.5941, 17.9689], "Karimnagar": [79.1288, 18.4386],
    "Nizamabad": [78.0941, 18.6725], "Khammam": [80.1514, 17.2473], "Mahbubnagar": [78.0035, 16.7488],
    "Nalgonda": [79.2684, 17.0577], "Adilabad": [78.5320, 19.6641], "Ramagundam": [79.4750, 18.7617],
    "Suryapet": [79.6239, 17.1439]
}

TG_CONNECTIONS = [
    ("Hyderabad", "Warangal", 0), ("Hyderabad", "Nalgonda", 1), ("Hyderabad", "Mahbubnagar", 2),
    ("Hyderabad", "Nizamabad", 8), ("Hyderabad", "Karimnagar", 17), ("Warangal", "Khammam", 3),
    ("Khammam", "Suryapet", 16), ("Suryapet", "Nalgonda", 30), ("Nizamabad", "Adilabad", 7),
    ("Karimnagar", "Ramagundam", 18), ("Ramagundam", "Adilabad", 9), ("Warangal", "Karimnagar", 25)
]

# Graph Utilities
def get_graph(mode):
    G, coords, conns = nx.Graph(), US_CITIES if mode == "US" else TG_CITIES, US_CONNECTIONS if mode == "US" else TG_CONNECTIONS
    for city in coords: G.add_node(city, pos=coords[city])
    for u, v, link_id in conns:
        if u in coords and v in coords: G.add_edge(u, v, id=link_id)
    return G, coords, conns

def get_link_id(G, u, v):
    return G[u][v]['id'] if G.has_edge(u, v) else 0

def recommend_modulation(gsnr_score):
    if gsnr_score >= 22.0: return "64QAM", "🟢 Excellent"
    elif gsnr_score >= 18.0: return "16QAM", "🔵 Good"
    elif gsnr_score >= 11.0: return "8QAM", "🟠 Moderate"
    elif gsnr_score >= 7.0:  return "QPSK", "🔴 Poor"
    else: return "BPSK", "⚫ Critical"

# Map Renderer
def render_map(coords, conns, mode):
    lines_data = [{"source": coords[u], "target": coords[v], "name": f"{u} ➝ {v}", "loss": "0.22 dB/km", "type": "SMF-28"} for u, v, _ in conns if u in coords and v in coords]
    nodes_data = [{"name": city, "coordinates": [lon, lat]} for city, (lon, lat) in coords.items()]
    lat, lon, zoom = (39.8283, -98.5795, 3) if mode == "US" else (17.8000, 79.0000, 6.5)

    return pdk.Deck(
        layers=[
            pdk.Layer("LineLayer", lines_data, get_source_position="source", get_target_position="target", get_color=[0, 255, 100], get_width=3, pickable=True),
            pdk.Layer("ScatterplotLayer", nodes_data, get_position="coordinates", get_color=[255, 50, 50], get_radius=20000 if mode == "US" else 5000, pickable=True),
            pdk.Layer("TextLayer", nodes_data, get_position="coordinates", get_text="name", get_color=[0, 255, 255], get_size=18, get_alignment_baseline="'bottom'")
        ],
        initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=zoom, pitch=30),
        map_style="mapbox://styles/mapbox/dark-v10",
        tooltip={"html": "<b>Route:</b> {name}<br/><b>Type:</b> {type}<br/><b>Loss:</b> {loss}", "style": {"backgroundColor": "#0f172a", "color": "white", "border": "1px solid #00ff62"}},
        height=400
    )

def process_login():
    email = st.session_state.email_input
    password = st.session_state.password_input

    if email and password:
        try:
            user = auth.sign_in_with_email_and_password(email, password)

            cookie_manager.set(
                "qot_auth_token",
                user["idToken"],
                max_age=86400
            )

            # ⭐ STEP 3: Reset the flag on successful login
            st.session_state.user = {"idToken": user["idToken"]}
            st.session_state.logged_out = False
            st.session_state.login_error = None

            st.rerun()

        except Exception:
            st.session_state.login_error = "❌ Authentication Failed"

# UI Layout: Login Screen
if not st.session_state.get("user"):
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("<br><br><br><div class='login-card'><h2 class='login-title'>🔐 Security Gateway</h2><p class='login-subtitle'>Authenticate to access the QOT Framework.</p></div>", unsafe_allow_html=True)
        with st.form("login_form", clear_on_submit=True):
            st.text_input("Operator Email", placeholder="admin@qot.com", label_visibility="collapsed", key="email_input")
            st.text_input("Authorization Key", type="password", placeholder="••••••••", label_visibility="collapsed", key="password_input")
            st.write("")
            st.form_submit_button("Authenticate", use_container_width=True, on_click=process_login)

        if st.session_state.login_error:
            st.error(st.session_state.login_error)

# UI Layout: Main Dashboard
else:
    with st.sidebar:
        with st.sidebar:
            if st.button("Logout", use_container_width=True):
                try:
                    cookie_manager.delete("qot_auth_token")
                except Exception:
                    pass # Ignore if already deleted
                st.session_state.user = None
                st.session_state.login_error = None
                st.session_state.logged_out = True
                st.rerun()
        st.markdown("---")
        st.markdown("## ⚙️ Configuration")
        network_mode = st.radio("Active Infrastructure:", ["🇺🇸 US National Backbone", "🇮🇳 Telangana Regional"], index=0)
        mode_key = "US" if "US" in network_mode else "TG"

        st.markdown("---")
        st.success("✅ AI Engine: Online")
        st.info(f"📍 Region: {mode_key}")

    st.title("📡 QOT Framework")
    st.markdown(f"#### Intelligent Lightpath Provisioning System ({mode_key} Region)")

    G, current_coords, current_conns = get_graph(mode_key)
    st.pydeck_chart(render_map(current_coords, current_conns, mode_key))

    st.markdown("### 🛠️ Connection Request")
    c1, c2, c3 = st.columns([1, 1, 1])
    cities = list(current_coords.keys())

    with c1: source = st.selectbox("Source Node", cities, index=6 if mode_key == "US" else 0)
    with c2: target = st.selectbox("Destination Node", [c for c in cities if c != source], index=3 if mode_key == "US" else 1)
    with c3:
        st.write("")
        st.write("")
        analyze = st.button("🚀 Analyze Route", type="primary")

    if analyze:
        st.markdown("---")
        
        # 1. SHOW HEADER & TABLE FIRST
        st.markdown(f"### 🔎 Path Analysis: {source} ➡ {target}")
        st.markdown("#### 📊 Comparative Route Analysis")
        
        # We need to calculate the paths first to show the table
        try:
            paths = list(nx.shortest_simple_paths(G, source, target))[:5]
            
            # Create a placeholder for the table so we can show it immediately
            table_placeholder = st.empty()
            
            # 2. RUN THE AI LOADING BELOW THE TABLE
            with st.spinner("Processing network graph & applying embeddings..."):
                # We do the heavy lifting (model prediction) inside the spinner
                results = []
                for i, p in enumerate(paths):
                    link_ids = [get_link_id(G, p[j], p[j+1]) for j in range(len(p)-1)]
                    padded = pad_sequences([link_ids], maxlen=MAX_PATH_LENGTH, padding='post', value=0)
                    
                    # AI Model Prediction
                    gsnr = model.predict(padded, verbose=0)[0][0]
                    if mode_key == "TG": 
                        gsnr = max(18.0, gsnr - (len(p)*0.5))
                    
                    rec, status = recommend_modulation(gsnr)
                    results.append({
                        "Path ID": f"Route #{i+1}", 
                        "Hops": len(link_ids), 
                        "Path Details": " ➔ ".join(p), 
                        "GSNR (dB)": gsnr, 
                        "Modulation": rec, 
                        "Status": status
                    })
                
                # Small artificial delay to make the spinner visible as requested
                time.sleep(1.0) 
                
                df = pd.DataFrame(results)
                best = df.loc[df['GSNR (dB)'].idxmax()]

            # 3. DISPLAY THE FINAL DATA
            # Update the table placeholder with the final results
            display_df = df.copy()
            display_df['GSNR (dB)'] = display_df['GSNR (dB)'].map('{:.2f}'.format)
            table_placeholder.dataframe(
                display_df, 
                column_config={
                    "Path ID": st.column_config.TextColumn("Route", width="small"), 
                    "Path Details": st.column_config.TextColumn("Trajectory", width="large"), 
                    "GSNR (dB)": st.column_config.TextColumn("Signal (dB)", width="small"), 
                    "Status": st.column_config.TextColumn("Quality", width="medium")
                }, 
                hide_index=True, 
                use_container_width=True
            )

            # 4. SHOW RECOMMENDATION AT THE VERY BOTTOM
            st.success(f"💡 AI Recommendation: {best['Path ID']} ({best['Modulation']})")
            m1, m2, m3 = st.columns(3)
            with m1: st.metric(label="Modulation Format", value=best['Modulation'])
            with m2: st.metric(label="Predicted GSNR", value=f"{best['GSNR (dB)']:.2f} dB")
            with m3: st.metric(label="Signal Status", value=best['Status'].split()[1])

        except nx.NetworkXNoPath:
            st.error("❌ No physical path exists between these nodes.")

    st.markdown("---")
    with st.expander("📖 Technical Legend & Glossary", expanded=True):
        st.table(pd.DataFrame({
            "Modulation (Short)": ["64QAM", "16QAM", "QPSK", "BPSK"],
            "Full Name": ["64-ary QAM", "16-ary QAM", "QPSK", "BPSK"],
            "Signal Quality (GSNR)": ["> 22 dB (Excellent)", "> 18 dB (Good)", "> 7 dB (Poor)", "< 7 dB (Critical)"],
            "Characteristics": ["Highest speed, Low noise tolerance", "Balanced speed and robustness", "Low speed, Very robust", "Emergency fallback only"]
        }))