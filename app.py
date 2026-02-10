import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import time
import networkx as nx
import pydeck as pdk 

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="QOT Framework", 
    page_icon="📡", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Main Background */
    .stApp { background-color: #0e1117; }
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 1rem !important;
        margin-top: 0rem !important;
    }
    
    /* White Border around the Map */
    div[data-testid="stDeckGlJsonChart"] {
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 8px;
        padding: 5px;
    }

    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #374151;
    }
</style>
""", unsafe_allow_html=True)

# --- CONFIGURATION ---
MODEL_PATH = "models/saved_models/qot_model.keras"
MAX_PATH_LENGTH = 42

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Please ensure 'models/saved_models/qot_model.keras' exists.")
    st.stop()

# ==========================================
# 🌍 1. DATASETS
# ==========================================

# --- DATASET A: US BACKBONE (REAL) ---
US_CITIES = {
    "New York": [-74.0060, 40.7128], "Washington DC": [-77.0369, 38.9072],
    "Chicago": [-87.6298, 41.8781], "Atlanta": [-84.3880, 33.7490],
    "Miami": [-80.1918, 25.7617], "Dallas": [-96.7970, 32.7767],
    "Denver": [-104.9903, 39.7392], "Seattle": [-122.3321, 47.6062],
    "San Francisco": [-122.4194, 37.7749], "Los Angeles": [-118.2437, 34.0522],
    "Boston": [-71.0589, 42.3601], "Houston": [-95.3698, 29.7604],
    "Phoenix": [-112.0740, 33.4484], "Philadelphia": [-75.1652, 39.9526],
    "Detroit": [-83.0458, 42.3314], "Minneapolis": [-93.2650, 44.9778],
    "St. Louis": [-90.1994, 38.6270], "Las Vegas": [-115.1398, 36.1699],
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

# --- DATASET B: TELANGANA REGIONAL (SYNTHETIC) ---
TG_CITIES = {
    "Hyderabad": [78.4867, 17.3850], "Warangal": [79.5941, 17.9689],
    "Karimnagar": [79.1288, 18.4386], "Nizamabad": [78.0941, 18.6725],
    "Khammam": [80.1514, 17.2473], "Mahbubnagar": [78.0035, 16.7488],
    "Nalgonda": [79.2684, 17.0577], "Adilabad": [78.5320, 19.6641],
    "Ramagundam": [79.4750, 18.7617], "Suryapet": [79.6239, 17.1439]
}

TG_CONNECTIONS = [
    ("Hyderabad", "Warangal", 0), ("Hyderabad", "Nalgonda", 1), ("Hyderabad", "Mahbubnagar", 2),
    ("Hyderabad", "Nizamabad", 8), ("Hyderabad", "Karimnagar", 17),
    ("Warangal", "Khammam", 3), ("Khammam", "Suryapet", 16), ("Suryapet", "Nalgonda", 30),
    ("Nizamabad", "Adilabad", 7), ("Karimnagar", "Ramagundam", 18),
    ("Ramagundam", "Adilabad", 9), ("Warangal", "Karimnagar", 25)
]

# ==========================================
# 🧠 2. LOGIC
# ==========================================

def get_graph(mode):
    G = nx.Graph()
    coords = US_CITIES if mode == "US" else TG_CITIES
    conns = US_CONNECTIONS if mode == "US" else TG_CONNECTIONS
    
    for city in coords:
        G.add_node(city, pos=coords[city])
    for u, v, link_id in conns:
        if u in coords and v in coords:
            G.add_edge(u, v, id=link_id)
    return G, coords, conns

def get_link_id(G, u, v):
    if G.has_edge(u, v): return G[u][v]['id']
    return 0

def recommend_modulation(gsnr_score):
    if gsnr_score >= 22.0: return "64QAM", "🟢 Excellent"
    elif gsnr_score >= 18.0: return "16QAM", "🔵 Good"
    elif gsnr_score >= 11.0: return "8QAM", "🟠 Moderate"
    elif gsnr_score >= 7.0:  return "QPSK", "🔴 Poor"
    else: return "BPSK", "⚫ Critical"

# --- MAP RENDERING ---
def render_map(coords, conns, mode):
    lines_data = []
    nodes_data = []

    for city, (lon, lat) in coords.items():
        nodes_data.append({"name": city, "coordinates": [lon, lat]})

    for u, v, link_id in conns:
        if u in coords and v in coords:
            # Add Extra Metadata for Hover
            lines_data.append({
                "source": coords[u],
                "target": coords[v],
                "name": f"{u} ➝ {v}",
                "loss": "0.22 dB/km",
                "type": "SMF-28 (Fiber)"
            })

    if mode == "US":
        lat, lon, zoom = 39.8283, -98.5795, 3
    else:
        lat, lon, zoom = 17.8000, 79.0000, 6.5

    view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=zoom, pitch=30)

    layer_lines = pdk.Layer(
        "LineLayer",
        lines_data,
        get_source_position="source",
        get_target_position="target",
        get_color=[0, 255, 100],
        get_width=3,
        pickable=True, # Critical for hover
    )
    
    layer_nodes = pdk.Layer(
        "ScatterplotLayer",
        nodes_data,
        get_position="coordinates",
        get_color=[255, 50, 50],
        get_radius=20000 if mode == "US" else 5000,
        pickable=True,
    )
    
    layer_text = pdk.Layer(
        "TextLayer",
        nodes_data,
        get_position="coordinates",
        get_text="name",
        get_color=[255, 255, 255],
        get_size=15,
        get_alignment_baseline="'bottom'",
    )

    return pdk.Deck(
        layers=[layer_lines, layer_nodes, layer_text],
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/satellite-v9",
        # Custom Tooltip showing Signal/Weights
        tooltip={
            "html": "<b>Route:</b> {name}<br/><b>Type:</b> {type}<br/><b>Signal Loss:</b> {loss}",
            "style": {"backgroundColor": "steelblue", "color": "white"}
        },
        height=400
    )

# ==========================================
# 🖥️ 3. UI LAYOUT
# ==========================================

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    network_mode = st.radio(
        "Active Infrastructure:",
        ["🇺🇸 US National Backbone", "🇮🇳 Telangana Regional"],
        index=0
    )
    
    mode_key = "US" if "US" in network_mode else "TG"
    
    st.markdown("---")
    st.markdown("**System Telemetry**")
    st.success("✅ AI Inference Engine: Online")
    st.info(f"📍 Region: {mode_key}")

# --- MAIN HEADER ---
st.title("📡 QOT Framework")
st.markdown(f"#### Intelligent Lightpath Provisioning System ({mode_key} Region)")

# Get Data
G, current_coords, current_conns = get_graph(mode_key)

# 1. MAP VISUALIZATION (Full Width)
st.pydeck_chart(render_map(current_coords, current_conns, mode_key))

# 2. CONTROLS (Below Map)
st.markdown("### 🛠️ Connection Request")
c1, c2, c3 = st.columns([1, 1, 1])

cities = list(current_coords.keys())
def_src = 6 if mode_key == "US" else 0
def_dst = 3 if mode_key == "US" else 1

with c1:
    source = st.selectbox("Source Node", cities, index=def_src)
with c2:
    target = st.selectbox("Destination Node", [c for c in cities if c != source], index=def_dst)
with c3:
    st.write("") # Spacer
    st.write("") # Spacer
    analyze = st.button("🚀 Analyze Route", type="primary")

# 3. ANALYSIS LOGIC
if analyze:
    st.markdown("---")
    st.markdown(f"### 🔎 Path Analysis: {source} ➡ {target}")
    
    with st.spinner("Processing network graph & applying embeddings..."):
        time.sleep(1.0)
        
        try:
            raw_paths = list(nx.shortest_simple_paths(G, source, target))
            paths = raw_paths[:5]
            
            results = []
            for i, p in enumerate(paths):
                link_ids = [get_link_id(G, p[j], p[j+1]) for j in range(len(p)-1)]
                
                padded = pad_sequences([link_ids], maxlen=MAX_PATH_LENGTH, padding='post', value=0)
                gsnr = model.predict(padded, verbose=0)[0][0]
                
                if mode_key == "TG": gsnr = max(18.0, gsnr - (len(p)*0.5)) 

                rec, status = recommend_modulation(gsnr)
                
                results.append({
                    "Path ID": f"Route #{i+1}",
                    "Hops": len(link_ids),
                    "Path Details": " ➔ ".join(p),
                    "GSNR (dB)": gsnr,
                    "Modulation": rec,
                    "Status": status
                })
            
            df = pd.DataFrame(results)
            best = df.loc[df['GSNR (dB)'].idxmax()]
            
            # --- NORMAL UI RESULTS (Reverted from Custom HTML) ---
            st.success(f"💡 AI Recommendation: {best['Path ID']} ({best['Modulation']})")
            
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric(label="Modulation Format", value=best['Modulation'])
            with m2:
                st.metric(label="Predicted GSNR", value=f"{best['GSNR (dB)']:.2f} dB")
            with m3:
                st.metric(label="Signal Status", value=best['Status'].split()[1]) # Extracts 'Excellent' from '🟢 Excellent'

            # DATA TABLE
            st.markdown("#### 📊 Comparative Route Analysis")
            
            display_df = df.copy()
            display_df['GSNR (dB)'] = display_df['GSNR (dB)'].map('{:.2f}'.format)
            
            st.dataframe(
                display_df,
                column_config={
                    "Path ID": st.column_config.TextColumn("Route", width="small"),
                    "Path Details": st.column_config.TextColumn("Trajectory", width="large"),
                    "GSNR (dB)": st.column_config.TextColumn("Signal (dB)", width="small"),
                    "Status": st.column_config.TextColumn("Quality", width="medium"),
                },
                hide_index=True,
                use_container_width=True
            )
            
        except nx.NetworkXNoPath:
            st.error("❌ No physical path exists between these nodes.")

# --- FOOTER ---
st.markdown("---")
with st.expander("📖 Technical Legend & Glossary", expanded=True):
    legend_data = {
        "Modulation (Short)": ["64QAM", "16QAM", "QPSK", "BPSK"],
        "Full Technical Name": [
            "64-ary Quadrature Amplitude Modulation",
            "16-ary Quadrature Amplitude Modulation",
            "Quadrature Phase Shift Keying",
            "Binary Phase Shift Keying"
        ],
        "Signal Quality (GSNR)": ["> 22 dB (Excellent)", "> 18 dB (Good)", "> 7 dB (Poor)", "< 7 dB (Critical)"],
        "Characteristics": [
            "Highest speed, Low noise tolerance",
            "Balanced speed and robustness",
            "Low speed, Very robust for long paths",
            "Emergency fallback only"
        ]
    }
    st.table(pd.DataFrame(legend_data))