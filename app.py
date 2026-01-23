import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import time
import networkx as nx
import pydeck as pdk 

# --- CONFIGURATION ---
MODEL_PATH = "models/saved_models/qot_model.keras"
MAX_PATH_LENGTH = 42

# --- PAGE CONFIG ---
st.set_page_config(page_title="QOT_Framework", page_icon="📡", layout="wide")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# --- 1. GEOSPATIAL DATA (REAL GPS COORDINATES) ---
CITY_COORDS = {
    "New York": [-74.0060, 40.7128],
    "Washington DC": [-77.0369, 38.9072],
    "Chicago": [-87.6298, 41.8781],
    "Atlanta": [-84.3880, 33.7490],
    "Miami": [-80.1918, 25.7617],
    "Dallas": [-96.7970, 32.7767],
    "Denver": [-104.9903, 39.7392],
    "Seattle": [-122.3321, 47.6062],
    "San Francisco": [-122.4194, 37.7749],
    "Los Angeles": [-118.2437, 34.0522],
    "Boston": [-71.0589, 42.3601],
    "Houston": [-95.3698, 29.7604],
    "Phoenix": [-112.0740, 33.4484],
    "Philadelphia": [-75.1652, 39.9526],
    "Detroit": [-83.0458, 42.3314],
    "Minneapolis": [-93.2650, 44.9778],
    "St. Louis": [-90.1994, 38.6270],
    "Las Vegas": [-115.1398, 36.1699]
}

# --- 2. BUILD NETWORK GRAPH ---
network_graph = nx.Graph()
connections = [
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

# Create Map Data
lines_data = []
nodes_data = []

for city in CITY_COORDS:
    nodes_data.append({"name": city, "coordinates": CITY_COORDS[city]})

for u, v, link_id in connections:
    if u in CITY_COORDS and v in CITY_COORDS:
        network_graph.add_edge(u, v, id=link_id)
        lines_data.append({
            "source": CITY_COORDS[u],
            "target": CITY_COORDS[v],
            "name": f"{u} -> {v}"
        })

# --- MAP RENDERING FUNCTION ---
def render_interactive_map(height=300):
    style_uri = "mapbox://styles/mapbox/satellite-v9"
    link_color = [0, 255, 100] 
    text_color = [255, 255, 255]

    layer_lines = pdk.Layer(
        "LineLayer",
        lines_data,
        get_source_position="source",
        get_target_position="target",
        get_color=link_color,
        get_width=3,
        pickable=True,
    )
    
    layer_nodes = pdk.Layer(
        "ScatterplotLayer",
        nodes_data,
        get_position="coordinates",
        get_color=[255, 50, 50],
        get_radius=120000,
        pickable=True,
    )
    
    layer_text = pdk.Layer(
        "TextLayer",
        nodes_data,
        get_position="coordinates",
        get_text="name",
        get_color=text_color,
        get_size=16,
        get_alignment_baseline="'bottom'",
    )

    view_state = pdk.ViewState(
        latitude=39.8283,
        longitude=-98.5795,
        zoom=3,
        pitch=0,
    )

    r = pdk.Deck(
        layers=[layer_lines, layer_nodes, layer_text],
        initial_view_state=view_state,
        map_style=style_uri,
        tooltip={"text": "{name}"},
        height=height
    )
    return r

# --- HELPER FUNCTIONS ---
def get_link_id(u, v):
    if network_graph.has_edge(u, v):
        return network_graph[u][v]['id']
    return 0

def recommend_modulation(gsnr_score):
    if gsnr_score >= 22.0: return "64QAM", "🟢 Excellent"
    elif gsnr_score >= 18.0: return "16QAM", "🔵 Good"
    elif gsnr_score >= 11.0: return "8QAM", "🟠 Moderate"
    elif gsnr_score >= 7.0:  return "QPSK", "🔴 Poor"
    else: return "BPSK", "⚫ Critical"

def find_routes(src, dst):
    try:
        raw_paths = list(nx.shortest_simple_paths(network_graph, src, dst))
        return raw_paths[:5] 
    except nx.NetworkXNoPath:
        return []

# --- SIDEBAR ---
with st.sidebar:
    st.title("🗺️ Network Topology")
    
    # Mini Map (Always Satellite)
    st.pydeck_chart(render_interactive_map(height=250))
    
    if st.button("🔍 View Full Network Map", type="secondary", use_container_width=True):
        st.session_state['show_fullscreen_map'] = True
    
    if st.button("❌ Close Map", type="secondary", use_container_width=True):
        st.session_state['show_fullscreen_map'] = False

    st.markdown("---")
    st.info("**System Status:** Online")

# --- MAIN PAGE ---
st.title("📡 Adaptive Machine Learning Framework")
st.subheader("Intelligent Lightpath Provisioning System")

# --- FULL SCREEN MAP OVERLAY ---
if st.session_state.get('show_fullscreen_map', False):
    st.write("### 🌍 National Backbone (Satellite View)")
    st.pydeck_chart(render_interactive_map(height=600))
    st.markdown("---")

# --- ROUTE OPTIMIZER UI ---
c1, c2, c3 = st.columns([1, 1, 1])
cities = list(CITY_COORDS.keys())

with c1:
    source_city = st.selectbox("📍 Source Node", cities, index=6) # Default Denver
with c2:
    dest_options = [c for c in cities if c != source_city]
    dest_city = st.selectbox("🎯 Destination Node", dest_options, index=3) # Default Miami
with c3:
    st.write("###")
    analyze_btn = st.button("🚀 Analyze Routes", type="primary", use_container_width=True)

if analyze_btn:
    st.write(f"### 🔎 Analyzing paths from **{source_city}** to **{dest_city}**...")
    
    with st.spinner("Calculating optimal routes & computing QoT..."):
        time.sleep(1.0)
        
        paths = find_routes(source_city, dest_city)
        
        if not paths:
            st.error(f"No physical path found between {source_city} and {dest_city}.")
        else:
            results_data = []
            
            for i, path_cities in enumerate(paths):
                path_ids = []
                for j in range(len(path_cities) - 1):
                    u, v = path_cities[j], path_cities[j+1]
                    path_ids.append(get_link_id(u, v))
                
                padded_seq = pad_sequences([path_ids], maxlen=MAX_PATH_LENGTH, padding='post', value=0)
                pred_gsnr = model.predict(padded_seq, verbose=0)[0][0]
                rec_fmt, status = recommend_modulation(pred_gsnr)
                path_str = " ➔ ".join(path_cities)
                
                results_data.append({
                    "Route Option": f"Path {i+1}",
                    "Hops": len(path_ids),
                    "Route Details": path_str,
                    "Predicted GSNR": f"{pred_gsnr:.2f} dB",
                    "Status": status,
                    "Recommendation": rec_fmt
                })
            
            df_results = pd.DataFrame(results_data)
            
            st.success(f"✅ Path Characterization Complete. {len(df_results)} Routes Analyzed.")
            st.dataframe(
                df_results, 
                column_config={
                    "Route Option": st.column_config.TextColumn("Option", width="small"),
                    "Hops": st.column_config.NumberColumn("Hops", width="small"),
                    "Route Details": st.column_config.TextColumn("Network Path", width="large"),
                    "Predicted GSNR": st.column_config.TextColumn("Signal Quality", width="medium"),
                },
                hide_index=True,
                use_container_width=True
            )
            
            best_path = df_results.iloc[0]
            st.info(f"💡 **AI Recommendation:** Select **{best_path['Route Option']}** ({best_path['Recommendation']}) for lowest latency.")
            
            st.write("### Signal Integrity Monitor (Best Path)")
            gsnr_val = float(best_path['Predicted GSNR'].split()[0])
            prog_val = float(min(1.0, max(0.0, gsnr_val / 30.0)))
            bar_color = "green" if gsnr_val > 18 else "orange" if gsnr_val > 10 else "red"
            st.markdown(f"""<style>.stProgress > div > div > div > div {{ background-color: {bar_color}; }}</style>""", unsafe_allow_html=True)
            st.progress(prog_val)

# --- FOOTER: GLOSSARY & LEGEND ---
st.markdown("---")
st.markdown("### 📖 Technical Reference: Modulation Formats & Standards")

# Create a clean DataFrame for the footer
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