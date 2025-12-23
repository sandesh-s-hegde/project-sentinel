import streamlit as st
import tempfile
import os
import matplotlib.pyplot as plt
from src.processor import SentinelProcessor
from src.finance import run_simulation

# Page Configuration
st.set_page_config(page_title="Project Sentinel", layout="wide", page_icon="👁️")

# Title & Abstract
st.title("👁️ Project Sentinel: Empirical Grounding of Operational Risk")
st.markdown("""
**Objective:** To bridge the gap between **Computer Vision** and **Real Options Analysis**. 
This tool extracts **Flow Entropy** ($\sigma_{flow}$) from unstructured shop-floor video feeds to generate dynamic inputs for financial risk models.
""")

# Sidebar: Simulation Parameters
with st.sidebar:
    st.header("1. Financial Parameters")
    S = st.number_input("Asset Value ($)", value=100.0)
    K = st.number_input("Exercise Cost ($)", value=95.0)
    T = st.slider("Time to Maturity (Years)", 0.1, 2.0, 1.0)
    r = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 4.0) / 100
    
    st.divider()
    
    st.header("2. Vision Engine")
    model_type = st.selectbox("YOLO Model Size", ["n (Nano)", "s (Small)"])
    st.info("Note: 'Nano' is optimized for CPU latency.")

# Main Interface
st.subheader("Step 1: Upload Surveillance Feed")
uploaded_file = st.file_uploader("Upload Shop Floor / Traffic Video (MP4)", type=['mp4'])

if uploaded_file:
    # WINDOWS SAFE FILE HANDLING
    # We create a temp file, write to it, and CLOSE it immediately so OpenCV can open it.
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
    tfile.write(uploaded_file.read())
    tfile.close() 
    
    video_path = tfile.name
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.video(video_path)
        st.caption("Raw Input Feed")
    
    # Run Button
    if st.button("🚀 Initialize Sentinel Engine", type="primary"):
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        # 1. Vision Processing
        processor = SentinelProcessor(model_size=model_type[0])
        
        def update_status(frame):
            status_text.text(f"Processing Frame: {frame} | Extracting Flow Vectors...")
            
        entropy, sigma_series = processor.process_video_feed(video_path, update_status)
        
        status_text.text("Vision Processing Complete. Running Stochastic Simulation...")
        progress_bar.progress(100)
        
        # 2. Financial Modeling
        dyn_prices, stat_prices, static_sigma = run_simulation(sigma_series, S, K, T, r)
        
        # 3. Visualization
        st.divider()
        st.subheader("Step 2: Empirical Results")
        
        # Plot A: The Volatility Input
        fig1, ax1 = plt.subplots(figsize=(10, 3))
        ax1.plot(sigma_series, color='#FF4B4B', label='Dynamic Sigma (Vision-Derived)')
        ax1.axhline(y=static_sigma, color='gray', linestyle='--', label=f'Static Mean ({static_sigma:.2f})')
        ax1.set_title("Operational Volatility Profile")
        ax1.set_ylabel("Volatility (σ)")
        ax1.legend()
        st.pyplot(fig1)
        
        # Plot B: The Option Price Output
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        ax2.plot(dyn_prices, color='#00CC96', label='Dynamic Option Value')
        ax2.plot(stat_prices, color='gray', linestyle='--', label='Static Option Value')
        ax2.fill_between(range(len(dyn_prices)), dyn_prices, stat_prices, alpha=0.1, color='#00CC96')
        ax2.set_title("Real Option Valuation: The Cost of 'Not Seeing'")
        ax2.set_ylabel("Option Price ($)")
        ax2.legend()
        st.pyplot(fig2)
        
        st.success("Simulation Complete. The divergence between the Green and Gray lines represents the value of Empirical Grounding.")

    # Cleanup (Best effort)
    try:
        os.remove(video_path)
    except:
        pass