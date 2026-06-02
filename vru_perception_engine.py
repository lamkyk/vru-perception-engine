import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION & PAGE SETUP ---
st.set_page_config(page_title="VRU Perception V&V Engine", page_icon="👁️", layout="wide")

# --- SYSTEMS ENGINEERING REQUIREMENTS (VRU SAFETY ENVELOPE) ---
SYS_REQS = {
    "min_confidence": 0.85,       # AI must be 85% sure it sees a human
    "min_iou": 0.70,              # Bounding box must overlap the actual human by 70%
    "critical_ttc_seconds": 2.5,  # If TTC drops below 2.5s, AEB (Auto-Braking) engages
    "max_critical_fn": 0          # ZERO False Negatives allowed inside the Critical TTC window
}

# --- DATA GENERATION (Simulating a Vehicle Approaching a VRU) ---
@st.cache_data
def simulate_vru_approach(ego_speed_mph, vru_type, lighting, occlusion):
    """Simulates 10 seconds (100 frames at 10Hz) of perception tracking during a VRU approach."""
    np.random.seed(42)
    frames = 100
    time_sec = np.linspace(0, 10, frames)
    
    # Ego vehicle speed in meters per second (m/s)
    ego_speed_ms = ego_speed_mph * 0.44704 
    
    # Distance closes over time. Start at 120 meters.
    distance_m = 120 - (ego_speed_ms * time_sec)
    distance_m = np.clip(distance_m, 0.1, 120) # Prevent negative distance
    
    # Time to Collision (TTC)
    ttc_sec = distance_m / ego_speed_ms
    
    # --- EDGE CASE DEGRADATION MODIFIERS ---
    mod_conf = 1.0
    mod_iou = 1.0
    
    # Lighting degrades confidence (AI struggles to classify pixels)
    if lighting == "Dusk/Dawn (Low Sun Glare)":
        mod_conf = 0.75
    elif lighting == "Night (Unlit Road)":
        mod_conf = 0.55
        
    # Occlusion degrades IoU (AI can't draw a tight box if the legs are hidden by a parked car)
    if occlusion == "Partial (Behind Parked Cars)":
        mod_iou = 0.80
    elif occlusion == "Severe (Emerging from Blind Alley)":
        mod_iou = 0.50
        mod_conf = 0.60 # Also hurts confidence

    # Small targets are harder to track
    if vru_type == "Child (Small Stature)":
        mod_conf *= 0.85
        mod_iou *= 0.85
        
    # Generate Telemetry Streams
    # Confidence increases as distance decreases (closer = easier to see)
    base_conf = np.clip(1.0 - (distance_m / 200), 0.1, 1.0)
    confidence = np.clip((base_conf * mod_conf) + np.random.normal(0, 0.08, frames), 0.05, 0.99)
    
    # IoU fluctuates but generally suffers from occlusion
    iou = np.clip((0.95 * mod_iou) + np.random.normal(0, 0.1, frames), 0.1, 0.99)
    
    # Simulate a "Ghosting" dropout event (hardware hiccup)
    if lighting != "Clear Daylight" or occlusion != "Clear Line of Sight":
        dropout_start = np.random.randint(40, 60)
        confidence[dropout_start:dropout_start+5] -= 0.40 # 500ms sensor dropout
        
    return pd.DataFrame({
        "Time_s": time_sec,
        "Distance_m": distance_m,
        "TTC_s": ttc_sec,
        "Detection_Confidence": confidence,
        "Bounding_Box_IoU": iou
    })

# --- UI ARCHITECTURE ---
st.title("VRU Perception Stack Validation Engine")
st.markdown("Automated edge-case evaluation for Vulnerable Road User (VRU) detection, focusing on Time-to-Collision (TTC) safety envelopes.")

tab_runner, tab_analytics = st.tabs([
    "⚙️ Edge Case Simulator & V&V Runner", 
    "📊 Executive Safety Analytics"
])

# ==========================================
# TAB 1: SIMULATOR & RUNNER
# ==========================================
with tab_runner:
    st.header("Scenario Configuration Matrix")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Kinematics")
        ego_speed = st.slider(
            "Ego Vehicle Speed (mph)", 15, 65, 35, 5,
            help="Velocity of the autonomous vehicle. Higher speeds drastically compress the Time-to-Collision (TTC) window, requiring the AI to detect the VRU at much further distances to safely stop."
        )
    with c2:
        st.subheader("Target Profile")
        vru_class = st.selectbox(
            "VRU Classification", 
            ["Adult Pedestrian", "Cyclist (Lateral Crossing)", "Child (Small Stature)"],
            help="Changes the pixel density and radar cross-section of the target. Children represent extreme edge cases due to their unpredictable kinematics and low camera profile."
        )
    with c3:
        st.subheader("Environmental Occlusion")
        lighting = st.selectbox(
            "Ambient Lighting", 
            ["Clear Daylight", "Dusk/Dawn (Low Sun Glare)", "Night (Unlit Road)"],
            help="Modulates optical sensor saturation. Low sun glare can completely blind camera optics, forcing reliance on LiDAR/Radar fusion."
        )
        occlusion = st.selectbox(
            "Spatial Occlusion", 
            ["Clear Line of Sight", "Partial (Behind Parked Cars)", "Severe (Emerging from Blind Alley)"],
            help="Simulates physical obstructions. Partial occlusion drastically lowers Bounding Box IoU (spatial accuracy) because the AI cannot predict the full geometry of the human body."
        )
        
    st.divider()
    
    col_run, col_reqs = st.columns([1, 1])
    with col_run:
        run_test = st.button("▶ Execute VRU Tracking Simulation", type="primary", use_container_width=True)
        st.caption("Executes a 100-frame (10-second) kinematic approach simulation.")
        
    with col_reqs:
        with st.expander("View Strict VRU Safety Requirements", expanded=True):
            st.info(
                f"**Critical Safety Envelopes:**\n\n"
                f"1. **Detection Floor:** Confidence must exceed **>{SYS_REQS['min_confidence']*100}%**.\n"
                f"2. **Spatial Accuracy (IoU):** Bounding box overlap must exceed **>{SYS_REQS['min_iou']*100}%** to prevent false steering actuation.\n"
                f"3. **Zero-Tolerance Window:** If Time-to-Collision (TTC) is under **{SYS_REQS['critical_ttc_seconds']}s**, a single frame dropping below the Detection Floor triggers a **Catastrophic Safety Failure**."
            )

    if run_test:
        with st.spinner('Generating pixel-level tracking data...'):
            df = simulate_vru_approach(ego_speed, vru_class, lighting, occlusion)
            
            # --- EVALUATION LOGIC ---
            # 1. Total frames where confidence was too low
            low_conf_frames = df[df["Detection_Confidence"] < SYS_REQS['min_confidence']]
            
            # 2. Total frames where IoU was too low
            low_iou_frames = df[df["Bounding_Box_IoU"] < SYS_REQS['min_iou']]
            
            # 3. CRITICAL: False Negatives inside the Danger Zone (TTC < 2.5s)
            danger_zone = df[df["TTC_s"] < SYS_REQS['critical_ttc_seconds']]
            critical_failures = danger_zone[danger_zone["Detection_Confidence"] < SYS_REQS['min_confidence']]
            
            # Cache to state
            st.session_state.update({
                'vru_df': df,
                'vru_class': vru_class,
                'conditions': f"{lighting} + {occlusion}",
                'crit_fails': len(critical_failures),
                'iou_fails': len(low_iou_frames)
            })
            
            st.success("Simulation Complete. Proceed to Executive Safety Analytics.")

# ==========================================
# TAB 2: EXECUTIVE ANALYTICS
# ==========================================
with tab_analytics:
    st.header("VRU Safety Qualification Report")
    
    if 'vru_df' not in st.session_state:
        st.warning("Awaiting tracking telemetry. Please execute a simulation in the Runner tab.")
    else:
        df = st.session_state['vru_df']
        crit_fails = st.session_state['crit_fails']
        iou_fails = st.session_state['iou_fails']
        
        # Absolute Zero Tolerance Logic
        if crit_fails > SYS_REQS['max_critical_fn']:
            status, color = "FAILED: LETHAL REGRESSION", "red"
        elif iou_fails > 15:
            status, color = "WARNING: SPATIAL DRIFT", "orange"
        else:
            status, color = "PASSED: SAFE TRACKING", "green"
            
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Target Profile", st.session_state['vru_class'])
        m2.metric(
            "Spatial (IoU) Failures", 
            iou_fails,
            help="Count of frames where the bounding box was too sloppy. High numbers here mean the car might brake for a pedestrian that is safely on the sidewalk."
        )
        m3.metric(
            "Critical TTC Failures", 
            crit_fails,
            help=f"ABSOLUTE METRIC: Count of times the AI lost track of the human when they were less than {SYS_REQS['critical_ttc_seconds']} seconds away. Any number above 0 is a catastrophic failure."
        )
        m4.markdown(f"### Status: :{color}[{status}]")
        
        st.divider()
        
        # --- CHART 1: Confidence vs TTC ---
        st.subheader(
            "Time-to-Collision (TTC) vs. Perception Confidence", 
            help="**HOW TO READ THIS CHART:**\n\nThe X-Axis is Time to Collision (TTC) moving backwards (from 10 seconds down to 0). As the car gets closer, TTC shrinks. \n\n**WHERE TO FOCUS:** Look at the shaded RED box on the left. This is the 'Danger Zone' (TTC < 2.5s). If the blue line (AI Confidence) dips below the dotted red line while inside that red box, the AI has 'dropped' a human right before hitting them. This is an instant test failure."
        )
        
        fig1, ax1 = plt.subplots(figsize=(12, 4))
        # Plot time backwards (closer to collision)
        ax1.plot(df["TTC_s"], df["Detection_Confidence"], color="#2980b9", linewidth=2, label="AI Confidence Score")
        ax1.axhline(y=SYS_REQS['min_confidence'], color="#e74c3c", linestyle="--", label="Confidence Floor (0.85)")
        
        # Highlight the Danger Zone
        ax1.axvspan(0, SYS_REQS['critical_ttc_seconds'], color='red', alpha=0.1, label="Critical AEB Zone (TTC < 2.5s)")
        
        # Highlight critical failures natively
        if crit_fails > 0:
            dz = df[df["TTC_s"] < SYS_REQS['critical_ttc_seconds']]
            c_fails = dz[dz["Detection_Confidence"] < SYS_REQS['min_confidence']]
            ax1.scatter(c_fails["TTC_s"], c_fails["Detection_Confidence"], color="red", zorder=5, label="Lethal Tracking Dropouts")
            
        ax1.invert_xaxis() # TTC goes from 10 down to 0
        ax1.set_xlim(10, 0)
        ax1.set_xlabel("Time to Collision (Seconds)")
        ax1.set_ylabel("Perception Confidence (0.0 - 1.0)")
        ax1.legend(loc="lower left")
        st.pyplot(fig1)
        
        # --- CHART 2: Spatial Accuracy (IoU) ---
        st.subheader(
            "Bounding Box Spatial Accuracy (IoU)", 
            help="**HOW TO READ THIS CHART:**\n\nIntersection over Union (IoU) measures how perfectly the AI's digital box wraps around the actual physical human. 1.0 is a perfect fit.\n\n**WHERE TO FOCUS:** The dotted orange line is the 0.70 safety threshold. If the green line drops below it, the box is 'drifting.' This often happens during partial occlusion (e.g., walking behind a streetlamp), causing the car's path planner to panic."
        )
        
        fig2, ax2 = plt.subplots(figsize=(12, 3.5))
        ax2.plot(df["Time_s"], df["Bounding_Box_IoU"], color="#27ae60", label="Intersection over Union (IoU)")
        ax2.axhline(y=SYS_REQS['min_iou'], color="#e67e22", linestyle="--", label="IoU Quality Floor (0.70)")
        
        ax2.set_xlabel("Elapsed Simulation Time (Seconds)")
        ax2.set_ylabel("Spatial Accuracy (IoU)")
        ax2.legend(loc="lower right")
        st.pyplot(fig2)