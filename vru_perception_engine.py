import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION & PAGE SETUP ---
st.set_page_config(page_title="Ultra-VRU Perception V&V Suite", page_icon="👁️", layout="wide")

# --- SYSTEMS ENGINEERING PASS/FAIL CRITERIA ---
SYS_REQS = {
    "min_confidence": 0.88,       # High bar for VRU classification certainty
    "min_iou": 0.75,              # Bounding box precision requirement to prevent phantom braking
    "critical_ttc_seconds": 3.0,  # Pre-crash safety envelope boundary
    "max_allowable_dropouts": 2,  # Maximum continuous frames allowed to fail inside danger zone
}

# --- STOCHASTIC SIMULATION MATHEMATICS ---
@st.cache_data
def run_high_fidelity_vru_sim(ego_speed_mph, vru_type, occlusion_state, environment_state):
    """
    Simulates a high-fidelity 15-second tracking sequence (150 frames at 10Hz)
    mapping complex physics, sensory attenuation, and spatial tracking profiles.
    """
    np.random.seed(1337) # Lock seed for deterministic regression testing
    frames = 150
    time_sec = np.linspace(0, 15, frames)
    
    # Kinematics translation
    ego_speed_ms = ego_speed_mph * 0.44704
    start_distance = 160 # Start farther out to capture tracking initialization
    distance_m = start_distance - (ego_speed_ms * time_sec)
    distance_m = np.clip(distance_m, 0.2, start_distance)
    
    ttc_sec = np.where(ego_speed_ms > 0, distance_m / ego_speed_ms, 99.0)
    
    # --- COMBINATORIAL EDGE-CASE PROFILES ---
    # Baseline sensory coefficients
    coef_conf = 1.0
    coef_iou = 1.0
    noise_sigma = 0.05
    
    # 1. Target Geometry Modifiers
    if vru_type == "Prone Pedestrian (Fallen/Injured)":
        coef_conf, coef_iou = 0.70, 0.65  # Low aspect ratio breaks standard anchor boxes
    elif vru_type == "Child (Upright, Small Stature)":
        coef_conf, coef_iou = 0.80, 0.80  # Low pixel height decreases classifier weight
    elif vru_type == "Prone Child (Fallen/Low Profile)":
        coef_conf, coef_iou = 0.55, 0.50  # Extreme edge case; highly unrepresented in training data
    elif vru_type == "Domestic Animal (Canine/Feline)":
        coef_conf, coef_iou = 0.75, 0.70  # Erratic four-legged kinematics
    elif vru_type == "Large Animal (Deer/Wildlife)":
        coef_conf, coef_iou = 0.85, 0.75  # High height but thin bounds split spatial groupings
    elif vru_type == "Stroller / Wheelchair User":
        coef_conf, coef_iou = 0.80, 0.65  # Complex visual occlusion compound structures

    # 2. Spatial Occlusion Profiles
    if occlusion_state == "Partially Hidden (10% - 40%)":
        coef_iou *= 0.85
    elif occlusion_state == "Severely Occluded (40% - 80%)":
        coef_iou *= 0.55
        coef_conf *= 0.80
    elif occlusion_state == "Intermittent (Cross-Traffic)":
        noise_sigma = 0.18 # Drastically amplify tracking noise variance
    elif occlusion_state == "Sudden Emergence (Blind Alley)":
        # Force 0 detection until target clears the occlusion zone at close range
        coef_conf *= 0.20 
        coef_iou *= 0.15

    # 3. Environmental Attenuation Profiles
    if environment_state == "Dusk / Dawn (Low Sun Glare)":
        coef_conf *= 0.75 # Sensor saturation
    elif environment_state == "Night (Active Headlights Only)":
        coef_conf *= 0.60
        noise_sigma += 0.04
    elif environment_state == "Night with High-Beam Glare":
        coef_conf *= 0.45 # Saturated pixel arrays
    elif environment_state == "Heavy Spray / Splash-back":
        coef_conf *= 0.70
        coef_iou *= 0.70 # Lens refraction errors

    # --- TIME-SERIES SYNTHESIS ---
    # Detection confidence naturally scales up as target approaches sensory focal ranges
    range_factor = np.clip(1.0 - (distance_m / start_distance), 0.1, 1.0)
    confidence = (range_factor * coef_conf) + np.random.normal(0, noise_sigma, frames)
    confidence = np.clip(confidence, 0.02, 0.98)
    
    iou = (0.92 * coef_iou) + np.random.normal(0, 0.08, frames)
    iou = np.clip(iou, 0.01, 0.99)
    
    # Overwrite sudden emergence masking logic
    if occlusion_state == "Sudden Emergence (Blind Alley)":
        # Target pops out precisely 4 seconds into the timeline
        hidden_indices = time_sec < 4.0
        confidence[hidden_indices] = np.random.uniform(0.01, 0.08, size=np.sum(hidden_indices))
        iou[hidden_indices] = np.random.uniform(0.01, 0.10, size=np.sum(hidden_indices))
        # Immediate tracking initialization spike
        recovery_indices = time_sec >= 4.0
        confidence[recovery_indices] = np.clip(confidence[recovery_indices] * 4.5, 0.05, 0.96)
        iou[recovery_indices] = np.clip(iou[recovery_indices] * 5.0, 0.05, 0.94)

    # Intermittent masking dropouts (cross-traffic passing)
    if occlusion_state == "Intermittent (Cross-Traffic)":
        dropout_blocks = [range(20, 30), range(75, 85)]
        for block in dropout_blocks:
            confidence[block] = np.random.uniform(0.02, 0.12, len(block))
            iou[block] = np.random.uniform(0.01, 0.15, len(block))

    return pd.DataFrame({
        "Time_s": time_sec,
        "Distance_m": distance_m,
        "TTC_s": ttc_sec,
        "Confidence": confidence,
        "IoU": iou
    })

# --- UI ARCHITECTURE ---
st.title("VRU Perception Detection & Safety Verification Suite")
st.markdown("Automated combinatorial edge-case verification for low-profile, occluded, and vulnerable road entities.")

tab_runner, tab_analytics = st.tabs([
    "⚙️ Combinatorial Edge-Case Runner", 
    "📊 Executive Diagnostics & Verification"
])

# ==========================================
# TAB 1: RUNNER
# ==========================================
with tab_runner:
    st.header("Scenario Parameter Matrix")
    
    col_vru, col_occ, col_env = st.columns(3)
    
    with col_vru:
        st.subheader("Target Entity Profile")
        sim_vru = st.selectbox(
            "Classification Category",
            [
                "Adult Pedestrian (Upright)",
                "Prone Pedestrian (Fallen/Injured)",
                "Child (Upright, Small Stature)",
                "Prone Child (Fallen/Low Profile)",
                "Cyclist (Lateral Crossing)",
                "E-Scooter Rider (High Velocity)",
                "Domestic Animal (Canine/Feline)",
                "Large Animal (Deer/Wildlife)",
                "Stroller / Wheelchair User"
            ],
            help="Defines the kinematic boundaries, structural dimensions, and aspect ratio signatures used for model classification evaluation."
        )
        sim_speed = st.slider(
            "Chassis Approach Velocity (mph)", 10, 75, 35, 5,
            help="Chassis velocity directly alters distance compression rates, tightening required latency bounds for network inference."
        )
        
    with col_occ:
        st.subheader("Spatial Occlusion Layer")
        sim_occlusion = st.selectbox(
            "Obstruction Configuration",
            [
                "Clear Line of Sight (0% Occlusion)",
                "Partially Hidden (10% - 40%)",
                "Severely Occluded (40% - 80%)",
                "Intermittent (Cross-Traffic)",
                "Sudden Emergence (Blind Alley)"
            ],
            help="Simulates visual boundaries. Sub-surface obstruction significantly limits ground-truth bounding box matching (IoU)."
        )
        
    with col_env:
        st.subheader("Environmental Context")
        sim_env = st.selectbox(
            "Atmospheric & Lighting State",
            [
                "Clear Daylight (100% Illumination)",
                "Dusk / Dawn (Low Sun Glare)",
                "Night (Active Headlights Only)",
                "Night with High-Beam Glare",
                "Heavy Spray / Splash-back"
            ],
            help="Modulates external lighting profiles to stress camera sensor arrays and back-end neural feature extractors."
        )

    st.divider()
    
    col_btn, col_sys_rules = st.columns([1, 1])
    with col_btn:
        run_sim = st.button("▶ Run Perception Regression Test", type="primary", use_container_width=True)
        st.caption("Processes a 150-frame time-series approach mapping the active configurations against systems validation floors.")
        
    with col_sys_rules:
        with st.expander("Systems Engineering Safety Boundary Specifications", expanded=True):
            st.info(
                f"**Verification Checkpoints:**\n\n"
                f"1. **Core Detection Threshold:** Multi-modal classification must remain **>={SYS_REQS['min_confidence']*100}%**.\n"
                f"2. **Spatial Alignment Floor:** Bounding box Intersection over Union (IoU) target is **>={SYS_REQS['min_iou']*100}%**.\n"
                f"3. **Zero-Tolerance Safety Window:** Inside a critical Time-to-Collision (TTC) of **{SYS_REQS['critical_ttc_seconds']}s**, "
                f"consecutive drops in classification confidence flag an immediate structural regression."
            )

    if run_sim:
        with st.spinner("Compiling time-series tracking metrics..."):
            sim_data = run_high_fidelity_vru_sim(sim_speed, sim_vru, sim_occlusion, sim_env)
            
            # Compute evaluation passes
            danger_zone = sim_data[sim_data["TTC_s"] <= SYS_REQS['critical_ttc_seconds']]
            failed_safety_frames = danger_zone[danger_zone["Confidence"] < SYS_REQS['min_confidence']]
            failed_iou_frames = sim_data[sim_data["IoU"] < SYS_REQS['min_iou']]
            
            st.session_state.update({
                'vru_metrics': sim_data,
                'active_vru': sim_vru,
                'active_config': f"{sim_occlusion} | {sim_env}",
                'safety_failures': len(failed_safety_frames),
                'iou_failures': len(failed_iou_frames)
            })
            st.success("Test execution complete. Visualization metrics generated in Analytics tab.")

# ==========================================
# TAB 2: ANALYTICS
# ==========================================
with tab_analytics:
    st.header("Perception Diagnostics Matrix")
    
    if 'vru_metrics' not in st.session_state:
        st.warning("No evaluation metrics found. Please trigger a regression pass within the Scenario Configuration tab.")
    else:
        df = st.session_state['vru_metrics']
        sf_count = st.session_state['safety_failures']
        iou_count = st.session_state['iou_failures']
        
        # Determine strict qualification status
        if sf_count > SYS_REQS['max_allowable_dropouts']:
            qual_status, qual_color = "CRITICAL FAIL: TRACK REJECTION", "red"
        elif iou_count > 35:
            qual_status, qual_color = "MARGINAL PASS: HIGH SPATIAL DRIFT", "orange"
        else:
            qual_status, qual_color = "PASS: STABLE BOUNDING ENVELOPE", "green"
            
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Evaluated Profile", st.session_state['active_vru'])
        r2.metric("Spatial Alignment Drops (IoU)", iou_count, help="Total execution frames where bounding-box structure dropped below nominal target precision benchmarks.")
        r3.metric("Danger-Zone Dropouts", sf_count, help="Critical frame instances inside the pre-crash window where target tracking validation was lost.")
        r4.markdown(f"### Verification Status:\n:{qual_color}[{qual_status}]")
        
        st.divider()
        
        # --- TIME-SERIES COMPONENT VISUALIZATIONS ---
        st.subheader(
            "Classification Confidence Convergence",
            help=(
                "**INSPECTION INSTRUCTIONS:**\n\n"
                "Focus on the shaded region on the right side of the timeline. This denotes the critical safety perimeter (TTC <= 3.0s). "
                "The blue trace tracking classification confidence should smoothly scale up towards 1.0 as the target nears the sensor array. "
                "Any dropouts dipping below the dashed line inside the critical window represent tracking loss, requiring instant logic modification."
            )
        )
        
        fig_c, ax_c = plt.subplots(figsize=(12, 4))
        ax_c.plot(df["Time_s"], df["Confidence"], color="#1f77b4", linewidth=2.5, label="Model Confidence Signature")
        ax_c.axhline(y=SYS_REQS['min_confidence'], color="#d62728", linestyle="--", alpha=0.8, label="Validation Limit (0.88)")
        
        # Compute critical safety window intersection boundaries
        if not df[df["TTC_s"] <= SYS_REQS['critical_ttc_seconds']].empty:
            danger_start_time = df[df["TTC_s"] <= SYS_REQS['critical_ttc_seconds']]["Time_s"].min()
            ax_c.axvspan(danger_start_time, df["Time_s"].max(), color="#d62728", alpha=0.12, label="Critical AEB Window (TTC <= 3s)")
            
            # Map critical fail artifacts natively onto plot array
            dz = df[df["Time_s"] >= danger_start_time]
            cf_pts = dz[dz["Confidence"] < SYS_REQS['min_confidence']]
            if not cf_pts.empty:
                ax_c.scatter(cf_pts["Time_s"], cf_pts["Confidence"], color="#d62728", s=60, zorder=5, label="Critical Dropouts")

        ax_c.set_xlabel("Scenario Timeline (Seconds)")
        ax_c.set_ylabel("Probability Score")
        ax_c.set_ylim(-0.05, 1.05)
        ax_c.legend(loc="upper left")
        st.pyplot(fig_c)
        
        st.divider()
        
        st.subheader(
            "Spatial Extent Overlap Matrix (IoU)",
            help=(
                "**INSPECTION INSTRUCTIONS:**\n\n"
                "Examine the tracking variance trace across the timeline. Sudden, deep valleys below the dashed orange floor indicate bounding box 'ballooning' or truncation. "
                "This typically manifests when a target is partially hidden (e.g., a child standing behind a stroller), causing the perception model to alternate between tracking the partial body and the full ensemble."
            )
        )
        
        fig_i, ax_i = plt.subplots(figsize=(12, 3.5))
        ax_i.plot(df["Time_s"], df["IoU"], color="#2ca02c", linewidth=2, label="Intersection over Union Trace")
        ax_i.axhline(y=SYS_REQS['min_iou'], color="#ff7f0e", linestyle="--", alpha=0.8, label="Spatial Quality Floor (0.75)")
        
        ax_i.set_xlabel("Scenario Timeline (Seconds)")
        ax_i.set_ylabel("IoU Accuracy Match")
        ax_i.set_ylim(-0.05, 1.05)
        ax_i.legend(loc="upper left")
        st.pyplot(fig_i)
