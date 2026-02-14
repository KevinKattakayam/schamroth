"""
Schamroth Window Detection - Pipeline App
Uses step-by-step verification for accurate results
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from detector_pipeline import SchamrothPipelineDetector, RiskLevel, PipelineStep

st.set_page_config(
    page_title="Schamroth Clubbing Detector",
    page_icon="🏥",
    layout="wide"
)

# Initialize Detector (cached)
@st.cache_resource
def get_detector():
    return SchamrothPipelineDetector()

st.title("🏥 Schamroth Window Detector")
st.markdown("**Pipeline-based detection** with step-by-step verification")

# Tabs
tabs = st.tabs(["📸 Scan", "📖 Guide", "ℹ️ About"])

with tabs[0]:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Image")
        uploaded = st.file_uploader("Upload photo of hands in Schamroth position", type=['jpg','png','jpeg'])
        
        if uploaded:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            st.image(image, caption="Uploaded Image", channels="BGR", use_column_width=True)
            
            if st.button("🔍 Analyze", type="primary"):
                with st.spinner("Running detection pipeline..."):
                    detector = get_detector()
                    result = detector.detect(image)
                    st.session_state.result = result
    
    with col2:
        st.subheader("Results")
        
        if 'result' in st.session_state:
            result = st.session_state.result
            
            # Show marked image if available
            if result.marked_image is not None and result.marked_image.size > 0:
                st.image(result.marked_image, caption="Detection Result", channels="BGR", use_column_width=True)
            
            # Show pipeline status
            st.subheader("Pipeline Status")
            
            steps = ["Step 1: Hands Check", "Step 2: Fingers Check", 
                     "Step 3: Geometry Check", "Step 4: Gap Analysis"]
            
            step_map = {
                PipelineStep.HANDS_CHECK: 0,
                PipelineStep.FINGERS_CHECK: 1,
                PipelineStep.GEOMETRY_CHECK: 2,
                PipelineStep.GAP_ANALYSIS: 3,
                PipelineStep.COMPLETE: 4
            }
            
            reached = step_map.get(result.step_reached, 0)
            
            for i, step in enumerate(steps):
                if i < reached or result.all_steps_passed:
                    st.success(f"✓ {step}")
                elif i == reached and not result.all_steps_passed:
                    st.error(f"✗ {step} - FAILED")
                else:
                    st.info(f"○ {step} - Pending")
            
            st.markdown("---")
            
            # Check if all steps passed
            if result.all_steps_passed:
                # Show results
                risk = result.risk_level
                if risk == RiskLevel.HEALTHY or risk == RiskLevel.LOW_RISK:
                    st.success(f"### ✅ {risk.value.upper().replace('_', ' ')}")
                elif risk == RiskLevel.BORDERLINE:
                    st.warning(f"### ⚠️ {risk.value.upper()}")
                else:
                    st.error(f"### ⚠️ {risk.value.upper().replace('_', ' ')}")
                
                # Metrics
                c1, c2, c3 = st.columns(3)
                c1.metric("Healthy", f"{result.healthy_percentage:.0f}%")
                c2.metric("Clubbing", f"{result.clubbing_percentage:.0f}%")
                c3.metric("Confidence", result.confidence_label)
                
                st.info(f"**{result.message}**")
                st.markdown(f"**Recommendation:** {result.recommendation}")
            else:
                st.error("### ⚠️ RETAKE REQUIRED")
                st.warning(f"**{result.message}**")
                st.markdown(f"**{result.recommendation}**")
                
                st.info("""
                **For all steps to pass, ensure:**
                1. ✅ Both hands visible in frame
                2. ✅ Hands SEPARATED (not overlapping)
                3. ✅ Index fingers touching nail-to-nail
                4. ✅ Good lighting, camera above hands
                """)
        else:
            st.info("Upload an image and click Analyze to see results")

with tabs[1]:
    st.subheader("📖 Detection Pipeline")
    
    st.markdown("""
    ### How Detection Works:
    
    The detector runs 4 verification steps in sequence:
    
    **Step 1: Hands Check**
    - Uses MediaPipe to detect hands
    - Must find exactly 2 hands
    
    **Step 2: Fingers Check**
    - Finds closest pair of fingertips
    - Verifies fingers are touching
    
    **Step 3: Geometry Check**
    - Verifies diamond shape between fingertips
    - Checks diamond has sufficient area
    
    **Step 4: Gap Analysis**
    - Analyzes brightness in diamond region
    - Computes edge and contrast scores
    - Generates final gap presence score
    
    ### Photo Requirements:
    - Both hands clearly visible
    - Hands SEPARATED (wrists apart)
    - Index fingers touching nail-to-nail
    - Good lighting, camera above hands
    """)

with tabs[2]:
    st.subheader("ℹ️ About")
    
    st.markdown("""
    ### Schamroth Window Test
    
    A clinical screening test for digital clubbing.
    
    **Healthy:** Diamond-shaped gap visible between nail folds
    
    **Clubbing:** No gap visible (nail beds meet)
    
    **Important:** This is a screening tool only. Consult a healthcare professional for diagnosis.
    """)

st.markdown("---")
st.caption("⚕️ For screening purposes only.")