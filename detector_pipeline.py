"""
================================================================================
SCHAMROTH PIPELINE DETECTOR - Step-by-Step Verification
================================================================================

Pipeline Steps:
1. HANDS CHECK    - Must detect exactly 2 hands via MediaPipe
2. FINGERS CHECK  - Find closest touching fingertips between hands
3. GEOMETRY CHECK - Verify diamond shape between fingertips
4. GAP ANALYSIS   - Analyze brightness/contrast in diamond region
5. CLASSIFICATION - Generate final result

Each step must pass before proceeding to next.
================================================================================
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SchamrothPipeline")


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class PipelineStep(Enum):
    """Pipeline verification steps"""
    HANDS_CHECK = "hands_check"
    FINGERS_CHECK = "fingers_check"
    GEOMETRY_CHECK = "geometry_check"
    GAP_ANALYSIS = "gap_analysis"
    COMPLETE = "complete"


class RiskLevel(Enum):
    HEALTHY = "healthy"
    LOW_RISK = "low_risk"
    BORDERLINE = "borderline"
    HIGH_RISK = "high_risk"
    CLUBBING = "clubbing"
    ERROR = "error"


@dataclass
class PipelineResult:
    """Result from pipeline detection"""
    # Pipeline status
    step_reached: PipelineStep
    all_steps_passed: bool
    step_results: Dict = field(default_factory=dict)
    
    # Detection results (if complete)
    risk_level: RiskLevel = RiskLevel.ERROR
    is_clubbing: bool = False
    healthy_percentage: float = 0.0
    clubbing_percentage: float = 0.0
    confidence_label: str = "LOW"
    
    # User messages
    message: str = ""
    recommendation: str = ""
    requires_retake: bool = True
    
    # Visual output
    marked_image: np.ndarray = field(default_factory=lambda: np.array([]))
    debug_info: Dict = field(default_factory=dict)


class SchamrothPipelineDetector:
    """
    Pipeline-based Schamroth detector with step-by-step verification.
    
    Each step must pass before proceeding:
    1. Hands Check → 2. Fingers Check → 3. Geometry Check → 4. Gap Analysis
    """
    
    # MediaPipe landmarks
    FINGERTIPS = {4: 'Thumb', 8: 'Index', 12: 'Middle', 16: 'Ring', 20: 'Pinky'}
    DIP_JOINTS = {4: 3, 8: 7, 12: 11, 16: 15, 20: 19}
    
    def __init__(self):
        """Initialize MediaPipe"""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        logger.info("SchamrothPipelineDetector initialized")
    
    def detect(self, image: np.ndarray) -> PipelineResult:
        """
        Run full detection pipeline with step-by-step verification.
        """
        if image is None or image.size == 0:
            return self._error_result("Invalid image")
        
        h, w = image.shape[:2]
        marked = image.copy()
        debug = {"image_size": (w, h)}
        step_results = {}
        
        # ================================================================
        # STEP 1: HANDS CHECK - Detect both hands
        # ================================================================
        logger.info("STEP 1: Checking for 2 hands...")
        hands_result = self._step1_hands_check(image, marked, w, h)
        step_results["hands_check"] = hands_result
        
        if not hands_result["passed"]:
            self._draw_step_failure(marked, 1, "HANDS CHECK FAILED", hands_result["reason"])
            return PipelineResult(
                step_reached=PipelineStep.HANDS_CHECK,
                all_steps_passed=False,
                step_results=step_results,
                message=f"⚠️ Step 1 Failed: {hands_result['reason']}",
                recommendation="Please retake with BOTH hands visible and SEPARATED (not overlapping).",
                requires_retake=True,
                marked_image=marked,
                debug_info=debug
            )
        
        self._draw_step_success(marked, 1, "✓ 2 HANDS DETECTED")
        hand1, hand2 = hands_result["hand1"], hands_result["hand2"]
        
        # ================================================================
        # STEP 2: FINGERS CHECK - Find touching fingertips
        # ================================================================
        logger.info("STEP 2: Checking for touching fingers...")
        fingers_result = self._step2_fingers_check(hand1, hand2, marked, w, h)
        step_results["fingers_check"] = fingers_result
        
        if not fingers_result["passed"]:
            self._draw_step_failure(marked, 2, "FINGERS CHECK FAILED", fingers_result["reason"])
            return PipelineResult(
                step_reached=PipelineStep.FINGERS_CHECK,
                all_steps_passed=False,
                step_results=step_results,
                message=f"⚠️ Step 2 Failed: {fingers_result['reason']}",
                recommendation="Please position index fingers TOUCHING nail-to-nail.",
                requires_retake=True,
                marked_image=marked,
                debug_info=debug
            )
        
        self._draw_step_success(marked, 2, f"✓ FINGERS TOUCHING ({fingers_result['distance']:.0f}px)")
        
        # ================================================================
        # STEP 3: GEOMETRY CHECK - Verify diamond shape
        # ================================================================
        logger.info("STEP 3: Checking diamond geometry...")
        geometry_result = self._step3_geometry_check(
            hand1, hand2, fingers_result, marked, w, h
        )
        step_results["geometry_check"] = geometry_result
        
        if not geometry_result["passed"]:
            self._draw_step_failure(marked, 3, "GEOMETRY CHECK FAILED", geometry_result["reason"])
            return PipelineResult(
                step_reached=PipelineStep.GEOMETRY_CHECK,
                all_steps_passed=False,
                step_results=step_results,
                message=f"⚠️ Step 3 Failed: {geometry_result['reason']}",
                recommendation="Please ensure fingers form a proper diamond shape.",
                requires_retake=True,
                marked_image=marked,
                debug_info=debug
            )
        
        self._draw_step_success(marked, 3, f"✓ DIAMOND DETECTED ({geometry_result['area']:.0f}px²)")
        
        # ================================================================
        # STEP 4: GAP ANALYSIS - Analyze diamond region
        # ================================================================
        logger.info("STEP 4: Analyzing gap...")
        gap_result = self._step4_gap_analysis(
            image, geometry_result, marked, w, h
        )
        step_results["gap_analysis"] = gap_result
        
        # Gap analysis always passes (gives a score)
        self._draw_step_success(marked, 4, f"✓ GAP SCORE: {gap_result['score']:.0%}")
        
        # ================================================================
        # STEP 5: CLASSIFICATION
        # ================================================================
        logger.info("STEP 5: Classifying result...")
        gap_score = gap_result["score"]
        
        # Convert to percentages
        healthy_pct = gap_score * 100
        clubbing_pct = 100 - healthy_pct
        
        # Classify
        if healthy_pct >= 65:
            risk = RiskLevel.HEALTHY
            is_clubbing = False
            msg = f"✅ HEALTHY - Clear diamond gap ({healthy_pct:.0f}%)"
            conf = "HIGH"
        elif healthy_pct >= 50:
            risk = RiskLevel.LOW_RISK
            is_clubbing = False
            msg = f"✅ LOW RISK - Good diamond gap ({healthy_pct:.0f}%)"
            conf = "MEDIUM"
        elif healthy_pct >= 35:
            risk = RiskLevel.BORDERLINE
            is_clubbing = False
            msg = f"⚠️ BORDERLINE - Unclear result ({healthy_pct:.0f}%)"
            conf = "LOW"
        elif healthy_pct >= 20:
            risk = RiskLevel.HIGH_RISK
            is_clubbing = True
            msg = f"⚠️ HIGH RISK - Reduced gap ({clubbing_pct:.0f}% clubbing)"
            conf = "MEDIUM"
        else:
            risk = RiskLevel.CLUBBING
            is_clubbing = True
            msg = f"❌ CLUBBING - No diamond gap ({clubbing_pct:.0f}% clubbing)"
            conf = "HIGH"
        
        # Recommendation
        if is_clubbing:
            rec = "Please consult a healthcare professional for evaluation."
        elif risk == RiskLevel.BORDERLINE:
            rec = "Consider retaking the photo in better lighting for more accurate results."
        else:
            rec = "No signs of clubbing detected. Continue monitoring."
        
        # Draw final result
        self._draw_final_result(marked, risk, healthy_pct, conf)
        
        return PipelineResult(
            step_reached=PipelineStep.COMPLETE,
            all_steps_passed=True,
            step_results=step_results,
            risk_level=risk,
            is_clubbing=is_clubbing,
            healthy_percentage=healthy_pct,
            clubbing_percentage=clubbing_pct,
            confidence_label=conf,
            message=msg,
            recommendation=rec,
            requires_retake=False,
            marked_image=marked,
            debug_info=debug
        )
    
    # ========================================================================
    # PIPELINE STEP IMPLEMENTATIONS
    # ========================================================================
    
    def _step1_hands_check(self, image: np.ndarray, marked: np.ndarray, 
                           w: int, h: int) -> Dict:
        """Step 1: Check if exactly 2 hands are detected"""
        # Resize for better MediaPipe detection
        max_dim = 1024
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            detect_img = cv2.resize(image, (int(w*scale), int(h*scale)))
        else:
            detect_img = image
            scale = 1.0
        
        rgb = cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        n_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
        
        if n_hands == 0:
            return {"passed": False, "reason": "No hands detected", "count": 0}
        elif n_hands == 1:
            return {"passed": False, "reason": "Only 1 hand detected (need 2)", "count": 1}
        
        # Success - got 2 hands
        return {
            "passed": True, 
            "count": 2,
            "hand1": results.multi_hand_landmarks[0].landmark,
            "hand2": results.multi_hand_landmarks[1].landmark,
            "scale": scale
        }
    
    def _step2_fingers_check(self, hand1, hand2, marked: np.ndarray, 
                              w: int, h: int) -> Dict:
        """
        Step 2: Find touching fingertips - MUST be INDEX fingers.
        
        The Schamroth test REQUIRES INDEX fingers only. If other fingers
        (middle, ring, etc.) are touching instead, reject the image.
        """
        INDEX_TIP = 8  # MediaPipe landmark for index fingertip
        
        # Collect ALL fingertips from BOTH hands
        all_tips = []
        
        for tip_idx, tip_name in self.FINGERTIPS.items():
            if tip_idx == 4:  # Skip thumbs
                continue
            p1 = (int(hand1[tip_idx].x * w), int(hand1[tip_idx].y * h))
            all_tips.append({'pos': p1, 'name': tip_name, 'hand': 1, 'idx': tip_idx})
            p2 = (int(hand2[tip_idx].x * w), int(hand2[tip_idx].y * h))
            all_tips.append({'pos': p2, 'name': tip_name, 'hand': 2, 'idx': tip_idx})
        
        # Find the CLOSEST pair of fingertips (must be from different hands)
        min_dist = float('inf')
        closest_pair = None
        
        for i, t1 in enumerate(all_tips):
            for t2 in all_tips[i+1:]:
                if t1['hand'] == t2['hand']:
                    continue
                dist = np.sqrt((t1['pos'][0]-t2['pos'][0])**2 + 
                              (t1['pos'][1]-t2['pos'][1])**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_pair = (t1, t2, dist)
        
        if closest_pair is None:
            return {"passed": False, "reason": "Could not find fingertip pair"}
        
        t1, t2, dist = closest_pair
        
        # Draw all fingertips
        for tip in all_tips:
            color = (0, 165, 255) if tip['idx'] == INDEX_TIP else (128, 128, 128)
            cv2.circle(marked, tip['pos'], 6, color, -1)
        
        # Max distance for touching = 30% of smaller dimension
        max_dist = min(w, h) * 0.30
        
        # ===================================================================
        # CHECK: Are the touching fingers INDEX fingers?
        # ===================================================================
        is_index_1 = (t1['idx'] == INDEX_TIP)
        is_index_2 = (t2['idx'] == INDEX_TIP)
        both_index = is_index_1 and is_index_2
        
        # Highlight the closest pair
        color = (0, 255, 0) if both_index else (0, 0, 255)  # Green if index, red if not
        cv2.circle(marked, t1['pos'], 14, color, -1)
        cv2.circle(marked, t2['pos'], 14, color, -1)
        cv2.line(marked, t1['pos'], t2['pos'], color, 3)
        cv2.putText(marked, t1['name'], (t1['pos'][0]-20, t1['pos'][1]-18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(marked, t2['name'], (t2['pos'][0]-20, t2['pos'][1]-18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # If NOT index fingers - REJECT!
        if not both_index:
            msg = f"{t1['name']} & {t2['name']} fingers touching, NOT Index!"
            cv2.putText(marked, "WRONG FINGERS!", (10, marked.shape[0]-60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(marked, "Touch INDEX fingers only", (10, marked.shape[0]-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            return {
                "passed": False,
                "reason": msg + " Schamroth test requires INDEX fingers touching nail-to-nail.",
                "distance": dist,
                "detected_fingers": (t1['name'], t2['name']),
                "tip1": t1['pos'],
                "tip2": t2['pos']
            }
        
        # Check distance
        if dist > max_dist:
            return {
                "passed": False,
                "reason": f"Index fingers too far apart ({dist:.0f}px). Touch fingertips together!",
                "distance": dist,
                "max_allowed": max_dist,
                "tip1": t1['pos'],
                "tip2": t2['pos']
            }
        
        # SUCCESS - Index fingers are touching!
        cv2.circle(marked, t1['pos'], 18, (0, 255, 0), 2)
        cv2.circle(marked, t2['pos'], 18, (0, 255, 0), 2)
        cv2.putText(marked, "INDEX FINGERS OK", (10, marked.shape[0]-30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return {
            "passed": True,
            "distance": dist,
            "tip1": t1['pos'],
            "tip2": t2['pos'],
            "name1": "Index",
            "name2": "Index",
            "idx1": INDEX_TIP,
            "idx2": INDEX_TIP
        }
    
    def _step3_geometry_check(self, hand1, hand2, fingers: Dict, 
                               marked: np.ndarray, w: int, h: int) -> Dict:
        """Step 3: Verify diamond geometry between fingertips"""
        # Get DIP joints (one below fingertip)
        dip1_idx = self.DIP_JOINTS.get(fingers['idx1'], fingers['idx1'] - 1)
        dip2_idx = self.DIP_JOINTS.get(fingers['idx2'], fingers['idx2'] - 1)
        
        dip1 = (int(hand1[dip1_idx].x * w), int(hand1[dip1_idx].y * h))
        dip2 = (int(hand2[dip2_idx].x * w), int(hand2[dip2_idx].y * h))
        
        # Diamond points: tip1 -> dip1 -> tip2 -> dip2
        diamond = np.array([
            fingers['tip1'],
            dip1,
            fingers['tip2'],
            dip2
        ], dtype=np.int32)
        
        # Calculate diamond area
        area = cv2.contourArea(diamond)
        
        # Minimum area check
        min_area = (fingers['distance'] * 0.3) ** 2
        
        if area < min_area:
            return {
                "passed": False,
                "reason": f"Diamond too small ({area:.0f}px² < {min_area:.0f}px²)",
                "area": area
            }
        
        # Calculate center
        center = (
            (fingers['tip1'][0] + fingers['tip2'][0]) // 2,
            (fingers['tip1'][1] + fingers['tip2'][1]) // 2
        )
        
        # Draw diamond
        cv2.circle(marked, dip1, 8, (255, 255, 0), -1)
        cv2.circle(marked, dip2, 8, (255, 255, 0), -1)
        cv2.polylines(marked, [diamond], True, (255, 0, 255), 3)
        cv2.circle(marked, center, 10, (255, 0, 255), -1)
        
        return {
            "passed": True,
            "area": area,
            "center": center,
            "diamond": diamond,
            "dip1": dip1,
            "dip2": dip2,
            "finger_dist": fingers['distance']  # Pass distance for gap ROI sizing
        }
    
    def _step4_gap_analysis(self, image: np.ndarray, geometry: Dict,
                            marked: np.ndarray, w: int, h: int) -> Dict:
        """
        Step 4: Analyze the GAP region for presence of diamond window.
        
        KEY INSIGHT: The Schamroth gap is the SMALL diamond-shaped space 
        RIGHT BETWEEN the touching fingertips - NOT the large area including DIP joints.
        We focus on a SMALL ROI centered exactly at the fingertip meeting point.
        """
        center = geometry['center']
        finger_distance = geometry.get('finger_dist', 60)
        
        # ===================================================================
        # CRITICAL: Use SMALL ROI focused on the exact gap between fingertips
        # The gap is typically only 5-15% of the finger distance
        # ===================================================================
        gap_roi_size = max(20, int(finger_distance * 0.8))  # Small, focused on gap
        
        x1 = max(0, center[0] - gap_roi_size)
        y1 = max(0, center[1] - gap_roi_size)
        x2 = min(w, center[0] + gap_roi_size)
        y2 = min(h, center[1] + gap_roi_size)
        
        roi = image[y1:y2, x1:x2]
        
        # Draw ROI with different color to show it's the GAP area
        cv2.rectangle(marked, (x1, y1), (x2, y2), (0, 255, 255), 3)  # Cyan = gap ROI
        cv2.putText(marked, "GAP ROI", (x1, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        if roi.size == 0:
            return {"score": 0.5, "reason": "Could not extract gap ROI"}
        
        # ===================================================================
        # ALGORITHM ALIGNMENT (Step 7 from Paper): PREPROCESSING
        # 1. Grayscale
        # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # 3. Gaussian Blur (noise reduction)
        # ===================================================================
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Apply Gaussian Blur
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        # ===================================================================
        # ALGORITHM ALIGNMENT (Step 8/9 from Paper): GAP SCORING
        # Analyze brightness in the diamond region
        # ===================================================================
        
        # Define diamond region mask for center of ROI
        h_roi, w_roi = roi.shape[:2]
        center_y, center_x = h_roi // 2, w_roi // 2
        
        mask = np.zeros((h_roi, w_roi), dtype=np.uint8)
        pts = np.array([
            [center_x, 0],      # Top
            [w_roi, center_y],  # Right
            [center_x, h_roi],  # Bottom
            [0, center_y]       # Left
        ], np.int32)
        cv2.fillPoly(mask, [pts], 255)
        
        # Extract pixels strictly within diamond mask
        masked_roi = cv2.bitwise_and(blurred, blurred, mask=mask)
        diamond_pixels = blurred[mask == 255]
        
        if diamond_pixels.size == 0:
            return {"score": 0.0, "reason": "Empty region", "brightness_score": 0.0}
            
        # Analysis Logic:
        # Healthy (Gap) = Center should be DARK (background showing through)
        # Clubbing (No Gap) = Center should be BRIGHT (skin filling gap)
        
        avg_brightness = np.mean(diamond_pixels)
        
        # Heuristic: Dark pixels are those < 80% of average brightness
        # This adapts to different lighting conditions
        dark_thresh = avg_brightness * 0.8
        dark_pixel_count = np.sum(diamond_pixels < dark_thresh)
        total_pixel_count = diamond_pixels.size
        
        # Gap Ratio = Percentage of DARK pixels in the diamond region
        # High ratio = Big dark gap = Healthy
        # Low ratio = Mostly bright skin = Clubbing
        gap_ratio = dark_pixel_count / total_pixel_count if total_pixel_count > 0 else 0
        
        # Normalize and boost score for better spread
        # Usually gap_ratio is 0.0 to 0.4. Map this to 0.0-1.0 range
        # Empirical scaling based on testing
        final_score = min(1.0, gap_ratio * 3.5) # Increased multiplier for sensitivity
        
        return {
            "score": float(final_score),
            "brightness_score": float(avg_brightness),
            "raw_ratio": float(gap_ratio),
            "roi_size": gap_roi_size,
            "finger_dist": finger_distance,
            "clahe_applied": True
        }
    
    # ========================================================================
    # VISUALIZATION HELPERS
    # ========================================================================
    
    def _draw_step_success(self, img: np.ndarray, step: int, text: str):
        """Draw step success indicator"""
        y = 25 + (step - 1) * 25
        cv2.putText(img, f"S{step}: {text}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def _draw_step_failure(self, img: np.ndarray, step: int, title: str, reason: str):
        """Draw step failure with details"""
        # Dark overlay
        cv2.rectangle(img, (5, 5), (400, 120), (0, 0, 0), -1)
        cv2.rectangle(img, (5, 5), (400, 120), (0, 0, 255), 3)
        
        cv2.putText(img, f"STEP {step} FAILED", (15, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, title, (15, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, reason[:45], (15, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        cv2.putText(img, "Please retake photo", (15, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _draw_final_result(self, img: np.ndarray, risk: RiskLevel, 
                           healthy_pct: float, conf: str):
        """Draw final result box"""
        colors = {
            RiskLevel.HEALTHY: (0, 200, 0),
            RiskLevel.LOW_RISK: (0, 200, 100),
            RiskLevel.BORDERLINE: (0, 180, 255),
            RiskLevel.HIGH_RISK: (0, 100, 255),
            RiskLevel.CLUBBING: (0, 0, 255),
        }
        color = colors.get(risk, (200, 200, 200))
        
        h = img.shape[0]
        cv2.rectangle(img, (5, h-80), (280, h-5), (0, 0, 0), -1)
        cv2.rectangle(img, (5, h-80), (280, h-5), color, 3)
        
        cv2.putText(img, "RESULT:", (15, h-55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(img, risk.value.upper().replace("_", " "), (90, h-55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(img, f"Healthy: {healthy_pct:.0f}% | Confidence: {conf}", (15, h-25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    def _error_result(self, msg: str) -> PipelineResult:
        """Create error result"""
        return PipelineResult(
            step_reached=PipelineStep.HANDS_CHECK,
            all_steps_passed=False,
            message=f"Error: {msg}",
            recommendation="Please try again with a valid image.",
            requires_retake=True
        )
    
    def process_file(self, path: str) -> PipelineResult:
        """Process image file"""
        img = cv2.imread(path)
        if img is None:
            return self._error_result(f"Could not load: {path}")
        return self.detect(img)
    
    def __del__(self):
        if hasattr(self, 'hands'):
            self.hands.close()


# Aliases for compatibility
SchamrothDetector = SchamrothPipelineDetector


# Test function
def detect_clubbing(image_or_path) -> PipelineResult:
    detector = SchamrothPipelineDetector()
    if isinstance(image_or_path, str):
        return detector.process_file(image_or_path)
    return detector.detect(image_or_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = detect_clubbing(sys.argv[1])
        print(f"Pipeline reached: {result.step_reached.value}")
        print(f"All steps passed: {result.all_steps_passed}")
        if result.all_steps_passed:
            print(f"Risk Level: {result.risk_level.value}")
            print(f"Healthy: {result.healthy_percentage:.1f}%")
        print(f"Message: {result.message}")
