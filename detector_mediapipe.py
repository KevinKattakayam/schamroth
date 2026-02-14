"""
Schamroth Window Detection - MediaPipe 2-Hand Edition
Strictly requires both hands to be detected for accurate results.

Requirements:
1. MUST detect 2 hands via MediaPipe
2. Index fingers must be TOUCHING (nail-to-nail)
3. Analyzes diamond gap between touching fingertips
4. Clear error when requirements not met
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SchamrothMediaPipe")


class RiskLevel(Enum):
    HEALTHY = "healthy"
    MOSTLY_HEALTHY = "mostly_healthy"
    BORDERLINE = "borderline"  
    LIKELY_CLUBBED = "likely_clubbed"
    CLUBBED = "clubbed"
    ERROR = "error"


@dataclass
class DetectionResult:
    """Complete detection result"""
    risk_level: RiskLevel
    is_clubbing: bool
    healthy_percentage: float
    clubbing_percentage: float
    confidence: str
    message: str
    recommendation: str
    hands_detected: int
    requires_retake: bool
    debug_info: Dict = field(default_factory=dict)
    marked_image: np.ndarray = field(default_factory=lambda: np.array([]))
    
    @property
    def is_valid(self) -> bool:
        return self.risk_level != RiskLevel.ERROR and not self.requires_retake


class SchamrothMediaPipeDetector:
    """
    Strict 2-Hand MediaPipe Detector
    
    Only gives results when BOTH hands are properly detected.
    Clear guidance when positioning is incorrect.
    """
    
    # MediaPipe landmark indices
    INDEX_TIP = 8
    INDEX_DIP = 7
    INDEX_PIP = 6
    
    def __init__(self):
        """Initialize MediaPipe hands"""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.3,  # Lower for better detection
            min_tracking_confidence=0.3
        )
        logger.info("SchamrothMediaPipeDetector initialized")
    
    def detect(self, image: np.ndarray) -> DetectionResult:
        """
        Detect Schamroth window with strict 2-hand requirement.
        
        Returns error result with clear guidance if 2 hands not detected.
        """
        if image is None or image.size == 0:
            return self._error("Invalid image")
        
        h, w = image.shape[:2]
        marked = image.copy()
        debug = {"image_size": (w, h)}
        
        # Resize large images for better MediaPipe performance
        max_dim = 1024
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            detect_img = cv2.resize(image, (int(w*scale), int(h*scale)))
        else:
            detect_img = image
            scale = 1.0
        
        # Run MediaPipe
        rgb = cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        # Count hands
        n_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
        debug["hands_detected"] = n_hands
        
        # ==================================================
        # STRICT REQUIREMENT: Must detect exactly 2 hands
        # ==================================================
        if n_hands < 2:
            return self._not_enough_hands(n_hands, marked, debug)
        
        # Get both hands
        hand1 = results.multi_hand_landmarks[0].landmark
        hand2 = results.multi_hand_landmarks[1].landmark
        
        # All fingertip landmarks
        FINGERTIPS = {4: 'Thumb', 8: 'Index', 12: 'Middle', 16: 'Ring', 20: 'Pinky'}
        DIP_JOINTS = {4: 3, 8: 7, 12: 11, 16: 15, 20: 19}  # DIP joint for each finger
        
        # Find the CLOSEST pair of fingertips between the two hands
        min_dist = float('inf')
        best_pair = None
        
        for tip_idx, name1 in FINGERTIPS.items():
            for tip_idx2, name2 in FINGERTIPS.items():
                p1 = self._get_landmark(hand1[tip_idx], w, h)
                p2 = self._get_landmark(hand2[tip_idx2], w, h)
                dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                
                if dist < min_dist:
                    min_dist = dist
                    best_pair = {
                        'tip1': p1, 'tip2': p2,
                        'name1': name1, 'name2': name2,
                        'dip1': self._get_landmark(hand1[DIP_JOINTS[tip_idx]], w, h),
                        'dip2': self._get_landmark(hand2[DIP_JOINTS[tip_idx2]], w, h),
                        'dist': dist
                    }
        
        tip1 = best_pair['tip1']
        tip2 = best_pair['tip2']
        dip1 = best_pair['dip1']
        dip2 = best_pair['dip2']
        tip_distance = best_pair['dist']
        
        # Max distance = 25% of smaller image dimension (more lenient)
        max_distance = min(w, h) * 0.25
        
        debug["tip_distance"] = tip_distance
        debug["max_distance"] = max_distance
        debug["closest_fingers"] = f"{best_pair['name1']} <-> {best_pair['name2']}"
        
        # ==================================================
        # REQUIREMENT: Fingers must be TOUCHING
        # ==================================================
        if tip_distance > max_distance:
            return self._fingers_not_touching(tip1, tip2, tip_distance, marked, debug)
        
        # Draw detected points
        cv2.circle(marked, tip1, 10, (0, 255, 0), -1)
        cv2.circle(marked, tip2, 10, (0, 255, 0), -1)
        cv2.circle(marked, dip1, 6, (255, 0, 0), -1)
        cv2.circle(marked, dip2, 6, (255, 0, 0), -1)
        cv2.line(marked, tip1, tip2, (0, 255, 0), 2)
        
        # Calculate diamond center (between fingertips)
        center = ((tip1[0] + tip2[0]) // 2, (tip1[1] + tip2[1]) // 2)
        cv2.circle(marked, center, 15, (255, 0, 255), 3)
        
        # Extract ROI for gap analysis
        roi_size = int(tip_distance * 1.5) or 100
        x1 = max(0, center[0] - roi_size)
        y1 = max(0, center[1] - roi_size)
        x2 = min(w, center[0] + roi_size)
        y2 = min(h, center[1] + roi_size)
        
        cv2.rectangle(marked, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return self._error("Could not extract ROI")
        
        # ==================================================
        # ANALYZE DIAMOND GAP
        # ==================================================
        gap_score = self._analyze_gap(roi)
        debug["gap_score"] = gap_score
        
        # Convert to percentages
        healthy_pct = gap_score * 100
        clubbing_pct = 100 - healthy_pct
        
        # Classify
        if healthy_pct >= 70:
            risk = RiskLevel.HEALTHY
            is_clubbing = False
            msg = f"✅ HEALTHY - Clear diamond gap detected ({healthy_pct:.0f}% healthy)"
        elif healthy_pct >= 55:
            risk = RiskLevel.MOSTLY_HEALTHY
            is_clubbing = False
            msg = f"✅ MOSTLY HEALTHY - Good diamond gap ({healthy_pct:.0f}% healthy)"
        elif healthy_pct >= 40:
            risk = RiskLevel.BORDERLINE
            is_clubbing = False
            msg = f"⚠️ BORDERLINE - Unclear result ({healthy_pct:.0f}% healthy)"
        elif healthy_pct >= 25:
            risk = RiskLevel.LIKELY_CLUBBED
            is_clubbing = True
            msg = f"⚠️ LIKELY CLUBBING - Reduced gap ({clubbing_pct:.0f}% clubbing indicators)"
        else:
            risk = RiskLevel.CLUBBED
            is_clubbing = True
            msg = f"❌ CLUBBING DETECTED - No diamond gap ({clubbing_pct:.0f}% clubbing indicators)"
        
        # Confidence based on clarity
        if healthy_pct > 75 or healthy_pct < 25:
            confidence = "HIGH"
        elif healthy_pct > 60 or healthy_pct < 40:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        # Recommendation
        if risk == RiskLevel.HEALTHY:
            rec = "No signs of clubbing. Continue monitoring."
        elif risk == RiskLevel.CLUBBED or risk == RiskLevel.LIKELY_CLUBBED:
            rec = "Please consult a healthcare professional for evaluation."
        else:
            rec = "Consider retaking the photo for more accurate results."
        
        # Annotate result
        self._annotate(marked, risk, healthy_pct, confidence)
        
        return DetectionResult(
            risk_level=risk,
            is_clubbing=is_clubbing,
            healthy_percentage=healthy_pct,
            clubbing_percentage=clubbing_pct,
            confidence=confidence,
            message=msg,
            recommendation=rec,
            hands_detected=2,
            requires_retake=False,
            debug_info=debug,
            marked_image=marked
        )
    
    def _get_landmark(self, lm, w: int, h: int) -> Tuple[int, int]:
        """Convert normalized landmark to pixel coords"""
        return (int(lm.x * w), int(lm.y * h))
    
    def _analyze_gap(self, roi: np.ndarray) -> float:
        """
        Analyze ROI for diamond gap presence.
        
        Returns 0-1 score (1 = clear gap = healthy, 0 = no gap = clubbing)
        """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 1. Center brightness analysis
        center_h, center_w = h // 2, w // 2
        size = min(h, w) // 4
        center_roi = gray[center_h-size:center_h+size, center_w-size:center_w+size]
        
        if center_roi.size == 0:
            return 0.5
        
        center_brightness = np.mean(center_roi)
        overall_brightness = np.mean(gray)
        
        # Gap should be brighter than surroundings
        brightness_ratio = center_brightness / max(overall_brightness, 1)
        brightness_score = min(1.0, max(0, (brightness_ratio - 0.8) / 0.4))
        
        # 2. Edge analysis (gap creates distinct edges)
        edges = cv2.Canny(gray, 50, 150)
        center_edges = edges[center_h-size:center_h+size, center_w-size:center_w+size]
        edge_density = np.sum(center_edges > 0) / max(center_edges.size, 1)
        edge_score = min(1.0, edge_density * 10)
        
        # 3. Contrast analysis
        std_dev = np.std(gray)
        contrast_score = min(1.0, std_dev / 50)
        
        # Combine scores
        gap_score = (
            brightness_score * 0.5 +
            edge_score * 0.3 +
            contrast_score * 0.2
        )
        
        return float(np.clip(gap_score, 0, 1))
    
    def _annotate(self, img: np.ndarray, risk: RiskLevel, 
                  healthy_pct: float, confidence: str) -> None:
        """Add visual annotations"""
        colors = {
            RiskLevel.HEALTHY: (0, 200, 0),
            RiskLevel.MOSTLY_HEALTHY: (100, 200, 100),
            RiskLevel.BORDERLINE: (0, 180, 255),
            RiskLevel.LIKELY_CLUBBED: (0, 100, 255),
            RiskLevel.CLUBBED: (0, 0, 255),
        }
        color = colors.get(risk, (200, 200, 200))
        
        cv2.rectangle(img, (5, 5), (250, 90), (0, 0, 0), -1)
        cv2.rectangle(img, (5, 5), (250, 90), color, 2)
        
        cv2.putText(img, f"2 HANDS DETECTED", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(img, risk.value.upper().replace("_", " "), (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(img, f"Healthy: {healthy_pct:.0f}% | Conf: {confidence}", (10, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def _not_enough_hands(self, n: int, marked: np.ndarray, debug: Dict) -> DetectionResult:
        """Return error when less than 2 hands detected"""
        cv2.rectangle(marked, (5, 5), (350, 160), (0, 0, 0), -1)
        cv2.rectangle(marked, (5, 5), (350, 160), (0, 0, 255), 3)
        
        cv2.putText(marked, f"ONLY {n} HAND DETECTED - NEED 2", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(marked, "Please retake with:", (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(marked, "1. BOTH hands visible in frame", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(marked, "2. Hands SEPARATED (not overlapping)", (10, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(marked, "3. Index fingers touching nail-to-nail", (10, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(marked, "4. Good lighting, camera above hands", (10, 155),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return DetectionResult(
            risk_level=RiskLevel.ERROR,
            is_clubbing=False,
            healthy_percentage=0,
            clubbing_percentage=0,
            confidence="NONE",
            message=f"⚠️ Only {n} hand detected - need 2 hands",
            recommendation="Please retake with both hands visible and separated. Keep fingers touching but hands apart.",
            hands_detected=n,
            requires_retake=True,
            debug_info=debug,
            marked_image=marked
        )
    
    def _fingers_not_touching(self, tip1, tip2, dist: float, 
                               marked: np.ndarray, debug: Dict) -> DetectionResult:
        """Return error when fingers are too far apart"""
        cv2.circle(marked, tip1, 15, (0, 0, 255), 3)
        cv2.circle(marked, tip2, 15, (0, 0, 255), 3)
        cv2.line(marked, tip1, tip2, (0, 0, 255), 2)
        
        cv2.rectangle(marked, (5, 5), (350, 120), (0, 0, 0), -1)
        cv2.rectangle(marked, (5, 5), (350, 120), (0, 128, 255), 3)
        
        cv2.putText(marked, "FINGERS NOT TOUCHING", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)
        cv2.putText(marked, f"Distance: {dist:.0f}px - too far apart", (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(marked, "Please retake with:", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(marked, "Index fingernails TOUCHING nail-to-nail", (10, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return DetectionResult(
            risk_level=RiskLevel.ERROR,
            is_clubbing=False,
            healthy_percentage=0,
            clubbing_percentage=0,
            confidence="NONE",
            message="⚠️ Fingers not touching - please bring index fingernails together",
            recommendation="Position your hands so index fingernails are touching nail-to-nail.",
            hands_detected=2,
            requires_retake=True,
            debug_info=debug,
            marked_image=marked
        )
    
    def _error(self, msg: str) -> DetectionResult:
        """Generic error result"""
        return DetectionResult(
            risk_level=RiskLevel.ERROR,
            is_clubbing=False,
            healthy_percentage=0,
            clubbing_percentage=0,
            confidence="NONE",
            message=f"Error: {msg}",
            recommendation="Please try again with a valid image.",
            hands_detected=0,
            requires_retake=True,
            debug_info={},
            marked_image=np.array([])
        )
    
    def process_file(self, path: str) -> DetectionResult:
        """Process image file"""
        img = cv2.imread(path)
        if img is None:
            return self._error(f"Could not load: {path}")
        return self.detect(img)
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'hands'):
            self.hands.close()


# Convenience function
def detect_clubbing(image_or_path) -> DetectionResult:
    """Quick detection function"""
    detector = SchamrothMediaPipeDetector()
    if isinstance(image_or_path, str):
        return detector.process_file(image_or_path)
    return detector.detect(image_or_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = detect_clubbing(sys.argv[1])
        print(f"Hands Detected: {result.hands_detected}")
        print(f"Requires Retake: {result.requires_retake}")
        if not result.requires_retake:
            print(f"Risk Level: {result.risk_level.value}")
            print(f"Healthy: {result.healthy_percentage:.1f}%")
            print(f"Message: {result.message}")
        else:
            print(f"Message: {result.message}")
            print(f"Recommendation: {result.recommendation}")
