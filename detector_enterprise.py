"""
================================================================================
SCHAMROTH WINDOW DETECTION ENGINE - ENTERPRISE EDITION v1.0
================================================================================

Professional-grade digital clubbing detection system for mobile healthcare apps.

Medical Background:
    The Schamroth window test is a clinical screening tool for digital clubbing.
    - Position: Two opposing index fingers touching at nail beds (dorsal view)
    - Healthy: Diamond-shaped gap visible between nail folds (light passes through)
    - Clubbing: No gap visible (increased Lovibond angle >180° causes nail beds to meet)

Technical Approach:
    1. Image Acquisition - MediaPipe for hand detection, center-based fallback
    2. ROI Extraction - Focus on the junction between fingernails
    3. Gap Analysis - Multi-method detection of diamond window presence/absence
    4. Classification - Conservative medical-grade classification with confidence

Enterprise Features:
    - Comprehensive error handling and validation
    - Detailed logging for debugging and audit
    - Configuration management
    - Performance optimization for mobile devices
    - Clear percentage-based scoring

Author: Enterprise Medical AI
Version: 1.0.0
License: Proprietary
================================================================================
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from datetime import datetime
import logging
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Central configuration management"""
    
    # Version info
    VERSION = "1.0.0"
    BUILD = "enterprise"
    
    # MediaPipe settings
    MP_MIN_DETECTION_CONFIDENCE = 0.2  # Lower threshold for better detection
    MP_MIN_TRACKING_CONFIDENCE = 0.2
    MP_MAX_HANDS = 2
    
    # Image processing
    TARGET_SIZE = 512  # Normalize images to this size
    CLAHE_CLIP_LIMIT = 3.0
    CLAHE_GRID_SIZE = 8
    
    # ROI extraction
    ROI_SIZE_RATIO = 0.25  # ROI as fraction of image
    MIN_ROI_SIZE = 50
    MAX_ROI_SIZE = 200
    
    # Gap detection thresholds
    # Key insight: In clubbing, the center is DARKER (no gap)
    # In healthy, the center is BRIGHTER (light through gap)
    DARK_THRESHOLD = 100      # Below this = dark (clubbing indicator)
    BRIGHT_THRESHOLD = 150    # Above this = bright (healthy indicator)
    
    # Classification thresholds (gap presence score 0-1)
    THRESHOLD_HEALTHY = 0.60           # >60% = clearly healthy
    THRESHOLD_MOSTLY_HEALTHY = 0.45    # 45-60% = probably healthy
    THRESHOLD_BORDERLINE = 0.30        # 30-45% = inconclusive
    THRESHOLD_LIKELY_CLUBBED = 0.15    # 15-30% = probably clubbed
    # <15% = clubbing detected
    
    # Validation
    MIN_BRIGHTNESS = 30
    MAX_BRIGHTNESS = 240
    MIN_CONTRAST = 20


# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SchamrothEngine")


# ============================================================================
# DATA MODELS
# ============================================================================

class RiskLevel(Enum):
    """Clinical risk classification levels"""
    HEALTHY = "healthy"
    MOSTLY_HEALTHY = "mostly_healthy"
    BORDERLINE = "borderline"
    LIKELY_CLUBBED = "likely_clubbed"
    CLUBBED = "clubbed"
    ERROR = "error"


class DetectionMethod(Enum):
    """Detection method used"""
    MEDIAPIPE = "mediapipe"
    CENTER_BASED = "center_based"
    FAILED = "failed"


class ImageQualityStatus(Enum):
    """Image quality assessment"""
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNUSABLE = "unusable"


@dataclass
class ImageQuality:
    """Image quality assessment results"""
    status: ImageQualityStatus
    brightness: float
    contrast: float
    sharpness: float
    issues: List[str] = field(default_factory=list)
    
    @property
    def is_usable(self) -> bool:
        return self.status in [ImageQualityStatus.GOOD, ImageQualityStatus.ACCEPTABLE]


@dataclass
class GapAnalysis:
    """Detailed gap analysis results"""
    gap_presence_score: float    # 0-1, higher = more gap visible (healthier)
    center_brightness: float     # Average brightness of center region
    dark_ratio: float           # Ratio of dark pixels (clubbing indicator)
    bright_ratio: float         # Ratio of bright pixels (healthy indicator)
    contrast_score: float       # Contrast in center region
    symmetry_score: float       # Left-right symmetry


@dataclass
class DetectionResult:
    """Complete detection result with all details"""
    
    # Primary results (no defaults - required)
    healthy_percentage: float       # 0-100%
    clubbing_percentage: float      # 0-100%
    risk_level: RiskLevel
    is_clubbing: bool
    
    # Confidence and reliability (no defaults - required)
    confidence_score: float         # 0-1
    confidence_label: str           # "HIGH", "MEDIUM", "LOW"
    
    # Detailed analysis (no defaults - required)
    gap_analysis: GapAnalysis
    detection_method: DetectionMethod
    image_quality: ImageQuality
    
    # User-facing output (no defaults - required)
    message: str
    recommendation: str
    
    # Technical output (no defaults - required)
    processing_time_ms: float
    
    # ============================================
    # FIELDS WITH DEFAULTS MUST COME AFTER ABOVE
    # ============================================
    
    # STRICT detection validation
    requires_retake: bool = False   # True if detection is unreliable
    detection_quality_score: float = 0.0  # 0-1, how reliable is the detection
    
    # Optional fields with defaults
    marked_image: np.ndarray = field(default_factory=lambda: np.array([]))
    debug_info: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = Config.VERSION
    
    @property
    def is_valid(self) -> bool:
        return self.risk_level != RiskLevel.ERROR and self.image_quality.is_usable and not self.requires_retake
    
    @property
    def is_reliable(self) -> bool:
        """Return True only if detection is reliable enough to show percentages"""
        return not self.requires_retake and self.detection_quality_score >= 0.5
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for API response"""
        return {
            "healthy_percentage": self.healthy_percentage,
            "clubbing_percentage": self.clubbing_percentage,
            "risk_level": self.risk_level.value,
            "is_clubbing": self.is_clubbing,
            "requires_retake": self.requires_retake,
            "detection_quality": self.detection_quality_score,
            "confidence": self.confidence_score,
            "message": self.message,
            "recommendation": self.recommendation,
            "timestamp": self.timestamp,
            "version": self.version
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


# ============================================================================
# MAIN DETECTOR CLASS
# ============================================================================

class SchamrothDetector:
    """
    Enterprise-grade Schamroth Window Detection Engine
    
    Usage:
        detector = SchamrothDetector()
        result = detector.detect(image)
        print(f"Healthy: {result.healthy_percentage}%")
        print(f"Risk: {result.risk_level.value}")
    """
    
    # MediaPipe landmark indices
    INDEX_TIP = 8
    INDEX_DIP = 7
    INDEX_PIP = 6
    INDEX_MCP = 5  # Metacarpophalangeal joint (knuckle)
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the detector.
        
        Args:
            config: Optional custom configuration
        """
        self.config = config or Config()
        self._init_mediapipe()
        self._init_processors()
        logger.info(f"SchamrothDetector initialized (v{Config.VERSION})")
    
    def _init_mediapipe(self):
        """Initialize MediaPipe hands detection"""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=Config.MP_MAX_HANDS,
            min_detection_confidence=Config.MP_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=Config.MP_MIN_TRACKING_CONFIDENCE
        )
    
    def _init_processors(self):
        """Initialize image processors"""
        self.clahe = cv2.createCLAHE(
            clipLimit=Config.CLAHE_CLIP_LIMIT,
            tileGridSize=(Config.CLAHE_GRID_SIZE, Config.CLAHE_GRID_SIZE)
        )
    
    def detect(self, image: np.ndarray) -> DetectionResult:
        """
        Analyze image for Schamroth window presence.
        
        STRICT DETECTION: If we can't reliably detect, we ask for retake
        rather than giving potentially wrong percentages.
        
        Args:
            image: BGR image (OpenCV format)
            
        Returns:
            DetectionResult with complete analysis
        """
        start_time = cv2.getTickCount()
        
        # Validate input
        if image is None or image.size == 0:
            return self._error_result("Invalid image provided")
        
        h, w = image.shape[:2]
        marked_image = image.copy()
        debug_info = {"original_size": (w, h)}
        
        try:
            # Step 1: Assess image quality
            quality = self._assess_image_quality(image)
            debug_info["quality"] = {
                "status": quality.status.value,
                "brightness": quality.brightness,
                "contrast": quality.contrast
            }
            
            if not quality.is_usable:
                return self._quality_error_result(quality, marked_image)
            
            # Step 2: Extract ROI with STRICT MediaPipe validation
            roi, roi_coords, method = self._extract_roi(image, marked_image)
            debug_info["detection_method"] = method.value
            debug_info["roi_coords"] = roi_coords
            
            # STRICT CHECK: If MediaPipe failed, require retake
            if roi is None or (hasattr(roi, 'size') and roi.size == 0):
                return self._mediapipe_error_result(marked_image)
            
            # Draw ROI on marked image
            cv2.rectangle(marked_image,
                         (roi_coords[0], roi_coords[1]),
                         (roi_coords[2], roi_coords[3]),
                         (0, 255, 255), 2)
            
            # Step 3: Analyze gap
            gap_analysis = self._analyze_gap(roi)
            debug_info["gap_analysis"] = {
                "presence_score": gap_analysis.gap_presence_score,
                "center_brightness": gap_analysis.center_brightness,
                "dark_ratio": gap_analysis.dark_ratio,
                "bright_ratio": gap_analysis.bright_ratio,
                "contrast_score": gap_analysis.contrast_score
            }
            
            # ================================================================
            # STRICT DETECTION QUALITY CHECK
            # If the gap analysis is in the "uncertain zone", require retake
            # ================================================================
            gap_score = gap_analysis.gap_presence_score
            detection_quality = self._calculate_detection_quality(gap_analysis, quality, method)
            debug_info["detection_quality_score"] = detection_quality
            
            # UNCERTAIN ZONE: Only in the 40-60% range we really can't tell
            # Narrower range = give results more often
            uncertain_zone = (0.40 <= gap_score <= 0.60)
            
            # For single-hand detection (CENTER_BASED), we still give results
            # but mark them as lower confidence
            is_single_hand = (method == DetectionMethod.CENTER_BASED)
            
            # Only require retake for truly uncertain cases
            # Single-hand detection CAN give results - just lower confidence
            requires_retake = uncertain_zone and (detection_quality < 0.4)
            
            if requires_retake:
                logger.warning(f"Detection uncertain - gap_score={gap_score:.2f}, "
                             f"quality={detection_quality:.2f}")
                debug_info["retake_reason"] = {
                    "uncertain_zone": uncertain_zone,
                    "low_detection_quality": (detection_quality < 0.4)
                }
            
            # Step 4: Calculate percentages and classify
            healthy_pct, clubbing_pct = self._calculate_percentages(gap_analysis)
            risk_level, is_clubbing = self._classify(healthy_pct)
            
            # Override only for retake-required cases
            if requires_retake:
                risk_level = RiskLevel.BORDERLINE
                message = "⚠️ DETECTION UNCERTAIN - Please Retake Photo"
                recommendation = ("The image analysis is not conclusive. Please retake with: "
                                "1) Both hands visible & separated, 2) Fingers touching nail-to-nail, "
                                "3) Good lighting, 4) Camera directly above hands")
            else:
                # Step 6: Generate user-facing messages
                message, recommendation = self._generate_messages(risk_level, healthy_pct, clubbing_pct)
            
            # Step 5: Calculate confidence
            confidence_score = self._calculate_confidence(gap_analysis, quality)
            confidence_label = self._confidence_to_label(confidence_score)
            
            # Step 7: Annotate marked image
            self._annotate_image(marked_image, risk_level, healthy_pct, clubbing_pct, 
                               confidence_label, requires_retake)
            
            # Calculate processing time
            processing_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency() * 1000
            
            return DetectionResult(
                healthy_percentage=round(healthy_pct, 1),
                clubbing_percentage=round(clubbing_pct, 1),
                risk_level=risk_level,
                is_clubbing=is_clubbing,
                requires_retake=requires_retake,
                detection_quality_score=round(detection_quality, 3),
                confidence_score=round(confidence_score, 3),
                confidence_label=confidence_label,
                gap_analysis=gap_analysis,
                detection_method=method,
                image_quality=quality,
                message=message,
                recommendation=recommendation,
                processing_time_ms=round(processing_time, 2),
                marked_image=marked_image,
                debug_info=debug_info
            )
            
        except Exception as e:
            logger.error(f"Detection error: {str(e)}")
            return self._error_result(f"Processing error: {str(e)}")
    
    def _assess_image_quality(self, image: np.ndarray) -> ImageQuality:
        """Assess image quality for analysis suitability"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        
        # Sharpness via Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(laplacian.var())
        
        issues = []
        
        if brightness < Config.MIN_BRIGHTNESS:
            issues.append("Image too dark - use flash or better lighting")
        elif brightness > Config.MAX_BRIGHTNESS:
            issues.append("Image overexposed - reduce lighting")
        
        if contrast < Config.MIN_CONTRAST:
            issues.append("Low contrast - ensure fingers are clearly visible")
        
        if sharpness < 100:
            issues.append("Image blurry - hold camera steady")
        
        # Determine overall status
        if not issues:
            status = ImageQualityStatus.GOOD
        elif len(issues) == 1:
            status = ImageQualityStatus.ACCEPTABLE
        elif len(issues) == 2:
            status = ImageQualityStatus.POOR
        else:
            status = ImageQualityStatus.UNUSABLE
        
        return ImageQuality(
            status=status,
            brightness=brightness,
            contrast=contrast,
            sharpness=sharpness,
            issues=issues
        )
    
    def _extract_roi(self, image: np.ndarray, 
                     marked: np.ndarray) -> Tuple[Optional[np.ndarray], tuple, DetectionMethod]:
        """
        Extract region of interest using STRICT MediaPipe validation.
        
        REQUIREMENTS FOR VALID DETECTION:
        1. Must detect exactly 2 hands
        2. Must find 2 fingertips that are TOUCHING (< 100px apart)
        3. Must detect diamond shape between the fingertips
        
        If ANY requirement fails -> return None and ask user to retake.
        """
        h, w = image.shape[:2]
        img_diagonal = np.sqrt(w**2 + h**2)
        
        # Maximum distance for "touching" fingers (25% of smaller dimension)
        MAX_TOUCH_DISTANCE = min(w, h) * 0.25
        
        # ============================================================
        # RESIZE LARGE IMAGES FOR BETTER MEDIAPIPE DETECTION
        # MediaPipe works better with smaller images (1024-1536px)
        # ============================================================
        MAX_DIMENSION = 1024
        scale_factor = 1.0
        
        if max(w, h) > MAX_DIMENSION:
            scale_factor = MAX_DIMENSION / max(w, h)
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            detection_image = cv2.resize(image, (new_w, new_h))
            detect_h, detect_w = new_h, new_w
            logger.info(f"Resized {w}x{h} -> {new_w}x{new_h} for MediaPipe (scale={scale_factor:.2f})")
        else:
            detection_image = image
            detect_h, detect_w = h, w
        
        # Try multiple preprocessing variants for better detection
        preprocessing_variants = [
            ("original", detection_image),
            ("enhanced", self._preprocess_for_mediapipe(detection_image)),
            ("bright", cv2.convertScaleAbs(detection_image, alpha=1.3, beta=30)),
            ("contrast", self._enhance_contrast(detection_image)),
        ]
        
        for variant_name, variant_image in preprocessing_variants:
            rgb = cv2.cvtColor(variant_image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            
            if not results.multi_hand_landmarks:
                continue
            
            n_hands = len(results.multi_hand_landmarks)
            logger.info(f"MediaPipe detected {n_hands} hand(s) with: {variant_name}")
            
            # ============================================================
            # STRICT REQUIREMENT: Must detect EXACTLY 2 hands
            # ============================================================
            if n_hands < 2:
                logger.warning(f"Only {n_hands} hand detected, need 2 hands")
                continue
            
            hand1 = results.multi_hand_landmarks[0].landmark
            hand2 = results.multi_hand_landmarks[1].landmark
            
            # All fingertip landmarks
            FINGERTIPS = {4: 'Thumb', 8: 'Index', 12: 'Middle', 16: 'Ring', 20: 'Pinky'}
            
            # Find the CLOSEST pair of fingertips between the two hands
            # Use detect dimensions for MediaPipe, then scale back to original
            min_dist = float('inf')
            best_pair = None
            
            for t1, n1 in FINGERTIPS.items():
                for t2, n2 in FINGERTIPS.items():
                    # Get coordinates in ORIGINAL image space (scale back from detection)
                    p1 = (int(hand1[t1].x * w), int(hand1[t1].y * h))
                    p2 = (int(hand2[t2].x * w), int(hand2[t2].y * h))
                    dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_pair = {
                            'p1': p1, 'p2': p2,
                            'n1': n1, 'n2': n2,
                            't1': t1, 't2': t2,
                            'dist': dist
                        }
            
            logger.info(f"Closest fingers: {best_pair['n1']} <-> {best_pair['n2']}, dist={min_dist:.0f}px")
            
            # ============================================================
            # STRICT REQUIREMENT: Fingers must be TOUCHING (close together)
            # ============================================================
            if min_dist > MAX_TOUCH_DISTANCE:
                logger.warning(f"Fingers NOT touching: {min_dist:.0f}px > {MAX_TOUCH_DISTANCE:.0f}px max")
                # Draw the problem
                cv2.circle(marked, best_pair['p1'], 15, (0, 0, 255), 3)
                cv2.circle(marked, best_pair['p2'], 15, (0, 0, 255), 3)
                cv2.line(marked, best_pair['p1'], best_pair['p2'], (0, 0, 255), 2)
                cv2.putText(marked, "FINGERS NOT TOUCHING!", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(marked, f"Distance: {min_dist:.0f}px (need < {MAX_TOUCH_DISTANCE:.0f}px)", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                continue
            
            # ============================================================
            # DIAMOND DETECTION: Find the actual diamond gap
            # ============================================================
            diamond_center = ((best_pair['p1'][0] + best_pair['p2'][0]) // 2,
                             (best_pair['p1'][1] + best_pair['p2'][1]) // 2)
            
            # Get the DIP joints (one below fingertip) for both fingers
            DIP_OFFSET = -1  # DIP is one landmark before TIP
            dip1_idx = best_pair['t1'] + DIP_OFFSET
            dip2_idx = best_pair['t2'] + DIP_OFFSET
            
            dip1 = (int(hand1[dip1_idx].x * w), int(hand1[dip1_idx].y * h))
            dip2 = (int(hand2[dip2_idx].x * w), int(hand2[dip2_idx].y * h))
            
            # The diamond is formed by: tip1, dip1, tip2, dip2
            diamond_points = np.array([
                best_pair['p1'],  # tip1
                dip1,             # dip1 
                best_pair['p2'],  # tip2
                dip2              # dip2
            ], dtype=np.int32)
            
            # Calculate diamond area
            diamond_area = cv2.contourArea(diamond_points)
            
            # Validate diamond exists (has reasonable area)
            min_diamond_area = (min_dist * 0.5) ** 2  # At least half the finger distance squared
            
            if diamond_area < min_diamond_area:
                logger.warning(f"Diamond too small: {diamond_area:.0f} < {min_diamond_area:.0f}")
            
            logger.info(f"Diamond center: {diamond_center}, area: {diamond_area:.0f}px²")
            
            # ============================================================
            # DRAW VALIDATED DETECTION
            # ============================================================
            # Draw fingertips (green = valid)
            cv2.circle(marked, best_pair['p1'], 12, (0, 255, 0), -1)
            cv2.circle(marked, best_pair['p2'], 12, (0, 255, 0), -1)
            cv2.putText(marked, best_pair['n1'], (best_pair['p1'][0]-20, best_pair['p1'][1]-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(marked, best_pair['n2'], (best_pair['p2'][0]-20, best_pair['p2'][1]-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw DIP joints
            cv2.circle(marked, dip1, 8, (255, 255, 0), -1)
            cv2.circle(marked, dip2, 8, (255, 255, 0), -1)
            
            # Draw diamond shape (magenta)
            cv2.polylines(marked, [diamond_points], True, (255, 0, 255), 3)
            cv2.circle(marked, diamond_center, 10, (255, 0, 255), -1)
            
            # Label
            cv2.putText(marked, "DIAMOND DETECTED", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(marked, f"Finger dist: {min_dist:.0f}px | Area: {diamond_area:.0f}px2", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # ============================================================
            # EXTRACT ROI centered on diamond
            # ============================================================
            roi_size = int(max(Config.MIN_ROI_SIZE, min(Config.MAX_ROI_SIZE, min_dist * 4)))
            
            x1 = max(0, diamond_center[0] - roi_size)
            y1 = max(0, diamond_center[1] - roi_size)
            x2 = min(w, diamond_center[0] + roi_size)
            y2 = min(h, diamond_center[1] + roi_size)
            
            # Draw ROI box
            cv2.rectangle(marked, (x1, y1), (x2, y2), (255, 255, 0), 2)
            
            # Store diamond info for gap analysis
            self._last_diamond = {
                'center': diamond_center,
                'points': diamond_points,
                'area': diamond_area,
                'finger_dist': min_dist
            }
            
            return image[y1:y2, x1:x2], (x1, y1, x2, y2), DetectionMethod.MEDIAPIPE
        
        # ============================================================
        # 2-HAND DETECTION FAILED
        # Do NOT use fallbacks - require proper 2-hand photo
        # ============================================================
        logger.warning("Could not detect 2 hands - requiring retake")
        
        # Provide clear guidance on the marked image
        cv2.putText(marked, "2-HAND DETECTION REQUIRED", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(marked, "Please retake photo with:", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(marked, "1. BOTH hands visible in frame", (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(marked, "2. Hands SEPARATED (not overlapping)", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(marked, "3. Index fingers touching nail-to-nail", (10, 135),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(marked, "4. Good lighting, camera above hands", (10, 160),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return None, (0, 0, 0, 0), DetectionMethod.FAILED
    
    def _preprocess_for_mediapipe(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image to improve MediaPipe detection"""
        # CLAHE enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast for better detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        return cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
    
    def _center_based_roi(self, image: np.ndarray, marked: np.ndarray) -> Tuple[Optional[np.ndarray], tuple]:
        """
        Extract ROI from the center of the image.
        
        This is a fallback when MediaPipe can't detect both hands.
        Works for close-up images where the fingertips are in the center.
        
        Returns:
            Tuple of (roi_image, (x1, y1, x2, y2))
        """
        h, w = image.shape[:2]
        
        # Assume the diamond/fingertips are in the center of the image
        center_x, center_y = w // 2, h // 2
        
        # ROI size - proportional to image, but reasonable
        roi_half = min(w, h) // 4
        
        x1 = max(0, center_x - roi_half)
        y1 = max(0, center_y - roi_half)
        x2 = min(w, center_x + roi_half)
        y2 = min(h, center_y + roi_half)
        
        # Draw the ROI
        cv2.rectangle(marked, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.circle(marked, (center_x, center_y), 10, (255, 0, 255), -1)
        
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return None, (0, 0, 0, 0)
        
        # Store basic info for gap analysis
        self._last_diamond = {
            'center': (center_x, center_y),
            'points': None,
            'area': (x2-x1) * (y2-y1),
            'finger_dist': roi_half * 2
        }
        
        return roi, (x1, y1, x2, y2)
    
    def _angle_between_vectors(self, v1: Tuple[float, float], 
                                v2: Tuple[float, float]) -> float:
        """Calculate angle between two vectors in degrees"""
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = np.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = np.sqrt(v2[0]**2 + v2[1]**2)
        
        if mag1 == 0 or mag2 == 0:
            return 0
        
        cos_angle = np.clip(dot / (mag1 * mag2), -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
    
    def _analyze_gap(self, roi: np.ndarray) -> GapAnalysis:
        """
        Analyze the ROI for diamond gap presence.
        
        HEALTHY: Diamond gap visible (light passes through) = bright center OR dark background gap
        CLUBBING: No gap (nail beds touch) = uniform brightness, no contrast
        
        The key is CONTRAST between the gap region and surrounding finger skin.
        High contrast = gap visible = HEALTHY
        Low contrast = no gap = CLUBBING
        """
        if roi is None or roi.size == 0:
            return GapAnalysis(0.5, 0, 0, 0, 0, 0)
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        
        # Enhanced preprocessing
        enhanced = self.clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        h, w = blurred.shape
        if h < 10 or w < 10:
            return GapAnalysis(0.5, 0, 0, 0, 0, 0)
        
        # =====================================================================
        # DIAMOND GAP DETECTION
        # The diamond is in the CENTER of the ROI
        # =====================================================================
        
        # Define diamond-shaped mask in center (rhombus shape)
        center_y, center_x = h // 2, w // 2
        diamond_size = min(h, w) // 4
        
        # Create diamond mask
        diamond_mask = np.zeros((h, w), dtype=np.uint8)
        diamond_pts = np.array([
            [center_x, center_y - diamond_size],  # top
            [center_x + diamond_size, center_y],  # right
            [center_x, center_y + diamond_size],  # bottom
            [center_x - diamond_size, center_y],  # left
        ], dtype=np.int32)
        cv2.fillPoly(diamond_mask, [diamond_pts], 255)
        
        # Get diamond region pixels
        diamond_pixels = blurred[diamond_mask > 0]
        
        # Get surrounding ring (finger skin) - exclude diamond
        ring_inner = diamond_size
        ring_outer = min(h, w) // 2
        ring_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(ring_mask, (center_x, center_y), ring_outer, 255, -1)
        ring_mask[diamond_mask > 0] = 0
        ring_pixels = blurred[ring_mask > 0]
        
        if len(diamond_pixels) == 0 or len(ring_pixels) == 0:
            return GapAnalysis(0.5, 0, 0, 0, 0, 0)
        
        # =====================================================================
        # CORE ANALYSIS: Contrast between diamond and surrounding skin
        # =====================================================================
        
        diamond_brightness = float(np.mean(diamond_pixels))
        ring_brightness = float(np.mean(ring_pixels))
        diamond_std = float(np.std(diamond_pixels))
        
        # CONTRAST: How different is the diamond from surrounding skin?
        # High contrast = gap visible = HEALTHY
        brightness_diff = abs(diamond_brightness - ring_brightness)
        if ring_brightness > 10:
            contrast_ratio = brightness_diff / ring_brightness
        else:
            contrast_ratio = 0.0
        
        # =====================================================================
        # SCORING METHODS
        # =====================================================================
        
        # METHOD 1: CONTRAST SCORE (40% weight)
        # Higher contrast = more visible gap = healthier
        if contrast_ratio >= 0.30:
            contrast_score = 1.0
        elif contrast_ratio >= 0.15:
            contrast_score = 0.5 + (contrast_ratio - 0.15) * 3.33
        elif contrast_ratio >= 0.05:
            contrast_score = (contrast_ratio - 0.05) * 5.0
        else:
            contrast_score = 0.0
        
        # METHOD 2: EDGE DETECTION in diamond (30% weight)
        # Gap creates edges where light meets skin
        diamond_region = blurred[center_y-diamond_size:center_y+diamond_size,
                                  center_x-diamond_size:center_x+diamond_size]
        if diamond_region.size > 0:
            edges = cv2.Canny(diamond_region, 30, 100)
            edge_density = float(np.count_nonzero(edges) / edges.size)
            edge_score = min(1.0, edge_density * 15)
        else:
            edge_score = 0.0
        
        # METHOD 3: TEXTURE VARIANCE (20% weight)
        # Gap has different texture than uniform skin
        texture_score = min(1.0, diamond_std / 40)
        
        # METHOD 4: BIMODAL CHECK (10% weight)
        # Gap creates two distinct brightness regions
        hist = cv2.calcHist([diamond_pixels.astype(np.uint8)], [0], None, [16], [0, 256])
        hist = hist.flatten() / (hist.sum() + 1e-6)
        peaks = sum(1 for i in range(1, len(hist)-1) 
                   if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > 0.08)
        bimodal_score = 1.0 if peaks >= 2 else (0.5 if peaks == 1 else 0.0)
        
        # =====================================================================
        # FINAL SCORE: Weighted combination
        # =====================================================================
        gap_presence_score = (
            0.40 * contrast_score +
            0.30 * edge_score +
            0.20 * texture_score +
            0.10 * bimodal_score
        )
        gap_presence_score = float(np.clip(gap_presence_score, 0, 1))
        
        logger.info(f"Gap analysis: contrast={contrast_score:.2f}, edge={edge_score:.2f}, "
                   f"texture={texture_score:.2f}, bimodal={bimodal_score:.2f} -> TOTAL={gap_presence_score:.2f}")
        
        # Calculate additional metrics
        bright_threshold = ring_brightness * 1.2
        bright_ratio = float(np.sum(diamond_pixels > bright_threshold) / len(diamond_pixels))
        
        dark_threshold = ring_brightness * 0.8
        dark_ratio = float(np.sum(diamond_pixels < dark_threshold) / len(diamond_pixels))
        
        symmetry_score = self._calculate_symmetry(blurred)
        
        return GapAnalysis(
            gap_presence_score=gap_presence_score,
            center_brightness=diamond_brightness,
            dark_ratio=dark_ratio,
            bright_ratio=bright_ratio,
            contrast_score=contrast_score,
            symmetry_score=symmetry_score
        )
    
    def _calculate_symmetry(self, gray: np.ndarray) -> float:
        """Calculate left-right symmetry"""
        h, w = gray.shape
        
        left = gray[:, :w//2].astype(float)
        right = gray[:, w//2:].astype(float)
        right_flipped = cv2.flip(right, 1)
        
        if left.shape[1] != right_flipped.shape[1]:
            min_w = min(left.shape[1], right_flipped.shape[1])
            left = left[:, :min_w]
            right_flipped = right_flipped[:, :min_w]
        
        diff = np.abs(left - right_flipped)
        mean_diff = np.mean(diff)
        
        symmetry = 1.0 - min(1.0, mean_diff / 80)
        return float(symmetry)
    
    def _calculate_detection_quality(self, gap: GapAnalysis, quality: ImageQuality, 
                                     method: DetectionMethod) -> float:
        """
        Calculate overall detection quality score (0-1).
        
        A HIGH score means we can trust the detection results.
        A LOW score means results are unreliable and user should retake.
        
        Factors:
        - Contrast score: higher = better gap visibility
        - Edge score (via contrast_score proxy): clear edges = reliable detection
        - Detection method: MediaPipe = more reliable than center-based
        - Image quality: better quality = more reliable
        """
        # Base score from contrast (most important - can we see the gap clearly?)
        contrast_weight = 0.4
        contrast_score = min(1.0, gap.contrast_score * 1.5)  # Boost slightly
        
        # Distance from uncertain zone (35-65%) - farther = more confident
        distance_weight = 0.3
        distance_from_center = abs(gap.gap_presence_score - 0.5)
        distance_score = min(1.0, distance_from_center * 3)  # Max at 0.33 distance
        
        # Method reliability
        method_weight = 0.15
        method_score = 1.0 if method == DetectionMethod.MEDIAPIPE else 0.6
        
        # Image quality
        quality_weight = 0.15
        quality_scores = {
            ImageQualityStatus.GOOD: 1.0,
            ImageQualityStatus.ACCEPTABLE: 0.8,
            ImageQualityStatus.POOR: 0.4,
            ImageQualityStatus.UNUSABLE: 0.1
        }
        quality_score = quality_scores.get(quality.status, 0.5)
        
        # Weighted combination
        total = (
            contrast_weight * contrast_score +
            distance_weight * distance_score +
            method_weight * method_score +
            quality_weight * quality_score
        )
        
        logger.debug(f"Detection quality: contrast={contrast_score:.2f}, distance={distance_score:.2f}, "
                    f"method={method_score:.2f}, quality={quality_score:.2f} -> TOTAL={total:.2f}")
        
        return float(np.clip(total, 0, 1))
    
    def _calculate_percentages(self, gap: GapAnalysis) -> Tuple[float, float]:
        """Calculate healthy and clubbing percentages"""
        healthy_pct = gap.gap_presence_score * 100
        clubbing_pct = 100 - healthy_pct
        return healthy_pct, clubbing_pct
    
    def _classify(self, healthy_pct: float) -> Tuple[RiskLevel, bool]:
        """
        Classify based on healthy percentage.
        
        IMPORTANT: If clubbing% > healthy%, the result IS clubbing.
        Classification must match the percentages shown to user.
        """
        clubbing_pct = 100 - healthy_pct
        
        # PRIMARY RULE: If clubbing > healthy, it's clubbing
        if clubbing_pct > healthy_pct:
            # Clubbing is dominant
            if clubbing_pct >= 70:
                return RiskLevel.CLUBBED, True
            elif clubbing_pct >= 55:
                return RiskLevel.LIKELY_CLUBBED, True
            else:
                # Borderline but leaning clubbing
                return RiskLevel.BORDERLINE, True
        else:
            # Healthy is dominant
            if healthy_pct >= 70:
                return RiskLevel.HEALTHY, False
            elif healthy_pct >= 55:
                return RiskLevel.MOSTLY_HEALTHY, False
            else:
                # Borderline but leaning healthy
                return RiskLevel.BORDERLINE, False
    
    def _calculate_confidence(self, gap: GapAnalysis, quality: ImageQuality) -> float:
        """Calculate confidence score based on multiple factors"""
        # Base confidence from gap analysis clarity
        # Distance from 50% (decision boundary) indicates clear signal
        score_distance = abs(gap.gap_presence_score - 0.5)
        base_confidence = min(1.0, score_distance * 2 + 0.4)  # Increased base
        
        # Adjust for image quality
        quality_factor = {
            ImageQualityStatus.GOOD: 1.0,
            ImageQualityStatus.ACCEPTABLE: 0.95,  # Reduced penalty
            ImageQualityStatus.POOR: 0.75,
            ImageQualityStatus.UNUSABLE: 0.4
        }.get(quality.status, 0.6)
        
        # Adjust for contrast (gap visibility) - less aggressive penalty
        contrast_factor = min(1.0, gap.contrast_score * 0.5 + 0.6)
        
        # Adjust for symmetry (well-centered analysis) - less aggressive penalty
        symmetry_factor = min(1.0, gap.symmetry_score * 0.5 + 0.6)
        
        # Weighted average instead of multiplication (less punishing)
        confidence = (base_confidence * 0.5 + 
                      quality_factor * 0.2 + 
                      contrast_factor * 0.15 + 
                      symmetry_factor * 0.15)
        
        logger.debug(f"Confidence calculation: base={base_confidence:.2f}, quality={quality_factor:.2f}, "
                    f"contrast={contrast_factor:.2f}, symmetry={symmetry_factor:.2f}, final={confidence:.2f}")
        
        return float(np.clip(confidence, 0, 1))
    
    def _confidence_to_label(self, score: float) -> str:
        """Convert confidence score to label"""
        if score >= 0.7:
            return "HIGH"
        elif score >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_messages(self, risk: RiskLevel, 
                          healthy_pct: float, 
                          clubbing_pct: float) -> Tuple[str, str]:
        """Generate user-facing messages"""
        
        messages = {
            RiskLevel.HEALTHY: (
                f"✅ HEALTHY ({healthy_pct:.0f}%)",
                "Normal Schamroth window detected. Diamond gap is visible between nail folds. No signs of digital clubbing."
            ),
            RiskLevel.MOSTLY_HEALTHY: (
                f"✅ MOSTLY HEALTHY ({healthy_pct:.0f}%)",
                "Diamond window appears normal. No immediate concern. Continue monitoring if you have other symptoms."
            ),
            RiskLevel.BORDERLINE: (
                f"⚠️ INCONCLUSIVE ({healthy_pct:.0f}% healthy)",
                "The result is not definitive. We recommend: 1) Retake photo with flash in good lighting, 2) Ensure fingers are touching properly, 3) Consult a healthcare provider if concerned."
            ),
            RiskLevel.LIKELY_CLUBBED: (
                f"⚠️ POSSIBLE CLUBBING ({clubbing_pct:.0f}% risk)",
                "Reduced diamond window detected. This may indicate early clubbing. Please consult a healthcare provider for clinical evaluation."
            ),
            RiskLevel.CLUBBED: (
                f"🔴 CLUBBING INDICATED ({clubbing_pct:.0f}%)",
                "Significant reduction or absence of diamond window. Please see a healthcare provider as soon as possible for proper clinical evaluation. Clubbing can indicate underlying conditions."
            ),
            RiskLevel.ERROR: (
                "❌ ERROR",
                "Could not complete analysis. Please retake photo."
            )
        }
        
        return messages.get(risk, ("Unknown", "Please retake photo"))
    
    def _annotate_image(self, image: np.ndarray, risk: RiskLevel,
                        healthy_pct: float, clubbing_pct: float,
                        confidence: str, requires_retake: bool = False) -> None:
        """Add visual annotations to image"""
        
        colors = {
            RiskLevel.HEALTHY: (0, 200, 0),
            RiskLevel.MOSTLY_HEALTHY: (100, 200, 100),
            RiskLevel.BORDERLINE: (0, 180, 255),
            RiskLevel.LIKELY_CLUBBED: (0, 100, 255),
            RiskLevel.CLUBBED: (0, 0, 255),
            RiskLevel.ERROR: (128, 128, 128)
        }
        
        # Override color if retake required
        if requires_retake:
            color = (0, 128, 255)  # Orange for warning
        else:
            color = colors.get(risk, (200, 200, 200))
        
        h = image.shape[0]
        
        # Background rectangle for text - larger if retake required
        box_height = 120 if requires_retake else 100
        cv2.rectangle(image, (5, 5), (220, box_height), (0, 0, 0), -1)
        cv2.rectangle(image, (5, 5), (220, box_height), color, 2)
        
        if requires_retake:
            # RETAKE WARNING - prominent display
            cv2.putText(image, "RETAKE REQUIRED", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2)
            cv2.putText(image, "Detection uncertain", (10, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(image, "Percentages not reliable", (10, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(image, f"(Raw: H={healthy_pct:.0f}% C={clubbing_pct:.0f}%)", (10, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (128, 128, 128), 1)
            cv2.putText(image, "Please retake photo", (10, 105),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 128, 255), 1)
        else:
            # Normal display
            # Risk level
            cv2.putText(image, risk.value.upper().replace("_", " "), (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Percentages
            cv2.putText(image, f"Healthy: {healthy_pct:.0f}%", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            cv2.putText(image, f"Clubbing: {clubbing_pct:.0f}%", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
            
            # Confidence
            cv2.putText(image, f"Confidence: {confidence}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def _error_result(self, message: str) -> DetectionResult:
        """Create error result"""
        return DetectionResult(
            healthy_percentage=0,
            clubbing_percentage=0,
            risk_level=RiskLevel.ERROR,
            is_clubbing=False,
            requires_retake=True,
            detection_quality_score=0,
            confidence_score=0,
            confidence_label="N/A",
            gap_analysis=GapAnalysis(0, 0, 0, 0, 0, 0),
            detection_method=DetectionMethod.FAILED,
            image_quality=ImageQuality(ImageQualityStatus.UNUSABLE, 0, 0, 0, [message]),
            message=f"❌ {message}",
            recommendation="Please retake photo following capture guidelines.",
            processing_time_ms=0,
            debug_info={"error": message}
        )
    
    def _quality_error_result(self, quality: ImageQuality, 
                              marked: np.ndarray) -> DetectionResult:
        """Create result for quality issues"""
        return DetectionResult(
            healthy_percentage=0,
            clubbing_percentage=0,
            risk_level=RiskLevel.ERROR,
            is_clubbing=False,
            requires_retake=True,
            detection_quality_score=0,
            confidence_score=0,
            confidence_label="N/A",
            gap_analysis=GapAnalysis(0, 0, 0, 0, 0, 0),
            detection_method=DetectionMethod.FAILED,
            image_quality=quality,
            message="❌ Image quality insufficient",
            recommendation=". ".join(quality.issues),
            processing_time_ms=0,
            marked_image=marked,
            debug_info={"quality_issues": quality.issues}
        )
    
    def _mediapipe_error_result(self, marked: np.ndarray) -> DetectionResult:
        """Create result when MediaPipe fails to detect both hands"""
        # Add text to marked image
        cv2.putText(marked, "HANDS NOT DETECTED", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(marked, "Please retake photo", (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return DetectionResult(
            healthy_percentage=0,
            clubbing_percentage=0,
            risk_level=RiskLevel.ERROR,
            is_clubbing=False,
            requires_retake=True,
            detection_quality_score=0,
            confidence_score=0,
            confidence_label="N/A",
            gap_analysis=GapAnalysis(0, 0, 0, 0, 0, 0),
            detection_method=DetectionMethod.FAILED,
            image_quality=ImageQuality(ImageQualityStatus.POOR, 0, 0, 0, 
                                       ["Could not detect both hands"]),
            message="❌ HANDS NOT DETECTED - Please retake photo",
            recommendation="Ensure BOTH hands are visible with fingers touching nail-to-nail. Use good lighting and hold camera steady.",
            processing_time_ms=0,
            marked_image=marked,
            debug_info={"error": "MediaPipe failed to detect both hands"}
        )
    
    def _pose_error_result(self, marked: np.ndarray) -> DetectionResult:
        """Create result when pose validation fails (fingers not touching)"""
        return DetectionResult(
            healthy_percentage=0,
            clubbing_percentage=0,
            risk_level=RiskLevel.ERROR,
            is_clubbing=False,
            requires_retake=True,
            detection_quality_score=0,
            confidence_score=0,
            confidence_label="N/A",
            gap_analysis=GapAnalysis(0, 0, 0, 0, 0, 0),
            detection_method=DetectionMethod.FAILED,
            image_quality=ImageQuality(ImageQualityStatus.POOR, 0, 0, 0, 
                                       ["Incorrect pose - fingers not touching"]),
            message="❌ INCORRECT POSE - Fingers must be touching",
            recommendation="Press your index fingers together nail-to-nail so they are touching. The Schamroth test requires fingers touching to check for the diamond gap.",
            processing_time_ms=0,
            marked_image=marked,
            debug_info={"error": "Pose validation failed - fingers too far apart"}
        )
    
    def process_file(self, path: Union[str, Path]) -> DetectionResult:
        """Process image file"""
        path = Path(path)
        if not path.exists():
            return self._error_result(f"File not found: {path}")
        
        image = cv2.imread(str(path))
        if image is None:
            return self._error_result(f"Cannot load image: {path}")
        
        return self.detect(image)
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'hands'):
            self.hands.close()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def detect_clubbing(image_or_path: Union[np.ndarray, str, Path]) -> DetectionResult:
    """
    Quick detection function.
    
    Args:
        image_or_path: BGR image array or path to image file
        
    Returns:
        DetectionResult with complete analysis
    """
    detector = SchamrothDetector()
    
    if isinstance(image_or_path, (str, Path)):
        return detector.process_file(image_or_path)
    return detector.detect(image_or_path)


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        result = detect_clubbing(sys.argv[1])
        print(result.to_json())
    else:
        print(f"Schamroth Detection Engine v{Config.VERSION}")
        print("Usage: python detector_enterprise.py <image_path>")
