"""
Schamroth API Server for Flutter Mobile App
============================================
Run this on your laptop, Flutter app connects to it.
Uses your EXACT detector_pipeline.py - no changes!
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import cv2
import numpy as np
import base64
import socket
import logging

from detector_pipeline import SchamrothPipelineDetector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize detector once
detector = SchamrothPipelineDetector()
logger.info("Detector initialized successfully")

# Mobile Web UI - Enterprise Edition
MOBILE_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Schamroth Window Test | Clinical Nail Assessment</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #0A2463;
            --primary-light: #1E3A8A;
            --accent: #1E8F7E;
            --accent-light: #2DD4BF;
            --success: #059669;
            --warning: #D97706;
            --danger: #DC2626;
            --bg-dark: #0F172A;
            --bg-card: #1E293B;
            --text-primary: #F8FAFC;
            --text-secondary: #94A3B8;
            --border: #334155;
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-dark);
            min-height: 100vh;
            color: var(--text-primary);
            line-height: 1.6;
        }
        
        /* Header */
        .header {
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
            padding: 20px;
            text-align: center;
            border-bottom: 1px solid var(--border);
            position: relative;
        }
        
        .header-badge {
            position: absolute;
            top: 12px;
            right: 12px;
            background: var(--accent);
            color: white;
            font-size: 10px;
            font-weight: 600;
            padding: 4px 8px;
            border-radius: 4px;
            letter-spacing: 0.5px;
        }
        
        .logo {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            margin-bottom: 8px;
        }
        
        .logo-icon {
            width: 48px;
            height: 48px;
            background: linear-gradient(135deg, var(--accent) 0%, var(--accent-light) 100%);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            box-shadow: 0 4px 12px rgba(30, 143, 126, 0.3);
        }
        
        .logo-text h1 {
            font-size: 20px;
            font-weight: 700;
            letter-spacing: -0.5px;
        }
        
        .logo-text p {
            font-size: 12px;
            color: var(--text-secondary);
            font-weight: 500;
        }
        
        /* Main Content */
        .container {
            max-width: 480px;
            margin: 0 auto;
            padding: 20px;
        }
        
        /* Medical Disclaimer Banner */
        .disclaimer-banner {
            background: rgba(217, 119, 6, 0.15);
            border: 1px solid rgba(217, 119, 6, 0.3);
            border-radius: 10px;
            padding: 12px 16px;
            margin-bottom: 20px;
            display: flex;
            gap: 10px;
            align-items: flex-start;
        }
        
        .disclaimer-banner .icon { font-size: 18px; }
        
        .disclaimer-banner p {
            font-size: 12px;
            color: var(--text-secondary);
            line-height: 1.5;
        }
        
        /* Instructions Card */
        .instructions-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        .instructions-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border);
        }
        
        .instructions-header h2 {
            font-size: 16px;
            font-weight: 600;
        }
        
        .instruction-steps {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        
        .instruction-step {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .step-number {
            width: 28px;
            height: 28px;
            background: var(--accent);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            font-weight: 700;
            flex-shrink: 0;
        }
        
        .step-text {
            font-size: 14px;
            color: var(--text-secondary);
        }
        
        .step-text strong {
            color: var(--accent-light);
        }
        
        /* Camera Card */
        .camera-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        #preview {
            width: 100%;
            max-height: 280px;
            border-radius: 12px;
            display: none;
            object-fit: contain;
            background: var(--bg-dark);
        }
        
        .placeholder {
            width: 100%;
            height: 200px;
            background: var(--bg-dark);
            border: 2px dashed var(--border);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            color: var(--text-secondary);
            transition: border-color 0.3s ease;
        }
        
        .placeholder:hover {
            border-color: var(--accent);
        }
        
        .placeholder-icon {
            font-size: 48px;
            margin-bottom: 12px;
            opacity: 0.6;
        }
        
        .placeholder-text {
            font-size: 14px;
            font-weight: 500;
        }
        
        .buttons {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-top: 16px;
        }
        
        .btn {
            padding: 14px 20px;
            border: none;
            border-radius: 10px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            transition: all 0.2s ease;
            font-family: inherit;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, var(--accent) 0%, #0F766E 100%);
            color: white;
            box-shadow: 0 4px 12px rgba(30, 143, 126, 0.25);
        }
        
        .btn-primary:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(30, 143, 126, 0.35);
        }
        
        .btn-secondary {
            background: var(--bg-dark);
            color: var(--text-primary);
            border: 1px solid var(--border);
        }
        
        .btn-secondary:hover {
            background: var(--primary-light);
            border-color: var(--primary-light);
        }
        
        input[type="file"] { display: none; }
        
        /* Loading */
        .loading {
            display: none;
            text-align: center;
            padding: 40px 20px;
        }
        
        .spinner {
            width: 48px;
            height: 48px;
            border: 3px solid var(--border);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin: 0 auto 16px;
        }
        
        @keyframes spin { to { transform: rotate(360deg); } }
        
        .loading-text {
            color: var(--text-secondary);
            font-size: 14px;
        }
        
        /* Pipeline Steps */
        .pipeline {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 20px;
            display: none;
        }
        
        .pipeline-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border);
        }
        
        .pipeline-header h3 {
            font-size: 14px;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .step {
            padding: 12px 16px;
            border-radius: 8px;
            margin: 6px 0;
            font-size: 13px;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 10px;
            transition: all 0.3s ease;
        }
        
        .step-passed {
            background: rgba(5, 150, 105, 0.15);
            border-left: 3px solid var(--success);
            color: #6EE7B7;
        }
        
        .step-failed {
            background: rgba(220, 38, 38, 0.15);
            border-left: 3px solid var(--danger);
            color: #FCA5A5;
        }
        
        .step-pending {
            background: rgba(100, 100, 100, 0.1);
            border-left: 3px solid var(--border);
            color: var(--text-secondary);
        }
        
        /* Metrics */
        .metrics {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin-bottom: 20px;
        }
        
        .metric-box {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
            text-align: center;
        }
        
        .metric-value {
            font-size: 28px;
            font-weight: 700;
            background: linear-gradient(135deg, var(--accent-light) 0%, var(--accent) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .metric-label {
            font-size: 11px;
            color: var(--text-secondary);
            margin-top: 4px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 500;
        }
        
        .risk-arrow {
            margin: 0 4px;
            font-size: 14px;
            opacity: 0.7;
        }
        
        .risk-label {
            font-size: 12px;
            font-weight: 600;
            color: var(--text-primary);
        }
        
        /* Result Card */
        .result-card {
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            display: none;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
        }
        
        .result-healthy {
            background: linear-gradient(135deg, #059669 0%, #047857 100%);
        }
        
        .result-warning {
            background: linear-gradient(135deg, #D97706 0%, #B45309 100%);
        }
        
        .result-danger {
            background: linear-gradient(135deg, #DC2626 0%, #B91C1C 100%);
        }
        
        .result-title {
            font-size: 24px;
            font-weight: 700;
            text-align: center;
            margin-bottom: 16px;
        }
        
        .result-message {
            background: rgba(0, 0, 0, 0.2);
            padding: 16px;
            border-radius: 10px;
            font-size: 14px;
            line-height: 1.6;
        }
        
        .result-recommendation {
            margin-top: 16px;
            padding-top: 16px;
            border-top: 1px solid rgba(255, 255, 255, 0.2);
            font-size: 13px;
            opacity: 0.9;
            display: flex;
            gap: 8px;
            align-items: flex-start;
        }
        
        /* Annotated Image Section */
        .annotated-section {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        .section-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border);
        }
        
        .section-header h3 {
            font-size: 14px;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .annotated-image {
            width: 100%;
            border-radius: 12px;
            border: 2px solid var(--accent);
            box-shadow: 0 4px 12px rgba(30, 143, 126, 0.2);
        }
        
        /* Footer */
        .footer {
            background: var(--bg-card);
            border-top: 1px solid var(--border);
            padding: 24px 20px;
            text-align: center;
            margin-top: 40px;
        }
        
        .footer-logo {
            font-size: 32px;
            margin-bottom: 12px;
        }
        
        .footer-brand {
            font-size: 16px;
            color: var(--text-primary);
            margin-bottom: 12px;
        }
        
        .footer-brand strong {
            font-weight: 700;
            letter-spacing: -0.3px;
        }
        
        .footer-disclaimer {
            font-size: 11px;
            color: var(--text-secondary);
            opacity: 0.7;
            line-height: 1.5;
            max-width: 320px;
            margin: 0 auto 16px;
        }
        
        .footer-credits {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
            font-size: 12px;
            color: var(--text-secondary);
            margin-bottom: 12px;
        }
        
        .author-name {
            color: var(--accent-light);
            font-weight: 600;
        }
        
        .footer-version {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-size: 11px;
            color: var(--text-secondary);
        }
        
        .version-badge {
            background: var(--border);
            padding: 4px 10px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 11px;
        }
        
        .divider {
            opacity: 0.5;
        }
        
        .year {
            opacity: 0.6;
        }
    </style>
</head>
<body>
    <!-- Header -->
    <header class="header">
        <span class="header-badge">BETA</span>
        <div class="logo">
            <div class="logo-icon">🩺</div>
            <div class="logo-text">
                <h1>Schamroth Window Test</h1>
                <p>Clinical Nail Assessment Tool</p>
            </div>
        </div>
    </header>
    
    <div class="container">
        <!-- Medical Disclaimer -->
        <div class="disclaimer-banner">
            <span class="icon">⚠️</span>
            <p><strong>Medical Disclaimer:</strong> This tool is for educational and screening purposes only. Results do not constitute medical diagnosis. Please consult a healthcare professional for clinical evaluation.</p>
        </div>
        
        <!-- Instructions -->
        <div class="instructions-card">
            <div class="instructions-header">
                <span>📋</span>
                <h2>How to Perform the Test</h2>
            </div>
            <div class="instruction-steps">
                <div class="instruction-step">
                    <span class="step-number">1</span>
                    <span class="step-text">Touch your <strong>index fingers</strong> nail-to-nail</span>
                </div>
                <div class="instruction-step">
                    <span class="step-number">2</span>
                    <span class="step-text">Hold hands against a <strong>dark background</strong></span>
                </div>
                <div class="instruction-step">
                    <span class="step-number">3</span>
                    <span class="step-text">Keep wrists <strong>apart</strong>, fingers visible</span>
                </div>
                <div class="instruction-step">
                    <span class="step-number">4</span>
                    <span class="step-text">Capture photo with <strong>good lighting</strong></span>
                </div>
            </div>
        </div>
        
        <!-- Camera Section -->
        <div class="camera-card">
            <div class="placeholder" id="placeholder">
                <div class="placeholder-icon">📷</div>
                <div class="placeholder-text">Tap to capture or select image</div>
            </div>
            <img id="preview" alt="Preview">
            <div class="buttons">
                <label class="btn btn-secondary">
                    📁 Gallery
                    <input type="file" id="fileInput" accept="image/*">
                </label>
                <label class="btn btn-primary">
                    📷 Camera
                    <input type="file" id="cameraInput" accept="image/*" capture="environment">
                </label>
            </div>
        </div>
        
        <!-- Loading -->
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p class="loading-text">Analyzing image with AI pipeline...</p>
        </div>
        
        <!-- Pipeline Steps -->
        <div class="pipeline" id="pipeline">
            <div class="pipeline-header">
                <span>🔬</span>
                <h3>Detection Pipeline</h3>
            </div>
            <div class="step step-pending" id="step1">Step 1: Hand Detection</div>
            <div class="step step-pending" id="step2">Step 2: Finger Alignment</div>
            <div class="step step-pending" id="step3">Step 3: Geometry Validation</div>
            <div class="step step-pending" id="step4">Step 4: Gap Analysis</div>
        </div>
        
        <!-- Metrics -->
        <div class="metrics" id="metrics" style="display: none;">
            <div class="metric-box">
                <div class="metric-value" id="healthyPct">--</div>
                <div class="metric-label">G_score</div>
            </div>
            <div class="metric-box">
                <div class="metric-value" id="clubbingPct">--</div>
                <div class="metric-label">Clubbing Prob.</div>
            </div>
            <div class="metric-box">
                <div class="metric-value" id="confidence">--</div>
                <div class="metric-label">Confidence</div>
            </div>
        </div>
        
        <!-- Annotated Image Output -->
        <div class="annotated-section" id="annotatedSection" style="display: none;">
            <div class="section-header">
                <span>🎯</span>
                <h3>Detection Analysis</h3>
            </div>
            <img id="annotatedImage" alt="Annotated Detection" class="annotated-image">
        </div>
        
        <!-- Result Card -->
        <div class="result-card" id="result">
            <div class="result-title" id="resultTitle">HEALTHY</div>
            <div class="result-message" id="resultMessage">Analysis complete.</div>
            <div class="result-recommendation" id="resultRec">📋 Recommendation will appear here.</div>
        </div>
    </div>
    
    <!-- Footer -->
    <footer class="footer">
        <div class="footer-logo">🩺</div>
        <div class="footer-brand">
            <strong>Schamroth Window Test</strong>
        </div>
        <p class="footer-disclaimer">
            For educational and research purposes only. Not a substitute for professional medical advice.
        </p>
        <div class="footer-credits">
            <span>Made by</span>
            <span class="author-name">Kevin Kattakayam</span>
        </div>
        <div class="footer-version">
            <span class="version-badge">Version 1.0</span>
            <span class="divider">•</span>
            <span class="year">© 2026</span>
        </div>
    </footer>
    
    <script>
        const preview = document.getElementById('preview');
        const placeholder = document.getElementById('placeholder');
        const loading = document.getElementById('loading');
        const result = document.getElementById('result');
        const pipeline = document.getElementById('pipeline');
        const metrics = document.getElementById('metrics');
        
        function handleImage(file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                preview.src = e.target.result;
                preview.style.display = 'block';
                placeholder.style.display = 'none';
                result.style.display = 'none';
                pipeline.style.display = 'none';
                metrics.style.display = 'none';
                analyzeImage(e.target.result);
            };
            reader.readAsDataURL(file);
        }
        
        async function analyzeImage(dataUrl) {
            loading.style.display = 'block';
            result.style.display = 'none';
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: dataUrl })
                });
                const data = await response.json();
                showResult(data);
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                loading.style.display = 'none';
            }
        }
        
        function showResult(data) {
            // Show pipeline steps
            pipeline.style.display = 'block';
            const stepMap = {
                'hands_check': 1, 'fingers_check': 2, 
                'geometry_check': 3, 'gap_analysis': 4, 'complete': 5
            };
            const reached = stepMap[data.step_reached] || 0;
            
            for (let i = 1; i <= 4; i++) {
                const step = document.getElementById('step' + i);
                step.className = 'step';
                if (i < reached || data.all_passed) {
                    step.className = 'step step-passed';
                    step.innerHTML = '✓ ' + step.textContent.replace('✓ ', '').replace('✗ ', '').replace('○ ', '');
                } else if (i === reached && !data.all_passed) {
                    step.className = 'step step-failed';
                    step.innerHTML = '✗ ' + step.textContent.replace('✓ ', '').replace('✗ ', '').replace('○ ', '') + ' - FAILED';
                } else {
                    step.className = 'step step-pending';
                    step.innerHTML = '○ ' + step.textContent.replace('✓ ', '').replace('✗ ', '').replace('○ ', '');
                }
            }
            
            // Show metrics with grid display
            metrics.style.display = 'grid';
            const gScore = Math.round(data.healthy_percentage || 0);
            
            // Get risk level interpretation based on G_score
            let riskLabel = '';
            if (gScore >= 65) riskLabel = 'Healthy';
            else if (gScore >= 50) riskLabel = 'Low Risk';
            else if (gScore >= 35) riskLabel = 'Borderline';
            else if (gScore >= 20) riskLabel = 'High Risk';
            else riskLabel = 'Clubbing';
            
            document.getElementById('healthyPct').innerHTML = gScore + '%<span class="risk-arrow">→</span><span class="risk-label">' + riskLabel + '</span>';
            document.getElementById('clubbingPct').textContent = Math.round(100 - gScore) + '%';
            document.getElementById('confidence').textContent = (data.confidence || 'N/A').replace('_', ' ');
            
            // Show annotated image if available
            const annotatedSection = document.getElementById('annotatedSection');
            const annotatedImage = document.getElementById('annotatedImage');
            if (data.annotated_image) {
                annotatedImage.src = 'data:image/jpeg;base64,' + data.annotated_image;
                annotatedSection.style.display = 'block';
            } else {
                annotatedSection.style.display = 'none';
            }
            
            // Show result card
            result.style.display = 'block';
            result.className = 'result-card';
            
            if (data.success && data.all_passed) {
                const risk = data.risk_level || '';
                if (risk.includes('healthy')) {
                    result.classList.add('result-healthy');
                    document.getElementById('resultTitle').textContent = '✅ HEALTHY';
                } else if (risk.includes('low') || risk.includes('border')) {
                    result.classList.add('result-warning');
                    document.getElementById('resultTitle').textContent = '⚠️ ' + risk.toUpperCase().replace('_', ' ');
                } else {
                    result.classList.add('result-danger');
                    document.getElementById('resultTitle').textContent = '⚠️ ' + risk.toUpperCase().replace('_', ' ');
                }
            } else {
                result.classList.add('result-warning');
                document.getElementById('resultTitle').textContent = data.requires_retake ? '⚠️ RETAKE REQUIRED' : '❌ CHECK RESULT';
            }
            
            document.getElementById('resultMessage').textContent = data.message || 'Analysis complete.';
            document.getElementById('resultRec').textContent = '📋 ' + (data.recommendation || 'No recommendation.');
        }
        
        document.getElementById('fileInput').addEventListener('change', (e) => {
            if (e.target.files[0]) handleImage(e.target.files[0]);
        });
        document.getElementById('cameraInput').addEventListener('change', (e) => {
            if (e.target.files[0]) handleImage(e.target.files[0]);
        });
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    """Mobile web interface"""
    return render_template_string(MOBILE_HTML)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'message': 'Schamroth API Server Running'})

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Analyze image for Schamroth test
    Expects: JSON with base64 encoded image
    Returns: Detection results
    """
    try:
        data = request.json
        
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        image_data = data['image']
        
        # Handle data URL format (data:image/jpeg;base64,...)
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 to image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Could not decode image'}), 400
        
        logger.info(f"Processing image: {img.shape}")
        
        # Run detector (your exact Python code!)
        result = detector.detect(img)
        
        # Encode annotated image to base64 for display
        annotated_image_b64 = None
        if result.marked_image is not None and result.marked_image.size > 0:
            _, buffer = cv2.imencode('.jpg', result.marked_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            annotated_image_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Prepare response
        response = {
            'success': True,
            'all_passed': result.all_steps_passed,
            'step_reached': result.step_reached.value,
            'risk_level': result.risk_level.value if result.risk_level else 'unknown',
            'healthy_percentage': round(result.healthy_percentage, 1),
            'clubbing_percentage': round(result.clubbing_percentage, 1),
            'confidence': result.confidence_label,
            'message': result.message,
            'recommendation': result.recommendation,
            'requires_retake': result.requires_retake,
            'annotated_image': annotated_image_b64
        }
        
        logger.info(f"Result: {response['risk_level']} ({response['healthy_percentage']}%)")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

def get_local_ip():
    """Get local IP address for phone access"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "localhost"

if __name__ == '__main__':
    local_ip = get_local_ip()
    port = 5000
    
    print("\n" + "="*60)
    print("🔬 SCHAMROTH API SERVER")
    print("="*60)
    print(f"\n📱 Configure Flutter app with this URL:")
    print(f"   http://{local_ip}:{port}")
    print(f"\n💻 Test endpoint:")
    print(f"   http://localhost:{port}/health")
    print("\n⚠️  Keep this running while using the app!")
    print("⚠️  Phone and laptop must be on SAME WiFi!")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=port, debug=False)
