import base64
import io
import os
import time
import torch
import torchvision.transforms as tfs
from PIL import Image
from flask import Flask, request, jsonify, render_template_string, send_from_directory
from flask_cors import CORS
from models.C2PNet import C2PNet

app = Flask(__name__)
CORS(app)

# Use MPS if available on Mac, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
print(f"[*] Initializing C2P-Net Web App on device: {device}")

MODELS = {
    'indoor': {'path': 'trained_models/ITS.pkl', 'net': None, 'name': 'C2P-Net Indoor (ITS)'},
    'outdoor': {'path': 'trained_models/OTS.pkl', 'net': None, 'name': 'C2P-Net Outdoor (OTS)'}
}

def load_model(model_key):
    if MODELS[model_key]['net'] is not None:
        return MODELS[model_key]['net']
    
    path = MODELS[model_key]['path']
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model checkpoint not found at {path}")
    
    print(f"[*] Loading model {model_key} from {path}...")
    net = C2PNet(gps=3, blocks=19).to(device)
    ckp = torch.load(path, map_location=device, weights_only=False)
    state_dict = ckp['model'] if 'model' in ckp else ckp
    net.load_state_dict(state_dict)
    net.eval()
    MODELS[model_key]['net'] = net
    print(f"[+] Loaded {model_key} successfully.")
    return net

# Preload indoor model
try:
    if os.path.exists('trained_models/ITS.pkl'):
        load_model('indoor')
except Exception as e:
    print(f"Warning: Could not preload ITS.pkl: {e}")

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>C2P-Net | Real-Time AI Dehazing Studio</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090d16;
      --card-bg: rgba(18, 24, 38, 0.75);
      --card-border: rgba(255, 255, 255, 0.08);
      --accent-cyan: #06b6d4;
      --accent-blue: #3b82f6;
      --accent-violet: #8b5cf6;
      --accent-gradient: linear-gradient(135deg, #06b6d4 0%, #3b82f6 50%, #8b5cf6 100%);
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --radius-lg: 20px;
      --radius-md: 12px;
      --radius-sm: 8px;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      font-family: 'Plus Jakarta Sans', sans-serif;
      background: var(--bg);
      background-image: 
        radial-gradient(at 0% 0%, rgba(59, 130, 246, 0.12) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(139, 92, 246, 0.12) 0px, transparent 50%),
        radial-gradient(at 50% 50%, rgba(6, 182, 212, 0.06) 0px, transparent 60%);
      background-attachment: fixed;
      color: var(--text-main);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }

    header {
      padding: 18px 36px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid var(--card-border);
      backdrop-filter: blur(16px);
      background: rgba(9, 13, 22, 0.85);
      position: sticky;
      top: 0;
      z-index: 100;
    }

    .logo {
      display: flex;
      align-items: center;
      gap: 12px;
      text-decoration: none;
    }

    .logo-icon {
      width: 38px;
      height: 38px;
      border-radius: 10px;
      background: var(--accent-gradient);
      display: flex;
      align-items: center;
      justify-content: center;
      color: #fff;
      font-weight: 800;
      font-size: 1.1rem;
      box-shadow: 0 4px 14px rgba(6, 182, 212, 0.4);
    }

    .logo-text h1 {
      font-size: 1.15rem;
      font-weight: 700;
      background: linear-gradient(90deg, #ffffff, #cbd5e1);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .logo-text span {
      font-size: 0.72rem;
      color: var(--text-muted);
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }

    .device-pill {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.75rem;
      padding: 6px 14px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.1);
      color: #38bdf8;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .device-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #22c55e;
      box-shadow: 0 0 8px #22c55e;
    }

    main {
      flex: 1;
      max-width: 1360px;
      width: 100%;
      margin: 0 auto;
      padding: 28px 24px;
      display: flex;
      flex-direction: column;
      gap: 22px;
    }

    .hero {
      text-align: center;
      max-width: 720px;
      margin: 0 auto 6px;
    }

    .hero h2 {
      font-size: 2.1rem;
      font-weight: 800;
      letter-spacing: -0.02em;
      margin-bottom: 8px;
      background: linear-gradient(135deg, #ffffff 40%, #93c5fd 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .hero p {
      color: var(--text-muted);
      font-size: 0.95rem;
      line-height: 1.5;
    }

    .grid-container {
      display: grid;
      grid-template-columns: 370px 1fr;
      gap: 24px;
    }

    @media (max-width: 1024px) {
      .grid-container {
        grid-template-columns: 1fr;
      }
    }

    .glass-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: var(--radius-lg);
      padding: 22px;
      backdrop-filter: blur(20px);
      box-shadow: 0 12px 40px rgba(0, 0, 0, 0.35);
    }

    .section-title {
      font-size: 0.8rem;
      font-weight: 700;
      color: #94a3b8;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 12px;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    /* Model Selector */
    .model-selector {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-bottom: 20px;
    }

    .model-btn {
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: var(--radius-md);
      padding: 12px 10px;
      color: var(--text-muted);
      cursor: pointer;
      text-align: left;
      transition: all 0.2s ease;
    }

    .model-btn:hover {
      background: rgba(255, 255, 255, 0.06);
      border-color: rgba(255, 255, 255, 0.15);
    }

    .model-btn.active {
      background: linear-gradient(135deg, rgba(6, 182, 212, 0.15), rgba(59, 130, 246, 0.15));
      border-color: #38bdf8;
      color: #ffffff;
      box-shadow: 0 0 16px rgba(56, 189, 248, 0.2);
    }

    .model-btn .title {
      font-weight: 700;
      font-size: 0.88rem;
      display: block;
      margin-bottom: 2px;
    }

    .model-btn .sub {
      font-size: 0.7rem;
      color: #94a3b8;
      display: block;
    }

    /* Speed & Quality Modes */
    .speed-modes {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      margin-bottom: 20px;
    }

    .speed-btn {
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: var(--radius-sm);
      padding: 9px 6px;
      color: var(--text-muted);
      cursor: pointer;
      text-align: center;
      transition: all 0.2s ease;
      font-size: 0.78rem;
      font-weight: 600;
    }

    .speed-btn:hover {
      background: rgba(255, 255, 255, 0.06);
    }

    .speed-btn.active {
      background: rgba(56, 189, 248, 0.18);
      border-color: #38bdf8;
      color: #38bdf8;
      font-weight: 700;
    }

    /* Upload Box */
    .dropzone {
      border: 2px dashed rgba(255, 255, 255, 0.15);
      border-radius: var(--radius-md);
      padding: 24px 14px;
      text-align: center;
      cursor: pointer;
      transition: all 0.2s ease;
      background: rgba(0, 0, 0, 0.2);
      margin-bottom: 20px;
    }

    .dropzone:hover, .dropzone.dragover {
      border-color: #38bdf8;
      background: rgba(56, 189, 248, 0.05);
    }

    .dropzone svg {
      width: 36px;
      height: 36px;
      stroke: #38bdf8;
      margin-bottom: 8px;
    }

    .dropzone p {
      font-size: 0.88rem;
      font-weight: 600;
      margin-bottom: 2px;
    }

    .dropzone span {
      font-size: 0.72rem;
      color: var(--text-muted);
    }

    input[type="file"] {
      display: none;
    }

    /* Quick Samples */
    .samples-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      margin-bottom: 20px;
    }

    .sample-item {
      aspect-ratio: 16/10;
      border-radius: 8px;
      overflow: hidden;
      cursor: pointer;
      border: 2px solid transparent;
      position: relative;
      transition: all 0.2s ease;
    }

    .sample-item:hover {
      transform: translateY(-2px);
      border-color: #38bdf8;
      box-shadow: 0 4px 12px rgba(56, 189, 248, 0.3);
    }

    .sample-item img {
      width: 100%;
      height: 100%;
      object-fit: cover;
    }

    .sample-item span {
      position: absolute;
      bottom: 0;
      left: 0;
      right: 0;
      background: rgba(0, 0, 0, 0.75);
      font-size: 0.65rem;
      font-weight: 600;
      padding: 2px 4px;
      text-align: center;
    }

    /* Action Button */
    .dehaze-btn {
      width: 100%;
      padding: 15px;
      border-radius: var(--radius-md);
      border: none;
      background: var(--accent-gradient);
      color: #ffffff;
      font-family: 'Plus Jakarta Sans', sans-serif;
      font-size: 0.95rem;
      font-weight: 700;
      cursor: pointer;
      box-shadow: 0 6px 20px rgba(6, 182, 212, 0.35);
      transition: all 0.25s ease;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
    }

    .dehaze-btn:hover:not(:disabled) {
      transform: translateY(-2px);
      box-shadow: 0 10px 26px rgba(6, 182, 212, 0.5);
    }

    .dehaze-btn:disabled {
      opacity: 0.45;
      cursor: not-allowed;
      transform: none;
    }

    /* Stage Viewer */
    .stage-card {
      display: flex;
      flex-direction: column;
      position: relative;
      min-height: 520px;
    }

    .stage-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
    }

    .mode-pills {
      display: flex;
      background: rgba(0, 0, 0, 0.3);
      padding: 4px;
      border-radius: 999px;
      border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .mode-pill {
      background: transparent;
      border: none;
      color: var(--text-muted);
      padding: 5px 12px;
      border-radius: 999px;
      font-size: 0.76rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
    }

    .mode-pill.active {
      background: rgba(255, 255, 255, 0.15);
      color: #ffffff;
    }

    .comparison-container {
      flex: 1;
      position: relative;
      width: 100%;
      height: 480px;
      border-radius: var(--radius-md);
      overflow: hidden;
      background: #000;
      display: flex;
      align-items: center;
      justify-content: center;
      user-select: none;
    }

    .empty-state {
      text-align: center;
      color: var(--text-muted);
      padding: 40px;
    }

    .empty-state svg {
      width: 54px;
      height: 54px;
      stroke: rgba(255, 255, 255, 0.2);
      margin-bottom: 14px;
    }

    .slider-img-wrapper {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
    }

    .slider-img-wrapper img {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      object-fit: contain;
    }

    .slider-img-wrapper.top {
      width: 50%;
      border-right: 2px solid #38bdf8;
      z-index: 2;
    }

    .slider-handle {
      position: absolute;
      top: 0;
      bottom: 0;
      left: 50%;
      width: 3px;
      background: #38bdf8;
      z-index: 3;
      cursor: ew-resize;
      box-shadow: 0 0 10px rgba(56, 189, 248, 0.9);
    }

    .slider-handle-button {
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      width: 32px;
      height: 32px;
      border-radius: 50%;
      background: #38bdf8;
      color: #090d16;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 0 12px rgba(56, 189, 248, 0.9);
    }

    .label-badge {
      position: absolute;
      top: 14px;
      padding: 5px 12px;
      border-radius: 6px;
      font-size: 0.72rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      backdrop-filter: blur(8px);
      z-index: 4;
      pointer-events: none;
    }

    .badge-before {
      left: 14px;
      background: rgba(239, 68, 68, 0.75);
      color: #fff;
    }

    .badge-after {
      right: 14px;
      background: rgba(34, 197, 94, 0.75);
      color: #fff;
    }

    .side-by-side {
      display: none;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
      width: 100%;
      height: 480px;
    }

    .side-panel {
      position: relative;
      border-radius: var(--radius-md);
      overflow: hidden;
      background: #000;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .side-panel img {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
    }

    .stats-bar {
      display: flex;
      gap: 16px;
      margin-top: 18px;
      padding: 12px 18px;
      background: rgba(0, 0, 0, 0.3);
      border-radius: var(--radius-md);
      border: 1px solid var(--card-border);
      align-items: center;
      justify-content: space-between;
    }

    .stat-item {
      display: flex;
      flex-direction: column;
      gap: 1px;
    }

    .stat-label {
      font-size: 0.68rem;
      text-transform: uppercase;
      color: var(--text-muted);
      letter-spacing: 0.05em;
    }

    .stat-val {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.95rem;
      font-weight: 700;
      color: #38bdf8;
    }

    .download-btn {
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid rgba(255, 255, 255, 0.15);
      color: #fff;
      padding: 8px 16px;
      border-radius: 8px;
      font-size: 0.82rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }

    .download-btn:hover {
      background: #38bdf8;
      color: #090d16;
      border-color: #38bdf8;
    }

    .loader {
      display: none;
      width: 20px;
      height: 20px;
      border: 2px solid rgba(255, 255, 255, 0.3);
      border-radius: 50%;
      border-top-color: #fff;
      animation: spin 0.7s linear infinite;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }
  </style>
</head>
<body>

  <header>
    <a href="#" class="logo">
      <div class="logo-icon">C2P</div>
      <div class="logo-text">
        <h1>C2P-Net Studio</h1>
        <span>CVPR 2023 Physics-Aware Image Dehazing</span>
      </div>
    </a>
    <div class="device-pill">
      <div class="device-dot"></div>
      <span>Acceleration: {{ device }} (Optimized)</span>
    </div>
  </header>

  <main>
    <div class="hero">
      <h2>Transform Foggy & Hazy Photos into Crisp Clarity</h2>
      <p>Instant hardware-optimized neural single image dehazing with zero system freeze.</p>
    </div>

    <div class="grid-container">
      <div class="glass-card">
        <div class="section-title">
          <svg width="15" height="15" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"></path></svg>
          1. Select Model
        </div>
        <div class="model-selector">
          <button class="model-btn active" id="btn-indoor" onclick="selectModel('indoor')">
            <span class="title">Indoor Scene</span>
            <span class="sub">ITS Checkpoint (42.56 dB)</span>
          </button>
          <button class="model-btn" id="btn-outdoor" onclick="selectModel('outdoor')">
            <span class="title">Outdoor Scene</span>
            <span class="sub">OTS Checkpoint (36.68 dB)</span>
          </button>
        </div>

        <div class="section-title">
          <svg width="15" height="15" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
          2. Speed / Quality Profile
        </div>
        <div class="speed-modes">
          <button class="speed-btn active" id="mode-fast" onclick="setQualityMode('fast')">⚡ Fast (~3s)</button>
          <button class="speed-btn" id="mode-balanced" onclick="setQualityMode('balanced')">⚖️ Balanced</button>
          <button class="speed-btn" id="mode-hd" onclick="setQualityMode('hd')">💎 Ultra HD</button>
        </div>

        <div class="section-title">
          <svg width="15" height="15" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
          3. Upload Hazy Image
        </div>
        <div class="dropzone" id="dropzone" onclick="document.getElementById('file-input').click()">
          <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path></svg>
          <p id="upload-text">Click or drag & drop image</p>
          <span>Auto-optimized without freezing</span>
          <input type="file" id="file-input" accept="image/*" onchange="handleFile(this.files[0])" />
        </div>

        <div class="section-title">
          <svg width="15" height="15" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path></svg>
          Or Test With Samples
        </div>
        <div class="samples-grid">
          <div class="sample-item" onclick="loadSample('/sample/indoor1', 'indoor')">
            <img src="/sample/indoor1" alt="Indoor 1" />
            <span>Indoor 1</span>
          </div>
          <div class="sample-item" onclick="loadSample('/sample/indoor2', 'indoor')">
            <img src="/sample/indoor2" alt="Indoor 2" />
            <span>Indoor 2</span>
          </div>
          <div class="sample-item" onclick="loadSample('/sample/outdoor1', 'outdoor')">
            <img src="/sample/outdoor1" alt="Outdoor 1" />
            <span>Outdoor 1</span>
          </div>
        </div>

        <button class="dehaze-btn" id="dehaze-btn" onclick="processDehaze()" disabled>
          <div class="loader" id="loader"></div>
          <span id="btn-label">Enhance & Dehaze Image</span>
        </button>
      </div>

      <div class="glass-card stage-card">
        <div class="stage-header">
          <div class="section-title" style="margin: 0;">Interactive Viewport</div>
          <div class="mode-pills" id="view-toggle" style="display: none;">
            <button class="mode-pill active" id="btn-split" onclick="setViewMode('split')">Split Slider</button>
            <button class="mode-pill" id="btn-side" onclick="setViewMode('side')">Side by Side</button>
          </div>
        </div>

        <div class="comparison-container" id="empty-box">
          <div class="empty-state">
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
            <h3>No Image Loaded Yet</h3>
            <p>Upload a hazy photo or click a sample to see C2P-Net restore it in seconds.</p>
          </div>
        </div>

        <div class="comparison-container" id="slider-container" style="display: none;">
          <div class="label-badge badge-before">Hazy (Original)</div>
          <div class="label-badge badge-after">Dehazed (Clear)</div>
          
          <div class="slider-img-wrapper">
            <img id="img-after" src="" alt="Dehazed Result" />
          </div>

          <div class="slider-img-wrapper top" id="top-layer">
            <img id="img-before" src="" alt="Original Hazy" />
          </div>

          <div class="slider-handle" id="slider-handle">
            <div class="slider-handle-button">
              <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M8 9l4-4 4 4m0 6l-4 4-4-4"></path></svg>
            </div>
          </div>
        </div>

        <div class="side-by-side" id="side-container">
          <div class="side-panel">
            <div class="label-badge badge-before">Hazy (Original)</div>
            <img id="side-before" src="" alt="Original Hazy" />
          </div>
          <div class="side-panel">
            <div class="label-badge badge-after">Dehazed (Clear)</div>
            <img id="side-after" src="" alt="Dehazed Result" />
          </div>
        </div>

        <div class="stats-bar" id="stats-bar" style="display: none;">
          <div style="display: flex; gap: 20px;">
            <div class="stat-item">
              <span class="stat-label">Model</span>
              <span class="stat-val" id="stat-model">ITS</span>
            </div>
            <div class="stat-item">
              <span class="stat-label">Inference Time</span>
              <span class="stat-val" id="stat-time">0.0s</span>
            </div>
            <div class="stat-item">
              <span class="stat-label">Resolution</span>
              <span class="stat-val" id="stat-res">620x460</span>
            </div>
          </div>
          <a id="btn-download" class="download-btn" href="#" download="c2pnet_dehazed.png">
            <svg width="15" height="15" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path></svg>
            Download HD Result
          </a>
        </div>

      </div>
    </div>
  </main>

  <script>
    let selectedModel = 'indoor';
    let qualityMode = 'fast';
    let currentImageBase64 = null;
    let isProcessing = false;

    function selectModel(model) {
      selectedModel = model;
      document.getElementById('btn-indoor').classList.toggle('active', model === 'indoor');
      document.getElementById('btn-outdoor').classList.toggle('active', model === 'outdoor');
    }

    function setQualityMode(mode) {
      qualityMode = mode;
      document.getElementById('mode-fast').classList.toggle('active', mode === 'fast');
      document.getElementById('mode-balanced').classList.toggle('active', mode === 'balanced');
      document.getElementById('mode-hd').classList.toggle('active', mode === 'hd');
    }

    function handleFile(file) {
      if (!file) return;
      document.getElementById('upload-text').innerText = file.name;
      const reader = new FileReader();
      reader.onload = function(e) {
        currentImageBase64 = e.target.result;
        previewHazy(currentImageBase64);
      };
      reader.readAsDataURL(file);
    }

    async function loadSample(sampleUrl, modelType) {
      selectModel(modelType);
      document.getElementById('upload-text').innerText = "Sample: " + sampleUrl.split('/').pop();
      const res = await fetch(sampleUrl);
      const blob = await res.blob();
      const reader = new FileReader();
      reader.onload = function(e) {
        currentImageBase64 = e.target.result;
        previewHazy(currentImageBase64);
        processDehaze();
      };
      reader.readAsDataURL(blob);
    }

    function previewHazy(base64) {
      document.getElementById('img-before').src = base64;
      document.getElementById('side-before').src = base64;
      document.getElementById('dehaze-btn').disabled = false;
      document.getElementById('empty-box').style.display = 'none';
      document.getElementById('slider-container').style.display = 'flex';
      document.getElementById('img-after').src = base64;
      document.getElementById('side-after').src = base64;
    }

    async function processDehaze() {
      if (!currentImageBase64 || isProcessing) return;
      isProcessing = true;
      const btn = document.getElementById('dehaze-btn');
      const loader = document.getElementById('loader');
      const label = document.getElementById('btn-label');
      
      btn.disabled = true;
      loader.style.display = 'block';
      label.innerText = 'Neural Dehazing...';

      try {
        const response = await fetch('/api/dehaze', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            image: currentImageBase64,
            model: selectedModel,
            mode: qualityMode
          })
        });

        const data = await response.json();
        if (data.error) {
          alert('Error: ' + data.error);
          return;
        }

        document.getElementById('img-after').src = data.dehazed_image;
        document.getElementById('side-after').src = data.dehazed_image;
        document.getElementById('btn-download').href = data.dehazed_image;
        
        document.getElementById('stat-model').innerText = selectedModel.toUpperCase();
        document.getElementById('stat-time').innerText = data.time_taken + 's';
        document.getElementById('stat-res').innerText = data.resolution;
        document.getElementById('stats-bar').style.display = 'flex';
        document.getElementById('view-toggle').style.display = 'flex';

        syncSliderImages();

      } catch (err) {
        alert('Request failed: ' + err.message);
      } finally {
        isProcessing = false;
        btn.disabled = false;
        loader.style.display = 'none';
        label.innerText = 'Enhance & Dehaze Image';
      }
    }

    const container = document.getElementById('slider-container');
    const topLayer = document.getElementById('top-layer');
    const handle = document.getElementById('slider-handle');
    let isDragging = false;

    function setSliderPosition(x) {
      const rect = container.getBoundingClientRect();
      let pos = (x - rect.left) / rect.width;
      if (pos < 0) pos = 0;
      if (pos > 1) pos = 1;
      const percent = pos * 100;
      topLayer.style.width = percent + '%';
      handle.style.left = percent + '%';
      syncSliderImages();
    }

    function syncSliderImages() {
      const w = container.offsetWidth;
      const beforeImg = document.getElementById('img-before');
      if (beforeImg) {
        beforeImg.style.width = w + 'px';
      }
    }

    window.addEventListener('resize', syncSliderImages);
    container.addEventListener('mousedown', (e) => { isDragging = true; setSliderPosition(e.clientX); });
    window.addEventListener('mouseup', () => { isDragging = false; });
    window.addEventListener('mousemove', (e) => { if (isDragging) setSliderPosition(e.clientX); });
    container.addEventListener('touchstart', (e) => { isDragging = true; setSliderPosition(e.touches[0].clientX); });
    window.addEventListener('touchend', () => { isDragging = false; });
    window.addEventListener('touchmove', (e) => { if (isDragging) setSliderPosition(e.touches[0].clientX); });

    function setViewMode(mode) {
      const isSplit = mode === 'split';
      document.getElementById('btn-split').classList.toggle('active', isSplit);
      document.getElementById('btn-side').classList.toggle('active', !isSplit);
      document.getElementById('slider-container').style.display = isSplit ? 'flex' : 'none';
      document.getElementById('side-container').style.display = isSplit ? 'none' : 'grid';
      if (isSplit) syncSliderImages();
    }

    const dropzone = document.getElementById('dropzone');
    dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
      if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });
  </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, device=str(device).upper())

@app.route('/sample/<sample_name>')
def get_sample(sample_name):
    sample_map = {
        'indoor1': 'data/SOTS/indoor/hazy/1400_1.png',
        'indoor2': 'data/SOTS/indoor/hazy/1401_1.png',
        'outdoor1': 'data/SOTS/outdoor/hazy/0001_0.8_0.2.jpg'
    }
    path = sample_map.get(sample_name)
    if path and os.path.exists(path):
        return send_from_directory(os.path.dirname(path), os.path.basename(path))
    return "Sample not found", 404

@app.route('/api/dehaze', methods=['POST'])
def api_dehaze():
    try:
        data = request.get_json(force=True)
        image_data = data.get('image')
        model_key = data.get('model', 'indoor')
        mode = data.get('mode', 'fast')

        if not image_data:
            return jsonify({'error': 'No image provided'}), 400

        if ',' in image_data:
            image_data = image_data.split(',', 1)[1]
        raw_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(raw_bytes)).convert('RGB')
        orig_w, orig_h = img.size

        # Optimized resolution scaling to guarantee zero freeze and snappy responsiveness
        # Fast: ~360px (ultra fast ~3-4s, sharp upscale)
        # Balanced: ~512px (~8s)
        # HD: ~720px
        dim_map = {'fast': 360, 'balanced': 512, 'hd': 720}
        target_max = dim_map.get(mode, 360)

        if max(orig_w, orig_h) > target_max:
            ratio = target_max / max(orig_w, orig_h)
            proc_w, proc_h = int(orig_w * ratio), int(orig_h * ratio)
            proc_img = img.resize((proc_w, proc_h), Image.Resampling.BILINEAR)
        else:
            proc_img = img
            proc_w, proc_h = orig_w, orig_h

        net = load_model(model_key)

        start_time = time.time()
        img_t = tfs.ToTensor()(proc_img).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = net(img_t)
            if device.type == 'mps':
                torch.mps.synchronize()
        
        pred = pred.clamp(0, 1).squeeze(0).cpu()
        time_taken = round(time.time() - start_time, 2)

        # Reconstruct back with high quality resampling
        out_pil = tfs.ToPILImage()(pred)
        if (proc_w, proc_h) != (orig_w, orig_h):
            out_pil = out_pil.resize((orig_w, orig_h), Image.Resampling.LANCZOS)

        # Clear MPS memory cache to prevent memory pressure
        if device.type == 'mps':
            torch.mps.empty_cache()

        buffered = io.BytesIO()
        out_pil.save(buffered, format="JPEG", quality=92)
        dehazed_b64 = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({
            'dehazed_image': dehazed_b64,
            'time_taken': time_taken,
            'resolution': f"{orig_w}x{orig_h}",
            'model': model_key
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print(f"[*] Starting C2P-Net Web App on http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=False)
