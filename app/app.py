"""
Flask web application for VEP classification.
Drag-and-drop interface for raw VEP files.
"""
from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from werkzeug.utils import secure_filename
from app.predict_from_raw import VEPPredictor
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
from pathlib import Path

# Resolve paths relative to this file so the app works regardless of cwd
APP_DIR = Path(__file__).resolve().parent
SAVED_MODELS_DIR = APP_DIR / "saved_models"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = str(APP_DIR / 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def _latest(pattern):
    matches = glob.glob(str(SAVED_MODELS_DIR / pattern))
    if not matches:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in {SAVED_MODELS_DIR}. "
            "Train a model first (see app/train_and_save_full_model.ipynb)."
        )
    return max(matches, key=os.path.getctime)

MODEL_PATH    = _latest("*.keras")
SCALER_PATH   = _latest("*.pkl")
METADATA_PATH = _latest("*.json")


predictor = VEPPredictor(MODEL_PATH, SCALER_PATH, METADATA_PATH)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def plot_to_base64(time, signal, prediction):
    """Generate a base64-encoded plot of the signal."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, signal, linewidth=1.5, color='#667eea')
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Signal (normalized)', fontsize=12)
    
    title = f'Predicted: {prediction["class"]} ({prediction["confidence"]:.1%})'
    if prediction.get('snr'):
        title += f' | SNR: {prediction["snr"]:.2f}'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    
    return img_str

@app.route('/')
def index():
    return render_template('index.html', 
                          classes=predictor.metadata['classes'],
                          model_info=predictor.metadata)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload a CSV file.'}), 400
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        # Predict
        result = predictor.predict(filepath)
        
        # Generate plot
        img_str = plot_to_base64(result['time'], result['signal'], result)
        
        # Clean up
        os.remove(filepath)
        
        # Format class names for display
        display_class = result['class'].replace('_', ' ')
        
        return jsonify({
            'success': True,
            'prediction': display_class,
            'confidence': f"{result['confidence']:.1%}",
            'snr': f"{result['snr']:.2f}",
            'probabilities': {k.replace('_', ' '): f"{v:.1%}" 
                            for k, v in result['probabilities'].items()},
            'plot': img_str,
            'filename': filename
        })
    
    except Exception as e:
        # Clean up on error
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model': os.path.basename(MODEL_PATH),
        'classes': predictor.metadata['classes'],
        'trained_on': predictor.metadata.get('training_date', 'unknown')
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("VEP Classifier Web App")
    print("="*60)
    print(f"Model: {os.path.basename(MODEL_PATH)}")
    print(f"Classes: {', '.join(predictor.metadata['classes'])}")
    print(f"Visit: http://localhost:8080")  # Changed from 5000
    print("="*60 + "\n")
    app.run(debug=False, port=8080, host='0.0.0.0')  # Changed from 5000