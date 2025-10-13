"""
Prediction pipeline for raw VEP files.
Handles preprocessing and classification using existing preprocessing functions.
"""
import json
import numpy as np
from core.preprocessing import preprocess_signal
from classifiers.CNN_classifier import CNN1D


class VEPPredictor:
    def __init__(self, model_path, scaler_path, metadata_path):
        """Load trained model and preprocessing components."""
        self.model, self.cnn = CNN1D.load_model(
            model_path, scaler_path, metadata_path
        )
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
    
    def preprocess_raw_file(self, filepath):
        """
        Preprocess a single raw VEP file using existing preprocessing pipeline.
        Returns preprocessed signal ready for model input.
        """
        preproc = self.metadata.get('preprocessing', {
            'normalize': True,
            'artifact_removal': False,
            'dwt_downsampling': True,
            'tmax': 400,
            'dwt_level': 4
        })
        
        time, signal, snr = preprocess_signal(
            filepath,
            normalize=preproc.get('normalize', True),
            do_artifact_removal=preproc.get('artifact_removal', False),
            do_dwt_downsampling=preproc.get('dwt_downsampling', True),
            tmax=preproc.get('tmax', 400),
            dwt_level=preproc.get('dwt_level', 4)
        )
        
        return time, signal, snr
    
    def predict(self, filepath):
        """
        Full pipeline: preprocess raw file and predict class.
        
        Returns
        -------
        dict with keys:
            'class': predicted class name
            'probabilities': dict of class probabilities
            'time': time axis
            'signal': preprocessed signal
            'snr': signal-to-noise ratio
            'confidence': prediction confidence (max probability)
        """
        # Preprocess
        time, signal, snr = self.preprocess_raw_file(filepath)
        
        # Use CNN's predict method
        # Prepare input
        X = self.cnn._as_array(signal)
        X_prep = self.cnn._prepare_X(X.reshape(1, -1), fit=False)
        
        # Predict
        probs = self.model.predict(X_prep, verbose=0)[0]
        pred_idx = np.argmax(probs)
        pred_class = self.cnn.label_decoder[pred_idx]
        
        # Format probabilities
        prob_dict = {self.cnn.label_decoder[i]: float(probs[i]) 
                     for i in range(len(probs))}
        
        return {
            'class': pred_class,
            'probabilities': prob_dict,
            'time': time,
            'signal': signal,
            'snr': float(snr),
            'confidence': float(probs[pred_idx])
        }