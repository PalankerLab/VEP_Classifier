import os
import warnings

# ============= MUST BE SET BEFORE IMPORTING SKLEARN =============
os.environ['PYTHONHASHSEED'] = str(42)
# ===================================================================

import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    confusion_matrix, classification_report
)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set all random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print(f"✓ Setup complete (seed={SEED})")

class LDA:
    def __init__(self, random_state=42, dim_reducer=None):
        self.random_state = random_state
        self.dim_reducer = dim_reducer
        self.scaler = StandardScaler()
        self.label_encoder = None
        self.label_decoder = None
        
        # Set random seeds again to ensure consistency
        random.seed(self.random_state)
        np.random.seed(self.random_state)

    def _prepare_X(self, X, fit=False):
        if fit:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        return X
    

    def fit_cv(self, X, y, n_splits, use_pca=False, use_ica=False):
        X = np.asarray(X)
        y = np.asarray(y)

        y_true, y_pred = [], []
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # --- Standardization ---
            X_train = self._prepare_X(np.vstack(X_train), fit=True)
            X_test = self._prepare_X(np.vstack(X_test), fit=False)

            # --- Dimensionality Reduction ---
            if self.dim_reducer:
                if use_pca:
                    X_train, X_test = self.dim_reducer.pca(X_train, X_test)
                elif use_ica:
                    X_train, X_test = self.dim_reducer.ica(X_train, X_test)
     
            # --- Train LDA ---
            lda = LinearDiscriminantAnalysis(
                solver='svd',  # 'svd' doesn't require tuning shrinkage
                store_covariance=False
            )

            lda.fit(X_train, y_train)
            preds = lda.predict(X_test)

            y_true.extend(y_test)
            y_pred.extend(preds)
        return np.array(y_true), np.array(y_pred)
    

    def _balanced_train_test_split(self, X, y, val_size=0.1):
        # Use numpy RandomState for reproducibility
        rng = np.random.RandomState(self.random_state)
        n_classes = len(np.unique(y))
        N = int(len(X)*val_size)
        k_per_class = int(N // n_classes)
        val_indices = []
        train_indices = []

        for class_id in np.unique(y):
            class_indices = np.where(y == class_id)[0]
            rng.shuffle(class_indices)

            # take k samples for this class
            test_cls = class_indices[:k_per_class]
            train_cls = class_indices[k_per_class:]

            val_indices.extend(test_cls)
            train_indices.extend(train_cls)
        val_indices = np.array(val_indices)
        train_indices = np.array(train_indices)

        X_train, X_test = X[train_indices], X[val_indices]
        y_train, y_test = y[train_indices], y[val_indices]
        return X_train, X_test, y_train, y_test
    
    def fit_train_val(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
        use_pca=False,
        use_ica=False,
        val_size=0.2,
    ):
        X = np.asarray(X)
        y = np.asarray(y)

        # --- Train / validation split ---
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = self._balanced_train_test_split(X, y, val_size=val_size)
        else:
            X_train, y_train = X, y
            X_val = np.asarray(X_val)
            y_val = np.asarray(y_val)

        # --- Feature preparation for training/prediction ---
        X_train_p = self._prepare_X(np.vstack(X_train), fit=True)
        X_val_p   = self._prepare_X(np.vstack(X_val),   fit=False)

        # --- Optional dimensionality reduction ---
        if self.dim_reducer:
            if use_pca:
                X_train_p, X_val_p = self.dim_reducer.pca(X_train_p, X_val_p)
            elif use_ica:
                X_train_p, X_val_p = self.dim_reducer.ica(X_train_p, X_val_p)

        # --- Train LDA ---
        model = LinearDiscriminantAnalysis(
            solver='svd',
            store_covariance=False
        )
        model.fit(X_train_p, y_train)

        # --- Validation prediction ---
        preds = model.predict(X_val_p)

        return np.array(y_val), np.array(preds), model, None


    def predict(self, model, X):
        X = np.asarray(X)
        X = self._prepare_X(np.vstack(X), fit=False)
        return model.predict(X)

    def evaluate(self, y_true, y_pred, verbose=True):
        labels = np.unique(np.concatenate([y_true, y_pred]))
        conf_matrix = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

        if verbose:
            plt.figure(figsize=(6,5)) 
            sns.heatmap(conf_matrix, annot=True, fmt=".2f", cmap="viridis", xticklabels=labels, yticklabels=labels, cbar_kws={'label': 'Proportion'}) 
            plt.title(f"Mean Confusion Matrix")
            plt.xlabel("Predicted Label") 
            plt.ylabel("True Label") 
            plt.tight_layout()
            plt.show()

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, average="weighted"),
            "confusion_matrix": conf_matrix,
            "report": classification_report(
                y_true, y_pred, output_dict=True, digits=3
            ),
        }