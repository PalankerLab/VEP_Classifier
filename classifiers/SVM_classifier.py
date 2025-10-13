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

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
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

class SVM:
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
     
            # --- Train SVM ---
            svm = SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced",
                random_state=self.random_state
            )

            svm.fit(X_train, y_train)
            preds = svm.predict(X_test)

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
            rng.shuffle(class_indices)  # Use rng instead of np.random

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
        compute_ig=False,
        ig_steps=50,
        ig_eps=1e-3,
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

        # Keep raw copies for IG attribution in original feature space
        X_val_raw = np.asarray(X_val)

        # --- Feature preparation for training/prediction ---
        X_train_p = self._prepare_X(np.vstack(X_train), fit=True)
        X_val_p   = self._prepare_X(np.vstack(X_val),   fit=False)

        # --- Optional dimensionality reduction ---
        if self.dim_reducer:
            if use_pca:
                X_train_p, X_val_p = self.dim_reducer.pca(X_train_p, X_val_p)
            elif use_ica:
                X_train_p, X_val_p = self.dim_reducer.ica(X_train_p, X_val_p)

        # --- Train SVM ---
        # decision_function_shape='ovr' makes multiclass decision_function -> (n_samples, n_classes)
        model = SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            decision_function_shape="ovr",
            random_state=self.random_state  # Added random_state here
        )
        model.fit(X_train_p, y_train)

        # --- Validation prediction ---
        preds = model.predict(X_val_p)

        # --- Integrated Gradients (optional) ---
        ig_attributions = None
        if compute_ig:
            ig_list = []
            for i in range(len(X_val_raw)):
                x_raw = X_val_raw[i]       # (F,) original feature space
                target_label = preds[i]    # predicted class (same as CNN logic)

                ig = self.integrated_gradients(
                    model=model,
                    x=x_raw,
                    target_label=target_label,
                    baseline=None,
                    steps=ig_steps,
                    eps=ig_eps,
                    use_pca=use_pca,
                    use_ica=use_ica,
                )
                ig_list.append(ig)

            ig_attributions = np.stack(ig_list, axis=0)  # (N_val, F)

        return np.array(y_val), np.array(preds), model, ig_attributions


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

    def _svm_score(self, model, X_prepared, target_class_index, target_label=None):
        """
        Returns a scalar score for the target class.
        - multiclass (ovr): decision_function -> (1, n_classes)
        - binary: decision_function -> (1,) or (1,); treat positive class score as +df
                 and negative class score as -df
        """
        df = model.decision_function(X_prepared)

        # Binary case
        if df.ndim == 1:
            # df shape (1,)
            df_val = float(df[0])
            # model.classes_ gives the label order; positive class is classes_[1]
            # If target_label is the "negative" class -> flip sign
            if target_label is not None and target_label == model.classes_[0]:
                return -df_val
            return df_val

        # Multiclass OVR case: df shape (1, n_classes)
        return float(df[0, target_class_index])

    def _finite_diff_grad(self, f_scalar, x, eps=1e-3):
        """
        Central finite difference gradient of scalar function f(x) wrt x.
        x: (F,)
        returns grad: (F,)
        """
        x = np.asarray(x, dtype=float)
        grad = np.zeros_like(x, dtype=float)

        for j in range(x.size):
            x_pos = x.copy()
            x_neg = x.copy()
            x_pos[j] += eps
            x_neg[j] -= eps
            grad[j] = (f_scalar(x_pos) - f_scalar(x_neg)) / (2.0 * eps)

        return grad

    def integrated_gradients(
        self,
        model,
        x,                      # (F,) in original feature space
        target_label,           # class label (not index)
        baseline=None,
        steps=50,
        eps=1e-3,
        use_pca=False,
        use_ica=False,
    ):
        """
        Integrated Gradients for SVM using finite-difference gradients.

        Returns attribution of shape (F,) in the ORIGINAL input feature space.
        """
        x = np.asarray(x, dtype=float).reshape(-1)

        if baseline is None:
            baseline = np.zeros_like(x, dtype=float)
        else:
            baseline = np.asarray(baseline, dtype=float).reshape(-1)

        # Map label -> index in model.classes_ (for multiclass OVR)
        classes = model.classes_
        target_class_index = int(np.where(classes == target_label)[0][0])

        # Scalar function f(x_raw) = decision score for target class
        def f_scalar(x_raw):
            X_prepared = self._prep_single(x_raw, model, use_pca=use_pca, use_ica=use_ica)
            return self._svm_score(
                model,
                X_prepared,
                target_class_index=target_class_index,
                target_label=target_label
            )

        # Interpolate inputs
        alphas = np.linspace(0.0, 1.0, steps, dtype=float)
        grads = []

        for a in alphas:
            x_a = baseline + a * (x - baseline)
            g = self._finite_diff_grad(f_scalar, x_a, eps=eps)
            grads.append(g)

        avg_grads = np.mean(np.stack(grads, axis=0), axis=0)  # (F,)
        ig = (x - baseline) * avg_grads                        # (F,)
        return ig

    def _prep_single(self, x_raw, model, use_pca=False, use_ica=False):
        """
        Takes x_raw shape (F,) in *original* feature space and returns
        X_prepared shape (1, F_prepared) as the model expects.
        """
        X = np.asarray(x_raw, dtype=float).reshape(1, -1)
        X = self._prepare_X(X, fit=False)

        if self.dim_reducer:
            if use_pca:
                # assumes dim_reducer.pca returns (X_train_red, X_test_red)
                # so we call it in a "transform-like" way:
                _, X = self.dim_reducer.pca(X_train=None, X_test=X)
            elif use_ica:
                _, X = self.dim_reducer.ica(X_train=None, X_test=X)

        return X