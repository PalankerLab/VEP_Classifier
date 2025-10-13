import os
import logging
import warnings
from core import config

# ============= MUST BE SET BEFORE IMPORTING TENSORFLOW =============
# Suppress TensorFlow warnings FIRST
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Changed to 3 for ERROR only
os.environ['PYTHONHASHSEED'] = str(42)

# For deterministic operations
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
# ===================================================================

# Now import everything
import numpy as np
import random
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
from datetime import datetime

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Flatten,
    Dense, Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import GlorotUniform, Zeros

# Suppress additional warnings
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Set all random seeds AFTER imports
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Configure TensorFlow for reproducibility
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print(f"✓ Setup complete (TF {tf.__version__}, seed={SEED})")

class CNN1D:
    """
    1D CNN classifier with:
    - unified preprocessing
    - consistent label encoding
    - CV / train-val / blind prediction
    """

    def __init__(
        self,
        random_state=42,
        learning_rate=5e-4,
        regularization="l2", # options: None, "l2", "early_stopping"
        l2_lambda=1e-3,
        dropout_rate=0.5,
        patience=20, # for early stopping
    ):
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.patience = patience

        self.scaler = StandardScaler()
        self.label_encoder = None
        self.label_decoder = None

        self.regularization = regularization
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    def _kernel_regularizer(self):
        if self.regularization == "l2":
            return l2(self.l2_lambda)
        return None
    
    def _build_model(self, input_shape, n_classes):
        kr = self._kernel_regularizer()
        
        kernel_init = GlorotUniform(seed=self.random_state)
        bias_init = Zeros()
        
        model = Sequential([
            Input(shape=input_shape),
            Conv1D(
                16, 5,
                padding="same",
                activation="relu",
                kernel_regularizer=kr,
                kernel_initializer=kernel_init,
                bias_initializer=bias_init,
            ),
            MaxPooling1D(2),
            Conv1D(
                32, 3,
                padding="same",
                activation="relu",
                kernel_regularizer=kr,
                kernel_initializer=kernel_init,
                bias_initializer=bias_init,
            ),
            MaxPooling1D(2),
            Flatten(),
            Dense(
                32,
                activation="relu",
                kernel_regularizer=kr,
                kernel_initializer=kernel_init,
                bias_initializer=bias_init,
            ),
            Dropout(self.dropout_rate, seed=self.random_state),
            Dense(
                n_classes,
                activation="softmax",
                kernel_initializer=kernel_init,
                bias_initializer=bias_init,
            ),
        ])
        model.compile(
            optimizer=Adam(self.learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model


    # ------------------------------------------------------------------
    # Encoding / preprocessing
    # ------------------------------------------------------------------
    def _encode_labels(self, y):
        if self.label_encoder is None:
            enc = pd.Series(y).astype("category")
            self.label_encoder = enc.cat.categories
            self.label_decoder = dict(enumerate(self.label_encoder))
            return enc.cat.codes.values
        return pd.Series(y).map(
            {c: i for i, c in enumerate(self.label_encoder)}
        ).values

    def _prepare_X(self, X, fit=False):
        if fit:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        return np.expand_dims(X, axis=-1)
    

    def _as_array(self, X):
        if isinstance(X, dict):
            return np.hstack([X[k] for k in sorted(X.keys())])
        return np.asarray(X)

    def _class_weights(self, y):
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y),
            y=y,
        )
        return dict(enumerate(weights))

    def _callbacks(self):
        callbacks = []
        if self.regularization == "early_stopping":
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.patience,
                    restore_best_weights=True,
                )
            )
        return callbacks

    # ------------------------------------------------------------------
    # Core training step
    # ------------------------------------------------------------------
    def _fit_once(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=16, n_classes=None):
        K.clear_session()
        
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)
        
        if n_classes is None:
            n_classes = len(np.unique(y_train))

        X_train = self._prepare_X(X_train, fit=True)
        y_train_cat = to_categorical(y_train, n_classes)

        model = self._build_model(X_train.shape[1:], n_classes)

        fit_kwargs = dict(
            epochs=epochs,
            batch_size=batch_size,
            class_weight=self._class_weights(y_train),
            callbacks=self._callbacks(),
            verbose=0,
        )

        # Only add validation if provided
        if X_val is not None and y_val is not None:
            X_val = self._prepare_X(X_val, fit=False)
            y_val_cat = to_categorical(y_val, n_classes)
            fit_kwargs["validation_data"] = (X_val, y_val_cat)

        history = model.fit(X_train, y_train_cat, **fit_kwargs)

        if self.regularization == "early_stopping":
            print(f"Stopped at val accuracy: {history.history['val_accuracy'][-1]:.4f} at epoch {len(history.history['val_accuracy'])}")
        return model, history

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------
    def _summarize(self, values):
        values = np.asarray(values, dtype=float)
        return {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)),   # sample SD across folds
            "per_fold": values,
        }
    
    def fit_cv(self, X, y, epochs=50, batch_size=16, n_splits=10):
        X = self._as_array(X)
        y_codes = self._encode_labels(y)

        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        # pooled predictions (for confusion matrix etc.)
        y_true_all, y_pred_all = [], []

        # per-fold metrics (for mean ± SD)
        fold_acc = []
        fold_balacc = []
        fold_f1 = []

        for fold, (tr, te) in enumerate(cv.split(X, y_codes), 1):
            model, _ = self._fit_once(
                X[tr], y_codes[tr],
                X[te], y_codes[te],
                epochs, batch_size,
            )

            preds = np.argmax(
                model.predict(self._prepare_X(X[te], fit=False), verbose=0),
                axis=1
            )

            y_true_fold = y_codes[te]
            y_pred_fold = preds

            # store pooled for later confusion matrix etc.
            y_true_all.extend(y_true_fold)
            y_pred_all.extend(y_pred_fold)

            # fold metrics
            fold_acc.append(accuracy_score(y_true_fold, y_pred_fold))
            fold_balacc.append(balanced_accuracy_score(y_true_fold, y_pred_fold))
            fold_f1.append(f1_score(y_true_fold, y_pred_fold, average="weighted"))

        cv_summary = {
            "accuracy": self._summarize(fold_acc),
            "balanced_accuracy": self._summarize(fold_balacc),
            "f1": self._summarize(fold_f1),
        }

        return self.decode(y_true_all), self.decode(y_pred_all), cv_summary


    # ------------------------------------------------------------------
    # Train / validation split
    # ------------------------------------------------------------------
    def _balanced_train_test_split(self, X, y, val_size=0.1):
        # Use the seeded random state
        np.random.seed(self.random_state)
        
        n_classes = len(np.unique(y))
        N = int(len(X)*val_size)
        k_per_class = int(N // n_classes)
        val_indices = []
        train_indices = []

        for class_id in np.unique(y):
            class_indices = np.where(y == class_id)[0]
            np.random.shuffle(class_indices)  # Now uses the seeded state

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
        self, X, y,
        X_val=None, y_val=None,
        epochs=25, batch_size=8,
        val_split = True,
        val_size=0.2,
        compute_ig=False, ig_steps=50,
    ):
        X = self._as_array(X)
        y_codes = self._encode_labels(y)

        assert not (compute_ig and not val_split), "To compute IG, validation split must be used."
        assert not (self.regularization == "early_stopping" and not val_split), "To use early stopping, validation split must be used."

        if val_split:
            if X_val is None or y_val is None:
                X_train, X_val, y_train, y_val = self._balanced_train_test_split(
                    X, y_codes, val_size=val_size
                )
            else:
                X_train = X
                y_train = y_codes
                X_val = self._as_array(X_val)
                y_val = self._encode_labels(y_val)

            model, _ = self._fit_once(
                X_train, y_train,
                X_val, y_val,
                epochs, batch_size,
            )

            # --- Validation prediction ---
            X_val_prepared = self._prepare_X(X_val, fit=False)
            probs = model.predict(X_val_prepared, verbose=0)
            pred_codes = np.argmax(probs, axis=1)

            # --- Integrated Gradients (optional) ---
            ig_attributions = None
            if compute_ig:
                ig_attributions = []
                for i in range(len(X_val)):
                    x_raw = X_val[i]                 # shape (T,)
                    target_class = pred_codes[i]     # or y_val[i] if you prefer

                    ig = self.integrated_gradients(
                        model=model,
                        x=x_raw,
                        target_class=target_class,
                        steps=ig_steps,
                    )
                    ig_attributions.append(ig)

                ig_attributions = np.stack(ig_attributions)  # (N_val, T)
                
            return (
                self.decode(y_val),
                self.decode(pred_codes),
                model,
                ig_attributions,
            )

        else: # just train on all data
            model, _ = self._fit_once(
                X, y_codes,
                X_val=None, y_val=None,
                epochs=epochs,
                batch_size=batch_size,
            )

            return None, None, model, None


    # ------------------------------------------------------------------
    # Leave-one-animal-out cross-validation
    # ------------------------------------------------------------------
    def fit_leave_one_animal_out(self, X, y, file_list, epochs=50, batch_size=16):
        """
        Leave-one-animal-out cross-validation.

        For each unique animal (identified via parse_filename_info on file_list),
        trains on all other animals and evaluates on the held-out animal.

        Returns
        -------
        results : dict with keys
            "per_animal"  – dict {animal_id: {"n_trials": int,
                                               "acc": float,
                                               "bal_acc": float,
                                               "f1": float}}
            "mean_by_animal" – {"acc", "bal_acc", "f1"} averaged over animals
            "mean_by_trial"  – {"acc", "bal_acc", "f1"} averaged over all trials
                               (= pooled predictions across all held-out folds)
        """
        from core.helpers import parse_filename_info

        X = self._as_array(X)
        y_codes = self._encode_labels(y)
        n_classes = len(np.unique(y_codes))

        animals = np.array([parse_filename_info(f)[0] for f in file_list])
        unique_animals = sorted(set(animals))

        per_animal = {}
        y_true_all, y_pred_all = [], []

        for animal in unique_animals:
            test_mask  = animals == animal
            train_mask = ~test_mask

            if train_mask.sum() == 0 or test_mask.sum() == 0:
                print(f"Skipping animal {animal}: insufficient data.")
                continue


            model, _ = self._fit_once(
                X[train_mask], y_codes[train_mask],
                X[test_mask],  y_codes[test_mask],
                epochs, batch_size,
                n_classes=n_classes,
            )

            preds = np.argmax(
                model.predict(self._prepare_X(X[test_mask], fit=False), verbose=0),
                axis=1,
            )

            y_true = y_codes[test_mask]
            y_true_all.extend(y_true)
            y_pred_all.extend(preds)

            per_animal[animal] = {
                "n_trials":  int(test_mask.sum()),
                "acc":       float(accuracy_score(y_true, preds)),
                "bal_acc":   float(balanced_accuracy_score(y_true, preds)),
                "f1":        float(f1_score(y_true, preds, average="weighted")),
            }

        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)

        mean_by_animal = {
            "acc":     float(np.mean([v["acc"]     for v in per_animal.values()])),
            "bal_acc": float(np.mean([v["bal_acc"] for v in per_animal.values()])),
            "f1":      float(np.mean([v["f1"]      for v in per_animal.values()])),
        }
        mean_by_trial = {
            "acc":     float(accuracy_score(y_true_all, y_pred_all)),
            "bal_acc": float(balanced_accuracy_score(y_true_all, y_pred_all)),
            "f1":      float(f1_score(y_true_all, y_pred_all, average="weighted")),
        }

        return {
            "per_animal":      per_animal,
            "mean_by_animal":  mean_by_animal,
            "mean_by_trial":   mean_by_trial,
        }

    # ------------------------------------------------------------------
    # Blind prediction
    # ------------------------------------------------------------------
    def predict(self, model, X):
        print("Predicting...")
        X = self._as_array(X) 
        X = self._prepare_X(X, fit=False)
        preds = np.argmax(model.predict(X), axis=1)
        return self.decode(preds)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def decode(self, y_codes):
        return np.array([self.label_decoder[i] for i in y_codes])


    def evaluate(self, y_true, y_pred, verbose=True, title=None):
        labels = list(self.label_encoder)
        conf_matrix = confusion_matrix(y_true, y_pred,labels=labels, normalize="true")

        if verbose:
            plt.figure(figsize=(6,5)) 
            sns.heatmap(conf_matrix, annot=True, fmt=".2f", cmap="viridis", xticklabels=labels, yticklabels=labels, cbar_kws={'label': 'Proportion'}) 
            plt.xlabel("Predicted Label") 
            plt.ylabel("True Label") 
            if title:
                plt.title(title)
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

    def integrated_gradients(
        self,
        model,
        x,                  # shape (T,)
        target_class,
        baseline=None,
        steps=50):
        """
        Integrated Gradients for a single 1D sample.
        Returns attribution of shape (T,)
        """
        if baseline is None:
            baseline = np.zeros_like(x)

        x = tf.convert_to_tensor(x, dtype=tf.float32)
        baseline = tf.convert_to_tensor(baseline, dtype=tf.float32)

        alphas = tf.linspace(0.0, 1.0, steps)

        with tf.GradientTape() as tape:
            tape.watch(x)

            # interpolate inputs
            interpolated = baseline + alphas[:, None] * (x - baseline)
            interpolated = tf.expand_dims(interpolated, axis=-1)  # (steps, T, 1)

            preds = model(interpolated, training=False)
            target = preds[:, target_class]

        grads = tape.gradient(target, interpolated)        # (steps, T, 1)
        avg_grads = tf.reduce_mean(grads, axis=0)          # (T, 1)

        ig = (x - baseline) * tf.squeeze(avg_grads)        # (T,)
        return ig.numpy()

    # ------------------------------------------------------------------
    # Model saving and loading
    # ------------------------------------------------------------------
    def save_model(self, model, save_dir="saved_models", prefix="model"):
        """
        Save model, scaler, and metadata to disk.
        
        Parameters
        ----------
        model : keras.Model
            Trained model to save
        save_dir : str
            Directory to save files
        prefix : str
            Prefix for saved files
            
        Returns
        -------
        dict with keys 'model_path', 'scaler_path', 'metadata_path'
        """
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{prefix}_{timestamp}"
        
        model_path = os.path.join(save_dir, f"{base_name}.keras")
        scaler_path = os.path.join(save_dir, f"{base_name}_scaler.pkl")
        metadata_path = os.path.join(save_dir, f"{base_name}_metadata.json")
        
        # Save model
        model.save(model_path)
        print(f"✓ Model saved to: {model_path}")
        
        # Save scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✓ Scaler saved to: {scaler_path}")
        
        # Save metadata
        metadata = {
            "training_date": timestamp,
            "n_classes": len(self.label_encoder) if self.label_encoder is not None else None,
            "classes": list(self.label_encoder) if self.label_encoder is not None else None,
            "label_decoder": self.label_decoder,
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "regularization": self.regularization,
                "l2_lambda": self.l2_lambda,
                "dropout_rate": self.dropout_rate,
                "patience": self.patience,
                "random_state": self.random_state,
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata saved to: {metadata_path}")
        
        return {
            "model_path": model_path,
            "scaler_path": scaler_path,
            "metadata_path": metadata_path,
        }
    
    @classmethod
    def load_model(cls, model_path, scaler_path, metadata_path):
        """
        Load a saved model with its scaler and metadata.
        
        Parameters
        ----------
        model_path : str
            Path to saved Keras model
        scaler_path : str
            Path to saved scaler
        metadata_path : str
            Path to saved metadata
            
        Returns
        -------
        tuple of (model, CNN1D instance)
        """
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Create CNN1D instance with saved hyperparameters
        hp = metadata["hyperparameters"]
        cnn = cls(
            random_state=hp["random_state"],
            learning_rate=hp["learning_rate"],
            regularization=hp["regularization"],
            l2_lambda=hp["l2_lambda"],
            dropout_rate=hp["dropout_rate"],
            patience=hp["patience"],
        )
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            cnn.scaler = pickle.load(f)
        
        # Load label encoder/decoder
        cnn.label_encoder = pd.Index(metadata["classes"])
        cnn.label_decoder = {int(k): v for k, v in metadata["label_decoder"].items()}
        
        # Load model
        model = tf.keras.models.load_model(model_path)
        
        print(f"✓ Model loaded from: {model_path}")
        print(f"✓ Classes: {metadata['classes']}")
        
        return model, cnn
    
    # ------------------------------------------------------------------
    # Train on full dataset and save
    # ------------------------------------------------------------------
    def fit_and_save(
        self,
        X, y,
        epochs=50,
        batch_size=16,
        save_dir="saved_models",
        model_name="model",
        device="unknown",
        n_classes=None,
        class_counts=None,
        preprocessing_info=None,
    ):
        """
        Train on full dataset and save model, scaler, and metadata.
        
        Parameters
        ----------
        X : array-like
            Feature matrix
        y : array-like
            Labels
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        save_dir : str
            Directory to save files
        model_name : str
            Prefix for saved files
        device : str
            Device name (e.g., "PRIMA")
        n_classes : int or str
            Number of classes (for metadata)
        class_counts : dict
            Class distribution (for metadata)
        preprocessing_info : dict
            Preprocessing parameters used
            
        Returns
        -------
        dict with saved file paths
        """
        print("=" * 60)
        print(f"Training {model_name} on full dataset")
        print("=" * 60)
        
        X = self._as_array(X)
        y_codes = self._encode_labels(y)
        
        # Dataset statistics
        unique, counts = np.unique(y, return_counts=True)
        print(f"\nDataset statistics:")
        print(f"Total samples: {len(X)}")
        for label, count in zip(unique, counts):
            print(f"  {label}: {count}")
        
        # Train on full dataset
        print(f"\nTraining for {epochs} epochs...")
        _, _, model, _ = self.fit_train_val(
            X, y,
            val_split=False,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Prepare extended metadata
        metadata_extra = {
            "device": device,
            "n_samples": len(X),
            "class_counts": class_counts or {label: int(count) for label, count in zip(unique, counts)},
            "training_params": {
                "epochs": epochs,
                "batch_size": batch_size,
            }
        }
        
        if preprocessing_info:
            metadata_extra["preprocessing"] = preprocessing_info
        
        # Save everything
        paths = self.save_model(model, save_dir=save_dir, prefix=model_name)
        
        # Update metadata with extra info
        with open(paths["metadata_path"], 'r') as f:
            metadata = json.load(f)
        
        metadata.update(metadata_extra)
        
        with open(paths["metadata_path"], 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("\n" + "=" * 60)
        print("Training complete!")
        print("=" * 60)
        print(f"Model files saved with prefix: {model_name}")
        print("=" * 60)
        
        return paths