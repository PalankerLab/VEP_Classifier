import pandas as pd
from itertools import product
import numpy as np
import os
import time
import traceback
import warnings

# --- Custom imports ---
from core.helpers import get_data
from classifiers.CNN_classifier import CNN1D
from core.preprocessing import preprocess_save_all
from core import config

warnings.filterwarnings(
    "ignore",
    message="The structure of `inputs` doesn't match the expected structure"
)


# ==========================================================
# CONFIG
# ==========================================================
DEVICE = "PRIMA_LE_DA"
N_SPLITS = 10
LABELS = ["BC_Only", "BC_and_RGC"]

RESULTS_DIR = os.path.join(config.OUTPUTS_DIR, "results", "hyperparam_results")
RESULTS_CSV = os.path.join(RESULTS_DIR, "results.csv")
LOG_FILE = os.path.join(RESULTS_DIR, "run.log")

os.makedirs(RESULTS_DIR, exist_ok=True)

# All devices to include in combined dataset
ALL_DEVICES = ["PRIMA_LE_DA", "PRIMA_RCS_DA", "MP20_LE_DA", "MP20_RCS_LA", "RB20_RCS_LA"]

# ==========================================================
# LOGGING
# ==========================================================
def log(msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ==========================================================
# HYPERPARAM GRID
# ==========================================================
HYPERPARAM_GRID = {
    # preprocessing params
    # for tmax make list autmatic of values in 10 ms steps from 20 to 400 ms
    "TMAX": list(range(20, 401, 10)),
    "NORMALIZE": [True],
    "SNR_FILTERING": [True],
    "ARTIFACT_REMOVAL": [False],
    "DWT_DOWNSAMPLING": [True],
    "SNR_THRESHOLD": [1.0],
    # learning params
    #"regularization": ["l2"],
    # "learning_rate": [1e-4, 5e-4, 1e-3],  # Most important learning param
    # "epochs": [30, 50, 100], 
    # "batch_size": [8, 16, 32], 
    # "dropout_rate": [0.2, 0.3, 0.4], 
    # "l2_lambda": [1e-4, 1e-3, 1e-2], 

    # standard values for final search
    "regularization": ["l2"],
    "learning_rate": [config.LR],  # Most important learning param
    "epochs": [config.EPOCHS],
    "batch_size": [config.BATCHSIZE],
    "dropout_rate": [config.DROPOUT],
    "l2_lambda": [config.L2_LAMBDA],
}


def generate_configs(grid):
    keys = list(grid.keys())
    for values in product(*grid.values()):
        yield dict(zip(keys, values))


# ==========================================================
# MAIN SEARCH LOOP
# ==========================================================
def hyperparam_search_cv():
    # Load existing results if present (resume-safe)
    if os.path.exists(RESULTS_CSV):
        df_results = pd.read_csv(RESULTS_CSV)
        log(f"Resuming search — {len(df_results)} configs already completed.")
    else:
        df_results = pd.DataFrame()
        log("Starting new hyperparameter search.")

    for idx, cfg in enumerate(generate_configs(HYPERPARAM_GRID), start=1):
        log("=" * 60)
        log(f"Config {idx}")
        for k, v in cfg.items():
            log(f"  {k}: {v}")

        try:
            # -------------------------------
            # Preprocessing ALL DEVICES
            # -------------------------------
            log("Preprocessing all devices...")
            preprocess_save_all(
                devices=ALL_DEVICES,
                normalize=cfg["NORMALIZE"],
                do_artifact_removal=cfg["ARTIFACT_REMOVAL"],
                tmax=cfg["TMAX"],
                do_dwt_downsampling=cfg["DWT_DOWNSAMPLING"],
                SNR_filtering=cfg["SNR_FILTERING"],
                SNR_threshold=cfg["SNR_THRESHOLD"],
                include_blind=False,
                labels=LABELS,
            )

            # -------------------------------
            # Load data for all conditions
            # -------------------------------
            X_prima_le_da, y_prima_le_da, _, _ = get_data(device="PRIMA_LE_DA", labels=LABELS)
            X_mp20_le_da, y_mp20_le_da, _, _ = get_data(device="MP20_LE_DA", labels=LABELS)
            X_prima_rcs_da, y_prima_rcs_da, _, _ = get_data(device="PRIMA_RCS_DA", labels=LABELS)
            X_mp20_rcs_la, y_mp20_rcs_la, _, _ = get_data(device="MP20_RCS_LA", labels=LABELS)
            X_rb20_rcs_la, y_rb20_rcs_la, _, _ = get_data(device="RB20_RCS_LA", labels=LABELS)
            # -------------------------------
            # Load COMBINED dataset from all devices
            # -------------------------------
            log("Loading combined dataset from all devices...")
            X_all = []
            y_all = []
            device_labels = []
            
            for device in ALL_DEVICES:
                X_device, y_device, _, _ = get_data(device=device, labels=LABELS)
                X_all.extend(X_device)
                y_all.extend(y_device)
                device_labels.extend([device] * len(X_device))
                log(f"  {device}: {len(X_device)} samples")
            
            X_all = np.array(X_all)
            y_all = np.array(y_all)
            device_labels = np.array(device_labels)
            
            log(f"Total combined dataset: {len(X_all)} samples")
            unique_labels, counts = np.unique(y_all, return_counts=True)
            for label, count in zip(unique_labels, counts):
                log(f"  {label}: {count}")

            # -------------------------------
            # 0) COMBINED DATASET (All Devices) -> Cross-validation
            # -------------------------------
            log("Running 10-fold CV on combined dataset...")
            cnn_cv_combined = CNN1D(
                regularization=cfg["regularization"],
                learning_rate=cfg["learning_rate"],
                dropout_rate=cfg["dropout_rate"],
                l2_lambda=cfg["l2_lambda"],
                patience=20,
            )

            y_true_combined, y_pred_combined, cv_results_combined = cnn_cv_combined.fit_cv(
                X_all, y_all,
                epochs=cfg["epochs"],
                batch_size=cfg["batch_size"],
                n_splits=N_SPLITS,
            )
            metrics_combined = cnn_cv_combined.evaluate(y_true_combined, y_pred_combined, verbose=False)

            # -------------------------------
            # 1) PRIMA_LE_DA -> PRIMA_LE_DA (Cross-validation)
            # -------------------------------
            log("Running 10-fold CV on PRIMA_LE_DA...")
            cnn_cv = CNN1D(
                regularization=cfg["regularization"],
                learning_rate=cfg["learning_rate"],
                dropout_rate=cfg["dropout_rate"],
                l2_lambda=cfg["l2_lambda"],
                patience=20,
            )

            y_true_prima_le_da, y_pred_prima_le_da, cv_results_prima = cnn_cv.fit_cv(
                X_prima_le_da, y_prima_le_da,
                epochs=cfg["epochs"],
                batch_size=cfg["batch_size"],
                n_splits=N_SPLITS,
            )
            metrics_prima_le_da = cnn_cv.evaluate(y_true_prima_le_da, y_pred_prima_le_da, verbose=False)

            # -------------------------------
            # 2) Train on PRIMA_LE_DA, test on MP20_LE_DA
            # -------------------------------
            log("Training on PRIMA_LE_DA, testing on MP20_LE_DA...")
            cnn_mp20_le_da = CNN1D(
                regularization=cfg["regularization"],
                learning_rate=cfg["learning_rate"],
                dropout_rate=cfg["dropout_rate"],
                l2_lambda=cfg["l2_lambda"],
                patience=20,
            )

            y_true_mp20_le_da, y_pred_mp20_le_da, _, _ = cnn_mp20_le_da.fit_train_val(
                X=X_prima_le_da,
                y=y_prima_le_da,
                X_val=X_mp20_le_da,
                y_val=y_mp20_le_da,
                epochs=cfg["epochs"],
                batch_size=cfg["batch_size"],
                val_split=True,
                compute_ig=False,
            )
            metrics_mp20_le_da = cnn_mp20_le_da.evaluate(y_true_mp20_le_da, y_pred_mp20_le_da, verbose=False)

            # -------------------------------
            # 3) Train on PRIMA_LE_DA, test on PRIMA_RCS_DA
            # -------------------------------
            log("Training on PRIMA_LE_DA, testing on PRIMA_RCS_DA...")
            cnn_prima_rcs_da = CNN1D(
                regularization=cfg["regularization"],
                learning_rate=cfg["learning_rate"],
                dropout_rate=cfg["dropout_rate"],
                l2_lambda=cfg["l2_lambda"],
                patience=20,
            )

            y_true_prima_rcs_da, y_pred_prima_rcs_da, _, _ = cnn_prima_rcs_da.fit_train_val(
                X=X_prima_le_da,
                y=y_prima_le_da,
                X_val=X_prima_rcs_da,
                y_val=y_prima_rcs_da,
                epochs=cfg["epochs"],
                batch_size=cfg["batch_size"],
                val_split=True,
                compute_ig=False,
            )
            metrics_prima_rcs_da = cnn_prima_rcs_da.evaluate(y_true_prima_rcs_da, y_pred_prima_rcs_da, verbose=False)

            # -------------------------------
            # 4) Train on PRIMA_LE_DA, test on MP20_RCS_LA
            # -------------------------------
            log("Training on PRIMA_LE_DA, testing on MP20_RCS_LA...")
            cnn_mp20_rcs_la = CNN1D(
                regularization=cfg["regularization"],
                learning_rate=cfg["learning_rate"],
                dropout_rate=cfg["dropout_rate"],
                l2_lambda=cfg["l2_lambda"],
                patience=20,
            )

            y_true_mp20_rcs_la, y_pred_mp20_rcs_la, _, _ = cnn_mp20_rcs_la.fit_train_val(
                X=X_prima_le_da,
                y=y_prima_le_da,
                X_val=X_mp20_rcs_la,
                y_val=y_mp20_rcs_la,
                epochs=cfg["epochs"],
                batch_size=cfg["batch_size"],
                val_split=True,
                compute_ig=False,
            )
            metrics_mp20_rcs_la = cnn_mp20_rcs_la.evaluate(y_true_mp20_rcs_la, y_pred_mp20_rcs_la, verbose=False)

            # -------------------------------
            # 5) Train on PRIMA_LE_DA, test on RB20_RCS_LA
            # -------------------------------
            log("Training on PRIMA_LE_DA, testing on RB20_RCS_LA...")
            cnn_rb20_rcs_la = CNN1D(
                regularization=cfg["regularization"],
                learning_rate=cfg["learning_rate"],
                dropout_rate=cfg["dropout_rate"],
                l2_lambda=cfg["l2_lambda"],
                patience=20,
            )

            y_true_rb20_rcs_la, y_pred_rb20_rcs_la, _, _ = cnn_rb20_rcs_la.fit_train_val(
                X=X_prima_le_da,
                y=y_prima_le_da,
                X_val=X_rb20_rcs_la,
                y_val=y_rb20_rcs_la,
                epochs=cfg["epochs"],
                batch_size=cfg["batch_size"],
                val_split=True,
                compute_ig=False,
            )
            metrics_rb20_rcs_la = cnn_rb20_rcs_la.evaluate(y_true_rb20_rcs_la, y_pred_rb20_rcs_la, verbose=False)

            # -------------------------------
            # Save results
            # -------------------------------
            result_row = {
                **cfg,
                # Combined dataset CV results
                "balanced_accuracy_combined": metrics_combined["balanced_accuracy"],
                "balanced_accuracy_combined_mean": cv_results_combined["balanced_accuracy"]["mean"],
                "balanced_accuracy_combined_sd": cv_results_combined["balanced_accuracy"]["sd"],
                # PRIMA_LE_DA CV results
                "balanced_accuracy_prima_le_da": metrics_prima_le_da["balanced_accuracy"],
                "balanced_accuracy_prima_le_da_mean": cv_results_prima["balanced_accuracy"]["mean"],
                "balanced_accuracy_prima_le_da_sd": cv_results_prima["balanced_accuracy"]["sd"],
                # Generalization test results
                "balanced_accuracy_mp20_le_da": metrics_mp20_le_da["balanced_accuracy"],
                "balanced_accuracy_prima_rcs_da": metrics_prima_rcs_da["balanced_accuracy"],
                "balanced_accuracy_mp20_rcs_la": metrics_mp20_rcs_la["balanced_accuracy"],
                "balanced_accuracy_rb20_rcs_la": metrics_rb20_rcs_la["balanced_accuracy"],
            }

            df_results = pd.concat(
                [df_results, pd.DataFrame([result_row])],
                ignore_index=True,
            )
            df_results.to_csv(RESULTS_CSV, index=False)

            log(
                f"DONE → COMBINED(CV) BalAcc: {metrics_combined['balanced_accuracy']:.4f} "
                f"(μ={cv_results_combined['balanced_accuracy']['mean']:.4f} ± "
                f"{cv_results_combined['balanced_accuracy']['sd']:.4f}), "
                f"F1: {metrics_combined['f1']:.4f} "
                f"(μ={cv_results_combined['f1']['mean']:.4f} ± "
                f"{cv_results_combined['f1']['sd']:.4f})"
            )
            log(
                f"PRIMA_LE_DA(CV) BalAcc: {metrics_prima_le_da['balanced_accuracy']:.4f} "
                f"(μ={cv_results_prima['balanced_accuracy']['mean']:.4f} ± "
                f"{cv_results_prima['balanced_accuracy']['sd']:.4f}), "
                f"F1: {metrics_prima_le_da['f1']:.4f} "
                f"(μ={cv_results_prima['f1']['mean']:.4f} ± "
                f"{cv_results_prima['f1']['sd']:.4f})"
            )
            log(
                f"MP20_LE_DA(Test) BalAcc: {metrics_mp20_le_da['balanced_accuracy']:.4f}, "
                f"F1: {metrics_mp20_le_da['f1']:.4f} | "
                f"PRIMA_RCS_DA(Test) BalAcc: {metrics_prima_rcs_da['balanced_accuracy']:.4f}, "
                f"F1: {metrics_prima_rcs_da['f1']:.4f} | "
                f"MP20_RCS_LA(Test) BalAcc: {metrics_mp20_rcs_la['balanced_accuracy']:.4f}, "
                f"F1: {metrics_mp20_rcs_la['f1']:.4f} | "
                f"RB20_RCS_LA(Test) BalAcc: {metrics_rb20_rcs_la['balanced_accuracy']:.4f}, "
                f"F1: {metrics_rb20_rcs_la['f1']:.4f} | "
            )

            log(f"Results saved to {RESULTS_CSV}")

        except Exception as e:
            log("ERROR during config execution")
            log(str(e))
            log(traceback.format_exc())
            continue

    log("Hyperparameter search finished.")


# ==========================================================
# ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    hyperparam_search_cv()