import pandas as pd
import numpy as np
import os
import time
import traceback
import warnings

# --- Custom imports ---
from core.helpers import get_data
from classifiers.CNN_classifier import CNN1D
from classifiers.SVM_classifier import SVM
from classifiers.LDA_classifier import LDA
from classifiers.RandomForest_classifier import RandomForest
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
MODELS = ["CNN1D", "SVM", "LDA", "RandomForest"]

LR = config.LR
EPOCHS = config.EPOCHS
BATCHSIZE = config.BATCHSIZE
DROPOUT = config.DROPOUT
L2_LAMBDA = config.L2_LAMBDA

RESULTS_DIR = os.path.join(config.OUTPUTS_DIR, "results", "model_comparison_results")
RESULTS_CSV = os.path.join(RESULTS_DIR, "results.csv")
LOG_FILE = os.path.join(RESULTS_DIR, "run.log")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================================================
# LOGGING
# ====================================================‚======
def log(msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

# ==========================================================
# MAIN SEARCH LOOP
# ==========================================================
def compare_models():
    # Load existing results if present (resume-safe)
    if os.path.exists(RESULTS_CSV):
        df_results = pd.read_csv(RESULTS_CSV)
        log(f"Resuming search — {len(df_results)} configs already completed.")
    else:
        df_results = pd.DataFrame()
        log("Starting new model comparison.")

    for idx, model in enumerate(MODELS):
        log("=" * 60)
        log(f"Model {idx}")

        try:
            # -------------------------------
            # Preprocessing
            # -------------------------------
            preprocess_save_all(
                normalize=True,
                do_artifact_removal=False,
                do_dwt_downsampling=True,
                tmax=400,
                SNR_filtering=True,
                SNR_threshold=1.0,
                include_blind=False,
                labels=LABELS,
            )

            # -------------------------------
            # Load data (PRIMA_LE_DA for CV)
            # -------------------------------
            X_prima, y_prima, _, _ = get_data(device="PRIMA_LE_DA", labels=LABELS)

             # -------------------------------
            # Load data (PRIMA_RCS_DA for CV)
            # -------------------------------
            X_prima_rcs, y_prima_rcs, _, _ = get_data(device="PRIMA_RCS_DA", labels=LABELS)


            # -------------------------------
            # Load data (MP20_LE_DA for external test)
            # -------------------------------
            X_mp20,  y_mp20,  _, _ = get_data(device="MP20_LE_DA", labels=LABELS)

            # -------------------------------
            # Load data (MP20_RCS_LA for external test)
            # -------------------------------
            X_mp20_la_rcs, y_mp20_la_rcs, _, _ = get_data(device="MP20_RCS_LA", labels=LABELS)

            # -------------------------------
            # LOAD data (mixed test)
            X_test_mixed, y_test_mixed, _, _ = get_data(device="TEST_ALL", labels=LABELS)
            # -------------------------------

            # -------------------------------
            # 1) PRIMA_LE_DA -> PRIMA_LE_DA (Cross-validation)
            # -------------------------------
            if model == "CNN1D":
                cnn_cv = CNN1D(regularization="l2", learning_rate=LR, l2_lambda=L2_LAMBDA, dropout_rate=DROPOUT)
                y_true_prima, y_pred_prima, _ = cnn_cv.fit_cv(X_prima, y_prima, epochs=EPOCHS, batch_size=BATCHSIZE, n_splits=10)
                metrics_prima = cnn_cv.evaluate(y_true_prima, y_pred_prima, verbose=False)

            elif model == "SVM":
                svm_cv = SVM()
                y_true, y_pred = svm_cv.fit_cv(X_prima, y_prima, n_splits=10)
                metrics_prima = svm_cv.evaluate(y_true, y_pred, verbose=False)

            elif model == "LDA":
                lda_cv = LDA()
                y_true, y_pred = lda_cv.fit_cv(X_prima, y_prima, n_splits=10)
                metrics_prima = lda_cv.evaluate(y_true, y_pred, verbose=False)

            elif model == "RandomForest":
                rf_cv = RandomForest()
                y_true, y_pred = rf_cv.fit_cv(X_prima, y_prima, n_splits=10)
                metrics_prima = rf_cv.evaluate(y_true, y_pred, verbose=False)

            # -------------------------------
            # 2) Train on PRIMA_LE_DA, test on MP20_LE_DA
            # -------------------------------
            if model == "CNN1D":
                cnn_xd = CNN1D(regularization="l2", learning_rate=LR, l2_lambda=L2_LAMBDA, dropout_rate=DROPOUT)
                y_true_mp20, y_pred_mp20, model_xd, _ = cnn_xd.fit_train_val(X=X_prima, y=y_prima, X_val=X_mp20, y_val=y_mp20, epochs=EPOCHS, batch_size=BATCHSIZE)
                metrics_mp20 = cnn_xd.evaluate(y_true_mp20, y_pred_mp20, verbose=False)

            elif model == "SVM":
                svm_xd = SVM()
                y_true_mp20, y_pred_mp20, model_xd, _ = svm_xd.fit_train_val(X=X_prima, y=y_prima, X_val=X_mp20, y_val=y_mp20)
                metrics_mp20 = svm_xd.evaluate(y_true_mp20, y_pred_mp20, verbose=False)

            elif model == "LDA":
                lda_xd = LDA()
                y_true_mp20, y_pred_mp20, model_xd, _ = lda_xd.fit_train_val(X=X_prima, y=y_prima, X_val=X_mp20, y_val=y_mp20)
                metrics_mp20 = lda_xd.evaluate(y_true_mp20, y_pred_mp20, verbose=False)

            elif model == "RandomForest":
                rf_xd = RandomForest()
                y_true_mp20, y_pred_mp20, model_xd, _ = rf_xd.fit_train_val(X=X_prima, y=y_prima, X_val=X_mp20, y_val=y_mp20)
                metrics_mp20 = rf_xd.evaluate(y_true_mp20, y_pred_mp20, verbose=False)

            # -------------------------------
            # 3) Train on PRIMA_LE_DA, test on MP20_LE_DA light anesthesia RCS
            # -------------------------------
            if model == "CNN1D":
                cnn_xd_la = CNN1D(regularization="l2", learning_rate=LR, l2_lambda=L2_LAMBDA, dropout_rate=DROPOUT)
                y_true_mp20_la_rcs, y_pred_mp20_la_rcs, _, _ = cnn_xd_la.fit_train_val(X=X_prima, y=y_prima, X_val=X_mp20_la_rcs, y_val=y_mp20_la_rcs, epochs=EPOCHS, batch_size=BATCHSIZE)
                metrics_mp20_la_rcs = cnn_xd_la.evaluate(y_true_mp20_la_rcs, y_pred_mp20_la_rcs, verbose=False)

            elif model == "SVM":
                svm_xd_la = SVM()
                y_true_mp20_la_rcs, y_pred_mp20_la_rcs, _, _ = svm_xd_la.fit_train_val(X=X_prima, y=y_prima, X_val=X_mp20_la_rcs, y_val=y_mp20_la_rcs)
                metrics_mp20_la_rcs = svm_xd_la.evaluate(y_true_mp20_la_rcs, y_pred_mp20_la_rcs, verbose=False)

            elif model == "LDA":
                lda_xd_la = LDA()
                y_true_mp20_la_rcs, y_pred_mp20_la_rcs, _, _ = lda_xd_la.fit_train_val(X=X_prima, y=y_prima, X_val=X_mp20_la_rcs, y_val=y_mp20_la_rcs)
                metrics_mp20_la_rcs = lda_xd_la.evaluate(y_true_mp20_la_rcs, y_pred_mp20_la_rcs, verbose=False)

            elif model == "RandomForest":
                rf_xd_la = RandomForest()
                y_true_mp20_la_rcs, y_pred_mp20_la_rcs, _, _ = rf_xd_la.fit_train_val(X=X_prima, y=y_prima, X_val=X_mp20_la_rcs, y_val=y_mp20_la_rcs)
                metrics_mp20_la_rcs = rf_xd_la.evaluate(y_true_mp20_la_rcs, y_pred_mp20_la_rcs, verbose=False)

            # -------------------------------
            # 4) Train on PRIMA_LE_DA, test on PRIMA_RCS_DA
            # -------------------------------
            if model == "CNN1D":
                cnn_xd_rcs = CNN1D(regularization="l2", learning_rate=LR, l2_lambda=L2_LAMBDA, dropout_rate=DROPOUT)
                y_true_prima_rcs, y_pred_prima_rcs, _, _ = cnn_xd_rcs.fit_train_val(X=X_prima, y=y_prima, X_val=X_prima_rcs, y_val=y_prima_rcs, epochs=EPOCHS, batch_size=BATCHSIZE)
                metrics_prima_rcs = cnn_xd_rcs.evaluate(y_true_prima_rcs, y_pred_prima_rcs, verbose=False)

            elif model == "SVM":
                svm_xd_rcs = SVM()
                y_true_prima_rcs, y_pred_prima_rcs, _, _ = svm_xd_rcs.fit_train_val(X=X_prima, y=y_prima, X_val=X_prima_rcs, y_val=y_prima_rcs)
                metrics_prima_rcs = svm_xd_rcs.evaluate(y_true_prima_rcs, y_pred_prima_rcs, verbose=False)

            elif model == "LDA":
                lda_xd_rcs = LDA()
                y_true_prima_rcs, y_pred_prima_rcs, _, _ = lda_xd_rcs.fit_train_val(X=X_prima, y=y_prima, X_val=X_prima_rcs, y_val=y_prima_rcs)
                metrics_prima_rcs = lda_xd_rcs.evaluate(y_true_prima_rcs, y_pred_prima_rcs, verbose=False)

            elif model == "RandomForest":
                rf_xd_rcs = RandomForest()
                y_true_prima_rcs, y_pred_prima_rcs, _, _ = rf_xd_rcs.fit_train_val(X=X_prima, y=y_prima, X_val=X_prima_rcs, y_val=y_prima_rcs)
                metrics_prima_rcs = rf_xd_rcs.evaluate(y_true_prima_rcs, y_pred_prima_rcs, verbose=False)

            # -------------------------------
            # 5) Train on PRIMA_LE_DA, test on mixed test set
            # -------------------------------
            if model == "CNN1D":
                cnn_xd_mixed = CNN1D(regularization="l2", learning_rate=LR, l2_lambda=L2_LAMBDA, dropout_rate=DROPOUT)
                y_true_mixed, y_pred_mixed, _, _ = cnn_xd_mixed.fit_train_val(X=X_prima, y=y_prima, X_val=X_test_mixed, y_val=y_test_mixed, epochs=EPOCHS, batch_size=BATCHSIZE)
                metrics_mixed = cnn_xd_mixed.evaluate(y_true_mixed, y_pred_mixed, verbose=False)

            elif model == "SVM":
                svm_xd_mixed = SVM()
                y_true_mixed, y_pred_mixed, _, _ = svm_xd_mixed.fit_train_val(X=X_prima, y=y_prima, X_val=X_test_mixed, y_val=y_test_mixed)
                metrics_mixed = svm_xd_mixed.evaluate(y_true_mixed, y_pred_mixed, verbose=False)

            elif model == "LDA":
                lda_xd_mixed = LDA()
                y_true_mixed, y_pred_mixed, _, _ = lda_xd_mixed.fit_train_val(X=X_prima, y=y_prima, X_val=X_test_mixed, y_val=y_test_mixed)
                metrics_mixed = lda_xd_mixed.evaluate(y_true_mixed, y_pred_mixed, verbose=False)

            elif model == "RandomForest":
                rf_xd_mixed = RandomForest()
                y_true_mixed, y_pred_mixed, _, _ = rf_xd_mixed.fit_train_val(X=X_prima, y=y_prima, X_val=X_test_mixed, y_val=y_test_mixed)
                metrics_mixed = rf_xd_mixed.evaluate(y_true_mixed, y_pred_mixed, verbose=False)
                
            # -------------------------------
            # Save results
            # -------------------------------
            result_row = {
                "model": model,
                "balanced_accuracy_prima": metrics_prima["balanced_accuracy"],
                "f1_prima": metrics_prima["f1"],
                "balanced_accuracy_mp20": metrics_mp20["balanced_accuracy"],
                "f1_mp20": metrics_mp20["f1"],
                "balanced_accuracy_mp20_la_rcs": metrics_mp20_la_rcs["balanced_accuracy"],
                "f1_mp20_la_rcs": metrics_mp20_la_rcs["f1"],
                "balanced_accuracy_prima_rcs": metrics_prima_rcs["balanced_accuracy"],
                "f1_prima_rcs": metrics_prima_rcs["f1"],
                "balanced_accuracy_mixed": metrics_mixed["balanced_accuracy"],
                "f1_mixed": metrics_mixed["f1"],
            }

            df_results = pd.concat(
                [df_results, pd.DataFrame([result_row])],
                ignore_index=True,
            )
            df_results.to_csv(RESULTS_CSV, index=False)

            log(
                f"Model: {model} | "
                f"PRIMA_LE_DA(CV) BalAcc: {metrics_prima['balanced_accuracy']:.4f}, "
                f"F1: {metrics_prima['f1']:.4f} | "
                f"MP20_LE_DA(Test) BalAcc: {metrics_mp20['balanced_accuracy']:.4f}, "
                f"F1: {metrics_mp20['f1']:.4f}"
                f"MP20_RCS_LA(Test) BalAcc: {metrics_mp20_la_rcs['balanced_accuracy']:.4f}, "
                f"F1: {metrics_mp20_la_rcs['f1']:.4f} "
                f"PRIMA_RCS_DA(Test) BalAcc: {metrics_prima_rcs['balanced_accuracy']:.4f}, "
                f"F1: {metrics_prima_rcs['f1']:.4f}"
                f"Mixed(Test) BalAcc: {metrics_mixed['balanced_accuracy']:.4f}, "
                f"F1: {metrics_mixed['f1']:.4f}"
            )
            log(f"Results saved to {RESULTS_CSV}")

        except Exception as e:
            log("ERROR during model comparison:")
            log(str(e))
            log(traceback.format_exc())
            continue

    log("Model comparison finished.")


# ==========================================================
# ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    compare_models()
