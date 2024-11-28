import subprocess
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import classification_report, f1_score, precision_score, accuracy_score, recall_score
from sklearn.model_selection import train_test_split



try:
    subprocess.check_output('nvidia-smi')
    GPU_AVAILABLE = True
except Exception: # this command not being found can raise quite a few different errors depending on the configuration
    print('No Nvidia GPU in system!')
    GPU_AVAILABLE = False

if GPU_AVAILABLE:
    import cudf
    import cp
    # from cuml.metrics import accuracy_score, f1_score, precision_score, recall_score
    from cuml.metrics import accuracy_score 

from src.machine_learning.gpu.ml import LogisticRegressionGPU

class TrainGPU():

    @classmethod
    def train(cls, embeddings, clustering_labels):

        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, 
            clustering_labels['Labels'], 
            test_size=0.2, 
            random_state=42
            )


        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_train = y_train.to_numpy().astype(np.int32)
        y_test = y_test.to_numpy().astype(np.int32)
        
        model = LogisticRegressionGPU.train(X_train, y_train).model
        cls.save(model, "data/processed/regression")
        y_pred = model.predict(X_test)

        print("Accuracy (Test Set):", accuracy_score(y_test, y_pred))
        print("F1 Score (Test Set):", f1_score(y_test, y_pred, average="weighted"))
        print("Precision (Test Set):", precision_score(y_test, y_pred, average="weighted"))
        print("Recall (Test Set):", recall_score(y_test, y_pred, average="weighted"))

        y_proba = model.predict_proba(X_test)

        def top_k_accuracy(predictions, true_labels, k=1):
            """
            Compute Top-k accuracy.
            
            Parameters:
            - predictions: 2D array of shape (n_samples, n_classes), model scores or probabilities.
            - true_labels: 1D array of shape (n_samples,), true label indices.
            - k: int, Top-k to compute accuracy for.
            
            Returns:
            - float, Top-k accuracy.
            """
            # Get indices of top-k predictions for each sample
            top_k_preds = np.argsort(predictions, axis=1)[:, -k:][:, ::-1]  # Top-k in descending order
            
            # Check if true label is in the top-k predictions
            correct = [true_label in top_k for true_label, top_k in zip(true_labels, top_k_preds)]
            
            # Calculate accuracy
            top_k_accuracy = np.mean(correct)
            return top_k_accuracy
        
        # Compute Top-1 and Top-5 accuracy
        top1_acc = top_k_accuracy(y_proba, y_test, k=1)
        top5_acc = top_k_accuracy(y_proba, y_test, k=5)

        print(f"Top-1 Accuracy: {top1_acc:.2f}")
        print(f"Top-5 Accuracy: {top5_acc:.2f}")
    
    def save(model, regression_folder):
        os.makedirs(regression_folder, exist_ok=True)
        with open(os.path.join(regression_folder, 'regression.pkl'), 'wb') as fout:
            pickle.dump({'model': model, 'model_type': 'regression'}, fout)    