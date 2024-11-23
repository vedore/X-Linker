import subprocess
import os
import pandas as pd


try:
    subprocess.check_output('nvidia-smi')
    GPU_AVAILABLE = True
except Exception: # this command not being found can raise quite a few different errors depending on the configuration
    print('No Nvidia GPU in system!')
    GPU_AVAILABLE = False

if GPU_AVAILABLE:
    import cudf
    import cp
    from cuml.model_selection import train_test_split
    # from cuml.metrics import accuracy_score, f1_score, precision_score, recall_score
    from cuml.metrics import accuracy_score 

from src.machine_learning.gpu.ml import LogisticRegressionGPU

class TrainGPU():

    def train(embeddings, clustering_labels):

        print(len(embeddings), len(clustering_labels))

        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, 
            clustering_labels['Labels'], 
            test_size=0.2, 
            random_state=42
            )


        embeddings_x = cudf.DataFrame(X_train.astype('int32'))
        clustering_labels_y = cudf.Series(y_train.astype('float32'))
        
        model = LogisticRegressionGPU.train(embeddings_x, clustering_labels_y).model
        y_pred = model.predict(X_test)
        
        print("Accuracy (Test Set):", accuracy_score(y_test, y_pred))

        """
        print("F1 Score (Test Set):", f1_score(y_test, y_pred, average="weighted"))
        print("Precision (Test Set):", precision_score(y_test, y_pred, average="weighted"))
        print("Recall (Test Set):", recall_score(y_test, y_pred, average="weighted"))
        """
        y_proba = model.predict_proba(X_test)

        # Function to calculate top-k accuracy
        def top_k_accuracy(y_true, y_proba, k=3):
            """
            Computes the top-k accuracy.

            Args:
                y_true (array): True labels.
                y_proba (array): Predicted probabilities (num_samples x num_classes).
                k (int): Number of top predictions to consider.

            Returns:
                float: Top-k accuracy score.
            """
            # Get the top-k predicted class indices
            top_k_preds = cp.argsort(y_proba, axis=1)[:, -k:]
            
            # Check if true labels are in the top-k predictions
            matches = cp.any(top_k_preds == y_true[:, None], axis=1)
            
            # Calculate the top-k accuracy
            return matches.mean()

        # Calculate metrics
        k = 3
        top_k_acc = top_k_accuracy(y_test, y_proba, k=k)
        print(f"Top-{k} Accuracy: {top_k_acc:.2f}")
