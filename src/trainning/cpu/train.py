import numpy as np
import time
import pickle
import os

from sklearn.metrics import classification_report, f1_score, precision_score, accuracy_score, recall_score
from sklearn.model_selection import train_test_split

from src.machine_learning.cpu.ml import LogisticRegressionCPU

        
class TrainCPU():

    @classmethod
    def train(cls, embeddings, clustering_labels):

        start_time = time.time()

        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, 
            clustering_labels['Labels'], 
            test_size=0.2, 
            random_state=42
            )
        
        y_train = y_train.to_numpy()
        model = LogisticRegressionCPU.train(X_train, y_train).model
        cls.save(model, "data/processed/regression")
        print(time.time() - start_time, 'time in ms')

        # Check the model's accuracy
        print("Training accuracy:", model.score(X_train, y_train))
        print("Test accuracy:", model.score(X_test, y_test))

        # Predict on the test set
        y_pred = model.predict(X_test)

        # print(classification_report(y_test, y_pred, zero_division=0))
        print("Accuracy (Test Set):", accuracy_score(y_test, y_pred))
        print("F1 Score (Test Set):", f1_score(y_test, y_pred, average="weighted"))
        print("Precision (Test Set):", precision_score(y_test, y_pred, average="weighted"))
        print("Recall (Test Set):", recall_score(y_test, y_pred, average="weighted"))

        y_proba = model.predict_proba(X_test)

        # Function to calculate top-k accuracy for CPU
        def top_k_accuracy(y_true, y_proba, k=3):
            """
            Computes the top-k accuracy on CPU.

            Args:
                y_true (array): True labels.
                y_proba (array): Predicted probabilities (num_samples x num_classes).
                k (int): Number of top predictions to consider.

            Returns:
                float: Top-k accuracy score.
            """
            # Convert y_true to a NumPy array if it's a pandas Series
            if not isinstance(y_true, np.ndarray):
                y_true = y_true.to_numpy()

            # Get the top-k predicted class indices
            top_k_preds = np.argsort(y_proba, axis=1)[:, -k:]
            
            # Check if true labels are in the top-k predictions
            matches = np.any(top_k_preds == y_true[:, None], axis=1)
            
            # Calculate the top-k accuracy
            return matches.mean()

        # Calculate metrics
        k = 3
        top_k_acc = top_k_accuracy(y_test, y_proba, k=k)
        print(f"Top-{k} Accuracy: {top_k_acc:.2f}")
     

    def save(model, regression_folder):
        os.makedirs(regression_folder, exist_ok=True)
        with open(os.path.join(regression_folder, 'regression.pkl'), 'wb') as fout:
            pickle.dump({'model': model, 'model_type': 'regression'}, fout)       


