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

        print("Accuracy (Test Set):", accuracy_score(y_test, y_pred))
        print("F1 Score (Test Set):", f1_score(y_test, y_pred, average="weighted"))
        print("Precision (Test Set):", precision_score(y_test, y_pred, average="weighted"))
        print("Recall (Test Set):", recall_score(y_test, y_pred, average="weighted"))

        y_proba = model.predict_proba(X_test)

        """
        # Define top-k
        top_k = 5

        # Get top-k predictions
        top_k_indices = np.argsort(y_proba, axis=1)[:, -top_k:]  # Top-k indices for each instance
        top_k_scores = np.sort(y_proba, axis=1)[:, -top_k:]  # Corresponding scores

        top_k_predictions = [
            {"indices": indices.tolist(), "scores": scores.tolist()}
            for indices, scores in zip(top_k_indices, top_k_scores)
        ]

        for i, pred in enumerate(top_k_predictions):
            print(f"Instance {i}: {pred}")
        """

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


