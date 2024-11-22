import numpy as np

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src.machine_learning.cpu.ml import LogisticRegressionCPU

        
class Train():

    @classmethod
    def train(cls, embeddings, clustering_labels):

        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, 
            clustering_labels, 
            test_size=0.2, 
            random_state=42
            )

        model = LogisticRegressionCPU.train(X_train, y_train).model

        # Check the model's accuracy
        print("Training accuracy:", model.score(X_train, y_train))
        print("Test accuracy:", model.score(X_test, y_test))

        # Predict on the test set
        y_pred = model.predict(X_test)

        print(classification_report(y_test, y_pred))

        # Test top-k accuracy on the test set
        top_k_acc = cls.top_k_accuracy(model, X_test, y_test)
        print(f"Top-3 Accuracy: {top_k_acc:.2f}")        

    @staticmethod
    def top_k_accuracy(model, X, y, k=3):
        probs = model.predict_proba(X)
        # Get the top-k predictions for each sample
        top_k_preds = np.argsort(probs, axis=1)[:, -k:]
        y_array = y.to_numpy()
        # Calculate top-k accuracy
        top_k_correct = [int(y_array[i] in top_k_preds[i]) for i in range(len(y_array))]
        return np.mean(top_k_correct)
