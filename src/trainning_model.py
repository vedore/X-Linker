import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# from cuML.model_selection import train_test_split

class TrainningModel():

    def __init__(self):

        self.embeddings = None
        self.clustering_df = None

        self.entity_ids = None
        self.clustering_labels = None
    
    def prepare_data(self, clustering_df, embeddings):
        self.embeddings = embeddings
        self.clustering_df = clustering_df
        self.entity_ids = clustering_df['EntityID']
        self.clustering_labels = clustering_df['ClusterLabel']

    def train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.embeddings, self.clustering_labels, test_size=0.2, random_state=42)

        matching_model = LogisticRegression(max_iter=1000, random_state=42)
        matching_model.fit(X_train, y_train)

        # Check the model's accuracy
        print("Training accuracy:", matching_model.score(X_train, y_train))
        print("Test accuracy:", matching_model.score(X_test, y_test))

        # Predict on the test set
        y_pred = matching_model.predict(X_test)

        print(classification_report(y_test, y_pred))

        # Test top-k accuracy on the test set
        top_k_acc = self.top_k_accuracy(matching_model, X_test, y_test)
        print(f"Top-3 Accuracy: {top_k_acc:.2f}")

    @staticmethod
    def top_k_accuracy(model, X, y, k=3):
        # Get probabilities for each class
        probs = model.predict_proba(X)
        # Get the top-k predictions for each sample
        top_k_preds = np.argsort(probs, axis=1)[:, -k:]
        
        y_array = y.to_numpy()
        
        # Calculate top-k accuracy
        top_k_correct = [int(y_array[i] in top_k_preds[i]) for i in range(len(y_array))]

        return np.mean(top_k_correct)

        

    