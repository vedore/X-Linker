import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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

        # Train a logistic regression classifier
        matching_model = RandomForestClassifier(max_depth=10, n_estimators=100, random_state=42)
        matching_model.fit(X_train, y_train)

        # Check the model's accuracy
        print("Training accuracy:", matching_model.score(X_train, y_train))
        print("Test accuracy:", matching_model.score(X_test, y_test))

        # Calculate top-3 accuracy on the test set
        print("Top-3 accuracy:", self.top_k_accuracy(matching_model, X_test, y_test, k=3))

    def top_k_accuracy(self, model, X, y, k=3):
        if hasattr(model, "predict_proba"):
            # Get probability predictions for each class
            probs = model.predict_proba(X)
            # Get indices of top-k predictions per sample
            top_k_preds = np.argsort(probs, axis=1)[:, -k:]
            # Check if true label is among the top-k predictions
            top_k_correct = [int(y.iloc[i] in top_k_preds[i]) for i in range(len(y))]
            return np.mean(top_k_correct)
        else:
            raise ValueError("Model does not support top-k accuracy calculation.")

        

    