# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split

from cuML.model_selection import train_test_split

class EntityLinking():

    def __init__(self):

        self.embeddings = None
        self.clustering_df = None

        self.entity_ids = None
        self.clustering_labels = None

    def train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.embeddings, self.clustering_labels, test_size=0.2, random_state=42)

        # Train a logistic regression classifier
        matching_model = LogisticRegression(max_iter=1000)
        matching_model.fit(X_train, y_train)

        # Check the model's accuracy
        print("Training accuracy:", matching_model.score(X_train, y_train))
        print("Test accuracy:", matching_model.score(X_test, y_test))

    
    def prepare_data(self, clustering_df, embeddings):
        self.embeddings = embeddings
        self.clustering_df = clustering_df
        self.entity_ids = clustering_df['EntityID']
        self.clustering_labels = clustering_df['ClusterLabel']

    