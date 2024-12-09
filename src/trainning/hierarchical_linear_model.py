import os
import pickle

import numpy as np

from sklearn.model_selection import train_test_split

from src.machine_learning.cpu.ml import AgglomerativeClusteringCPU, KMeansCPU, LogisticRegressionCPU

class HieararchicalLinearModel():

    def __init__(self, model=None, model_type=None):
        self.model = model
        self.model_type = model_type

    def save(self, linear_model_folder):
        os.makedirs(linear_model_folder, exist_ok=True)
        with open(os.path.join(linear_model_folder, 'linear_model.pkl'), 'wb') as fout:
            pickle.dump({'model': self.model, 'model_type': self.model_type}, fout)

    @classmethod
    def load(cls, linear_model_folder):
        linear_model_path = os.path.join(linear_model_folder, 'vectorizer.pkl')
        assert os.path.exists(linear_model_path), f"{linear_model_path} does not exist"
        with open(linear_model_path, 'rb') as fclu:
            data = pickle.load(fclu)
        return cls(model=data['model'], model_type=data['model_type'])
    
    @classmethod
    # X_data = embeddings, y_data = cluster labels
    def execute_pipeline(cls, X_data, y_data):

        X_train, X_test, y_train, y_test = train_test_split(
            X_data, 
            y_data, 
            test_size=0.2, 
            random_state=42
            )
        
        # Regression Class
        root_model = LogisticRegressionCPU.train(X_train, y_train).model

        y_proba = root_model.predict_proba(X_test)
                
        def get_top_k_indices(y_proba, k):
            top_k_indices = np.argsort(y_proba, axis=1)[:, -k:][:, ::-1]
            # top_k_probabilities = np.take_along_axis(y_proba, top_k_indices, axis=1)
            return top_k_indices

        top_k_indices = get_top_k_indices(y_proba, 2)

        """
            pegar os top 5 clusters de cada row, e fazer os filhos, 
            e com os 5 desses fazer outros 5
            Para testar tentar com 2 

            Fazer 3 niveis, root, child, leaf.
            Começa-se por fazer uma LogisticRegression na root, 
            Depois fazer para cada cluster, outro algoritmo de clustering e 
            para esse clustering realizar outra logistic regression,
            E fazer isto novemente, para a leaf, depois tem se de fazer o predict
            top_k.
            Prever o top_k clusters do root_classifier, 
            Depois para cada top_k cluster, utilizar o respetivo child classifier 
            para prever o top_k desse classifiear, depois faz se novamente, 
            No final combina-se todos os scores de todos os scores para o ultimo
            top_k labels
        """

        def execute_child_pipeline(X_child_data, y_child_data, child_top_k_indices):
            cluster_labels_algorithms = {}
            linear_algorithms = {}

            # Embeddings, Label To Filter
            def get_embeddings_from_cluster_label(X_data, y_data, label):
                return X_data[y_data == label] 
            
            # Going One By One, Revamp the pipeline
            # for each cluster inside top_k_indices
            for child_cluster in np.unique(top_k_indices):
                print('Cluster', child_cluster)
                # Mais tempo, mas menos memoria
                child_filtered_embeddings = get_embeddings_from_cluster_label(X_child_data, y_child_data, child_cluster)

                print('Started Clustering')
                # agglomerative_clustering = AgglomerativeClusteringCPU.train(filtered_embeddings.toarray()) 
                kmeans_clustering = KMeansCPU.train(child_filtered_embeddings) 
                child_cluster_labels = kmeans_clustering.get_labels() 
                # Save The Algorithm   
                cluster_labels_algorithms[child_cluster] = child_cluster_labels

                child_X_train, child_X_test, child_y_train, child_y_test = train_test_split(
                child_filtered_embeddings, 
                child_cluster_labels, 
                test_size=0.2, 
                random_state=42
                )

                print('Started Logistic')
                logistic_regression = LogisticRegressionCPU.train(child_X_train, child_y_train)
                # Save The Algorithm
                linear_algorithms[child_cluster] = logistic_regression.model
            
            return {'cluster_labels': cluster_labels_algorithms, 'linear_models': linear_algorithms}

        samples = {}
        counter = 0
        for top_k_list in top_k_indices:
            samples[counter] = execute_child_pipeline(X_data, y_data, top_k_list)
            counter += 1

        print(samples)

        
                
