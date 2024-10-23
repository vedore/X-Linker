from pecos.xmc.xtransformer.model import XTransformer
from pecos.utils.featurization.text.preprocess import Preprocessor
from pecos.utils.cluster_util import ClusterChain


"""
    Load a PECOS-EL model from disk.
"""
def load_model(model_dir, clustering_method="pifa"):

    # loads the XTransformer model
    custom_xtf = XTransformer.load(f"{model_dir}/xtransformer")

    # TF-IDF (Term Frequency-Inverse Document Frequency) model
    # Convert text into numerical features
    tfidf_model = Preprocessor.load(f"{model_dir}/tfidf_model")

    # The cluster_chain represents a hierarchical clustering of labels (or instances), which helps in organizing and speeding up the prediction by breaking it down into clusters.
    cluster_chain = ClusterChain.load(f"{model_dir}/cluster_chain_{clustering_method}")

    return custom_xtf, tfidf_model, cluster_chain