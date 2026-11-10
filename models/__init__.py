from .complement_nb import ComplementNaiveBayesModel
from .multinomial_nb import MultinomialNaiveBayesModel
# NEURAL NETWORK DISABLED (needs an NVIDIA GPU): uncomment to use it.
# from .neural_network import NeuralNetworkClassifier, NeuralNetworkModel
from .tfidf import TfidfModel


MODELS = {
    'tfidf': TfidfModel,
    'complement_nb': ComplementNaiveBayesModel,
    'multinomial_nb': MultinomialNaiveBayesModel,
    # NEURAL NETWORK DISABLED (needs an NVIDIA GPU): uncomment to use it.
    # 'neural_network': NeuralNetworkModel,
}


def get_model(name, **kwargs):
    if name not in MODELS:
        raise ValueError(f'Unknown model {name!r}; choose from {tuple(MODELS)}.')
    return MODELS[name](**kwargs)
