from .complement_nb import ComplementNaiveBayesModel
from .multinomial_nb import MultinomialNaiveBayesModel
from .neural_network import NeuralNetworkClassifier, NeuralNetworkModel
from .tfidf import TfidfModel


MODELS = {
    'tfidf': TfidfModel,
    'complement_nb': ComplementNaiveBayesModel,
    'multinomial_nb': MultinomialNaiveBayesModel,
    'neural_network': NeuralNetworkModel,
}


def get_model(name, **kwargs):
    if name not in MODELS:
        raise ValueError(f'Unknown model {name!r}; choose from {tuple(MODELS)}.')
    return MODELS[name](**kwargs)
