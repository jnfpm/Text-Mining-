from sklearn.naive_bayes import MultinomialNB

from .features import TextModel


class MultinomialNaiveBayesModel(TextModel):
    def __init__(self, alpha=1.0, fit_prior=True, **features):
        super().__init__(**features)
        self.alpha = alpha
        self.fit_prior = fit_prior

    def build_classifier(self):
        return MultinomialNB(alpha=self.alpha, fit_prior=self.fit_prior)

    def parameter_grid(self):
        return {
            'features__ngram_range': [(1, 1), (1, 2)],
            'classifier__alpha': [0.1, 0.5, 1.0],
        }
