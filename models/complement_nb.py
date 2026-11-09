from sklearn.naive_bayes import ComplementNB

from .features import TextModel


class ComplementNaiveBayesModel(TextModel):
    def __init__(self, alpha=1.0, fit_prior=True, norm=False, **features):
        super().__init__(**features)
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.norm = norm

    def build_classifier(self):
        return ComplementNB(alpha=self.alpha, fit_prior=self.fit_prior, norm=self.norm)

    def parameter_grid(self):
        return {
            'features__ngram_range': [(1, 1), (1, 2)],
            'classifier__alpha': [0.1, 0.5, 1.0],
        }
