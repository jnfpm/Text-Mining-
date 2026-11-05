from sklearn.naive_bayes import ComplementNB

from .features import build_text_features


class ComplementNaiveBayesModel:
    def __init__(self, representation='tfidf', ngram_range=(1, 2), min_df=2,
                 max_df=1.0, max_features=20_000, stop_words=None,
                 sublinear_tf=True, alpha=1.0, fit_prior=True, norm=False):
        self.representation = representation
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.stop_words = stop_words
        self.sublinear_tf = sublinear_tf
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.norm = norm

    def build_features(self):
        if self.representation == 'word2vec':
            raise ValueError('Naive Bayes requires nonnegative count or TF-IDF features.')
        return build_text_features(
            self.representation, self.ngram_range, self.min_df, self.max_df,
            self.max_features, self.stop_words, self.sublinear_tf,
        )

    def build_classifier(self):
        return ComplementNB(alpha=self.alpha, fit_prior=self.fit_prior,
                            norm=self.norm)

    def parameter_grid(self):
        return {
            'features__ngram_range': [(1, 1), (1, 2)],
            'classifier__alpha': [0.1, 0.5, 1.0],
        }
