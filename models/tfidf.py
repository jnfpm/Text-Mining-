from sklearn.svm import LinearSVC

from .features import build_text_features, build_word2vec_features


class TfidfModel:
    def __init__(self, representation='tfidf', ngram_range=(1, 2), min_df=2,
                 max_df=1.0, max_features=20_000, stop_words=None,
                 sublinear_tf=True, C=1.0, class_weight=None, max_iter=5000,
                 dual='auto',
                 random_state=39, vector_size=100, window=5, min_count=2,
                 word2vec_epochs=5):
        self.representation = representation
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.stop_words = stop_words
        self.sublinear_tf = sublinear_tf
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.dual = dual
        self.random_state = random_state
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.word2vec_epochs = word2vec_epochs

    def build_features(self):
        if self.representation == 'word2vec':
            return build_word2vec_features(
                self.vector_size, self.window, self.min_count,
                self.word2vec_epochs, self.random_state,
            )
        return build_text_features(
            self.representation, self.ngram_range, self.min_df, self.max_df,
            self.max_features, self.stop_words, self.sublinear_tf,
        )

    def build_classifier(self):
        return LinearSVC(C=self.C, class_weight=self.class_weight,
                         max_iter=self.max_iter, dual=self.dual,
                         random_state=self.random_state)

    def parameter_grid(self):
        if self.representation == 'word2vec':
            return {'classifier__C': [0.5, 1.0, 2.0]}
        return {
            'features__ngram_range': [(1, 1), (1, 2)],
            'classifier__C': [0.5, 1.0, 2.0],
        }
