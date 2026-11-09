from sklearn.svm import LinearSVC

from .features import TextModel, build_word2vec_features


class TfidfModel(TextModel):
    """Linear SVM on TF-IDF, counts or mean Word2Vec vectors."""

    def __init__(self, C=1.0, class_weight=None, max_iter=5000, dual='auto', random_state=39,
                 vector_size=100, window=5, min_count=2, word2vec_epochs=5, **features):
        super().__init__(**features)
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
            return build_word2vec_features(self.vector_size, self.window, self.min_count,
                                           self.word2vec_epochs, self.random_state)
        return super().build_features()

    def build_classifier(self):
        return LinearSVC(C=self.C, class_weight=self.class_weight, max_iter=self.max_iter,
                         dual=self.dual, random_state=self.random_state)

    def parameter_grid(self):
        grid = {'classifier__C': [0.5, 1.0, 2.0]}
        if self.representation != 'word2vec':
            grid['features__ngram_range'] = [(1, 1), (1, 2)]
        return grid
