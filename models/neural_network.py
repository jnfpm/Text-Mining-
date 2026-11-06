import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from sklearn.utils.validation import check_is_fitted


class NeuralNetworkClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, max_tokens=20_000, sequence_length=400, embedding_dim=128,
                 dense_units=64, dropout=0.3, learning_rate=0.001, epochs=20,
                 batch_size=64, validation_split=0.1, patience=None, device='auto',
                 random_state=39, verbose=1):
        self.max_tokens = max_tokens
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.dense_units = dense_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.patience = patience
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def build_network(self, vectorizer, class_count):
        import tensorflow as tf
 
        inputs = tf.keras.Input(shape=(), dtype=tf.string)
        x = vectorizer(inputs)
        x = tf.keras.layers.Embedding(
            input_dim=vectorizer.vocabulary_size(),
            output_dim=self.embedding_dim, mask_zero=True,
        )(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(self.dense_units, activation='relu')(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)
        outputs = tf.keras.layers.Dense(class_count, activation='softmax')(x)
        return tf.keras.Model(inputs, outputs)

    def _select_device(self, tf):
        if self.device not in {'auto', 'cpu', 'gpu'}:
            raise ValueError("device must be 'auto', 'cpu', or 'gpu'.")
        gpus = tf.config.list_physical_devices('GPU')
        if self.device == 'gpu' and not gpus:
            raise RuntimeError('device="gpu" requires a GPU visible to TensorFlow.')
        if self.device != 'cpu' and gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError:
                    pass
            return '/GPU:0'
        return '/CPU:0'

    def fit(self, X, y):
        if not 0 <= self.validation_split < 1:
            raise ValueError('validation_split must be between 0 and 1.')
        try:
            import tensorflow as tf
        except ImportError as error:
            raise ImportError(
                'The neural model requires requirements-neural.txt '
                'or requirements-gpu.txt.'
            ) from error
        texts = np.asarray(X, dtype=str)
        labels = np.asarray(y)
        if texts.ndim != 1 or labels.ndim != 1 or len(texts) != len(labels):
            raise ValueError('Provide one text string and one class label per row.')
        self.label_encoder_ = LabelEncoder().fit(labels)
        encoded = self.label_encoder_.transform(labels)
        self.classes_ = self.label_encoder_.classes_
        if len(self.classes_) < 2:
            raise ValueError('At least two classes are required.')
        self.device_ = self._select_device(tf)
        tf.keras.utils.set_random_seed(self.random_state)
        if self.validation_split:
            train_texts, valid_texts, train_y, valid_y = train_test_split(
                texts, encoded, test_size=self.validation_split,
                random_state=self.random_state, stratify=encoded,
            )
        else:
            train_texts, train_y = texts, encoded
            valid_texts = None
        with tf.device(self.device_):
            self.vectorizer_ = tf.keras.layers.TextVectorization(
                max_tokens=self.max_tokens, standardize=None,
                output_mode='int', output_sequence_length=self.sequence_length,
            )
            vocabulary_data = tf.data.Dataset.from_tensor_slices(
                train_texts.tolist()
            ).batch(self.batch_size)
            self.vectorizer_.adapt(vocabulary_data)
            network = self.build_network(self.vectorizer_, len(self.classes_))
            network.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                loss='sparse_categorical_crossentropy', metrics=['accuracy'],
            )
            train_data = tf.data.Dataset.from_tensor_slices(
                (train_texts.tolist(), train_y)
            ).shuffle(len(train_texts), seed=self.random_state).batch(self.batch_size)
            valid_data = None
            if valid_texts is not None:
                valid_data = tf.data.Dataset.from_tensor_slices(
                    (valid_texts.tolist(), valid_y)
                ).batch(self.batch_size)
            callbacks = []
            if self.patience is not None and valid_data is not None:
                callbacks.append(tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=self.patience,
                    restore_best_weights=True,
                ))
            history = network.fit(
                train_data, validation_data=valid_data, epochs=self.epochs,
                callbacks=callbacks, verbose=self.verbose,
            )
        self.network_ = network
        self.history_ = history.history
        return self

    def predict_proba(self, X):
        check_is_fitted(self, 'network_')
        import tensorflow as tf

        texts = np.asarray(X, dtype=str)
        if texts.ndim != 1:
            raise ValueError('Provide a one-dimensional collection of text strings.')
        data = tf.data.Dataset.from_tensor_slices(texts.tolist()).batch(self.batch_size)
        with tf.device(self.device_):
            return np.asarray(self.network_.predict(data, verbose=0))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class NeuralNetworkModel:
    def __init__(self, max_tokens=20_000, sequence_length=400, embedding_dim=128,
                 dense_units=64, dropout=0.3, learning_rate=0.001, epochs=20,
                 batch_size=64, validation_split=0.1, patience=None, device='auto',
                 random_state=39, verbose=1):
        self.max_tokens = max_tokens
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.dense_units = dense_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.patience = patience
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def build_features(self):
        return FunctionTransformer(validate=False)

    def build_classifier(self):
        return NeuralNetworkClassifier(**vars(self))

    def parameter_grid(self):
        return {
            'classifier__embedding_dim': [64, 128],
            'classifier__dropout': [0.2, 0.4],
        }
