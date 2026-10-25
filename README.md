# Text-Mining-

Run `prediction_model.ipynb` with the course Python environment (dependencies in
`requirements.txt`). The first CPU baseline uses the existing normalization and
tokenizer, TF-IDF unigrams/bigrams (up to 20,000 features), and one-vs-rest
logistic regression. It uses the complete labeled dataset, with an approximately
80/20 training/validation split for evaluation, then fits a fresh final model on
all labeled songs for official test predictions.

`preprocessing.py` contains tokenization and optional Word2Vec embeddings;
`model.py` builds and evaluates the pipeline, and `utils.py` loads the data. Existing modules and
notebooks stay at the project root until the planned folder migration.

The loader prefers `data/processed/train.csv` and `test.csv`, then the
`data/normalized_x_train.csv` and `normalized_x_test.csv` exports from
`preprocessing.ipynb`. Missing tags are restored by song ID. Without those saved
outputs, it applies `text_quality.normalize_lyrics` to the raw lyrics; API-derived
artist/year corrections cannot be recovered, and this model uses lyrics only.

The current dataset has one `tag` per song, so this run predicts one category.
The pipeline also accepts multi-hot targets when actual multilabel annotations
are available. Identical tokenized lyrics stay together in an approximately
80/20 training/validation split; TF-IDF is fitted on training lyrics only.
`build_word2vec_model()` provides the original mean-embedding alternative.

Evaluation follows TM05/TM06: accuracy, weighted precision/recall/F1, macro/micro
F1, probability-based ROC AUC, and per-category confusion matrices. Official test
predictions are matched to the sample-submission template by ID; template labels
are never used for evaluation. The submission uses the final model fitted on
all labeled data; its held-out evaluation belongs to the earlier split model.

Checks: `python -m unittest discover -s tests -v`.
