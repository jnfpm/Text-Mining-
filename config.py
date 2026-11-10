MODEL = 'tfidf'

PREPROCESSING = {
    'normalize_unicode': True,
    'repair_mixed_script': False,
    'remove_section_headers': False,
    'remove_performer_names': True,
    'remove_production_credits': True,
    'remove_unknown_markers': True,
    'remove_bracket_symbols': False,
    'remove_sung_brackets': False,
    'remove_unbracketed_headers': True,
    'remove_repeated_lines': False,
    'lowercase': True,
    'strip_punctuation': False,
    'preserve_lines': True,
    'input_fields': 'lyrics',   # 'lyrics' | 'lyrics_artist' | 'lyrics_artist_year' | 'all'
    'n_jobs': -1,               # clean lyrics in parallel on all CPU cores
}

MODEL_PARAMETERS = {
    'tfidf': {
        'representation': 'tfidf',
        'ngram_range': (1, 2),
        'max_features': 20_000,
        'min_df': 2,
        'C': 1.0,
        'class_weight': 'balanced',
    },
    'complement_nb': {
        'representation': 'tfidf',
        'ngram_range': (1, 2),
        'max_features': 20_000,
        'min_df': 2,
        'alpha': 1.0,
    },
    'multinomial_nb': {
        'representation': 'count',
        'ngram_range': (1, 2),
        'max_features': 20_000,
        'min_df': 2,
        'alpha': 1.0,
    },
    # NEURAL NETWORK DISABLED (needs an NVIDIA GPU): uncomment to use it.
    # 'neural_network': {         # used without --halving-search; other settings use the class defaults
    #     'model_name': 'distilroberta-base',
    #     'max_length': 256,
    #     'learning_rate': 3e-5,
    #     'batch_size': 32,
    #     'max_epochs': 3,
    # },
}

# NEURAL NETWORK DISABLED (needs an NVIDIA GPU): uncomment to use it.
# # Neural search (run_gpu.py): 40 random candidates, the best third survive each
# # round and train on 3x more songs. Took 4.7 h on the RTX A5000.
# NEURAL_SEARCH = {
#     'n_candidates': 40,
#     'factor': 3,
#     'scoring': 'accuracy',      # f1_macro and f1_weighted are logged too
#     'output_path': 'data/submission_neural.csv',
# }

# _PREPROCESSING_SPACE = {
#     'preprocessing__input_fields': ['lyrics', 'lyrics_artist', 'lyrics_artist_year', 'all'],
#     'preprocessing__remove_section_headers': [False, True],
#     'preprocessing__remove_repeated_lines': [False, True],
#     'preprocessing__lowercase': [False, True],
#     'preprocessing__strip_punctuation': [False, True],
#     'remove_duplicates': [False, True],
# }
# _TRAINING_SPACE = {
#     'classifier__truncation': ['head', 'head_tail'],
#     'classifier__pooling': ['cls', 'mean'],
#     'classifier__dropout': [0.1, 0.2, 0.3],
#     'classifier__learning_rate': [1.5e-5, 2e-5, 3e-5, 4e-5, 5e-5],
#     'classifier__weight_decay': [0.0, 0.01, 0.05],
#     'classifier__warmup_ratio': [0.06, 0.1],
#     'classifier__batch_size': [16, 32],
#     'classifier__class_weight': [None, 'sqrt', 'balanced'],
#     'classifier__label_smoothing': [0.0, 0.1],
# }
# # roberta-base only at 256 tokens: at 512 it is ~4x slower than the distilled models.
# NEURAL_SEARCH_SPACE = [
#     {**_PREPROCESSING_SPACE, **_TRAINING_SPACE,
#      'classifier__model_name': ['distilroberta-base', 'distilbert-base-uncased'],
#      'classifier__max_length': [256, 384, 512]},
#     {**_PREPROCESSING_SPACE, **_TRAINING_SPACE,
#      'classifier__model_name': ['roberta-base'],
#      'classifier__max_length': [256]},
# ]

RUN = {
    'data_dir': 'data',
    'output_path': 'data/predictions.csv',
    'report_dir': 'reports/runs',
    'validation_size': 0.2,
    'cv_folds': 3,
    'random_state': 39,
    'n_jobs': 1,
    'sample_size': None,
    'remove_duplicates': False,
    'grid_search': False,
}

SEARCH_GRID = {
    'preprocessing__remove_section_headers': [False, True],
    'remove_duplicates': [False, True],
}
