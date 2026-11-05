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
    'neural_network': {
        'max_tokens': 20_000,
        'sequence_length': 400,
        'embedding_dim': 128,
        'dense_units': 64,
        'dropout': 0.3,
        'epochs': 20,
        'batch_size': 64,
        'patience': None,
        'device': 'auto',
    },
}

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
    'preprocessing__remove_performer_names': [True],
    'preprocessing__remove_production_credits': [True],
    'preprocessing__remove_unknown_markers': [True],
    'preprocessing__remove_unbracketed_headers': [True],
    'remove_duplicates': [False, True],
}
