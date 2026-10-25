# file used to store relevant functions that we may need
# In a notebook, install dependencies with:
#!pip install pandas requests python-dotenv unidecode nltk
# and then:
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

import re
import os
import time
from pathlib import Path
import nltk
import pandas as pd
import requests
from dotenv import load_dotenv
from unidecode import unidecode
from nltk.tokenize.treebank import TreebankWordDetokenizer

def get_song_info(song_id):
    '''
    Faz uma chamada(request) à API, dá o id da música e recebe(get) o id, o titulo, o artista, e a data de acordo 
    com a base de dados do Genius
    '''
    load_dotenv()
    token = os.getenv('GENIUS_API_KEY')
    if not token:
        raise ValueError('Set GENIUS_API_KEY before requesting song metadata.')
    r = requests.get(f'https://api.genius.com/songs/{song_id}', #site do genius para fazer calls a API
                     headers={'Authorization': f'Bearer {token}'}, timeout=20) #usa a API key no .env para ser auturizado a fazer a call
    
    r.raise_for_status() # verefica se pedido correu bem
    
    s = r.json()['response']['song'] # transforma a resposta que vem em json em um dict, vai para a informação util do pedido
    
    # nas traduções o primary_artist é o tradutor; o artista original vem em 'translation_of'
    artist = s['primary_artist']['name']
    
    # devolve um dicionario com a id,titulo, artista e year tirado do json
    for rel in s['song_relationships']:
        if rel['relationship_type'] == 'translation_of' and rel['songs']:
            artist = rel['songs'][0]['primary_artist']['name']
    
    return {
        'id': song_id,
        'title': s['title'],
        'artist': artist,
        'release_date': s.get('release_date'),
    }
    
def get_info_batch(ids : pd.DataFrame or list[int],title = False,artist = False,release_date = False) : 
    ''' 
    Recebe uma lista ou dataframe de song_id e transforma em um dataframe com os argumentos que queremos saídos do Genius
    '''
    rows = []
    
    for song_id in ids:
        try:
            rows.append(get_song_info(song_id))
        except requests.RequestException:   # 404 (música apagada), timeout, erro na rede
            rows.append({'id': song_id})    # fica só o id, o resto NaN
        time.sleep(0.25)
    # recebe todos os id e transforma-os em um dataframe
    
    songs = pd.DataFrame(rows).set_index('id')

    wanted = [name for name, keep in [('title', title), ('artist', artist), ('release_date', release_date)] if keep]
    return songs.reindex(columns=wanted)

def artist_from_title(title):
    # Strip common translation suffixes before trying to extract the artist
    t = re.sub(
        r'\s*(english\s+)?translations?\s*$'
        r'|\s*traduc\w*\s*$'           # Spanish/Portuguese: traducción, tradução
        r'|\s*traduction\s*$'          # French
        r'|\s*\(.*?translat\w*\)\s*$', # e.g. "(Official English Translation)"
        '', title, flags=re.I
    )
    parts = re.split(r'\s[-–—]\s', t, maxsplit=1)
    return parts[0].strip() if len(parts) == 2 else None

def fix_translation_artists(df):
    '''
    Recebe um dataframe com as colunas id e title, já só com as linhas cujo
    artista é uma conta da plataforma Genius (e.g. "Genius English Translations",
    "Rap Genius") e devolve um novo dataframe com id e o artista real extraído
    do título. Devolve None quando o título não segue o formato "Artista - Título".
    '''
    result = df[['id']].copy()
    result['artist'] = df['title'].fillna('').map(artist_from_title)

    return result


def summarize_years(frame, ids, split, status):
    years = pd.to_numeric(
        frame.loc[frame['id'].isin(ids), 'year'],
        errors='coerce'
    ).dropna()

    return {
        'split': split,
        'status': status,
        'rows_with_year': len(years),
        'minimum_year': years.min() if not years.empty else None,
        'years_below_1900': int((years < 1900).sum()),
        'maximum_year': years.max() if not years.empty else None,
    }



def padding_lyrics(tokenized_lyrics_list, max_len=None, padding_token="0"):
    """
    Applies padding to a list of tokenized lyrics so they all have the same length.
    """
    # 1. If no len is provided, automatically find the length of the longest song
    if max_len is None:
        max_len = max(len(lyrics) for lyrics in tokenized_lyrics_list)
    
    padded_dataset = []
    
    for lyrics in tokenized_lyrics_list:
        if len(lyrics) < max_len:
            # If the song is shorter, append the padding_token at the END (Post-padding)
            num_padding_needed = max_len - len(lyrics)
            padded_lyrics = lyrics + [padding_token] * num_padding_needed
        else:
            # If the song is longer than max_len, truncate it
            padded_lyrics = lyrics[:max_len]
            
        padded_dataset.append(padded_lyrics)
        
    return padded_dataset


def load_prediction_data(data_dir='data'):
    """Load saved preprocessing outputs, restoring training labels by song ID."""
    from text_quality import normalize_lyrics

    data_dir = Path(data_dir)
    candidates = [
        (data_dir / 'processed/train.csv', data_dir / 'processed/test.csv'),
        (data_dir / 'normalized_x_train.csv', data_dir / 'normalized_x_test.csv'),
        (data_dir / 'train.csv', data_dir / 'test.csv'),
    ]
    for train_path, test_path in candidates:
        if train_path.exists() or test_path.exists():
            if not train_path.exists() or not test_path.exists():
                missing = train_path if not train_path.exists() else test_path
                raise FileNotFoundError(f'Missing matching dataset: {missing}')
            break
    else:
        raise FileNotFoundError(f'No training/test CSV pair found in {data_dir}.')

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    for frame, path in [(train, train_path), (test, test_path)]:
        missing = {'id', 'lyrics'} - set(frame.columns)
        if missing:
            raise ValueError(f'{path} is missing columns: {sorted(missing)}')
        if frame['id'].isna().any() or frame['id'].duplicated().any():
            raise ValueError(f'{path} must contain unique, non-missing song IDs.')

    if 'tag' not in train.columns:
        raw_path = data_dir / 'train.csv'
        labels = pd.read_csv(raw_path, usecols=['id', 'tag'])
        if labels['id'].isna().any() or labels['id'].duplicated().any():
            raise ValueError(f'{raw_path} must contain unique, non-missing song IDs.')
        train = train.merge(labels, on='id', how='left', sort=False,
                            validate='one_to_one', indicator=True)
        if train['_merge'].ne('both').any():
            raise ValueError('Processed training data contains unknown song IDs.')
        train = train.drop(columns='_merge')

    if train['tag'].isna().any() or train['tag'].astype(str).str.strip().eq('').any():
        raise ValueError('Training data contains missing genre labels.')

    if train_path == data_dir / 'train.csv':
        print('No saved preprocessing outputs found; normalizing raw lyrics. '
              'Saved metadata corrections are unavailable; the model uses lyrics only.')
        train['lyrics'] = train['lyrics'].map(normalize_lyrics)
        test['lyrics'] = test['lyrics'].map(normalize_lyrics)
    else:
        print(f'Loaded preprocessed datasets: {train_path} and {test_path}')
    return train, test


def check_for_duplicates(dataset1, dataset2, dataset1_name='train', dataset2_name='test'):
    """Audit exact non-empty normalized lyrics without changing either dataset."""
    from text_quality import normalize_lyrics

    frames = []
    summaries = []
    for name, dataset in [(dataset1_name, dataset1), (dataset2_name, dataset2)]:
        lyrics = dataset['lyrics'].map(normalize_lyrics)
        usable = lyrics.map(lambda text: isinstance(text, str) and bool(text.strip()))
        frame = dataset.loc[usable].copy()
        frame['lyrics'] = lyrics.loc[usable]
        counts = frame['lyrics'].value_counts()
        repeated = counts[counts > 1]
        conflicts = (frame.groupby('lyrics')['tag'].nunique().gt(1).sum()
                     if 'tag' in frame.columns else None)
        summaries.append({
            'dataset': name,
            'rows': len(dataset),
            'non_empty_lyrics': len(frame),
            'duplicate_lyric_groups': len(repeated),
            'duplicate_extra_rows': int((repeated - 1).sum()),
            'conflicting_genre_groups': conflicts,
        })
        frames.append(frame)

    shared = set(frames[0]['lyrics']) & set(frames[1]['lyrics'])
    for summary, frame in zip(summaries, frames):
        summary['shared_lyric_groups'] = len(shared)
        summary['shared_rows'] = int(frame['lyrics'].isin(shared).sum())
    shared_rows = pd.concat([
        frame.loc[frame['lyrics'].isin(shared)].assign(dataset=name)
        for name, frame in zip([dataset1_name, dataset2_name], frames)
    ], ignore_index=True)
    return {'summary': pd.DataFrame(summaries), 'shared_rows': shared_rows}

