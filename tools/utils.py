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
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
    token = os.getenv('GENIUS_API_KEY')
    if not token:
        raise RuntimeError('GENIUS_API_KEY is not configured in the environment or .env file.')

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


def median_years(training_dataset):
    """Return the median numeric release year for each artist in training data."""
    required_columns = {'artist', 'year'}
    missing_columns = required_columns.difference(training_dataset.columns)
    if missing_columns:
        raise ValueError(
            f"training_dataset must contain columns: {', '.join(sorted(missing_columns))}."
        )

    artists = training_dataset['artist'].astype('string').str.strip()
    artists = artists.replace('', pd.NA)
    years = pd.to_numeric(training_dataset['year'], errors='coerce')
    valid_rows = artists.notna() & years.notna()

    return years.loc[valid_rows].groupby(artists.loc[valid_rows]).median().rename(
        'median_year'
    )


def checking_duplicates(
    dataset1,
    dataset2,
    dataset1_name='dataset1',
    dataset2_name='dataset2',
    lyrics_column='lyrics',
):
    """Report repeated lyrics in each dataset and exact matches between them.

    Missing and whitespace-only lyrics are excluded. The returned
    ``within_dataset`` frames contain each duplicated row's ID (when present)
    and lyrics; ``shared_rows`` contains matching IDs and lyrics from both
    datasets.
    """
    for dataset, name in ((dataset1, dataset1_name), (dataset2, dataset2_name)):
        if lyrics_column not in dataset.columns:
            raise ValueError(
                f"{name} must contain a {lyrics_column!r} column to check duplicates."
            )

    def usable_lyrics(dataset):
        lyrics = dataset[lyrics_column]
        return lyrics.notna() & lyrics.astype('string').str.strip().ne('')

    def duplicate_rows(dataset, usable):
        return dataset.loc[
            usable & dataset[lyrics_column].duplicated(keep=False),
            [column for column in ('id', lyrics_column) if column in dataset.columns],
        ].copy()

    usable1 = usable_lyrics(dataset1)
    usable2 = usable_lyrics(dataset2)
    duplicates1 = duplicate_rows(dataset1, usable1)
    duplicates2 = duplicate_rows(dataset2, usable2)
    duplicate_lyrics1 = int(duplicates1[lyrics_column].nunique())
    duplicate_lyrics2 = int(duplicates2[lyrics_column].nunique())

    matches1 = dataset1.loc[usable1, [lyrics_column]].copy()
    matches1['_row1'] = dataset1.index[usable1]
    matches2 = dataset2.loc[usable2, [lyrics_column]].copy()
    matches2['_row2'] = dataset2.index[usable2]
    if 'id' in dataset1.columns:
        matches1[f'{dataset1_name}_id'] = dataset1.loc[usable1, 'id'].to_numpy()
    if 'id' in dataset2.columns:
        matches2[f'{dataset2_name}_id'] = dataset2.loc[usable2, 'id'].to_numpy()

    shared_rows = matches1.merge(
        matches2,
        on=lyrics_column,
        how='inner',
        suffixes=(f'_{dataset1_name}', f'_{dataset2_name}'),
    )

    summary = pd.DataFrame([
        {
            'dataset': dataset1_name,
            'rows_with_lyrics': int(usable1.sum()),
            'unique_duplicate_lyrics': duplicate_lyrics1,
            'duplicate_rows': int(duplicates1.shape[0]),
        },
        {
            'dataset': dataset2_name,
            'rows_with_lyrics': int(usable2.sum()),
            'unique_duplicate_lyrics': duplicate_lyrics2,
            'duplicate_rows': int(duplicates2.shape[0]),
        },
        {
            'dataset': f'{dataset1_name} vs {dataset2_name}',
            'rows_with_lyrics': None,
            'unique_duplicate_lyrics': int(shared_rows[lyrics_column].nunique()),
            'duplicate_rows': int(len(shared_rows)),
        },
    ])

    return {
        'summary': summary,
        'within_dataset': {
            dataset1_name: duplicates1,
            dataset2_name: duplicates2,
        },
        'shared_rows': shared_rows,
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
